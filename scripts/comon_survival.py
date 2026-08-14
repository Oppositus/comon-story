"""comon_survival.py — БЛОК 1.4: выживаемость стратегий Comon.

План исследования, Блок 1.4. Вход — панель Блока 0 и классы смерти
(comon_death_class.py). Правила Блока 0: дата смерти = ПОСЛЕДНИЙ ТОРГОВЫЙ ДЕНЬ,
«не стартовала» (ноль торговых дней) исключается из знаменателя — это пустые
заготовки, а не проигравшие системы.

Что считаем:
  1. Каплан-Мейер — кривая дожития, медианный срок жизни, доля доживших до 1/3/5 лет
     (живые = цензурированные на дату выкачки; интервалы по Гринвуду);
  2. функция риска (hazard) по месяцам жизни — когда именно умирают;
  3. KM по когортам рождения — растёт ли смертность у поздних когорт;
  4. конкурирующие риски (Aalen–Johansen): раздельные кривые для разорения,
     деградации, заброшенности, ухода на пике. ЭТО ГЛАВНОЕ для нас: «автор потерял
     интерес» таким риском не описывается, а «система систематически поехала» — описывается;
  5. Cox — что предсказывает смерть. Ковариаты меряются за ПЕРВЫЙ ГОД жизни, риск
     считается с конца первого года (иначе immortal time bias). Без followerCount —
     поле у мёртвых обнулено (Блок 0).

Реализации KM / Aalen–Johansen / Cox (Breslow) написаны здесь: lifelines не установлен.

Текстовый вывод. Запуск: python comon_survival.py
С флагом --charts дополнительно рисует графики 1, 2, 31 серии в images/ — стиль в
comon_charts.py. Числа на картинке берутся из тех же массивов, что и таблицы выше.
"""
import gzip
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
LOG = ROOT / "results" / "comon_survival.log"
ASOF = date(2026, 8, 5)             # дата выкачки: живые цензурируются здесь
NPROC = 8
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


# ── Каплан-Мейер ─────────────────────────────────────────────────────────────
def km(t, e):
    """Оценка Каплана-Мейера. Возвращает (времена событий, S(t), стандартная ошибка)."""
    order = np.argsort(t)
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    n = len(t)
    S, var, out_s, out_v = 1.0, 0.0, [], []
    for ti in times:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        if at_risk == 0:
            break
        S *= 1.0 - d / at_risk
        var += d / (at_risk * (at_risk - d)) if at_risk > d else 0.0
        out_s.append(S)
        out_v.append(S * np.sqrt(var))            # Гринвуд
    return times, np.array(out_s), np.array(out_v)


def km_at(times, S, se, x):
    """S(x) и её стандартная ошибка в точке x."""
    if len(times) == 0 or x < times[0]:
        return 1.0, 0.0
    i = int(np.searchsorted(times, x, side="right") - 1)
    return float(S[i]), float(se[i])


def km_median(times, S):
    i = np.where(S <= 0.5)[0]
    return float(times[i[0]]) if len(i) else np.nan


def aalen_johansen(t, e, cause, causes):
    """Кумулятивная функция инцидентности по причинам (конкурирующие риски).

    e = 1 при любой смерти, cause — метка причины. Возвращает {причина: (times, CIF)}.
    """
    order = np.argsort(t)
    t, e, cause = t[order], e[order], np.asarray(cause)[order]
    times = np.unique(t[e == 1])
    S = 1.0
    cif = {c: [] for c in causes}
    acc = {c: 0.0 for c in causes}
    for ti in times:
        at_risk = int((t >= ti).sum())
        if at_risk == 0:
            break
        mask = (t == ti) & (e == 1)
        d_all = int(mask.sum())
        for c in causes:
            d_c = int((mask & (cause == c)).sum())
            acc[c] += S * d_c / at_risk           # S(t−) × хазард причины
            cif[c].append(acc[c])
        S *= 1.0 - d_all / at_risk
    return times, {c: np.array(v) for c, v in cif.items()}


# ── Cox, частичное правдоподобие Бреслоу ─────────────────────────────────────
def cox(X, t, e):
    """Оценка Cox (Breslow). Возвращает (β, se, z, p)."""
    order = np.argsort(-t)                        # по убыванию времени
    X, t, e = X[order], t[order], e[order]
    n, k = X.shape

    def nll(b):
        eta = X @ b
        w = np.exp(eta - eta.max())
        cum = np.cumsum(w)                        # риск-множество: все с временем ≥ t_i
        ll = np.sum(eta[e == 1]) - np.sum(np.log(cum[e == 1]) + eta.max())
        return -ll

    res = optimize.minimize(nll, np.zeros(k), method="BFGS")
    b = res.x
    # 🔴 hess_inv из BFGS — грубое приближение, его SE прыгали между прогонами.
    # Считаем наблюдённую информацию честно: численный гессиан в оптимуме.
    h = 1e-5
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h, h
            H[i, j] = H[j, i] = (nll(b + ei + ej) - nll(b + ei - ej)
                                 - nll(b - ei + ej) + nll(b - ei - ej)) / (4 * h * h)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = b / se
    return b, se, z, 2 * (1 - stats.norm.cdf(np.abs(z)))


def first_year(sid):
    """Метрики ПЕРВОГО ГОДА жизни: доходность, вола, просадка, число торговых дней."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if not s:
        return None
    s = s[::-1]
    d = np.array([x["date"] for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s])
    d0 = date.fromisoformat(d[0])
    keep = np.array([(date.fromisoformat(x) - d0).days <= 365 for x in d])
    r1 = r[keep]
    tr = r1[r1 != 0.0]
    if len(tr) < 20:
        return None
    eq = np.cumprod(1.0 + r1)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"id": sid, "y1_ret": float(eq[-1] - 1.0),
            "y1_vol": float(tr.std(ddof=1) * np.sqrt(252)),
            "y1_dd": dd, "y1_ndays": int(len(tr))}


# ── графики (только при --charts; стиль и сохранение — comon_charts.py) ──────
# --charts-only: нарисовать и выйти, не считая Cox (он читает тысячи файлов
# рядов и нужен лишь таблице 1.4.6). Правка стиля не должна стоить полного прогона.
ONLY = "--charts-only" in sys.argv
CHARTS = ONLY or "--charts" in sys.argv


COX_CACHE = ROOT / "results" / "comon_survival_cox.npz"


def chart_cox(labs, b, se, n, nev):
    """График 31: форест-плот Кокса — что предсказывает смерть стратегии."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=900, bottom=0.30)
    f.subplots_adjust(left=0.30)
    b, se = np.asarray(b, float), np.asarray(se, float)
    hr, lo, hi = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    order = np.argsort(np.abs(b))            # снизу вверх: от слабого к сильному
    ys = np.arange(len(order))
    ax.axvline(1.0, color=ch.INK, lw=1.4, zorder=3)
    shown = set()
    for y, i in zip(ys, order):
        sig = not (lo[i] <= 1.0 <= hi[i])
        color = ch.BLUE if sig else ch.PALE
        key = "sig" if sig else "ns"
        lab = None
        if key not in shown:
            shown.add(key)
            lab = ("интервал не накрывает единицу — признак работает" if sig
                   else "интервал накрывает единицу — связи не видно")
        ax.plot([lo[i], hi[i]], [y, y], color=color, lw=2.6, zorder=4, label=lab)
        ax.plot([hr[i]], [y], marker="o", ms=10, color=color, zorder=5)
        # у признака, чей HR стоит вплотную к единице, подпись сдвигается вбок:
        # по центру её перечёркивает сама опорная вертикаль
        near = abs(hr[i] - 1.0) < 0.05
        ax.annotate(ch.n_(hr[i], 3), xy=(hr[i], y),
                    xytext=(14, 14) if near else (0, 17),
                    textcoords="offset points", fontsize=11, color=color,
                    ha="left" if near else "center", va="bottom",
                    fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
    ax.set_xticklabels([ch.n_(v, 1) for v in (0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)])
    ax.set_xlim(0.6, 1.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([labs[i] for i in order], fontsize=11)
    ax.set_ylim(-0.6, len(ys) - 0.05)        # запас под подпись верхней строки
    ax.grid(axis="y", visible=False)
    # легенда — слева внизу: справа внизу под ней оказывалась подпись самого
    # слабого признака, он стоит у единицы в нижней строке
    ax.legend(loc="upper right", handlelength=2.4, labelspacing=0.55)
    ax.set_xlabel("во сколько раз меняется риск смерти")
    ax.set_title("Смерть предсказывают порог входа и активность, а не доходность")
    ch.note(ax,
            f"Регрессия Кокса на {ch.n_(n)} стратегиях, проживших не меньше года "
            f"(смертей {ch.n_(nev)}). Признаки измерены за ПЕРВЫЙ год жизни, риск "
            f"считается с конца первого года — иначе долгожители «предсказывались» "
            f"бы тем, что долго жили.",
            "Левее единицы — признак продлевает жизнь, правее — укорачивает. "
            "Признаки стандартизованы, поэтому их влияние сравнимо между собой; "
            "шкала логарифмическая, так что равные отношения рисков дают равные "
            "отрезки.",
            "Счётчик подписчиков в модель не входит: у закрытых стратегий это поле "
            "обнулено площадкой. Доходность первого года — единственный признак, "
            "чей интервал накрывает единицу.")
    ch.save(f, 31, "cox-forest")
    print("    " + "; ".join(f"{labs[i]}: HR {hr[i]:.3f} [{lo[i]:.3f}; {hi[i]:.3f}]"
                             for i in order[::-1]), flush=True)


def chart_km(times, S, se, med, n, ndeath, t):
    """График 1 (список §4): кривая дожития Каплана—Мейера с полосой Гринвуда."""
    import comon_charts as ch

    f, ax = ch.fig()
    x = times / 365.25
    lo = np.clip(S - 1.96 * se, 0, 1)
    hi = np.clip(S + 1.96 * se, 0, 1)
    # Систем много → интервал ±0,5–0,8 п.п.: полоса физически тоньше линии.
    # Обмер показал 2–3 px на сторону, а штриховые границы распадались в бахрому
    # вплотную к кривой — грязь без информации. Оставлена одна заливка потемнее,
    # а ширина интервала сказана числами в сноске.
    ax.fill_between(x, lo, hi, step="post", color=ch.BAND, lw=0, zorder=2)
    ax.step(x, S, where="post", color=ch.BLUE, lw=1.8, zorder=3)

    # медиана: горизонталь 50 % до пересечения с кривой
    my = med / 365.25
    ax.plot([0, my], [0.5, 0.5], color=ch.GREY, ls=":", lw=1.2, zorder=2)
    ax.plot([my, my], [0, 0.5], color=ch.GREY, ls=":", lw=1.2, zorder=2)
    ax.annotate(f"половина систем не доживает\nдо {med:.0f} дней",
                xy=(my, 0.5), xytext=(my + 0.55, 0.62), fontsize=11,
                arrowprops=dict(arrowstyle="->", color=ch.GREY, lw=1.0))

    pts = []
    names = {1: "1 год", 3: "3 года", 5: "5 лет"}
    for yr in (1, 3, 5):
        s_, e_ = km_at(times, S, se, yr * 365.25)
        pts.append((yr, s_, e_))
        ax.plot([yr], [s_], marker="o", ms=8, color=ch.BLUE, zorder=4,
                markeredgecolor=ch.SURFACE, markeredgewidth=1.6)
        ch.stack_label(ax, (yr, s_), f"{100*s_:.1f} %".replace(".", ","),
                       names[yr], dx=7, color=ch.BLUE)

    # ось обрезана: до 8 лет доживают единицы, а растянутая до 13,9 года шкала
    # оставляла 44 % панели пустыми и сжимала первые полгода — главный сюжет
    xmax = 8.0
    tail = int((t > xmax * 365.25).sum())
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, int(xmax) + 1))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([i / 10 for i in range(11)])     # деление на 50 % — иначе медиану не считать с оси
    ax.set_xlabel("возраст системы, лет")
    ax.set_ylabel("доля доживших")
    ax.set_title("Сколько живут торгующие системы")
    ch.pct(ax)
    ch.note(ax,
            f"Оценка Каплана—Мейера: {ch.n_(n)} торговавших систем, из них "
            f"закрыты {ch.n_(ndeath)}, остальные учтены как живые на 5 августа "
            f"2026 года.",
            "Светлая полоса — 95 % доверительный интервал. Он почти сливается с "
            "линией, потому что систем много: ±0,8 процентного пункта на годе "
            "и ±0,5 на трёх и пяти годах.",
            f"Ось обрезана на восьми годах: систем старше — {tail} из {ch.n_(n)}.")
    p = ch.save(f, 1, "km-survival")
    print(f"    ось X 0–{xmax:.0f} лет (обрезан хвост до {x.max():.1f}: "
          f"{tail} систем старше 8 лет), "
          f"ось Y 0–100 % с делением на 50 %; медиана {med:.0f} дн; "
          + "; ".join(f"{yr} г = {100*s:.1f} % ±{100*1.96*e:.1f}"
                      for yr, s, e in pts), flush=True)
    return p


def chart_hazard(rows):
    """График 2 (список §4): годовой темп смертности по 13 интервалам жизни."""
    import comon_charts as ch

    f, ax = ch.fig()
    xs, ys = [], []
    for m0, m1, _, _, hy in rows:
        xs.append(max(m0, 0.4))                    # 0 не ложится на лог-ось
        ys.append(hy)
    xs.append(rows[-1][1])
    ys.append(ys[-1])
    ax.step(xs, ys, where="post", color=ch.ORANGE, lw=2.2, zorder=3)

    # подпись первой ступени стоит СПРАВА от вертикального сброса на первом
    # месяце: слева она оказывалась ровно на линии, и та перечёркивала текст
    ch.stack_label(ax, (1.15, rows[0][4]), f"{100*rows[0][4]:.0f} %",
                   "в первый месяц жизни", dx=6, up=False, color=ch.ORANGE)
    # и последняя — СПРАВА от сброса на 84-м месяце, на самом плато: слева её
    # перечёркивал вертикальный участок ступени (поймано проверкой в save)
    last = rows[-1]
    ch.stack_label(ax, (95.0, last[4]), f"{100*last[4]:.0f} %",
                   "после семи лет", dx=6, color=ch.ORANGE)

    ax.set_xscale("log")
    # деление на половине месяца — иначе левая седьмая часть панели с данными
    # остаётся без разметки; правый край шире последнего деления, чтобы подпись
    # «20 лет» не свисала за рамку
    ticks = [0.5, 1, 3, 6, 12, 24, 60, 120, 240]
    names = ["2 недели", "1 мес.", "3 мес.", "6 мес.", "1 год", "2 года", "5 лет",
             "10 лет", "20 лет"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(names)
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.4, 300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("возраст системы (шкала неравномерная: первые месяцы показаны крупнее)")
    ax.set_ylabel("вероятность закрыться в течение года")
    ax.set_title("Чем дольше система прожила, тем реже умирает")
    ch.pct(ax)
    ch.note(ax, "Годовой темп смертности внутри каждого из 13 интервалов возраста:",
            "доля закрывшихся за интервал, приведённая к году.")
    p = ch.save(f, 2, "hazard-by-age")
    print("    ось X 0,4–240 мес (лог), ось Y 0–100 %; ступени: "
          + "; ".join(f"{m0}–{m1} мес {100*hy:.0f} %" for m0, m1, _, _, hy in rows),
          flush=True)
    return p


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    deaths = pd.read_csv(DIR / "deaths.csv.gz", index_col="id", low_memory=False)
    df = panel.join(deaths[["mech", "fin", "death_date"]], how="left")

    # ── время жизни: рождение → последний ТОРГОВЫЙ день (или цензура на дату выкачки)
    born = pd.to_datetime(df["created_at"], errors="coerce")
    last_tr = pd.to_datetime(df["last_trade"], errors="coerce")
    alive = df["is_live"] == True                                       # noqa: E712
    started = df["n_trade"].fillna(0) > 0
    end = last_tr.where(~alive, pd.Timestamp(ASOF))
    life = (end - born).dt.days
    ok = started & born.notna() & life.notna() & (life >= 0)
    t = life[ok].to_numpy(float)
    e = (~alive[ok]).to_numpy(int)

    say(f"популяция: {len(df)} систем")
    say(f"исключены как «не стартовала» (0 торговых дней): {int((~started).sum())} "
        f"({100*(~started).mean():.1f} %)")
    say(f"в анализе выживаемости: {int(ok.sum())} — событий (смертей) {int(e.sum())}, "
        f"цензурировано (живые) {int((e == 0).sum())}")
    say(f"дата цензурирования: {ASOF}")

    say()
    say("=" * 92)
    say("1.4.1 КАПЛАН-МЕЙЕР: сколько живут торгующие системы")
    say("=" * 92)
    times, S, se = km(t, e)
    med = km_median(times, S)
    say(f"медианный срок жизни: {med:.0f} дней ({med/365.25:.2f} года)")
    say(f'{"дожили до":<14} {"доля":>8} {"95 % ДИ":>18} {"в живых из 1000":>17}')
    for yr in (0.5, 1, 2, 3, 5, 7, 10):
        s_, e_ = km_at(times, S, se, yr * 365.25)
        lo, hi = max(0, s_ - 1.96 * e_), min(1, s_ + 1.96 * e_)
        say(f'{yr:>6.1f} года   {100*s_:>7.1f}% {f"{100*lo:.1f}–{100*hi:.1f}%":>18} '
            f'{1000*s_:>17.0f}')

    say()
    say("=" * 92)
    say("1.4.2 ФУНКЦИЯ РИСКА: когда именно умирают (по месяцам жизни)")
    say("=" * 92)
    say(f'{"месяц жизни":<14} {"под риском":>11} {"умерло":>8} {"риск смерти":>13} '
        f'{"в год":>9}')
    hz_rows = []
    for m0 in (0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84):
        m1 = {0: 1, 1: 2, 2: 3, 3: 6, 6: 9, 9: 12, 12: 18, 18: 24, 24: 36, 36: 48,
              48: 60, 60: 84, 84: 240}[m0]
        a, b = m0 * 30.44, m1 * 30.44
        at_risk = int((t >= a).sum())
        d = int(((t >= a) & (t < b) & (e == 1)).sum())
        if at_risk == 0:
            continue
        h = d / at_risk
        hy = 1 - (1 - h) ** (365.25 / (b - a))
        hz_rows.append((m0, m1, at_risk, d, hy))
        say(f'{f"{m0}–{m1}":<14} {at_risk:>11,} {d:>8,} {100*h:>12.1f}% '
            f'{100*hy:>8.1f}%')
    say("«риск смерти» — доля умерших за интервал среди доживших до его начала;")
    say("«в год» — тот же риск, приведённый к годовому темпу.")

    if CHARTS:
        say()
        say("── графики 1 и 2 ─────────────────────────────────────────────────")
        chart_km(times, S, se, med, int(ok.sum()), int(e.sum()), t)
        chart_hazard(hz_rows)
        if ONLY:
            # Cox считается минуты (Pool по тысячам рядов) — рисуем его по
            # сохранённым коэффициентам, если они есть от полного прогона.
            if COX_CACHE.exists():
                d = np.load(COX_CACHE, allow_pickle=False)
                chart_cox([str(x) for x in d["labs"]], d["b"], d["se"],
                          int(d["n"]), int(d["nev"]))
            say("--charts-only: остальные разделы и лог пропущены")
            return

    say()
    say("=" * 92)
    say("1.4.3 КОГОРТЫ РОЖДЕНИЯ: меняется ли смертность со временем")
    say("=" * 92)
    coh = born[ok].dt.year
    say(f'{"когорта":<10} {"систем":>7} {"медиана жизни, дн":>18} {"дожили 1 год":>14} '
        f'{"дожили 3 года":>15}')
    for y in sorted(coh.dropna().unique()):
        m = (coh == y).to_numpy()
        if m.sum() < 30:
            continue
        ti, Si, sei = km(t[m], e[m])
        s1, _ = km_at(ti, Si, sei, 365.25)
        s3, _ = km_at(ti, Si, sei, 3 * 365.25)
        # когорта незрелая, если самая старая её система не прожила столько наблюдения
        mx = t[m].max()
        f1 = "" if mx >= 365.25 else "*"
        f3 = "" if mx >= 3 * 365.25 else "*"
        say(f'{int(y):<10} {int(m.sum()):>7,} {km_median(ti, Si):>18.0f} '
            f'{f"{100*s1:.1f}%{f1}":>14} {f"{100*s3:.1f}%{f3}":>15}')
    say("* — когорта не наблюдалась столько времени: оценка неполная, читать нельзя.")
    say("Поздние когорты моложе — KM учитывает это цензурированием, но ДИ у них шире.")

    say()
    say("=" * 92)
    say("1.4.4 КОНКУРИРУЮЩИЕ РИСКИ (Aalen–Johansen): от ЧЕГО умирают")
    say("=" * 92)
    cause = df.loc[ok, "mech"].fillna("прочее").to_numpy()
    causes = ["заброшена", "смерть у пика", "деградация", "обычная просадка",
              "недоказано", "недостаточно истории"]
    cause = np.where(np.isin(cause, causes), cause, "недостаточно истории")
    ti, cifs = aalen_johansen(t, e, cause, causes)
    say(f'{"причина":<24}' + "".join(f'{f"{y} год":>11}' for y in (1, 3, 5, 10))
        + f'{"систем":>9}')
    say("-" * 92)
    for c in causes:
        cif = cifs[c]
        vals = []
        for yr in (1, 3, 5, 10):
            i = int(np.searchsorted(ti, yr * 365.25, side="right") - 1)
            vals.append(cif[i] if i >= 0 else 0.0)
        say(f'{c:<24}' + "".join(f'{100*v:>10.1f}%' for v in vals)
            + f'{int((cause == c).sum()):>9,}')
    say("-" * 92)
    tot = sum(cifs[c][int(np.searchsorted(ti, 10 * 365.25, side="right") - 1)]
              for c in causes)
    say(f'{"ВСЕ ПРИЧИНЫ":<24}' + " " * 33 + f'{100*tot:>10.1f}%')
    say("Числа — накопленная вероятность умереть ИМЕННО ОТ ЭТОЙ причины к сроку X,")
    say("с учётом того, что другие причины могут забрать систему раньше. Суммируются.")
    say("🔴 «недостаточно истории» — не причина, а признание нехватки мощности: у системы")
    say("нет 250 торговых дней до пика, чтобы отличить свой риск от поломки. Это половина")
    say("смертей, поэтому ниже — та же декомпозиция, где для таких систем причина берётся")
    say("по ФИНАНСОВОМУ ИСХОДУ (ось I), который известен всегда.")

    say()
    say("=" * 92)
    say("1.4.4б ОБЪЕДИНЁННАЯ ПРИЧИНА (ось II где есть вердикт, иначе ось I)")
    say("=" * 92)
    fin = df.loc[ok, "fin"].fillna("нет данных").to_numpy()
    mech = df.loc[ok, "mech"].fillna("прочее").to_numpy()
    cause2 = np.where(
        fin == "разорение", "разорение",
        np.where(mech == "деградация", "деградация",
                 np.where(mech == "заброшена", "заброшена",
                          np.where(fin == "прибыль", "ушла в прибыли",
                                   "убыток без вердикта"))))
    causes2 = ["разорение", "деградация", "убыток без вердикта", "заброшена",
               "ушла в прибыли"]
    ti2, cifs2 = aalen_johansen(t, e, cause2, causes2)
    say(f'{"причина":<24}' + "".join(f'{f"{y} год":>11}' for y in (1, 3, 5, 10))
        + f'{"систем":>9}')
    say("-" * 92)
    for c in causes2:
        cif = cifs2[c]
        vals = []
        for yr in (1, 3, 5, 10):
            i = int(np.searchsorted(ti2, yr * 365.25, side="right") - 1)
            vals.append(cif[i] if i >= 0 else 0.0)
        say(f'{c:<24}' + "".join(f'{100*v:>10.1f}%' for v in vals)
            + f'{int((cause2 == c).sum()):>9,}')

    say()
    say("=" * 92)
    say("1.4.5 ЧТО ПЕРЕНОСИМО НА НАС: смертность без «автор потерял интерес»")
    say("=" * 92)
    own = ["разорение", "деградация", "убыток без вердикта"]
    say(f'{"срок":<8} {"рыночная причина":>18} {"решение автора":>17} '
        f'{"в т.ч. разорение":>18} {"в т.ч. деградация":>19}')
    for yr in (1, 3, 5, 10):
        i = int(np.searchsorted(ti2, yr * 365.25, side="right") - 1)
        a = sum(cifs2[c][i] for c in own)
        b = sum(cifs2[c][i] for c in ("заброшена", "ушла в прибыли"))
        say(f'{yr:>4} год {100*a:>17.1f}% {100*b:>16.1f}% '
            f'{100*cifs2["разорение"][i]:>17.1f}% {100*cifs2["деградация"][i]:>18.1f}%')
    say("Рыночные причины — то, что может случиться с работающей машиной; решение")
    say("автора (заброшена / ушёл на пике) таким риском не описывается.")

    say()
    say("=" * 92)
    say("1.4.6 COX: что предсказывает смерть (ковариаты — первый год жизни)")
    say("=" * 92)
    cand = df.index[ok & (life >= 365)].tolist()
    with Pool(NPROC) as pool:
        fy = [x for x in pool.imap_unordered(first_year, cand, chunksize=100) if x]
    F = pd.DataFrame(fy).set_index("id")
    sub = F.join(df[["min_sum", "tariff_type", "is_live"]], how="inner")
    sub["t"] = (life.reindex(sub.index) - 365).clip(lower=1)     # риск с конца 1-го года
    sub["e"] = (~alive.reindex(sub.index)).astype(int)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["y1_ret", "y1_vol", "y1_dd", "y1_ndays", "min_sum", "t"])
    sub = sub[sub["min_sum"] > 0]
    cols = ["y1_ret", "y1_vol", "y1_ddepth", "y1_ndays", "log_minsum"]
    sub["log_minsum"] = np.log10(sub["min_sum"])
    sub["y1_ddepth"] = -sub["y1_dd"]        # ГЛУБИНА просадки: больше = хуже
    X = sub[cols].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)                               # z-скоры: β сравнимы
    b, se, z, p = cox(X, sub["t"].to_numpy(float), sub["e"].to_numpy(int))
    say(f"выборка: {len(sub)} систем, проживших ≥ 1 года (событий {int(sub['e'].sum())})")
    say(f'{"признак первого года":<28} {"β":>8} {"HR на 1 σ":>11} {"z":>8} {"p":>9}')
    say("-" * 68)
    names = {"y1_ret": "доходность за 1-й год", "y1_vol": "волатильность 1-го года",
             "y1_ddepth": "ГЛУБИНА просадки 1-го года",
             "y1_ndays": "торговых дней в 1-й год",
             "log_minsum": "log10(порог входа minSum)"}
    for i, c in enumerate(cols):
        say(f'{names[c]:<28} {b[i]:>8.3f} {np.exp(b[i]):>11.3f} {z[i]:>8.2f} '
            f'{p[i]:>9.2e}')
    say("HR на 1 σ — во сколько раз меняется мгновенный риск смерти при росте признака")
    say("на одно стандартное отклонение. HR > 1 — умирают чаще, < 1 — живут дольше.")

    if CHARTS:
        short = {"y1_ret": "доходность\nпервого года",
                 "y1_vol": "волатильность\nпервого года",
                 "y1_ddepth": "глубина просадки\nпервого года",
                 "y1_ndays": "торговых дней\nв первый год",
                 "log_minsum": "порог входа\n(логарифм)"}
        np.savez(COX_CACHE, labs=np.array([short[c] for c in cols]), b=b, se=se,
                 n=len(sub), nev=int(sub["e"].sum()))
        say()
        say("── график 31 ────────────────────────────────────────────────────")
        chart_cox([short[c] for c in cols], b, se, len(sub), int(sub["e"].sum()))

    LOG.write_text("\n".join(_lines) + "\n")
    say(f"\nлог: {LOG}")


if __name__ == "__main__":
    main()
