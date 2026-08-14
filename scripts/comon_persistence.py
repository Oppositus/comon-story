"""comon_persistence.py — БЛОК 3: персистентность результата.

План исследования, Блок 3. Главный вопрос для нас — НЕ «умеют ли
торговать на Comon», а СКОЛЬКО ФОРВАРДНЫХ МЕСЯЦЕВ ВООБЩЕ ИНФОРМАТИВНЫ на этом рынке.
Форвардом принято судить о качестве системы; здесь измеряется, с какого горизонта
форвард перестаёт быть шумом.

Дизайн — классический Carhart (квинтили формирования → период удержания), с тремя
поправками под специфику площадки:

🔴 СМЕРТЬ — ИСХОД, А НЕ ВЫБЫВАНИЕ. Исключить умерших в периоде удержания значит
измерить персистентность среди выживших, то есть переоткрыть survivorship bias,
который Блок 1 только что измерил. Здесь доходность после смерти = 0 (капитал вышел
в кэш) — экономически это ровно то, что получает подписчик.

🔴 ОКНА ПЕРЕКРЫВАЮТСЯ. Сетка дат формирования полугодовая, окна до 3 лет → соседние
наблюдения одной системы почти те же данные. Пулить их в одну регрессию нельзя
(эффективное N в разы меньше номинального). Считаем статистику ВНУТРИ когорты (одна
дата формирования = одна кросс-секция), затем усредняем когортные оценки — Fama-MacBeth.

🔴 ВИТРИННЫЕ ПОЛЯ НЕ ИМЕЮТ ИСТОРИИ. rating/followers в карточке — снимок на момент
выкачки. Проверить «предсказывал ли рейтинг будущее» нельзя в принципе: на дату
формирования его значение неизвестно. Поэтому раздел 3.4 честно ограничен кросс-секцией
(что рейтинг отражает СЕЙЧАС), а предиктивная часть проверяется на восстановимом из
ряда аналоге витринной метрики — доходности за последние 365 дней.

Разделы:
  3.1 матрица ранговых корреляций предиктор × горизонт удержания;
  3.2 квинтили формирования → доходность, Sharpe и СМЕРТНОСТЬ в периоде удержания;
  3.3 test-retest reliability по длине окна → сколько форвардных лет информативны;
  3.4 витринный рейтинг и подписчики: что они отражают (кросс-секция);
  3.5 «умные ли деньги»: за чем идут подписки — за качеством или за недавним ростом;
  3.6 что это значит для длины форварда.

Текстовый вывод. Запуск: python comon_persistence.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "persistence.csv.gz"
NPROC = 8

ASOF = date(2026, 8, 5)
FORM = [0.5, 1.0, 2.0, 3.0]          # длины окна формирования, годы
HOLD = [0.5, 1.0, 2.0, 3.0]          # длины окна удержания, годы
EDGE_DAYS = 30                       # допуск на «ряд покрывает окно целиком», дни
DEAD_DAYS = 90                       # нет сделок столько дней -> система мертва (Блок 1)
MIN_TRADE_FRAC = 0.10                # активность в окне формирования (как в Блоке 2)
MIN_TRADE_ABS = 30                   # торговых дней в окне формирования — минимум
MIN_COHORT = 30                      # систем в когорте, иначе кросс-секция не считается
NQ = 5                               # число квинтилей
REF_SHARPE = 1.5
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def _grid():
    """Полугодовая сетка дат формирования (ordinal)."""
    out = []
    for y in range(2012, 2027):
        for m in (1, 7):
            d = date(y, m, 1)
            if d < ASOF:
                out.append(d)
    return out


T0S = _grid()


def _win(d, r, lo, hi):
    """Срез ряда по календарному окну [lo, hi) в ordinal-днях."""
    i0, i1 = np.searchsorted(d, lo), np.searchsorted(d, hi)
    return d[i0:i1], r[i0:i1]


def _sh(dd, rr):
    """Sharpe арифметический (mean/std·√freq) + вола, по фактической частоте."""
    if len(rr) < 20:
        return np.nan, np.nan
    sd = rr.std(ddof=1)
    if sd <= 0:
        return np.nan, np.nan
    yrs = max((dd[-1] - dd[0]) / 365.25, 1e-9)
    freq = len(rr) / yrs
    return float(rr.mean() / sd * np.sqrt(freq)), float(sd * np.sqrt(freq))


def rows(sid):
    """Все (система × дата формирования × длина окна) наблюдения одной системы."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return []
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if len(s) < 60:
        return []
    s = s[::-1]
    d = np.array([date.fromisoformat(x["date"]).toordinal() for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s], dtype=np.float64)
    d0, dN = int(d[0]), int(d[-1])
    out = []
    for t0d in T0S:
        t0 = t0d.toordinal()
        for F in FORM:
            lo = t0 - int(round(F * 365.25))
            # ряд обязан покрывать окно формирования целиком (иначе «стаж» фиктивный)
            if d0 > lo + EDGE_DAYS or dN < t0 - EDGE_DAYS:
                continue
            dd, rr = _win(d, r, lo, t0)
            ntr = int((rr != 0).sum())
            if len(rr) < 20 or ntr < MIN_TRADE_ABS or ntr / len(rr) < MIN_TRADE_FRAC:
                continue
            sh, vol = _sh(dd, rr)
            if not np.isfinite(sh):
                continue
            eq = np.cumprod(1.0 + rr)
            peak = np.maximum.accumulate(eq)
            rec = {"id": sid, "t0": t0d.isoformat(), "F": F,
                   "f_sh": sh, "f_vol": vol,
                   "f_ret": float(eq[-1] - 1.0),                 # витринная метрика
                   "f_dd": float(np.min(eq / peak - 1.0)),
                   "f_ntr": ntr, "f_age": (t0 - d0) / 365.25}
            for H in HOLD:
                hi = t0 + int(round(H * 365.25))
                if hi > ASOF.toordinal():
                    continue
                dh, rh = _win(d, r, t0, hi)
                # доходность за окно удержания: дни после смерти отсутствуют в ряде,
                # что эквивалентно нулевой доходности — деньги подписчика в кэше
                k = f"h{H:g}"
                rec[f"{k}_ret"] = float(np.prod(1.0 + rh) - 1.0) if len(rh) else 0.0
                tr = dh[rh != 0]
                # жива на конец окна = была сделка в последние DEAD_DAYS дней окна
                rec[f"{k}_alive"] = bool(len(tr) and tr[-1] >= hi - DEAD_DAYS)
                # ряд покрывает окно целиком -> Sharpe окна считать корректно
                full = len(dh) > 20 and dh[-1] >= hi - EDGE_DAYS
                rec[f"{k}_sh"] = _sh(dh, rh)[0] if full else np.nan
            out.append(rec)
    return out


def fm(g, xcol, ycol, method="spearman", mincoh=MIN_COHORT):
    """Fama-MacBeth: статистика внутри когорты, затем среднее по когортам.

    Возвращает (среднее, t, число когорт, медианный размер когорты).
    """
    vals, sizes = [], []
    for _, sub in g.groupby("t0"):
        s = sub[[xcol, ycol]].dropna()
        if len(s) < mincoh:
            continue
        if method == "spearman":
            v = stats.spearmanr(s[xcol], s[ycol]).statistic
        else:
            v = np.corrcoef(s[xcol], s[ycol])[0, 1]
        if np.isfinite(v):
            vals.append(float(v))
            sizes.append(len(s))
    if len(vals) < 3:
        return np.nan, np.nan, len(vals), 0
    a = np.array(vals)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    return float(a.mean()), float(t), len(a), int(np.median(sizes))


CHARTS = "--charts" in sys.argv


def chart_quintiles(rows):
    """График 7: что получил купивший «лучших» — доход и смертность по квинтилям."""
    import comon_charts as ch

    f, axes = ch.fig(h_px=1120, bottom=0.22, nrows=2, sharex=True)
    a0, a1 = axes
    labs = [r[0] for r in rows]
    ret = [r[1] for r in rows]
    dead = [r[2] for r in rows]
    xs = np.arange(len(labs))
    cols = [ch.RED] + [ch.BLUE] * (len(labs) - 2) + [ch.GREEN]

    a0.axhline(0, color=ch.PALE, lw=1.2, zorder=1)
    a0.bar(xs, ret, width=0.6, color=cols, zorder=3)
    # запас сверху и снизу: иначе подписи крайних столбцов выходили за рамку
    a0.set_ylim(min(ret) - 0.05, max(ret) * 1.35)
    for i, v in enumerate(ret):
        a0.annotate(f"{100*v:+.1f} %".replace(".", ","), xy=(i, v),
                    xytext=(0, 7 if v >= 0 else -8), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=12,
                    fontweight="bold")
    ch.pct(a0)
    a0.set_ylabel("доходность за два года")
    a0.set_title("Отбор по прошлому работает — но даёт половину обещанного")

    a1.bar(xs, dead, width=0.6, color=cols, zorder=3)
    for i, v in enumerate(dead):
        a1.annotate(f"{100*v:.1f} %".replace(".", ","), xy=(i, v), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=12,
                    fontweight="bold")
    ch.pct(a1)
    a1.set_ylim(0, max(dead) * 1.22)
    a1.set_ylabel("умерло за два года")
    a1.set_xticks(xs)
    a1.set_xticklabels(labs)
    a1.set_xlabel("группы по Sharpe за предыдущие два года: от худших к лучшим")
    ch.note(a1,
            "Стратегии делятся на пять равных групп по прошлому Sharpe, затем "
            "измеряется, что они принесли следующие два года.",
            "Разница между крайними группами по смертности надёжнее разницы по "
            "доходности — и практически важнее: умершая стратегия бьёт по счёту "
            "сильнее, чем слабая.")
    p = ch.save(f, 7, "persistence-quintiles")
    print("    " + "; ".join(f"{l.replace(chr(10), ' ')}: {100*r:+.1f} % / "
                             f"умерло {100*d:.1f} %" for l, r, d in rows), flush=True)
    return p


def chart_rho_curves(curves):
    """График 8: сколько прошлого содержится в будущем — по подвыборкам."""
    import comon_charts as ch

    f, ax = ch.fig()
    cols = [ch.BLUE, ch.AQUA, ch.ORANGE, ch.VIOLET]
    mks = ["o", "s", "^", "D"]
    for (lab, xs, ys), c, mk in zip(curves, cols, mks):
        ax.plot(xs, ys, color=c, lw=2.4, marker=mk, ms=8, zorder=3, label=lab)
    ax.axhline(0, color=ch.PALE, lw=1.2, zorder=1)
    ax.legend(loc="upper left", handlelength=3.0, labelspacing=0.6)
    ax.set_xlabel("длина отрезка истории, лет")
    ax.set_ylabel("насколько прошлое связано с будущим")
    ax.set_title("Мусор от не-мусора прошлое отличает; хороших между собой — почти нет")
    ch.note(ax,
            "Связь Sharpe двух соседних непересекающихся окон одинаковой длины. "
            "Чем выше, тем больше в наблюдаемом результате настоящего качества, а не "
            "случая.",
            "Линии расходятся: по всей популяции связь растёт с длиной окна, а внутри "
            "уже приличных стратегий — падает.")
    p = ch.save(f, 8, "persistence-rho-curves")
    for lab, xs, ys in curves:
        print(f"    {lab}: " + ", ".join(f"T={a:g}г {b:+.3f}" for a, b in zip(xs, ys)),
              flush=True)
    return p


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    ids = panel.index[panel["n_pts"].notna()].tolist()
    say(f"систем с рядом: {len(ids):,}")
    say(f"сетка дат формирования: {len(T0S)} шт, {T0S[0]} … {T0S[-1]} (шаг 6 мес)")
    with Pool(NPROC) as pool:
        acc = []
        for chunk in pool.imap_unordered(rows, ids, chunksize=200):
            acc.extend(chunk)
    P = pd.DataFrame(acc)
    P.to_csv(OUT, index=False, compression="gzip")
    say(f"наблюдений (система × дата × длина окна): {len(P):,}, "
        f"уникальных систем: {P['id'].nunique():,}")
    say()
    say("Фильтр окна формирования: ряд покрывает окно целиком (±30 дней), "
        f"≥ {MIN_TRADE_ABS} торговых дней, активность ≥ {MIN_TRADE_FRAC:.0%}.")
    say("Фильтр по ИЗМЕРИМОСТИ, не по результату — как в Блоке 2.")

    # ── 3.1 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("3.1 ПРЕДСКАЗЫВАЕТ ЛИ ПРОШЛОЕ БУДУЩЕЕ: ранговая корреляция предиктор × горизонт")
    say("=" * 104)
    say("Spearman ρ внутри каждой когорты (одна дата формирования), затем среднее по")
    say("когортам ± t (Fama-MacBeth). Цель — доходность окна удержания, дни после")
    say("смерти = 0 доходности.")
    say()
    preds = [("f_sh", "Sharpe окна"), ("f_ret", "доходность окна (витрина)"),
             ("f_vol", "волатильность"), ("f_dd", "макс. просадка окна")]
    for suf, tgt in (("_ret", "ДОХОДНОСТЬ окна удержания (все, после смерти 0)"),
                     ("_sh", "SHARPE окна удержания (только выжившие)")):
        say()
        say(f"### цель — {tgt}")
        say()
        for F in FORM:
            g = P[P["F"] == F]
            say(f"── окно формирования {F:g} года ─────────────────────────────────────")
            hdr = f'{"предиктор":<28}' + "".join(f'{f"H={H:g}г":>20}' for H in HOLD)
            say(hdr)
            say("-" * len(hdr))
            for c, lab in preds:
                row = f"{lab:<28}"
                for H in HOLD:
                    y = f"h{H:g}{suf}"
                    if y not in g.columns:
                        row += f'{"—":>20}'
                        continue
                    m, t, nc, nsz = fm(g, c, y)
                    row += f'{"—":>20}' if not np.isfinite(m) else \
                        f'{f"{m:+.3f} (t={t:+.1f})":>20}'
                say(row)
            m, t, nc, nsz = fm(g, "f_sh", f"h{HOLD[0]:g}{suf}")
            say(f"  когорт: {nc}, медианный размер кросс-секции: {nsz} систем")
            say()
    say("Читать так: ρ = +0.10 значит, что ранг прошлого результата объясняет ~1 % ранга")
    say("будущего. ρ = +0.30 — уже рабочий сигнал отбора.")
    say()
    say("🔴 ЗАЧЕМ ДВЕ ЦЕЛИ. По доходности вола и просадка обгоняют Sharpe, но часть")
    say("этого — механика, а не предсказание: при одинаковом среднем ретёрне высокая")
    say("вола сама по себе опускает МЕДИАНУ накопленной доходности (volatility drag,")
    say("E[log(1+r)] < log(1+E[r])). Цель «Sharpe удержания» от drag свободна: если")
    say("преимущество риск-метрик сохраняется и там — оно настоящее.")

    # ── 3.2 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("3.2 КВИНТИЛИ: что получил подписчик, купивший «лучших» по прошлому Sharpe")
    say("=" * 104)
    q_rows = []
    for F, H in ((1.0, 1.0), (2.0, 2.0), (3.0, 1.0)):
        y, a, sh = f"h{H:g}_ret", f"h{H:g}_alive", f"h{H:g}_sh"
        g = P[(P["F"] == F)].dropna(subset=[y]).copy()
        if not len(g):
            continue
        cohorts = []
        for t0, sub in g.groupby("t0"):
            if len(sub) < MIN_COHORT * 2:
                continue
            sub = sub.copy()
            sub["q"] = pd.qcut(sub["f_sh"].rank(method="first"), NQ, labels=False) + 1
            for qq, s2 in sub.groupby("q"):
                cohorts.append({"t0": t0, "q": int(qq), "n": len(s2),
                                "med_ret": s2[y].median(),
                                # усечённое среднее: сырое ломается единичными
                                # артефактами витрины (есть система с +2 877 000 %)
                                "mean_ret": stats.trim_mean(s2[y].to_numpy(), 0.01),
                                "dead": 1.0 - s2[a].mean(),
                                "med_sh": s2[sh].median(),
                                "f_sh": s2["f_sh"].median()})
        if not cohorts:
            continue
        C = pd.DataFrame(cohorts)
        say(f"── формирование {F:g} г → удержание {H:g} г "
            f"({C['t0'].nunique()} когорт, медиана {int(C.groupby('t0')['n'].sum().median())} систем)")
        hdr = (f'{"квинтиль":<26} {"S окна форм.":>13} {"медиана дох.":>14} '
               f'{"усеч. средняя":>14} {"S в удерж.":>12} {"умерло":>9}')
        say(hdr)
        say("-" * len(hdr))
        for qq in range(1, NQ + 1):
            s2 = C[C["q"] == qq]
            lab = {1: "Q1 (худшие)", NQ: f"Q{NQ} (лучшие)"}.get(qq, f"Q{qq}")
            if F == 2.0 and H == 2.0:
                q_rows.append((lab.replace(" (худшие)", "\nхудшие")
                               .replace(" (лучшие)", "\nлучшие"),
                               float(100 * s2["med_ret"].mean()) / 100,
                               float(s2["dead"].mean())))
            say(f'{lab:<26} {s2["f_sh"].mean():>13.2f} {100*s2["med_ret"].mean():>13.1f}% '
                f'{100*s2["mean_ret"].mean():>13.1f}% {s2["med_sh"].mean():>12.2f} '
                f'{100*s2["dead"].mean():>8.1f}%')
        # спред Q5-Q1 по когортам -> t
        piv = C.pivot_table(index="t0", columns="q", values="med_ret")
        if NQ in piv.columns and 1 in piv.columns:
            sp = (piv[NQ] - piv[1]).dropna()
            t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))) if len(sp) > 2 else np.nan
            say(f'  спред Q{NQ}−Q1 по медианной доходности: {100*sp.mean():+.1f} пп, '
                f't = {t:+.2f} ({len(sp)} когорт)')
        pivd = C.pivot_table(index="t0", columns="q", values="dead")
        if NQ in pivd.columns and 1 in pivd.columns:
            spd = (pivd[NQ] - pivd[1]).dropna()
            td = spd.mean() / (spd.std(ddof=1) / np.sqrt(len(spd))) if len(spd) > 2 else np.nan
            say(f'  спред Q{NQ}−Q1 по смертности: {100*spd.mean():+.1f} пп, t = {td:+.2f}')
        say()
    say("«умерло» — доля систем квинтиля, не совершивших ни одной сделки в последние")
    say(f"{DEAD_DAYS} дней окна удержания (определение смерти из Блока 1).")
    say("«S в удерж.» считается только по системам, чей ряд покрывает окно целиком —")
    say("то есть по ВЫЖИВШИМ; доходность — по всем, с нулями после смерти.")

    # ── 3.3 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("3.3 СКОЛЬКО ФОРВАРДНЫХ ЛЕТ ИНФОРМАТИВНЫ (test-retest reliability)")
    say("=" * 104)
    say("Корреляция Sharpe двух СМЕЖНЫХ НЕПЕРЕСЕКАЮЩИХСЯ окон одинаковой длины T.")
    say("Если истинное качество системы постоянно, эта корреляция и есть доля")
    say("«истинного» в наблюдаемом Sharpe: ρ(T) = σ²_истин / (σ²_истин + шум(T)),")
    say("а шум ~ 1/T. Отсюда — сколько лет нужно, чтобы измерение перестало быть шумом.")
    say()
    say("⚠️ Оценка условна на выживание: Sharpe второго окна существует только у систем,")
    say("доживших до его конца. Умирают преимущественно плохие (3.2) — их отсев сужает")
    say("разброс истинного качества, то есть скорее ЗАНИЖАЕТ ρ. Верхняя граница честности")
    say("здесь неизвестна; трактуем полученные ρ как нижнюю оценку.")
    say()
    hdr = (f'{"длина окна T, лет":<20} {"когорт":>8} {"систем (медиана)":>18} '
           f'{"ρ(S_прошл, S_буд)":>20} {"t":>8}')
    say(hdr)
    say("-" * len(hdr))
    rel = []
    for T in FORM:
        if T not in HOLD:
            continue
        g = P[P["F"] == T]
        m, t, nc, nsz = fm(g, "f_sh", f"h{T:g}_sh")
        if np.isfinite(m):
            rel.append((T, m))
            say(f'{T:<20g} {nc:>8} {nsz:>18} {m:>20.3f} {t:>8.2f}')
    say()
    if len(rel) >= 3:
        Tv = np.array([x[0] for x in rel])
        rv = np.array([x[1] for x in rel])
        fit = optimize.least_squares(lambda c: Tv / (Tv + c[0]) - rv, [2.0], bounds=(1e-3, 1e3))
        c = float(fit.x[0])
        say(f"Фит ρ(T) = T / (T + c) даёт c = {c:.2f} года.")
        say("c — это горизонт, на котором сигнал и шум равны (ρ = 0.5).")
        say()
        hdr2 = f'{"горизонт форварда":<24} {"доля сигнала ρ":>18} {"смысл":<44}'
        say(hdr2)
        say("-" * len(hdr2))
        for T, lab in ((0.25, "3 месяца"), (0.5, "6 месяцев"), (1.0, "1 год"),
                       (2.0, "2 года"), (3.0, "3 года"), (5.0, "5 лет")):
            rho = T / (T + c)
            sense = ("шум" if rho < 0.25 else "слабый сигнал" if rho < 0.45 else
                     "сигнал сравним с шумом" if rho < 0.6 else "измерение осмысленно")
            say(f'{lab:<24} {rho:>18.2f} {sense:<44}')
    say()
    say("── 3.3б ТА ЖЕ ВЕЛИЧИНА СРЕДИ ПРИЛИЧНЫХ СИСТЕМ ──────────────────────────")
    say("🔴 Персистентность по ВСЕЙ популяции во многом разделяет мусор и не-мусор:")
    say("система с S = −3 стабильно остаётся мусором, и это надувает ρ. Вопрос")
    say("другой — различимы ли между собой те, кто уже прошёл фильтр измеримости.")
    say("Ниже ρ пересчитана внутри подвыборок по результату окна формирования.")
    say()
    P["rank_sh"] = P.groupby(["t0", "F"])["f_sh"].rank(pct=True)
    hdr = (f'{"подвыборка":<34} {"T=0.5г":>14} {"T=1г":>14} {"T=2г":>14} {"T=3г":>14}')
    say(hdr)
    say("-" * len(hdr))
    subs = [("вся выборка", lambda x: x),
            ("S окна формирования > 0", lambda x: x[x["f_sh"] > 0]),
            ("S окна формирования > 1", lambda x: x[x["f_sh"] > 1]),
            ("верхний квинтиль когорты", lambda x: x[x["rank_sh"] > 0.8])]
    curves = []
    for lab, fsub in subs:
        row = f"{lab:<34}"
        xs_c, ys_c = [], []
        for T in FORM:
            g = fsub(P[P["F"] == T])
            m, t, nc, nsz = fm(g, "f_sh", f"h{T:g}_sh", mincoh=20)
            row += f'{"—":>14}' if not np.isfinite(m) else f'{f"{m:+.3f}({nc})":>14}'
            if np.isfinite(m):
                xs_c.append(T)
                ys_c.append(m)
        if len(xs_c) >= 3:
            curves.append((lab, xs_c, ys_c))
        say(row)
    if CHARTS:
        chart_quintiles(q_rows)
        chart_rho_curves(curves)
    say("В скобках — число когорт, по которым усреднено (требуется ≥ 20 систем в")
    say("кросс-секции; у длинных окон верхний квинтиль столько не набирает).")
    say()
    # сколько форварда нужно, чтобы результат стал статистически значимым
    S = REF_SHARPE
    need = 4.0 * (1 + S * S / 2) / (S * S)
    say(f"Теоретический ориентир для системы с S = {S}: чтобы t-статистика Sharpe")
    say(f"достигла 2, нужно T = 4(1+S²/2)/S² = {need:.1f} года непрерывного форварда")
    say("(при условии, что edge всё это время не сломался).")

    # ── 3.4 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("3.4 ВИТРИННЫЙ РЕЙТИНГ И ПОДПИСЧИКИ: что они отражают")
    say("=" * 104)
    say("🔴 ОГРАНИЧЕНИЕ: rating/followers — снимок на момент выкачки, истории у них нет.")
    say("Проверить «предсказывал ли рейтинг будущее» невозможно в принципе. Ниже —")
    say("только кросс-секция ЖИВЫХ систем: с чем рейтинг связан сегодня.")
    say()
    S2 = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    L = panel[panel["is_live"] == True].copy()                          # noqa: E712
    L = L.join(S2[["sharpe_ar", "activity", "n_trade"]], how="left", rsuffix="_m")
    L = L[(L["n_trade_m"] >= 100) & (L["activity"] >= 0.10)]
    say(f"живых систем с измеримым рядом: {len(L):,}")
    say()
    hdr = f'{"пара":<52} {"Spearman ρ":>12} {"n":>7}'
    say(hdr)
    say("-" * len(hdr))
    pairs = [("rating", "sharpe_ar", "рейтинг витрины ↔ Sharpe за всю жизнь"),
             ("rating", "cagr", "рейтинг витрины ↔ CAGR за всю жизнь"),
             ("rating", "card_365", "рейтинг витрины ↔ доходность за 365 дней"),
             ("rating", "maxdd", "рейтинг витрины ↔ макс. просадка"),
             ("rating", "years", "рейтинг витрины ↔ возраст системы"),
             ("rating", "followers", "рейтинг витрины ↔ число подписчиков"),
             ("followers", "sharpe_ar", "подписчики ↔ Sharpe за всю жизнь"),
             ("followers", "card_365", "подписчики ↔ доходность за 365 дней"),
             ("followers", "maxdd", "подписчики ↔ макс. просадка"),
             ("followers", "years", "подписчики ↔ возраст системы")]
    for a, b, lab in pairs:
        s = L[[a, b]].dropna()
        if len(s) < 30:
            continue
        rho = stats.spearmanr(s[a], s[b]).statistic
        say(f"{lab:<52} {rho:>12.3f} {len(s):>7,}")

    # ── 3.5 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("3.5 УМНЫЕ ЛИ ДЕНЬГИ: за чем идут подписки")
    say("=" * 104)
    say("Регрессия log(1+подписчики) на стандартизованные предикторы (живые системы).")
    say("Коэффициент = изменение log-подписчиков на 1 σ предиктора.")
    say()
    X = L[["sharpe_ar", "card_365", "maxdd", "years", "rating"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1) & L["followers"].notna()
    X, yv = X[ok], np.log1p(L.loc[ok, "followers"].to_numpy())
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    s2 = resid @ resid / (len(yv) - A.shape[1])
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    hdr = f'{"предиктор":<34} {"коэф.":>10} {"SE":>9} {"t":>8}'
    say(hdr)
    say("-" * len(hdr))
    names = ["константа", "Sharpe за всю жизнь", "доходность за 365 дней",
             "макс. просадка", "возраст системы", "рейтинг витрины"]
    for nm, b, s in zip(names, beta, se):
        say(f"{nm:<34} {b:>10.3f} {s:>9.3f} {b/s:>8.2f}")
    r2 = 1 - resid.var() / yv.var()
    say(f"n = {len(yv):,}, R² = {r2:.3f}")

    (ROOT / "results" / "comon_persistence.log").write_text(
        "\n".join(_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
