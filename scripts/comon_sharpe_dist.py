"""comon_sharpe_dist.py — БЛОК 2: распределение Sharpe в дикой природе.

План исследования, Блок 2. Вопрос: какой Sharpe вообще бывает у реально торгующих
систем — и как часто. Ответ нужен как внешняя шкала: изнутри любого одного проекта
вопрос «это скромный результат или наглый?» не решается принципиально.

🔴 ДВЕ КОНВЕНЦИИ SHARPE. Арифметическая — mean/std·√252, по среднему дневных
ретёрнов и БЕЗ вычета безрисковой ставки. Геометрическая (CAGR/vol, записана в
панели Блока 0) систематически НИЖЕ: эффект Йенсена, тем сильнее чем выше вола.
Считаем обе и всюду говорим, какая именно перед вами.

🔴 ФИЛЬТР ИЗМЕРИМОСТИ ОБЯЗАТЕЛЕН (следствие Блока 1 и разбора правил площадки): 35 %
популяции — 66 авторов-фабрик с медианой жизни 15 дней, 21.9 % архива — единовременная
чистка 2014 года. Считать распределение по всей популяции значит описывать фабрики,
а не торговые системы. Фильтры заданы по ИЗМЕРИМОСТИ (длина ряда, число торговых
дней), а не по результату — отбор по доходности исказил бы изучаемое распределение.

Разделы:
  2.1 распределение S: вся популяция / живые / мёртвые;
  2.2 сколько держат S ≥ 1.5 на горизонте ≥ 3 лет;
  2.3 survivorship bias в пунктах Sharpe — цена взгляда только на витрину;
  2.4 держится ли S с ростом длины ряда (гипотеза: высокий S у молодых = отбор);
  2.5 честная проверка той же гипотезы ВНУТРИ системы: S первого года против S остатка;
  2.6 реалистичный потолок — верхние квантили среди доживших до 5 лет.

Текстовый вывод. Запуск: python comon_sharpe_dist.py
"""
import gzip
import json
import subprocess
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit", "cards")       # первичных данных нет в репозитории — см. DATA.md
LOG = ROOT / "results" / "comon_sharpe_dist.log"
OUT = DIR / "sharpe.csv.gz"
NPROC = 8

WIN_FROM, WIN_TILL = "2022-10-12", "2026-08-03"   # окно сравнения: последние ~4 года
MIN_TRADE_DAYS = 100      # торговых дней за всю жизнь — иначе Sharpe = шум
MIN_ACTIVITY = 0.10       # доля торговых дней в ряде: ниже — система стоит, а не торгует
REF_SHARPE = 1.5          # ориентир «хорошей» системы, с которым сверяется популяция
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def metrics(sid):
    """Sharpe в обеих конвенциях + разрезы по времени. Ряд — только rValue (Блок 0)."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if len(s) < 30:
        return None
    s = s[::-1]
    d = np.array([x["date"] for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s])
    n = len(r)
    yrs = max((date.fromisoformat(d[-1]) - date.fromisoformat(d[0])).days / 365.25, 1e-9)
    freq = n / yrs                                   # фактическая частота (~365)
    sd = r.std(ddof=1)
    if sd <= 0:
        return None
    eq = np.cumprod(1.0 + r)
    ntr = int((r != 0).sum())
    out = {"id": sid, "n": n, "n_trade": ntr, "years": yrs, "activity": ntr / n,
           # ── арифметическая конвенция: среднее ретёрнов, без безрисковой ставки
           "sharpe_ar": float(r.mean() / sd * np.sqrt(freq)),
           # ── геометрическая версия (в панели Блока 0)
           "sharpe_geo": float(((eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1)
                               / (sd * np.sqrt(freq))),
           "vol": float(sd * np.sqrt(freq)),
           "ret_total": float(eq[-1] - 1.0)}
    # t-статистика Sharpe: SE ≈ sqrt((1 + S²/2)/N) по числу ЛЕТ наблюдения
    S = out["sharpe_ar"]
    out["t_sharpe"] = float(S * np.sqrt(yrs) / np.sqrt(1 + S * S / 2))
    # разрез «первый год против остатка» — честная проверка гипотезы про молодых
    d0 = date.fromisoformat(d[0])
    y1 = np.array([(date.fromisoformat(x) - d0).days <= 365 for x in d])
    for lab, m in (("y1", y1), ("rest", ~y1)):
        rr = r[m]
        if len(rr) > 60 and rr.std(ddof=1) > 0:
            out[f"sh_{lab}"] = float(rr.mean() / rr.std(ddof=1) * np.sqrt(freq))
    return out


def rusfar():
    """Дневная ставка денежного рынка RUSFAR с ISS — что приносит капитал без риска.

    Доступна с 2019-01-09. Возвращает {дата: ставка в долях годовых}. Кэш на диске.
    """
    f = DIR / "rusfar.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for y in range(2019, 2027):
        url = ("https://iss.moex.com/iss/history/engines/stock/markets/index/boards/"
               f"MMIX/securities/RUSFAR.json?iss.meta=off&from={y}-01-01&till={y}-12-31")
        start = 0
        while True:
            j = json.loads(subprocess.check_output(
                ['curl', '-sL', f'{url}&start={start}'], text=True))
            cols, data = j["history"]["columns"], j["history"]["data"]
            if not data:
                break
            i_d, i_c = cols.index("TRADEDATE"), cols.index("CLOSE")
            for row in data:
                if row[i_c] is not None:
                    out[row[i_d]] = float(row[i_c]) / 100.0
            start += len(data)
    f.write_text(json.dumps(out))
    return out


def window_metrics(args):
    """Sharpe системы НА ЗАДАННОМ ОКНЕ: сырой и избыточный над ставкой."""
    sid, lo, hi = args
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    s_ = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if not s_:
        return None
    s_ = s_[::-1]
    d = [x["date"] for x in s_]
    r = np.array([float(x["rValue"] or 0.0) for x in s_])
    m = np.array([(lo <= x <= hi) for x in d])
    if m.sum() < 250:
        return None
    dd = [x for x, k in zip(d, m) if k]
    rr = r[m]
    yrs = max((date.fromisoformat(dd[-1]) - date.fromisoformat(dd[0])).days / 365.25, 1e-9)
    if yrs < 3.0 or rr.std(ddof=1) <= 0:            # требуем почти всё окно
        return None
    freq = len(rr) / yrs
    RF = _RF
    rf_d = np.array([RF.get(x, np.nan) for x in dd])
    rf_d = pd.Series(rf_d).ffill().bfill().to_numpy() / freq   # годовая -> на 1 точку
    ex = rr - rf_d
    return {"id": sid, "n": len(rr), "years": yrs,
            "sh_win": float(rr.mean() / rr.std(ddof=1) * np.sqrt(freq)),
            "sh_win_ex": float(ex.mean() / ex.std(ddof=1) * np.sqrt(freq)),
            "vol_win": float(rr.std(ddof=1) * np.sqrt(freq)),
            "cagr_win": float(np.cumprod(1 + rr)[-1] ** (1 / yrs) - 1),
            "rf_avg": float(np.mean(rf_d) * freq)}


def structure(sid):
    """Состав портфеля из карточки: доли Fut / Micex (акции) / Gko (облигации) / Money.

    Нужно, чтобы отделить СРОЧНЫЕ стратегии. У фьючерса контанго съедает безрисковую
    ставку, поэтому его P&L — уже ИЗБЫТОЧНАЯ доходность; у стратегии на акциях
    доходность полная. Сравнивать срочные корректно именно со срочными, иначе вычет
    ставки получается односторонним.
    ⚠️ structure — снимок на момент выкачки (у архивных — на момент архивации),
    а не средний состав за жизнь.
    """
    f = DIR / "cards" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        c = json.loads(gzip.open(f, "rt").read()).get("data") or {}
    except Exception:                                                  # noqa: BLE001
        return None
    st = c.get("structure") or []
    if not st:
        return None
    d = {x["id"]: x["value"] for x in st}
    return {"id": sid, "fut": d.get("Fut", 0.0), "eq": d.get("Micex", 0.0),
            "bond": d.get("Gko", 0.0), "cash": d.get("Money", 0.0)}


def q(x, ps=(0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    return [float(np.nanquantile(x, p)) for p in ps]


# ── графики 4, 5, 19 утверждённого списка ──
# --charts: дорисовать после расчёта; --charts-only: нарисовать из sharpe.csv.gz,
# который этот же скрипт и пишет, не пересчитывая ряды 19 тысяч систем.
ONLY = "--charts-only" in sys.argv
CHARTS = ONLY or "--charts" in sys.argv
BINS = (1, 2, 3, 5, 7)


def chart_tail(G):
    """График 4: сжатие хвоста Sharpe по длине истории."""
    import comon_charts as ch

    f, ax = ch.fig()
    xs = list(BINS)
    rows = {"максимум": [], "p99": [], "p95": [], "p90": []}
    ns = []
    for yr in xs:
        s = G[G["years"] >= yr]["sharpe_ar"]
        ns.append(len(s))
        rows["максимум"].append(s.max())
        rows["p99"].append(s.quantile(.99))
        rows["p95"].append(s.quantile(.95))
        rows["p90"].append(s.quantile(.90))

    ax.axhline(2.0, color=ch.PALE, lw=1.0, ls="--", zorder=1)

    # подписи у линий пробовали — на длинных историях все четыре величины
    # сходятся в полтора пункта, и подписи ложились на сами линии; в легенде
    # каждая линия опознаётся цветом, штрихом и формой маркера сразу
    styles = [("самая лучшая стратегия", "-", "o", 2.4, ch.ORANGE),
              ("p99 — верхний процент", "--", "s", 1.6, ch.BLUE),
              ("p95", ":", "^", 1.6, ch.AQUA),
              ("p90 — верхняя десятая часть", "-.", "D", 1.6, ch.VIOLET)]
    for (lab, ls, mk, lw, col), key in zip(styles, ("максимум", "p99", "p95", "p90")):
        ax.plot(xs, rows[key], ls=ls, marker=mk, ms=6, lw=lw, color=col, zorder=3,
                label=lab)
    ax.legend(loc="upper right", handlelength=3.4, labelspacing=0.7)

    # сторона подписи выбрана под наклон кривой в каждой точке: у круто
    # падающей линии свободно то сверху, то справа-сверху
    offs = {1: (0, 12, "center"), 2: (9, 7, "left"), 3: (9, 7, "left"),
            5: (9, 12, "left"), 7: (-4, 12, "right")}   # последняя — над линией 2,0
    for x, v in zip(xs, rows["максимум"]):
        dx, dy, ha = offs[x]
        ax.annotate(f"{v:.2f}".replace(".", ","), xy=(x, v), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11, fontweight="bold",
                    ha=ha)
    ax.annotate("Sharpe 2,0 — очень редкий результат", xy=(7.3, 2.0),
                xytext=(0, 6), textcoords="offset points", fontsize=10,
                color=ch.GREY, ha="left")

    ax.set_xlim(0.5, 10.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"≥ {y} года" if y in (1, 2, 3, 4) else f"≥ {y} лет"
                        for y in xs])
    ax.set_ylim(0, 21)
    ax.set_xlabel("длина истории стратегии")
    ax.set_ylabel("Sharpe")
    ax.set_title("Чем длиннее история, тем ниже лучший результат")
    ch.note(ax,
            "Верхние точки распределения Sharpe среди стратегий с историей не короче "
            "указанной. p90 — уровень, выше которого только десятая часть выборки; "
            "p99 — сотая.",
            "Стратегий в группах: " + ", ".join(
                f"{'≥ ' + str(y)} — {n:,}".replace(",", " ") for y, n in zip(xs, ns))
            + ".")
    p = ch.save(f, 4, "sharpe-tail-compression")
    print("    максимум по группам: "
          + "; ".join(f"≥{y} лет {v:.2f} (n={n})"
                      for y, v, n in zip(xs, rows["максимум"], ns))
          + f"; p90 {rows['p90'][0]:.2f}→{rows['p90'][-1]:.2f}", flush=True)
    return p


def chart_slope(P):
    """График 5: у каждой стратегии первый год и остаток жизни — двумя точками."""
    import comon_charts as ch

    top = P[P["sh_y1"] >= 2.0]
    f, ax = ch.fig()
    lo, hi = -3.0, 8.0
    out = 0
    for a, b in zip(top["sh_y1"], top["sh_rest"]):
        if a > hi or b > hi or b < lo:
            out += 1
        # цвет по исходу: удержала планку · осталась в плюсе · ушла в убыток
        col, z = (ch.GREEN, 3) if b >= 1.5 else ((ch.RED, 3) if b < 0
                                                 else (ch.PALE, 2))
        ax.plot([0, 1], [min(a, hi), min(max(b, lo), hi)], color=col,
                lw=0.8, alpha=0.75 if col != ch.PALE else 0.5, zorder=z,
                solid_capstyle="round")
    ma, mb = top["sh_y1"].median(), top["sh_rest"].median()
    ax.plot([0, 1], [ma, mb], color=ch.INK, lw=3.2, zorder=6,
            solid_capstyle="round")
    ax.plot([0, 1], [ma, mb], marker="o", ms=10, color=ch.INK, zorder=7, ls="none",
            markeredgecolor=ch.SURFACE, markeredgewidth=1.8)
    for c, lab in ((ch.GREEN, "удержали Sharpe 1,5 и выше"),
                   (ch.PALE, "остались в плюсе, но ниже"),
                   (ch.RED, "ушли в убыток")):
        ax.plot([], [], color=c, lw=2.4, label=lab)
    ax.plot([], [], color=ch.INK, lw=3.2, label="медиана группы")
    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.6)
    ch.stack_label(ax, (0, ma), f"{ma:.2f}".replace(".", ","), "медиана", dx=-8,
                   ha="right")
    ch.stack_label(ax, (1, mb), f"{mb:.2f}".replace(".", ","), "медиана", dx=8)

    ax.axhline(1.5, color=ch.PALE, lw=1.0, ls="--", zorder=1)
    # подпись уровня — вне пучка: посередине его пересекают все 319 линий
    ax.annotate("Sharpe 1,5", xy=(1.09, 1.5), xytext=(0, 6),
                textcoords="offset points", fontsize=10, color=ch.GREY, ha="left")
    ax.axhline(0, color="#888888", lw=1.0, zorder=1)

    kept = int((top["sh_rest"] >= 1.5).sum())
    lost = int((top["sh_rest"] < 0).sum())
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["первый год жизни", "вся оставшаяся жизнь"])
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Sharpe")
    ax.set_title("Что стало с теми, кто начал блестяще")
    ch.note(ax,
            f"Каждая тонкая линия — одна из {len(top)} стратегий, показавших "
            f"Sharpe 2,0 и выше в первый год. Жирная линия — медиана группы.",
            f"Удержали 1,5 и выше: {kept} ({100*kept/len(top):.1f} %). "
            f"Ушли в убыток: {lost} ({100*lost/len(top):.1f} %). "
            f"Линий, выходящих за пределы оси: {out}.")
    p = ch.save(f, 5, "sharpe-first-year-vs-rest")
    print(f"    {len(top)} линий, медиана {ma:.2f} → {mb:.2f}; удержали ≥1.5 {kept}; "
          f"в минусе {lost}; за осью {out}; ось Y {lo}…{hi}", flush=True)
    return p


def chart_hist(G):
    """График 19: распределение Sharpe — все, живые, закрытые."""
    import comon_charts as ch

    live = G["is_live"] == True                                         # noqa: E712
    f, ax = ch.fig()
    lo, hi, step = -5.0, 5.0, 0.25
    edges = np.arange(lo, hi + step, step)
    sets = [("все", G["sharpe_ar"], "-", 2.0, ch.INK),
            ("живые", G.loc[live, "sharpe_ar"], "--", 2.0, ch.GREEN),
            ("закрытые", G.loc[~live, "sharpe_ar"], ":", 2.0, ch.RED)]
    clipped, meds = 0, {}
    for lab, v, ls, lw, col in sets:
        clipped += int(((v < lo) | (v > hi)).sum()) if lab == "все" else 0
        h, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        ax.step(edges[:-1], h / len(v), where="post", ls=ls, lw=lw, color=col,
                zorder=3, label=lab)
        meds[lab] = m = float(v.median())
        # медианы помечены треугольниками на оси: три подписи рядом с числами
        # слиплись бы — сами числа ушли в сноску
        ax.plot([m], [0.0], marker="v", ms=8, color=col, clip_on=False, zorder=5)

    # подписи кривых собраны в легенду: три надписи у самих кривых неизбежно
    # ложились на них — в середине поля кривые почти смыкаются
    ax.legend(loc="upper right", handlelength=3.2, labelspacing=0.7,
              title="распределение Sharpe")

    ax.set_xlim(lo, hi)
    ax.set_xticks(range(-5, 6))
    ax.set_xlabel("Sharpe за всю жизнь стратегии")
    ax.set_ylabel("доля стратегий своей выборки")
    ax.set_title("Медиана рынка — около нуля")
    ch.pct(ax, decimals=0)
    ch.note(ax,
            f"Все {ch.n_(len(G))} стратегии, прошедшие фильтр измеримости: живых "
            f"{ch.n_(int(live.sum()))}, закрытых {ch.n_(int((~live).sum()))}. "
            f"Каждая кривая нормирована на свою выборку.",
            f"Треугольники на оси — медианы: все "
            f"{ch.n_(meds['все'], 2, sign=True)}, живые "
            f"{ch.n_(meds['живые'], 2, sign=True)}, закрытые "
            f"{ch.n_(meds['закрытые'], 2, sign=True)}.",
            f"Значения за пределами шкалы (|Sharpe| > 5) отнесены к крайним "
            f"столбцам: таких {clipped}.")
    p = ch.save(f, 19, "sharpe-histogram")
    print(f"    корзины {lo}…{hi} шагом {step}; медианы: все "
          f"{G['sharpe_ar'].median():.2f}, живые {G.loc[live,'sharpe_ar'].median():.2f}, "
          f"закрытые {G.loc[~live,'sharpe_ar'].median():.2f}; вне шкалы {clipped}",
          flush=True)
    return p


def draw_all(M):
    """Все графики этого скрипта — из его же артефакта sharpe.csv.gz."""
    G = M[(M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)]
    P = G.dropna(subset=["sh_y1", "sh_rest"])
    P = P[P["years"] >= 2]
    print(f"графики: выборка {len(G):,} систем, пар «первый год / остаток» {len(P):,}"
          .replace(",", " "), flush=True)
    chart_tail(G)
    chart_slope(P)
    chart_hist(G)


_RF = {}


def main():
    global _RF
    _RF = rusfar()
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    ids = panel.index[panel["n_pts"].notna()].tolist()
    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(metrics, ids, chunksize=200) if x]
    M = pd.DataFrame(rows).set_index("id")
    M = M.join(panel[["is_live", "owner_id", "followers", "min_sum"]], how="left")
    M.to_csv(OUT, compression="gzip")
    if CHARTS:
        draw_all(M)

    say(f"систем с рядом: {len(M):,}")
    good = (M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)
    say(f"фильтр измеримости (≥{MIN_TRADE_DAYS} торговых дней и активность ≥"
        f"{MIN_ACTIVITY:.0%}): {int(good.sum()):,} систем ({100*good.mean():.1f} %)")
    say("Фильтр по ИЗМЕРИМОСТИ, не по результату: отбор по доходности исказил бы")
    say("само распределение, которое мы изучаем.")
    G = M[good]
    live = G["is_live"] == True                                         # noqa: E712

    say()
    say("=" * 100)
    say("2.1 РАСПРЕДЕЛЕНИЕ SHARPE (арифметическая конвенция)")
    say("=" * 100)
    hdr = (f'{"выборка":<26} {"систем":>7} {"p5":>7} {"p25":>7} {"медиана":>8} '
           f'{"p75":>7} {"p90":>7} {"p95":>7} {"p99":>7}')
    say(hdr)
    say("-" * len(hdr))
    for lab, sub in (("вся популяция (фильтр)", G), ("живые", G[live]),
                     ("мёртвые", G[~live])):
        say(f'{lab:<26} {len(sub):>7,}' + "".join(f'{v:>7.2f}' for v in q(sub["sharpe_ar"])))
    say()
    say(f'{"для справки — геометрический (CAGR/vol):":<26}')
    for lab, sub in (("вся популяция (фильтр)", G), ("живые", G[live])):
        say(f'{lab:<26} {len(sub):>7,}' + "".join(f'{v:>7.2f}' for v in q(sub["sharpe_geo"])))
    say("Геометрический систематически ниже — эффект Йенсена. Сравнивать два числа")
    say("можно только внутри одной конвенции.")
    say()
    say("🔴 Sharpe везде БЕЗ вычета безрисковой ставки. При ставке 15–20 %")
    say("годовых, характерной для последних лет, вычет сдвинул бы всю картину вниз")
    say("примерно на 0.3–0.7 единицы Sharpe при воле 30 %.")

    say()
    say("=" * 100)
    say("2.2 СКОЛЬКО ДЕРЖАТ S ≥ 1.5 НА ДЛИННОМ ГОРИЗОНТЕ")
    say("=" * 100)
    say(f'{"горизонт":<18} {"систем":>8} {"S ≥ 1.0":>16} {"S ≥ 1.5":>16} '
        f'{"S ≥ 2.0":>16} {"S ≥ 2.5":>16}')
    say("-" * 100)
    for yr in (1, 2, 3, 5, 7):
        sub = G[G["years"] >= yr]
        if not len(sub):
            continue
        row = f'≥ {yr} года{"":<9}'[:18] + f'{len(sub):>8,}'
        for th in (1.0, 1.5, 2.0, 2.5):
            m = sub["sharpe_ar"] >= th
            row += f'{f"{int(m.sum()):,} ({100*m.mean():.1f}%)":>16}'
        say(row)
    say()
    sub3 = G[G["years"] >= 3]
    m15 = sub3["sharpe_ar"] >= REF_SHARPE
    sig = m15 & (sub3["t_sharpe"] >= 2)
    say(f'На горизонте ≥ 3 лет S ≥ 1.5 показали {int(m15.sum())} систем из {len(sub3)} '
        f'({100*m15.mean():.1f} %),')
    say(f'из них статистически значимо (t ≥ 2): {int(sig.sum())} '
        f'({100*sig.mean():.1f} % выборки).')
    say(f'Медиана t-статистики Sharpe у прошедших порог: '
        f'{sub3.loc[m15, "t_sharpe"].median():.2f}')

    say()
    say("=" * 100)
    say("2.3 SURVIVORSHIP BIAS В ПУНКТАХ SHARPE — цена взгляда только на витрину")
    say("=" * 100)
    say(f'{"горизонт":<14} {"живые: медиана":>16} {"все: медиана":>15} {"разница":>9} '
        f'{"живые p90":>11} {"все p90":>9} {"разница":>9}')
    say("-" * 100)
    for yr in (1, 2, 3, 5):
        a = G[(G["years"] >= yr) & live]["sharpe_ar"]
        b = G[G["years"] >= yr]["sharpe_ar"]
        if len(a) < 20:
            continue
        say(f'{f"≥ {yr} года":<14} {a.median():>16.2f} {b.median():>15.2f} '
            f'{a.median()-b.median():>+9.2f} {a.quantile(.9):>11.2f} '
            f'{b.quantile(.9):>9.2f} {a.quantile(.9)-b.quantile(.9):>+9.2f}')
    say("Разница — сколько Sharpe приписывает себе тот, кто смотрит только на живые")
    say("системы. Это и есть цена survivorship bias, выраженная в пунктах Sharpe.")

    say()
    say("=" * 100)
    say("2.4 ДЕРЖИТСЯ ЛИ S С РОСТОМ ДЛИНЫ РЯДА")
    say("=" * 100)
    say(f'{"длина истории":<18} {"систем":>7} {"медиана S":>11} {"p90 S":>8} '
        f'{"p99 S":>8} {"доля S≥1.5":>12} {"медиана волы":>14}')
    say("-" * 100)
    for lo, hi, lab in ((0, 1, "< 1 года"), (1, 1.5, "1–1.5 года"),
                        (1.5, 3, "1.5–3 года"), (3, 5, "3–5 лет"),
                        (5, 7, "5–7 лет"), (7, 100, "7 лет и больше")):
        sub = G[(G["years"] >= lo) & (G["years"] < hi)]
        if len(sub) < 10:
            continue
        say(f'{lab:<18} {len(sub):>7,} {sub["sharpe_ar"].median():>11.2f} '
            f'{sub["sharpe_ar"].quantile(.9):>8.2f} {sub["sharpe_ar"].quantile(.99):>8.2f} '
            f'{100*(sub["sharpe_ar"] >= 1.5).mean():>11.1f}% '
            f'{100*sub["vol"].median():>13.1f}%')
    say("⚠️ Это СМЕШАННОЕ сравнение: длинные ряды принадлежат выжившим, короткие —")
    say("всем подряд. Чистая проверка гипотезы — ниже, внутри одних и тех же систем.")

    say()
    say("=" * 100)
    say("2.5 ЧЕСТНАЯ ПРОВЕРКА: S ПЕРВОГО ГОДА против S ОСТАВШЕЙСЯ ЖИЗНИ (одни системы)")
    say("=" * 100)
    P = G.dropna(subset=["sh_y1", "sh_rest"])
    P = P[P["years"] >= 2]
    say(f'систем с ≥ 2 годами истории и обеими оценками: {len(P):,}')
    say(f'{"величина":<28} {"медиана":>10} {"p25":>9} {"p75":>9} {"p90":>9}')
    say("-" * 70)
    for lab, col in (("S первого года", "sh_y1"), ("S оставшейся жизни", "sh_rest")):
        say(f'{lab:<28} {P[col].median():>10.2f} {P[col].quantile(.25):>9.2f} '
            f'{P[col].quantile(.75):>9.2f} {P[col].quantile(.9):>9.2f}')
    d = P["sh_rest"] - P["sh_y1"]
    say(f'{"падение S после 1-го года":<28} {d.median():>10.2f} {d.quantile(.25):>9.2f} '
        f'{d.quantile(.75):>9.2f} {d.quantile(.9):>9.2f}')
    say()
    top = P[P["sh_y1"] >= 2.0]
    say(f'Из систем, показавших S ≥ 2.0 в ПЕРВЫЙ год ({len(top)} шт):')
    say(f'  медиана их S в оставшейся жизни: {top["sh_rest"].median():.2f}')
    say(f'  удержали S ≥ 1.5 дальше: {int((top["sh_rest"] >= 1.5).sum())} '
        f'({100*(top["sh_rest"] >= 1.5).mean():.1f} %)')
    say(f'  ушли в минус: {int((top["sh_rest"] < 0).sum())} '
        f'({100*(top["sh_rest"] < 0).mean():.1f} %)')
    say("Гипотеза плана «высокий S у молодых — эффект отбора и везения» проверяется")
    say("именно здесь: тот же набор систем, разные отрезки их собственной жизни.")

    say()
    say("=" * 100)
    say("2.6 РЕАЛИСТИЧНЫЙ ПОТОЛОК: верхние квантили среди доживших до 5 лет")
    say("=" * 100)
    L = G[G["years"] >= 5]
    say(f'систем с ≥ 5 годами истории: {len(L):,} '
        f'(из {len(G):,} прошедших фильтр измеримости — {100*len(L)/len(G):.1f} %)')
    say(f'{"квантиль":<12}' + "".join(f'{p:>10}' for p in
                                      ("медиана", "p75", "p90", "p95", "p99", "макс")))
    vals = [L["sharpe_ar"].median(), L["sharpe_ar"].quantile(.75),
            L["sharpe_ar"].quantile(.90), L["sharpe_ar"].quantile(.95),
            L["sharpe_ar"].quantile(.99), L["sharpe_ar"].max()]
    say(f'{"Sharpe":<12}' + "".join(f'{v:>10.2f}' for v in vals))
    say()
    best = L.nlargest(10, "sharpe_ar")[["years", "sharpe_ar", "t_sharpe", "vol",
                                        "ret_total", "is_live"]]
    say("Десять лучших по Sharpe среди доживших до 5 лет:")
    say(f'{"id":>8} {"лет":>6} {"S":>7} {"t":>7} {"вола":>8} {"итог":>12} {"жива":>6}')
    for i, r in best.iterrows():
        say(f'{i:>8} {r["years"]:>6.1f} {r["sharpe_ar"]:>7.2f} {r["t_sharpe"]:>7.2f} '
            f'{100*r["vol"]:>7.1f}% {100*r["ret_total"]:>11.0f}% '
            f'{"да" if r["is_live"] else "нет":>6}')

    # ── 2.7: одно окно, одна ставка, прямое сравнение ────────────────────────
    say()
    say("=" * 100)
    say(f"2.7 ВСЕ НА ОДНОМ ОКНЕ И ОДНОЙ СТАВКЕ ({WIN_FROM}..{WIN_TILL})")
    say("=" * 100)
    args = [(int(i), WIN_FROM, WIN_TILL) for i in M.index]
    with Pool(NPROC) as pool:
        wr = [x for x in pool.imap_unordered(window_metrics, args, chunksize=200) if x]
    W = pd.DataFrame(wr).set_index("id").join(M[["is_live", "n_trade", "activity"]])
    W = W[(W["n_trade"] >= MIN_TRADE_DAYS) & (W["activity"] >= MIN_ACTIVITY)]
    say(f"систем, торговавших ≥ 3 лет внутри окна: {len(W):,}")
    say(f"средняя ставка денежного рынка RUSFAR за окно: "
        f"{100*W['rf_avg'].median():.1f} % годовых")
    say()
    say(f'{"метрика":<34} {"p25":>8} {"медиана":>9} {"p75":>8} {"p90":>8} {"p95":>8} '
        f'{"p99":>8} {"макс":>8}')
    say("-" * 100)
    for lab, col in (("Sharpe сырой", "sh_win"),
                     ("Sharpe СВЕРХ ставки (избыточный)", "sh_win_ex")):
        v = W[col]
        say(f'{lab:<34} {v.quantile(.25):>8.2f} {v.median():>9.2f} {v.quantile(.75):>8.2f} '
            f'{v.quantile(.90):>8.2f} {v.quantile(.95):>8.2f} {v.quantile(.99):>8.2f} '
            f'{v.max():>8.2f}')
    say(f'{"годовая доходность, %":<34} {100*W["cagr_win"].quantile(.25):>8.1f} '
        f'{100*W["cagr_win"].median():>9.1f} {100*W["cagr_win"].quantile(.75):>8.1f} '
        f'{100*W["cagr_win"].quantile(.90):>8.1f} {100*W["cagr_win"].quantile(.95):>8.1f} '
        f'{100*W["cagr_win"].quantile(.99):>8.1f} {100*W["cagr_win"].max():>8.1f}')
    say(f'{"волатильность, %":<34} {100*W["vol_win"].quantile(.25):>8.1f} '
        f'{100*W["vol_win"].median():>9.1f} {100*W["vol_win"].quantile(.75):>8.1f} '
        f'{100*W["vol_win"].quantile(.90):>8.1f} {100*W["vol_win"].quantile(.95):>8.1f} '
        f'{100*W["vol_win"].quantile(.99):>8.1f} {100*W["vol_win"].max():>8.1f}')

    say()
    say(f'ориентир S = {REF_SHARPE} на этом окне → перцентиль '
        f'{100*(W["sh_win"] < REF_SHARPE).mean():.1f} '
        f'(выше него {int((W["sh_win"] >= REF_SHARPE).sum())} систем из {len(W)})')
    say()
    say("🔴 Вычет ставки — не придирка: свободный капитал зарабатывает её без риска.")
    say("Sharpe сверх ставки отвечает на вопрос «что даёт торговля СВЕРХ того, что дал бы")
    say("тот же капитал, просто лежащий в денежном рынке».")

    # ── 2.8: сравнение с СРОЧНЫМИ стратегиями ────────────────────────────────
    say()
    say("=" * 100)
    say("2.8 ЧЕСТНОЕ СРАВНЕНИЕ: только СРОЧНЫЕ стратегии (та же природа P&L)")
    say("=" * 100)
    say("🔴 Методическая поправка. У фьючерса контанго съедает безрисковую ставку, поэтому")
    say("его P&L — УЖЕ избыточная доходность, и вычитать из него RUSFAR значит вычесть")
    say("ставку дважды. У стратегии на акциях доходность полная, вычет правомерен.")
    say("Сравниваем срочных со срочными по СЫРОМУ Sharpe — он для обеих сторон уже")
    say("избыточен по построению.")
    say()
    with Pool(NPROC) as pool:
        st_rows = [x for x in pool.imap_unordered(structure, M.index.tolist(),
                                                  chunksize=300) if x]
    ST = pd.DataFrame(st_rows).set_index("id")
    say(f"систем с известным составом портфеля: {len(ST):,}")
    say(f'{"класс":<10} {"медиана доли":>13} {"систем с долей > 50 %":>23}')
    for lab, c in (("фьючерсы", "fut"), ("акции РФ", "eq"),
                   ("облигации", "bond"), ("деньги", "cash")):
        say(f'{lab:<10} {ST[c].median():>12.1f}% {int((ST[c] > 50).sum()):>23,}')
    say()
    WS = W.join(ST, how="inner")
    say(f'{"выборка (окно 2022-10..2026-08)":<34} {"систем":>7} {"медиана S":>11} '
        f'{"p75":>8} {"p90":>8} {"p95":>8} {"макс":>8}')
    say("-" * 100)
    for lab, m in (("все системы окна", WS.index == WS.index),
                   ("срочные: доля фьючерсов > 50 %", WS["fut"] > 50),
                   ("срочные: доля фьючерсов > 20 %", WS["fut"] > 20),
                   ("акционные: акции > 50 %", WS["eq"] > 50)):
        sub = WS[m]
        if len(sub) < 15:
            say(f'{lab:<34} {len(sub):>7,}  — мало систем для квантилей')
            continue
        say(f'{lab:<34} {len(sub):>7,} {sub["sh_win"].median():>11.2f} '
            f'{sub["sh_win"].quantile(.75):>8.2f} {sub["sh_win"].quantile(.90):>8.2f} '
            f'{sub["sh_win"].quantile(.95):>8.2f} {sub["sh_win"].max():>8.2f}')
    fut = WS[WS["fut"] > 20]
    if len(fut) >= 15:
        say()
        say(f'срочные (доля фьючерсов > 20 %, {len(fut)} шт): медиана волатильности '
            f'{100*fut["vol_win"].median():.1f} %, '
            f'выше S = {REF_SHARPE} держатся {int((fut["sh_win"] >= REF_SHARPE).sum())} штук')

    say()
    say("=" * 100)
    say("ЧТО ЗНАЧИТ ТОТ ИЛИ ИНОЙ SHARPE НА ФОНЕ ПОПУЛЯЦИИ")
    say("=" * 100)
    for ref, lab in ((REF_SHARPE, "S = 1.5"), (2.5, "S = 2.5"), (2.9, "S = 2.9")):
        for yr in (3, 5):
            sub = G[G["years"] >= yr]
            pct = 100 * (sub["sharpe_ar"] < ref).mean()
            say(f'{lab:<26} на горизонте ≥ {yr} лет — перцентиль {pct:.1f} '
                f'(выше {int((sub["sharpe_ar"] >= ref).sum())} систем из {len(sub)})')

    say()
    say(f"файл: {OUT}")
    LOG.write_text("\n".join(_lines) + "\n")
    say(f"лог: {LOG}")


if __name__ == "__main__":
    if ONLY:
        draw_all(pd.read_csv(OUT, index_col="id"))
    else:
        main()
