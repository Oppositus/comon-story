"""comon_leverage_risk.py — БЛОК 4: плечо, риск и «дорогая доходность».

План исследования, Блок 4. Вопрос блока: чем куплена высокая доходность публичных
систем — мастерством или просто размером риска? Лежат ли они на одном фронте
эффективности? Ответ требует отделить плечо от мастерства.

🔴 ГЛАВНАЯ ЛОВУШКА БЛОКА — ВЫЖИВАНИЕ. Связь «выше вола → выше доходность» на витрине
измерена по тем, кто дожил. Системы с высокой волой умирают чаще (Блок 1), и их
доходность в момент смерти отрицательна. Поэтому каждая связь считается ТРИЖДЫ: по всей
популяции, отдельно по живым, отдельно по мёртвым. Расхождение этих трёх картин и есть
цена риска, которую витрина не показывает.

🔴 ЕСТЕСТВЕННЫЙ ЭКСПЕРИМЕНТ. Чистое плечо масштабирует доходность и волу одинаково,
поэтому Sharpe к нему инвариантен — в теории. На практике плечо стоит денег
(проскальзывание растёт с размером, стоп-ауты чаще, ГО дороже). Раздел 4.2 измеряет эту
цену на парах систем ОДНОГО АВТОРА с почти одинаковой логикой (корреляция дневных ≥ 0.8)
и разным масштабом риска: если S падает с ростом плеча — плечо не бесплатно.

Разделы:
  4.1 объясняется ли доходность просто волатильностью;
  4.2 цена плеча на близнецах одного автора;
  4.3 совместное распределение доходности и просадки, отношение MAR;
  4.4 хвосты: скос, эксцесс и связь формы распределения со смертью.

Текстовый вывод. Запуск: python comon_leverage_risk.py
"""
import gzip
import json
import sys
from datetime import date
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "twins.csv.gz"
NPROC = 8

MIN_TRADE_DAYS = 100      # фильтр измеримости данных — тот же, что в Блоках 2 и 3
MIN_ACTIVITY = 0.10
MIN_OVERLAP = 250         # общих дней у пары систем одного автора
TWIN_RHO = 0.80           # корреляция дневных, выше которой считаем «одна логика»
MAX_PAIRS_AUTHOR = 50     # чтобы автор-фабрика не забил выборку своими сотнями пар
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def series(sid):
    """Дневной ряд системы: (даты-ordinal, ретёрны). Семантика ряда — из Блока 0."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if len(s) < 60:
        return None
    s = s[::-1]
    d = np.array([date.fromisoformat(x["date"]).toordinal() for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s], dtype=np.float64)
    return sid, d, r


def wstats(d, r):
    """Метрики на общем окне пары: (Sharpe, вола, CAGR, maxDD)."""
    sd = r.std(ddof=1)
    if sd <= 0 or len(r) < 20:
        return None
    yrs = max((d[-1] - d[0]) / 365.25, 1e-9)
    freq = len(r) / yrs
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    cagr = (eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0
    return (float(r.mean() / sd * np.sqrt(freq)), float(sd * np.sqrt(freq)),
            float(cagr), float(np.min(eq / peak - 1.0)))


def qtab(sub, col, ps=(0.05, 0.25, 0.50, 0.75, 0.95)):
    return [float(np.nanquantile(sub[col], p)) for p in ps]


def ols(y, X, names):
    """OLS с константой; печатает коэффициенты, SE, t. Возвращает R²."""
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    s2 = resid @ resid / (len(y) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
    for nm, b, s in zip(["константа"] + names, beta, se):
        say(f'{nm:<30} {b:>12.4f} {s:>10.4f} {b/s:>9.2f}')
    return 1 - resid.var() / y.var()


CHARTS = "--charts" in sys.argv


def chart_vol_deciles(G):
    """График 6: децили волатильности — доходность и выживаемость двумя панелями."""
    import comon_charts as ch

    f, axes = ch.fig(h_px=1140, bottom=0.22, nrows=2, sharex=True,
                     gridspec_kw={"height_ratios": [1.35, 1]})
    a0, a1 = axes
    d = sorted(G["vdec"].unique())
    cagr = [G[G["vdec"] == x]["cagr"].median() for x in d]
    surv = [G[G["vdec"] == x]["live"].mean() for x in d]
    vol = [G[G["vdec"] == x]["vol"].median() for x in d]

    a0.axhline(0, color=ch.PALE, lw=1.2, zorder=1)
    a0.plot(d, cagr, color=ch.BLUE, lw=2.4, marker="o", ms=8, zorder=3)
    ch.pct(a0)
    a0.set_ylabel("медианная годовая\nдоходность")   # в две строки: панель низкая
    a0.set_title("Больше риска — не больше денег, а меньше шансов выжить")
    # обе подписи уводятся ВНУТРЬ поля: у краёв они садились на рамку панели
    # и на заголовок, а первая ещё и читалась как продолжение заголовка
    a0.set_ylim(-1.28, 0.30)
    # линия падает: у левой точки свободно сверху, у правой — снизу; запас по
    # оси до −105 % добавлен, чтобы подписи помещались внутрь поля
    ch.stack_label(a0, (d[0], cagr[0]), f"{100*cagr[0]:+.1f} %".replace(".", ","),
                   "самые спокойные", dx=9, color=ch.BLUE)
    ch.stack_label(a0, (d[-1], cagr[-1]), f"{100*cagr[-1]:+.1f} %".replace(".", ","),
                   "самые рискованные", dx=-9, up=False, ha="right", color=ch.BLUE)

    a1.plot(d, surv, color=ch.ORANGE, lw=2.4, marker="s", ms=8, zorder=3)
    ch.pct(a1)
    a1.set_ylim(-0.045, max(surv) * 1.25)
    a1.set_ylabel("доля выживших")
    a1.set_xlabel("десятые части популяции по волатильности: от самых спокойных к самым рискованным")
    a1.set_xticks(d)
    # обе подписи уходят вниз: в начале линия не строго монотонна и цепляла
    # подпись, поставленную сверху
    ch.stack_label(a1, (d[0], surv[0]), f"{100*surv[0]:.1f} %".replace(".", ","),
                   "каждая четвёртая", dx=9, up=False, color=ch.ORANGE)
    ch.stack_label(a1, (d[-1], surv[-1]), f"{100*surv[-1]:.1f} %".replace(".", ","),
                   "каждая пятнадцатая", dx=-9, up=False, ha="right", color=ch.ORANGE)
    ch.note(a1,
            "Каждая точка — десятая часть популяции: слева самые спокойные системы, "
            f"справа самые резкие (медианная волатильность от {100*vol[0]:.0f} % до "
            f"{100*vol[-1]:.0f} % годовых).",
            "Две панели, а не две шкалы на одном поле: величины разной природы, и "
            "общая ось создала бы ложное впечатление их соизмеримости.")
    p = ch.save(f, 6, "volatility-deciles")
    print("    CAGR по децилям: " + "; ".join(f"{100*c:+.1f}" for c in cagr)
          + " | выжившие: " + "; ".join(f"{100*s:.1f}" for s in surv), flush=True)
    return p


def chart_twins(T):
    """График 33: цена плеча — ΔSharpe против логарифма отношения волатильностей."""
    import comon_charts as ch

    D = T[["d_sh", "L"]].dropna()
    D = D[(D["L"] > 0) & np.isfinite(D["d_sh"])]
    f, ax = ch.fig()
    x = np.log(D["L"].to_numpy())
    y = D["d_sh"].to_numpy()
    ax.axhline(0, color=ch.PALE, lw=1.4, zorder=1)
    ax.scatter(x, y, s=26, color=ch.BLUE, alpha=0.55, lw=0.6,
               edgecolors=ch.SURFACE, zorder=3)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=ch.ORANGE, lw=2.6, zorder=4)
    ax.annotate(f"удвоение риска той же логикой стоит "
                f"{abs(b1*np.log(2)):.2f} пункта Sharpe".replace(".", ","),
                xy=(xs[-1], b0 + b1 * xs[-1]), xytext=(-6, -18),
                textcoords="offset points", fontsize=11, color=ch.ORANGE, ha="right")
    ax.set_xlabel("во сколько раз вторая система рискованнее первой")
    ax.set_ylabel("разница Sharpe между ними")
    ax.set_title("Плечо не создаёт качества: та же логика крупнее — тот же Sharpe "
                 "или чуть хуже")
    tick = [1, 1.5, 2, 3, 5]
    ax.set_xticks([np.log(v) for v in tick])
    ax.set_xticklabels([str(v).replace(".", ",") for v in tick])
    ch.note(ax,
            f"Каждая точка — пара систем ОДНОГО автора: та же торговая идея, "
            f"разный масштаб риска. Пар: {ch.n_(len(D))}.",
            "Линия — подгонка. Если бы плечо было бесплатным, она была бы "
            "горизонтальной на нуле.")
    p = ch.save(f, 33, "leverage-twins")
    print(f"    пар {len(D)}, наклон {b1:+.3f} на единицу log-плеча, "
          f"медиана ΔS {np.median(y):+.3f}", flush=True)
    return p


def chart_dd_by_return(G, bins):
    """График 32: просадка почти не зависит от доходности."""
    import comon_charts as ch

    # подписи делений двухстрочные, сноска на четыре строки — нужна высота
    f, ax = ch.fig(h_px=1060, bottom=0.28)
    labs, meds, ns, deep = [], [], [], []
    for lo, hi, lab in bins:
        sub = G[(G["cagr"] > lo) & (G["cagr"] <= hi)]
        if not len(sub):
            continue
        labs.append(lab.replace("−", "−"))
        meds.append(float(sub["maxdd"].median()))
        ns.append(len(sub))
        deep.append(float((sub["maxdd"] < -0.50).mean()))
    xs = np.arange(len(labs))
    # цветом выделена середина — те группы, про которые и говорит заголовок;
    # прежняя раскраска красила крайние (худшую и лучшую) одним цветом и
    # предлагала читателю искать смысл, которого нет
    cols = [ch.PALE if i in (0, 1) else ch.BLUE for i in range(len(labs))]
    ax.bar(xs, meds, width=0.62, color=cols, zorder=3)
    for i, (v, n) in enumerate(zip(meds, ns)):
        ax.annotate(f"{100*v:.1f} %".replace(".", ","), xy=(i, v), xytext=(0, -8),
                    textcoords="offset points", ha="center", va="top", fontsize=12,
                    fontweight="bold", color=ch.INK)
        ax.annotate(f"{n:,}".replace(",", " "), xy=(i, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color=ch.GREY)
    mid = [m for m, l in zip(meds, labs) if l not in (labs[0], labs[1], labs[-1])]
    lvl = float(np.median(mid))
    ax.axhline(lvl, color=ch.INK, lw=1.4, ls="--", zorder=4)
    # пояснение уходит в пустую нижнюю треть поля и связывается с линией
    # выноской: поверх столбцов серый текст не читался
    ax.annotate(f"в середине просадка стоит на месте — около {-100*lvl:.0f} %",
                xy=(2.5, lvl), xytext=(2.5, min(meds) * 0.72),
                fontsize=12, color=ch.INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.45", fc=ch.SURFACE, ec="#c9c8c3",
                          lw=0.9, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=ch.GREY, lw=1.2,
                                shrinkB=6))
    ax.set_xticks(xs)
    # подписи двух левых групп смыкались в одну фразу — разводим переносом
    ax.set_xticklabels([l.replace("убыток ", "убыток\n") for l in labs], fontsize=11)
    ax.set_ylim(min(meds) * 1.18, 0.075)
    ax.annotate(f"{-100*lvl:.0f} %", xy=(len(labs) - 0.55, lvl), xytext=(0, 5),
                textcoords="offset points", fontsize=11, color=ch.INK, ha="right",
                va="bottom")
    ch.pct(ax)
    ax.set_xlabel("группа по годовой доходности")
    ax.set_ylabel("медианная максимальная просадка")
    ax.set_title("Среди прибыльных стратегий просадка не зависит от доходности")
    ch.note(ax,
            "Числа сверху — сколько стратегий в группе. Стратегия, зарабатывающая "
            "50 % в год, проседает не глубже той, что зарабатывает 5 %: глубина "
            "провала определяется принятым риском, а не результатом.",
            "Две бледные группы слева выпадают из правила по другой причине: у них "
            "просадка и есть их итог. Правая группа — самые доходные — снова "
            "проседает глубже: за доходность выше 60 % годовых уже платят риском.")
    p = ch.save(f, 32, "drawdown-by-return")
    print("    " + "; ".join(f"{l}: {100*m:.1f} % (n={n})"
                             for l, m, n in zip(labs, meds, ns)), flush=True)
    return p


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    S = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    M = S.join(panel[["cagr", "maxdd", "skew", "kurt", "owner_id", "is_live",
                      "zero_share", "years"]], how="inner", rsuffix="_p")
    good = (M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)
    G = M[good].copy()
    G["live"] = G["is_live"] == True                                    # noqa: E712
    G["mar"] = G["cagr"] / G["maxdd"].abs().clip(lower=1e-6)
    say(f"систем с рядом: {len(M):,}; прошли фильтр измеримости "
        f"(≥{MIN_TRADE_DAYS} торговых дней, активность ≥{MIN_ACTIVITY:.0%}): {len(G):,}")
    say(f"из них живых: {int(G['live'].sum()):,}, мёртвых: {int((~G['live']).sum()):,}")

    # ── 4.1 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("4.1 ОБЪЯСНЯЕТСЯ ЛИ ДОХОДНОСТЬ ПРОСТО ВОЛАТИЛЬНОСТЬЮ")
    say("=" * 104)
    say("Децили по годовой волатильности. Если риск оплачивается — Sharpe не должен")
    say("падать с ростом волы; если доходность = просто плечо — CAGR растёт, Sharpe стоит.")
    say()
    G["vdec"] = pd.qcut(G["vol"].rank(method="first"), 10, labels=False) + 1
    hdr = (f'{"дециль волы":<12} {"вола, %":>9} {"CAGR, %":>9} {"Sharpe":>8} '
           f'{"maxDD, %":>10} {"живых":>8} {"систем":>8}')
    say(hdr)
    say("-" * len(hdr))
    for dd, sub in G.groupby("vdec"):
        say(f'{int(dd):<12} {100*sub["vol"].median():>9.1f} {100*sub["cagr"].median():>9.1f} '
            f'{sub["sharpe_ar"].median():>8.2f} {100*sub["maxdd"].median():>10.1f} '
            f'{100*sub["live"].mean():>7.1f}% {len(sub):>8,}')
    say()
    say("То же отдельно по живым и по мёртвым — разница и есть невидимая на витрине цена:")
    say()
    hdr = (f'{"дециль волы":<12} {"вола, %":>9} {"CAGR живых":>12} {"CAGR мёртвых":>14} '
           f'{"S живых":>9} {"S мёртвых":>11}')
    say(hdr)
    say("-" * len(hdr))
    for dd, sub in G.groupby("vdec"):
        a, b = sub[sub["live"]], sub[~sub["live"]]
        say(f'{int(dd):<12} {100*sub["vol"].median():>9.1f} '
            f'{100*a["cagr"].median():>11.1f}% {100*b["cagr"].median():>13.1f}% '
            f'{a["sharpe_ar"].median():>9.2f} {b["sharpe_ar"].median():>11.2f}')
    say()
    for lab, sub in (("вся популяция", G), ("живые", G[G["live"]]),
                     ("мёртвые", G[~G["live"]])):
        rho_cv = stats.spearmanr(sub["vol"], sub["cagr"]).statistic
        rho_sv = stats.spearmanr(sub["vol"], sub["sharpe_ar"]).statistic
        say(f'{lab:<16} ρ(вола, CAGR) = {rho_cv:+.3f}   '
            f'ρ(вола, Sharpe) = {rho_sv:+.3f}   n = {len(sub):,}')
    say()
    say("Регрессия CAGR на волатильность (вся популяция, доли, не проценты):")
    say(f'{"":<30} {"коэф.":>12} {"SE":>10} {"t":>9}')
    r2 = ols(G["cagr"].to_numpy(), G[["vol"]].to_numpy(), ["волатильность"])
    say(f"R² = {r2:.3f}")
    if CHARTS:
        chart_vol_deciles(G)

    # ── 4.2 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("4.2 ЦЕНА ПЛЕЧА: близнецы одного автора с разным масштабом риска")
    say("=" * 104)
    say(f"Пары систем одного автора, корреляция дневных ≥ {TWIN_RHO} на общем окне")
    say(f"≥ {MIN_OVERLAP} дней — то есть одна логика, разный размер. Чистое плечо обязано")
    say("оставлять Sharpe неизменным. Всё, что от него отклоняется, — цена плеча.")
    say()
    vc = G["owner_id"].value_counts()
    multi = vc[vc >= 2].index.tolist()
    cand = G[G["owner_id"].isin(multi)].index.tolist()
    say(f"авторов с ≥ 2 измеримыми системами: {len(multi):,}, "
        f"систем у них: {len(cand):,}")
    # ряды грузим один раз для ВСЕЙ измеримой выборки: они нужны и здесь (пары),
    # и в 4.4б (предиктивный разрез скоса)
    with Pool(NPROC) as pool:
        ser = {x[0]: (x[1], x[2]) for x in
               pool.imap_unordered(series, G.index.tolist(), chunksize=100) if x}
    say(f"рядов загружено: {len(ser):,}")
    rows = []
    for oid in multi:
        ids = [i for i in G.index[G["owner_id"] == oid] if i in ser]
        if len(ids) < 2:
            continue
        for k, (a, b) in enumerate(combinations(sorted(ids), 2)):
            if k >= MAX_PAIRS_AUTHOR:
                break
            da, ra = ser[a]
            db, rb = ser[b]
            common, ia, ib = np.intersect1d(da, db, return_indices=True)
            if len(common) < MIN_OVERLAP:
                continue
            xa, xb = ra[ia], rb[ib]
            if xa.std(ddof=1) <= 0 or xb.std(ddof=1) <= 0:
                continue
            rho = float(np.corrcoef(xa, xb)[0, 1])
            if not np.isfinite(rho) or rho < TWIN_RHO:
                continue
            sa, sb = wstats(common, xa), wstats(common, xb)
            if sa is None or sb is None:
                continue
            # A = менее рискованная нога пары, B = более рискованная
            if sa[1] > sb[1]:
                sa, sb, a, b = sb, sa, b, a
            rows.append({"owner": oid, "lo": a, "hi": b, "n": len(common), "rho": rho,
                         "sh_lo": sa[0], "sh_hi": sb[0], "vol_lo": sa[1], "vol_hi": sb[1],
                         "cagr_lo": sa[2], "cagr_hi": sb[2],
                         "dd_lo": sa[3], "dd_hi": sb[3],
                         "L": sb[1] / max(sa[1], 1e-9)})
    T = pd.DataFrame(rows)
    if not len(T):
        say("пар-близнецов не найдено")
    else:
        T["d_sh"] = T["sh_hi"] - T["sh_lo"]
        T["k_cagr"] = T["cagr_hi"] / T["cagr_lo"].replace(0, np.nan)
        T["k_dd"] = T["dd_hi"] / T["dd_lo"].replace(0, np.nan)
        T.to_csv(OUT, index=False, compression="gzip")
        say(f"найдено пар-близнецов: {len(T):,} у {T['owner'].nunique():,} авторов "
            f"(медиана общего окна {int(T['n'].median())} дней, медиана ρ {T['rho'].median():.2f})")
        say()
        T["lbin"] = pd.cut(T["L"], [0.99, 1.2, 1.5, 2.0, 3.0, 1e9],
                           labels=["1.0–1.2", "1.2–1.5", "1.5–2.0", "2.0–3.0", "> 3"])
        hdr = (f'{"плечо L = вола_B/вола_A":<24} {"пар":>6} {"S ноги A":>10} '
               f'{"S ноги B":>10} {"ΔS = B−A":>11} {"CAGR_B/CAGR_A":>15} {"DD_B/DD_A":>11}')
        say(hdr)
        say("-" * len(hdr))
        for lb, sub in T.groupby("lbin", observed=True):
            say(f'{str(lb):<24} {len(sub):>6,} {sub["sh_lo"].median():>10.2f} '
                f'{sub["sh_hi"].median():>10.2f} {sub["d_sh"].median():>11.2f} '
                f'{sub["k_cagr"].median():>15.2f} {sub["k_dd"].median():>11.2f}')
        say()
        d = T["d_sh"].dropna()
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        say(f'ΔS по всем парам: медиана {d.median():+.3f}, среднее {d.mean():+.3f}, '
            f't = {t:+.2f} (n = {len(d):,})')
        say("🔴 t здесь завышен: пары одного автора не независимы. Смотреть на знак и")
        say("величину медианы, а не на уровень значимости.")
        say()
        say("Регрессия ΔS на log(плечо) по парам:")
        say(f'{"":<30} {"коэф.":>12} {"SE":>10} {"t":>9}')
        ok = T[["d_sh", "L"]].dropna()
        r2 = ols(ok["d_sh"].to_numpy(), np.log(ok[["L"]].to_numpy()), ["log(плечо)"])
        say(f"R² = {r2:.3f}")
        say()
        say("Если плечо бесплатно — коэффициент 0. Отрицательный означает, что удвоение")
        say("риска той же логикой СТОИТ столько-то пунктов Sharpe.")
        if CHARTS:
            chart_twins(T)

    # ── 4.3 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("4.3 ЧЕМ ПЛАТЯТ ЗА ДОХОДНОСТЬ: совместное распределение CAGR и просадки")
    say("=" * 104)
    hdr = (f'{"группа по CAGR":<22} {"систем":>8} {"медиана maxDD":>15} '
           f'{"DD глубже −50 %":>17} {"DD глубже −80 %":>17} {"медиана MAR":>13}')
    say(hdr)
    say("-" * len(hdr))
    bins = [(-9e9, -0.10, "убыток хуже −10 %"), (-0.10, 0.0, "убыток до −10 %"),
            (0.0, 0.15, "0…15 % годовых"), (0.15, 0.30, "15…30 %"),
            (0.30, 0.60, "30…60 %"), (0.60, 9e9, "> 60 %")]
    for lo, hi, lab in bins:
        sub = G[(G["cagr"] > lo) & (G["cagr"] <= hi)]
        if not len(sub):
            continue
        say(f'{lab:<22} {len(sub):>8,} {100*sub["maxdd"].median():>14.1f}% '
            f'{100*(sub["maxdd"] < -0.50).mean():>16.1f}% '
            f'{100*(sub["maxdd"] < -0.80).mean():>16.1f}% {sub["mar"].median():>13.2f}')
    if CHARTS:
        chart_dd_by_return(G, bins)
    say()
    say("MAR = CAGR / |макс. просадка|: сколько годовой доходности приходится на единицу")
    say("худшего провала. Считается по всей истории системы.")
    say()
    hdr = f'{"выборка":<20} {"p5":>8} {"p25":>8} {"медиана":>9} {"p75":>8} {"p95":>8}'
    say(hdr)
    say("-" * len(hdr))
    for lab, sub in (("максDD, все", G), ("максDD, живые", G[G["live"]]),
                     ("максDD, мёртвые", G[~G["live"]])):
        say(f'{lab:<20}' + "".join(f'{100*v:>8.1f}' for v in qtab(sub, "maxdd")))
    for lab, sub in (("MAR, все", G), ("MAR, живые", G[G["live"]])):
        say(f'{lab:<20}' + "".join(f'{v:>8.2f}' for v in qtab(sub, "mar")))
    say()
    say(f'доля систем с просадкой глубже −50 %: {100*(G["maxdd"] < -0.50).mean():.1f} % '
        f'(живых {100*(G[G["live"]]["maxdd"] < -0.50).mean():.1f} %)')
    say(f'доля систем с просадкой глубже −80 %: {100*(G["maxdd"] < -0.80).mean():.1f} % '
        f'(живых {100*(G[G["live"]]["maxdd"] < -0.80).mean():.1f} %)')

    # ── 4.4 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("4.4 ХВОСТЫ: форма распределения и её связь со смертью")
    say("=" * 104)
    say("Положительный скос — профиль трендследящих (много мелких минусов, редкие крупные")
    say("плюсы). Отрицательный — профиль продажи риска (много мелких плюсов, редкий обвал).")
    say("Второй тип на коротком окне даёт КРАСИВЫЙ Sharpe и убивает счёт на хвосте.")
    say()
    say(f'доля систем с положительным скосом: {100*(G["skew"] > 0).mean():.1f} %')
    say(f'медианный скос: {G["skew"].median():+.2f}, медианный эксцесс: '
        f'{G["kurt"].median():.1f}')
    say()
    G["sbin"] = pd.cut(G["skew"], [-1e9, -1.0, -0.3, 0.3, 1.0, 1e9],
                       labels=["< −1 (продажа риска)", "−1…−0.3", "−0.3…+0.3 (симметрия)",
                               "+0.3…+1", "> +1 (трендовые)"])
    hdr = (f'{"группа по скосу":<24} {"систем":>8} {"Sharpe":>8} {"CAGR, %":>9} '
           f'{"maxDD, %":>10} {"эксцесс":>9} {"живых":>8} {"лет жизни":>10}')
    say(hdr)
    say("-" * len(hdr))
    for lb, sub in G.groupby("sbin", observed=True):
        say(f'{str(lb):<24} {len(sub):>8,} {sub["sharpe_ar"].median():>8.2f} '
            f'{100*sub["cagr"].median():>9.1f} {100*sub["maxdd"].median():>10.1f} '
            f'{sub["kurt"].median():>9.1f} {100*sub["live"].mean():>7.1f}% '
            f'{sub["years"].median():>10.1f}')
    say()
    for lab, a, b in (("скос ↔ Sharpe", "skew", "sharpe_ar"),
                      ("скос ↔ CAGR", "skew", "cagr"),
                      ("скос ↔ maxDD", "skew", "maxdd"),
                      ("эксцесс ↔ Sharpe", "kurt", "sharpe_ar"),
                      ("скос ↔ срок жизни", "skew", "years")):
        s = G[[a, b]].dropna()
        say(f'{lab:<22} Spearman ρ = {stats.spearmanr(s[a], s[b]).statistic:+.3f}')
    say()
    say("── 4.4б СКОС КАК ПРОГНОЗ, А НЕ ДИАГНОЗ ──────────────────────────────────")
    say("🔴 Всё выше считано по ПОЛНОЙ истории, включая день краха. Тогда отрицательный")
    say("скос — не «профиль продажи риска», а «обвал уже случился»: один катастрофический")
    say("день одновременно топит скос, Sharpe и CAGR, и связь между ними механическая.")
    say("Честная проверка — скос ПЕРВОЙ половины истории против результата ВТОРОЙ.")
    say()
    hrows = []
    for sid, (d, r) in ser.items():
        h = len(r) // 2
        r1, r2, d1, d2 = r[:h], r[h:], d[:h], d[h:]
        if min(len(r1), len(r2)) < 100 or min((r1 != 0).sum(), (r2 != 0).sum()) < 30:
            continue
        w1, w2 = wstats(d1, r1), wstats(d2, r2)
        if w1 is None or w2 is None:
            continue
        hrows.append({"id": sid, "skew1": float(stats.skew(r1)),
                      "kurt1": float(stats.kurtosis(r1, fisher=False)),
                      "sh1": w1[0], "vol1": w1[1],
                      "sh2": w2[0], "cagr2": w2[2], "dd2": w2[3]})
    H = pd.DataFrame(hrows).set_index("id").join(G[["live"]], how="left")
    say(f"систем с двумя полноценными половинами: {len(H):,}")
    say()
    H["sbin1"] = pd.cut(H["skew1"], [-1e9, -1.0, -0.3, 0.3, 1.0, 1e9],
                        labels=["< −1", "−1…−0.3", "−0.3…+0.3", "+0.3…+1", "> +1"])
    hdr = (f'{"скос 1-й половины":<20} {"систем":>8} {"S 1-й пол.":>12} '
           f'{"S 2-й пол.":>12} {"CAGR 2-й, %":>13} {"maxDD 2-й, %":>14} {"живых":>8}')
    say(hdr)
    say("-" * len(hdr))
    for lb, sub in H.groupby("sbin1", observed=True):
        say(f'{str(lb):<20} {len(sub):>8,} {sub["sh1"].median():>12.2f} '
            f'{sub["sh2"].median():>12.2f} {100*sub["cagr2"].median():>12.1f}% '
            f'{100*sub["dd2"].median():>13.1f}% {100*sub["live"].mean():>7.1f}%')
    say()
    say(f'{"связь":<38} {"Spearman ρ":>12}')
    say("-" * 52)
    for lab, a, b in (("скос 1-й пол. ↔ Sharpe ТОЙ ЖЕ половины", "skew1", "sh1"),
                      ("скос 1-й пол. ↔ Sharpe 2-й половины", "skew1", "sh2"),
                      ("скос 1-й пол. ↔ просадка 2-й половины", "skew1", "dd2"),
                      ("эксцесс 1-й пол. ↔ Sharpe 2-й половины", "kurt1", "sh2"),
                      ("Sharpe 1-й пол. ↔ Sharpe 2-й половины", "sh1", "sh2"),
                      ("вола 1-й пол. ↔ Sharpe 2-й половины", "vol1", "sh2")):
        s = H[[a, b]].dropna()
        say(f'{lab:<38} {stats.spearmanr(s[a], s[b]).statistic:>12.3f}')
    say()
    say("Первая строка — механическая связь внутри одного окна (эталон завышения).")
    say("Вторая — настоящая предсказательная сила формы распределения.")

    (ROOT / "results" / "comon_leverage_risk.log").write_text(
        "\n".join(_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
