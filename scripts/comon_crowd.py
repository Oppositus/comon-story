"""comon_crowd.py — БЛОК 9 (закрытие): подписчики на ЧЕСТНЫХ метриках.

План исследования, Блок 9. Первый проход
(2026-08-05, `comon_followers.py`) считался ЦЕЛИКОМ на витринных полях карточки:
`annualAverageProfit`, `maxDrawDown`, `conditionalValueAtRisk`, `riskLevel`. Тогда
рядов живых систем ещё не было. Теперь они есть (Блоки 2-8), и главный вопрос блока
проверяем прямо: реагирует ли толпа на КАЧЕСТВО или на ПОКАЗАННЫЕ ЧИСЛА.

🔴 ЧТО ЗДЕСЬ НЕЛЬЗЯ. Истории подписок в API нет (проверено семью эндпойнтами) —
только текущий срез. Поэтому ни причинности, ни реакции на просадку тут не будет:
всё, что можно, — связи в срезе. «Толпа выбирает X» ниже везде означает «системы с
большим X сейчас имеют больше подписчиков», а не «подписчик посмотрел на X».

🔴 ЧЕМ ЧЕСТНЫЕ МЕТРИКИ ОТЛИЧАЮТСЯ ОТ ВИТРИННЫХ. Витрина показывает ПОЖИЗНЕННЫЕ
величины, поэтому основная база сравнения — тоже пожизненные метрики из рядов
(панель Блока 0): один горизонт с тем, что видит подписчик. Дополнительно берутся
ОКОННЫЕ метрики (2022-10..2026-08, ≥ 3 лет внутри) — только там осмыслен избыточный
Sharpe над ставкой, потому что ставка за 2008-2026 несравнима.

Разделы (нумерация продолжает первый проход, разделы 9.1-9.3 которого — срез витрины):
  9.4 честные метрики против витринных: на что похожа популярность на самом деле;
  9.5 Sharpe, который реально несут деньги (money-weighted против equal-weighted);
  9.6 платят ли за презентацию (описание, теги, аватар, рейтинг, premium);
  9.7 эффект автора: выбирают системы или людей;
  9.8 чувствительность к цене и класс активов;
  9.9 всё сразу: ранговая регрессия популярности;
  9.10 можно ли сегментировать спрос или есть только среднее.

Текстовый вывод. Запуск: python comon_crowd.py
"""
import gzip
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit", "cards")       # первичных данных нет в репозитории — см. DATA.md
LOG = ROOT / "results" / "comon_crowd.log"
NPROC = 8

WIN_FROM, WIN_TILL = "2022-10-12", "2026-08-04"   # окно Блоков 7 и 8
MIN_WIN_DAYS = 250
MIN_WIN_YEARS = 3.0
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Прогон читает карточки и ряды всех живых систем; величины для картинок
# кладутся рядом с логом, чтобы оформление правилось без пересчёта.
CACHE = ROOT / "results" / "comon_crowd_charts.npz"


def chart_popularity_model(rows, r2, n, n_auth):
    """График 30: форест-плот модели популярности с кластерными интервалами."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=1000, bottom=0.26)
    f.subplots_adjust(left=0.30)
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=ch.INK, lw=1.4, zorder=3)
    shown = set()
    for y, r in zip(ys, rows):
        sig = abs(r["b"]) >= 2 * r["se"]
        color = ch.BLUE if sig else ch.PALE
        lab = None
        key = "sig" if sig else "ns"
        if key not in shown:
            shown.add(key)
            lab = ("интервал не накрывает ноль — связь есть"
                   if sig else "интервал накрывает ноль — связи не видно")
        ax.plot([r["b"] - 1.96 * r["se"], r["b"] + 1.96 * r["se"]], [y, y],
                color=color, lw=2.4, zorder=4, label=lab)
        ax.plot([r["b"]], [y], marker="o", ms=9, color=color, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["lab"] for r in rows], fontsize=11)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-0.45, 0.55)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", handlelength=2.4, labelspacing=0.55)
    ax.set_xlabel("вклад признака в популярность, в стандартных отклонениях ранга")
    # 🔴 Заголовок — на фигуре, а не на панели: панель начинается на трети ширины
    # (слева длинные подписи признаков), и заголовок, центрованный по ней,
    # обрезался правым краем файла.
    f.suptitle("Подписки идут за витринной доходностью и рейтингом, а не за качеством",
               fontsize=14, fontweight="bold", y=0.975, va="top")
    ch.note(ax,
            f"Модель на {ch.n_(n)} живых стратегиях с рядом: ранг числа подписчиков "
            f"объясняется рангами признаков, коэффициенты стандартизованы и потому "
            f"сравнимы между собой; объяснено {ch.n_(100*r2, 1)} % разброса.",
            f"Интервалы кластерно-робастные по авторам ({ch.n_(n_auth)} кластеров): "
            f"стратегии одного человека зависимы, и обычные интервалы были бы уже "
            f"настоящих. Честный Sharpe — единственный признак качества в модели — "
            f"ноль накрывает.")
    ch.save(f, 30, "popularity-forest")
    print("    " + "; ".join(f'{r["lab"]}: {r["b"]:+.3f}±{1.96*r["se"]:.3f}'
                             for r in rows), flush=True)


# ── признаки карточки: презентация, тариф, состав ─────────────────────────────
def card_feats(sid):
    f = DIR / "cards" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        c = json.loads(gzip.open(f, "rt").read()).get("data") or {}
    except Exception:                                                   # noqa: BLE001
        return None
    txt = (c.get("textDescription") or c.get("description") or "")
    tags = c.get("tags") or []
    st = {x["id"]: x["value"] for x in (c.get("structure") or [])}
    tar = c.get("autoFollowingTariffDetails") or []
    mgmt, succ = [], []
    import re
    for t in tar:
        d = t.get("description") or ""
        m = re.search(r"([\d,\.]+)\s*%\s*годовых\s*от\s*СЧА", d)
        s = re.search(r"([\d,\.]+)\s*%\s*от\s*инвестиционного\s*дохода", d)
        mgmt.append(float(m.group(1).replace(",", ".")) / 100 if m else 0.0)
        succ.append(float(s.group(1).replace(",", ".")) / 100 if s else 0.0)
    return {"id": sid,
            "desc_len": len(txt), "has_desc": int(len(txt) > 0),
            "n_tags": len(tags), "has_tags": int(len(tags) > 0),
            "has_avatar": int(bool(c.get("authorAvatarUrl"))),
            "premium": int(bool(c.get("premium"))),
            "rating": float(c.get("strategyRating") or 0.0),
            "accuracy": (float(c["accuracyFollowingPercent"])
                         if c.get("accuracyFollowingPercent") is not None else np.nan),
            "n_options": len(c.get("options") or []),
            "card_30": c.get("profit30Days"), "card_90": c.get("profit90Days"),
            # 🔴 Отсутствие ключа означает НОЛЬ в этом активе, а не «неизвестно».
            # Раньше здесь стоял np.nan, и ниже выборка резалась по fut.notna() —
            # то есть из разреза классов вылетали 910 систем из 1 425, причём
            # не случайно, а все, у кого нет фьючерсной строки (чистые акции+деньги).
            "has_struct": bool(st),
            "fut": st.get("Fut", 0.0), "eq_": st.get("Micex", 0.0),
            "bond": st.get("Gko", 0.0), "cash": st.get("Money", 0.0),
            "n_tariff": len(tar),
            "mgmt": min(mgmt) if mgmt else np.nan,
            "succ": max(succ) if succ else np.nan}


# ── оконные метрики (для избыточного Sharpe) ──────────────────────────────────
_RF = {}


def _init(rf):
    global _RF
    _RF = rf


def win_metrics(sid):
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        s_ = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    except Exception:                                                   # noqa: BLE001
        return None
    if not s_:
        return None
    s_ = s_[::-1]
    d = [x["date"] for x in s_]
    r = np.array([float(x["rValue"] or 0.0) for x in s_], dtype=np.float64)
    m = np.array([(WIN_FROM <= x <= WIN_TILL) for x in d])
    if m.sum() < MIN_WIN_DAYS:
        return None
    dd = [x for x, k in zip(d, m) if k]
    rr = r[m]
    yrs = max((date.fromisoformat(dd[-1]) - date.fromisoformat(dd[0])).days / 365.25, 1e-9)
    if yrs < MIN_WIN_YEARS or rr.std(ddof=1) <= 0:
        return None
    freq = len(rr) / yrs
    rf_d = pd.Series([_RF.get(x, np.nan) for x in dd]).ffill().bfill().to_numpy() / freq
    ex = rr - rf_d
    eq = np.cumprod(1.0 + rr)
    peak = np.maximum.accumulate(eq)
    neg = rr[rr < 0]
    return {"id": sid,
            "w_sh": float(rr.mean() / rr.std(ddof=1) * np.sqrt(freq)),
            "w_sh_ex": float(ex.mean() / ex.std(ddof=1) * np.sqrt(freq)),
            "w_vol": float(rr.std(ddof=1) * np.sqrt(freq)),
            "w_cagr": float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0,
            "w_maxdd": float(np.min(eq / peak - 1.0)),
            "w_sortino": (float(rr.mean() / neg.std(ddof=1) * np.sqrt(freq))
                          if len(neg) > 5 and neg.std(ddof=1) > 0 else np.nan)}


# ── статистика ────────────────────────────────────────────────────────────────
def sp(a, b):
    """Ранговая корреляция Спирмена с числом пар."""
    d = pd.concat([pd.to_numeric(a, errors="coerce"),
                   pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(d) < 30:
        return np.nan, len(d)
    return float(sst.spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic), len(d)


def psp(df, x, y, ctrl):
    """Частичная ранговая корреляция x и y при контроле ctrl (на рангах)."""
    cols = [x, y] + list(ctrl)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 50:
        return np.nan, len(d)
    R = d.rank()
    C = np.column_stack([np.ones(len(R))] + [R[c].to_numpy() for c in ctrl])

    def res(v):
        b, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b
    a1, a2 = res(R[x].to_numpy()), res(R[y].to_numpy())
    if a1.std() == 0 or a2.std() == 0:
        return np.nan, len(d)
    return float(np.corrcoef(a1, a2)[0, 1]), len(d)


def wq(v, w, p):
    """Взвешенный квантиль."""
    d = pd.DataFrame({"v": v, "w": w}).dropna()
    d = d[d["w"] > 0].sort_values("v")
    if not len(d):
        return np.nan
    cw = d["w"].cumsum() / d["w"].sum()
    return float(d["v"].to_numpy()[np.searchsorted(cw.to_numpy(), p)])


def wmean(v, w, trim=0.05):
    """Взвешенное среднее с усечением хвостов.

    🔴 Усечение обязательно: у нескольких живых систем ряд почти константный
    (вола ~1e-6), и обычный Sharpe у них улетает в десятки тысяч — среднее по
    сырым величинам показывает не популяцию, а эти три строки.
    """
    d = pd.DataFrame({"v": v, "w": w}).dropna()
    d = d[d["w"] > 0]
    if not len(d):
        return np.nan
    lo, hi = d["v"].quantile(trim), d["v"].quantile(1 - trim)
    d = d[(d["v"] >= lo) & (d["v"] <= hi)]
    return float((d["v"] * d["w"]).sum() / d["w"].sum()) if len(d) else np.nan


def tmean(v, trim=0.05):
    """Усечённое среднее (то же обоснование, что и в wmean)."""
    v = pd.Series(v).dropna()
    if not len(v):
        return np.nan
    lo, hi = v.quantile(trim), v.quantile(1 - trim)
    v = v[(v >= lo) & (v <= hi)]
    return float(v.mean()) if len(v) else np.nan


def gini(v):
    s = np.sort(np.asarray(v, dtype=float))
    n = len(s)
    if n == 0 or s.sum() == 0:
        return np.nan
    return float((2 * np.sum((np.arange(1, n + 1)) * s)) / (n * s.sum()) - (n + 1) / n)


def draw_from_cache():
    # 🔴 График 17 (витринная просадка против реальной) живёт НЕ здесь, а в
    # comon_panel.py: у живых стратегий эти две величины совпадают тождественно
    # (ρ = +1.000, все точки на диагонали), и расхождение появляется только на
    # полной популяции, где витрина архивной системы смягчает провал.
    d = np.load(CACHE, allow_pickle=False)
    chart_popularity_model(
        [{"lab": str(l), "b": float(b), "se": float(e)}
         for l, b, e in zip(d["labs"], d["b"], d["se"])],
        float(d["r2"]), int(d["n"]), int(d["n_auth"]))


def main():
    if ONLY:
        draw_from_cache()
        return
    from comon_sharpe_dist import rusfar                                # noqa: E402
    rf = rusfar()
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    ids = panel.index.tolist()

    say("=" * 104)
    say("БЛОК 9 (ЗАКРЫТИЕ). ПОДПИСЧИКИ НА ЧЕСТНЫХ МЕТРИКАХ")
    say("=" * 104)
    say("первый проход 2026-08-05 считался на витринных полях; здесь — на рядах")
    say()

    with Pool(NPROC) as pool:
        cf = [x for x in pool.imap_unordered(card_feats, ids, chunksize=200) if x]
    C = pd.DataFrame(cf).set_index("id")
    with Pool(NPROC, initializer=_init, initargs=(rf,)) as pool:
        wm = [x for x in pool.imap_unordered(win_metrics, ids, chunksize=200) if x]
    Wm = pd.DataFrame(wm).set_index("id")

    D = panel.join(C, how="left", rsuffix="_c").join(Wm, how="left")
    D["live"] = D["is_live"] == True                                    # noqa: E712
    D["age"] = [(date(2026, 8, 4) - date.fromisoformat(str(x)[:10])).days / 365.25
                if isinstance(x, str) else np.nan for x in D["created_at"]]
    D["abs_dd"] = -D["maxdd"]           # положительная величина риска (пожизненная)
    D["abs_dd_card"] = -D["card_maxdd"]
    D["abs_cvar"] = -D["card_cvar"]
    L = D[D["live"] & D["followers"].notna()].copy()
    LS = L[L["n_pts"].notna() & L["sharpe"].notna()].copy()   # живые с рядами

    say(f'живых систем в панели: {int(D["live"].sum()):,}; из них с рядами: {len(LS):,}; '
        f'подписок у них: {int(LS["followers"].sum()):,}')
    say(f'живых с ≥ {MIN_WIN_YEARS:.0f} годами внутри окна {WIN_FROM}..{WIN_TILL}: '
        f'{int(LS["w_sh"].notna().sum()):,}')
    say()

    # ── 9.4 честные метрики против витринных ──────────────────────────────────
    say("=" * 104)
    say("9.4 ЧЕСТНЫЕ МЕТРИКИ ПРОТИВ ВИТРИННЫХ: НА ЧТО ПОХОЖА ПОПУЛЯРНОСТЬ")
    say("=" * 104)
    say("Вопрос блока: витрина показывает annualAverageProfit / maxDrawDown / CVaR.")
    say("Если корреляция популярности с ЧЕСТНЫМ качеством слабее, чем с ПОКАЗАННЫМИ")
    say("числами, значит толпа реагирует на витрину, а не на систему.")
    say()
    say("Сначала — насколько витринные поля вообще соответствуют рядам (живые с рядами):")
    hdr = f'{"пара":<52} {"ρ Спирмена":>12} {"n":>7}'
    say(hdr)
    say("-" * len(hdr))
    for lab, a, b in (
            ("витринная год. доходность ↔ честный CAGR", "card_annual", "cagr"),
            ("витринная просадка ↔ честная просадка (обе глубиной)",
             "abs_dd_card", "abs_dd"),
            ("витринный CVaR (глубиной) ↔ честная волатильность", "abs_cvar", "vol"),
            ("витринный CVaR (глубиной) ↔ честная просадка", "abs_cvar", "abs_dd"),
            ("ярлык риска (1-3) ↔ честная волатильность", "risk_level", "vol"),
            ("ярлык риска (1-3) ↔ честная просадка", "risk_level", "abs_dd"),
            ("рейтинг площадки ↔ честный Sharpe", "rating", "sharpe")):
        r_, n_ = sp(LS[a], LS[b])
        say(f'{lab:<52} {r_:>+12.3f} {n_:>7,}')
    say()
    say("Теперь корреляция ПОПУЛЯРНОСТИ (число подписчиков) с тем и другим:")
    hdr = f'{"величина":<52} {"ρ с подписчиками":>18} {"n":>7}'
    say(hdr)
    say("-" * len(hdr))
    say("  ВИТРИННЫЕ ПОЛЯ (то, что видно в карточке):")
    for lab, col in (("годовая доходность (annualAverageProfit)", "card_annual"),
                     ("доходность за 365 дней", "card_365"),
                     ("доходность за 30 дней", "card_30"),
                     ("максимальная просадка (глубина, +)", "abs_dd_card"),
                     ("CVaR «прогноз просадки» (глубина, +)", "abs_cvar"),
                     ("ярлык уровня риска (1-3)", "risk_level"),
                     ("рейтинг площадки", "rating"),
                     ("точность следования, %", "accuracy")):
        r_, n_ = sp(LS[col], LS["followers"])
        say(f'    {lab:<50} {r_:>+16.3f} {n_:>7,}')
    say("  ЧЕСТНЫЕ МЕТРИКИ ИЗ РЯДА (пожизненные, тот же горизонт, что у витрины):")
    for lab, col in (("CAGR", "cagr"),
                     ("Sharpe", "sharpe"),
                     ("Sortino", "sortino"),
                     ("волатильность", "vol"),
                     ("максимальная просадка (глубина, +)", "abs_dd"),
                     ("скос", "skew"),
                     ("эксцесс", "kurt"),
                     ("возраст, лет", "age"),
                     ("доля торговых дней", "n_trade")):
        r_, n_ = sp(LS[col], LS["followers"])
        say(f'    {lab:<50} {r_:>+16.3f} {n_:>7,}')
    say("  ЧЕСТНЫЕ МЕТРИКИ НА ОДНОМ ОКНЕ (2022-10..2026-08, где есть ставка):")
    for lab, col in (("CAGR на окне", "w_cagr"),
                     ("Sharpe на окне (сырой)", "w_sh"),
                     ("Sharpe на окне ИЗБЫТОЧНЫЙ над ставкой", "w_sh_ex"),
                     ("Sortino на окне", "w_sortino"),
                     ("волатильность на окне", "w_vol"),
                     ("просадка на окне (глубина, +)", "w_maxdd")):
        v = -LS[col] if col == "w_maxdd" else LS[col]
        r_, n_ = sp(v, LS["followers"])
        say(f'    {lab:<50} {r_:>+16.3f} {n_:>7,}')
    say()
    say("ЧАСТИЧНЫЕ КОРРЕЛЯЦИИ (что остаётся, если убрать эффект доходности и возраста):")
    hdr = f'{"величина":<52} {"ρ парная":>10} {"ρ частичная":>13} {"n":>7}'
    say(hdr)
    say("-" * len(hdr))
    for lab, col in (("честный Sharpe", "sharpe"),
                     ("честный Sortino", "sortino"),
                     ("честная просадка (глубина, +)", "abs_dd"),
                     ("витринная просадка (глубина, +)", "abs_dd_card"),
                     ("витринный CVaR (глубина, +)", "abs_cvar"),
                     ("ярлык уровня риска", "risk_level")):
        p0, _ = sp(LS[col], LS["followers"])
        p1, n_ = psp(LS, col, "followers", ["card_annual", "age"])
        say(f'{lab:<52} {p0:>+10.3f} {p1:>+13.3f} {n_:>7,}')
    say("(контроль: витринная годовая доходность + возраст — то, что видно первым)")
    say()

    # ── 9.5 Sharpe, который несут деньги ──────────────────────────────────────
    say("=" * 104)
    say("9.5 SHARPE, ПОД КОТОРЫМ ЛЕЖАТ ДЕНЬГИ (money-weighted против equal-weighted)")
    say("=" * 104)
    say("Питает Блок 2: одно дело — какой Sharpe БЫВАЕТ на витрине, другое — под каким")
    say("Sharpe лежат реальные подписки. Веса — число подписчиков.")
    say()
    for base, lab_b, sub in (("sharpe", "ПОЖИЗНЕННЫЙ Sharpe (живые с рядами)", LS),
                             ("w_sh", "Sharpe НА ОКНЕ, сырой", LS[LS["w_sh"].notna()]),
                             ("w_sh_ex", "Sharpe НА ОКНЕ, ИЗБЫТОЧНЫЙ над ставкой",
                              LS[LS["w_sh_ex"].notna()])):
        v, w = sub[base], sub["followers"]
        say(f'{lab_b} (n = {len(sub):,}, подписок {int(w.sum()):,}):')
        hdr = (f'  {"вес":<22} {"p10":>8} {"p25":>8} {"медиана":>9} {"p75":>8} '
               f'{"p90":>8} {"ср. усеч.":>10}')
        say(hdr)
        say("  " + "-" * (len(hdr) - 2))
        say(f'  {"равновзвешенно":<22} {v.quantile(.1):>8.2f} {v.quantile(.25):>8.2f} '
            f'{v.median():>9.2f} {v.quantile(.75):>8.2f} {v.quantile(.9):>8.2f} '
            f'{tmean(v):>10.2f}')
        say(f'  {"по подпискам":<22} {wq(v, w, .1):>8.2f} {wq(v, w, .25):>8.2f} '
            f'{wq(v, w, .5):>9.2f} {wq(v, w, .75):>8.2f} {wq(v, w, .9):>8.2f} '
            f'{wmean(v, w):>10.2f}')
        say()
    say("Доля СИСТЕМ и доля ПОДПИСОК по порогам качества:")
    hdr = (f'{"критерий":<44} {"систем":>9} {"доля систем":>13} {"подписок":>10} '
           f'{"доля подписок":>15}')
    say(hdr)
    say("-" * len(hdr))
    tot_s, tot_w = len(LS), LS["followers"].sum()
    for lab, msk in (("пожизненный Sharpe < 0", LS["sharpe"] < 0),
                     ("пожизненный Sharpe ≥ 1", LS["sharpe"] >= 1),
                     ("пожизненный Sharpe ≥ 1.5", LS["sharpe"] >= 1.5),
                     ("пожизненная просадка глубже −30 %", LS["maxdd"] < -0.30),
                     ("пожизненная просадка глубже −50 %", LS["maxdd"] < -0.50),
                     ("избыточный Sharpe на окне < 0", LS["w_sh_ex"] < 0),
                     ("избыточный Sharpe на окне ≥ 1", LS["w_sh_ex"] >= 1)):
        m = msk.fillna(False)
        say(f'{lab:<44} {int(m.sum()):>9,} {100*m.sum()/tot_s:>12.1f}% '
            f'{int(LS.loc[m, "followers"].sum()):>10,} '
            f'{100*LS.loc[m, "followers"].sum()/tot_w:>14.1f}%')
    say()

    # ── 9.6 презентация ───────────────────────────────────────────────────────
    say("=" * 104)
    say("9.6 ПЛАТЯТ ЛИ ЗА ПРЕЗЕНТАЦИЮ")
    say("=" * 104)
    say("Отделяем «выбрали за результат» от «выбрали за оформление»: признаки подачи")
    say("против популярности, парно и при контроле доходности с возрастом.")
    say()
    hdr = (f'{"признак подачи":<40} {"ρ парная":>10} {"ρ при контроле":>16} {"n":>7}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col in (("длина описания, символов", "desc_len"),
                     ("описание вообще есть (0/1)", "has_desc"),
                     ("число тегов", "n_tags"),
                     ("теги есть (0/1)", "has_tags"),
                     ("аватар автора есть (0/1)", "has_avatar"),
                     ("рейтинг площадки", "rating"),
                     ("число заполненных опций карточки", "n_options"),
                     ("число тарифных планов", "n_tariff")):
        p0, _ = sp(LS[col], LS["followers"])
        p1, n_ = psp(LS, col, "followers", ["card_annual", "age"])
        say(f'{lab:<40} {p0:>+10.3f} {p1:>+16.3f} {n_:>7,}')
    say(f'(premium-метки нет ни у одной живой системы: '
        f'{int(LS["premium"].fillna(0).sum())} шт — признак исключён)')
    say()
    say("Группы по длине описания (медианы внутри группы; медиана подписчиков по")
    say("витрине равна нулю, поэтому вместо неё — доля систем, у кого подписчик есть):")
    Q = LS[LS["desc_len"].notna()].copy()
    Q["g"] = pd.cut(Q["desc_len"], [-1, 0, 200, 800, 10 ** 9],
                    labels=["нет описания", "1-200", "201-800", "> 800"])
    hdr = (f'{"описание":<16} {"систем":>7} {"подписок":>10} {"с подписчиками":>16} '
           f'{"честный Sharpe":>15} {"CAGR, %":>9} {"возраст, лет":>13}')
    say(hdr)
    say("-" * len(hdr))
    for g, sub in Q.groupby("g", observed=True):
        say(f'{str(g):<16} {len(sub):>7,} {int(sub["followers"].sum()):>10,} '
            f'{100*float((sub["followers"] > 0).mean()):>15.1f}% '
            f'{sub["sharpe"].median():>15.2f} '
            f'{100*sub["cagr"].median():>9.1f} {sub["age"].median():>13.1f}')
    say()

    # ── 9.7 эффект автора ─────────────────────────────────────────────────────
    say("=" * 104)
    say("9.7 ЭФФЕКТ АВТОРА: ВЫБИРАЮТ СИСТЕМЫ ИЛИ ЛЮДЕЙ")
    say("=" * 104)
    A = LS[LS["owner_id"].notna()].copy()
    ao = A.groupby("owner_id")["followers"].sum().sort_values(ascending=False)
    say(f'авторов среди живых с рядами: {len(ao):,}; подписок: {int(ao.sum()):,}')
    say(f'Джини подписок ПО АВТОРАМ: {gini(ao.to_numpy()):.3f} '
        f'(по системам в первом проходе было 0.931)')
    for k in (5, 10, 50):
        say(f'  доля топ-{k:<3} авторов в подписках: '
            f'{100*ao.head(k).sum()/ao.sum():.1f} %')
    say(f'  авторов без единого подписчика: {int((ao == 0).sum()):,} '
        f'({100*float((ao == 0).mean()):.1f} %)')
    say()
    multi = A.groupby("owner_id").filter(lambda g: len(g) >= 2)
    say(f'авторов с ≥ 2 живыми системами: {multi["owner_id"].nunique():,} '
        f'(систем {len(multi):,})')
    if len(multi) > 50:
        y = np.log1p(multi["followers"].to_numpy(dtype=float))
        g = multi["owner_id"].to_numpy()
        gm = pd.Series(y).groupby(g).transform("mean").to_numpy()
        ss_b = float(((gm - y.mean()) ** 2).sum())
        ss_t = float(((y - y.mean()) ** 2).sum())
        say(f'  доля разброса популярности (log1p подписчиков), объяснённая ЛИЧНОСТЬЮ '
            f'автора: {100*ss_b/ss_t:.1f} %')
        # переносится ли: популярность системы против средней по ДРУГИМ системам автора
        s = multi.groupby("owner_id")["followers"].transform("sum")
        n = multi.groupby("owner_id")["followers"].transform("size")
        loo = (s - multi["followers"]) / (n - 1)
        r_, n_ = sp(loo, multi["followers"])
        say(f'  ρ(подписчики системы, средние подписчики ДРУГИХ систем того же автора) '
            f'= {r_:+.3f} (n = {n_:,})')
        r2, n2 = sp(multi.groupby("owner_id")["sharpe"].transform(
            lambda v: (v.sum() - v) / max(len(v) - 1, 1)), multi["followers"])
        say(f'  ρ(подписчики системы, средний честный Sharpe ДРУГИХ систем автора) '
            f'= {r2:+.3f} (n = {n2:,})')
    say()
    say("Топ-10 авторов по подпискам (все их живые системы с рядами):")
    hdr = (f'{"автор":<26} {"систем":>7} {"подписок":>10} {"медиана Sharpe":>15} '
           f'{"медиана CAGR, %":>16} {"медиана maxDD, %":>17}')
    say(hdr)
    say("-" * len(hdr))
    for oid in ao.head(10).index:
        sub = A[A["owner_id"] == oid]
        nm = str(sub["author"].iloc[0])[:24]
        say(f'{nm:<26} {len(sub):>7} {int(sub["followers"].sum()):>10,} '
            f'{sub["sharpe"].median():>15.2f} {100*sub["cagr"].median():>16.1f} '
            f'{100*sub["maxdd"].median():>17.1f}')
    say()

    # ── 9.8 цена и класс активов ──────────────────────────────────────────────
    say("=" * 104)
    say("9.8 ЧУВСТВИТЕЛЬНОСТЬ К ЦЕНЕ И КЛАСС АКТИВОВ")
    say("=" * 104)
    hdr = (f'{"признак":<40} {"ρ парная":>10} {"ρ при контроле":>16} {"n":>7}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col in (("порог входа minSum, ₽", "min_sum"),
                     ("ставка от СЧА, % годовых", "mgmt"),
                     ("есть плата от дохода (доля)", "succ"),
                     ("лимит средств moneyLimit", "money_limit")):
        p0, _ = sp(LS[col], LS["followers"])
        p1, n_ = psp(LS, col, "followers", ["card_annual", "age"])
        say(f'{lab:<40} {p0:>+10.3f} {p1:>+16.3f} {n_:>7,}')
    say("(контроль: витринная годовая доходность + возраст)")
    say()
    S = LS[LS["has_struct"] == True].copy()                             # noqa: E712
    if len(S):
        # 🔴 Деньги — НЕ класс активов, а состояние счёта на день выгрузки.
        # Прежняя версия ставила «деньги» пятым классом наравне с акциями, и в него
        # попадала почти половина витрины: у этих систем медианная волатильность
        # 45.7 % и просадка −44.6 %, то есть это активно торгующие стратегии,
        # которые на день снимка просто были вне позиции. Класс теперь определяется
        # по доминирующему РИСКОВОМУ активу (доля от вложенной части), а доля денег
        # выводится отдельной строкой как характеристика момента.
        def cls(r):
            inv = r["fut"] + r["eq_"] + r["bond"]
            if inv <= 0:
                return "вне рынка на день выгрузки"
            for name, v in (("фьючерсы", r["fut"]), ("акции", r["eq_"]),
                            ("облигации", r["bond"])):
                if v / inv >= 0.5:
                    return name
            return "смешанные"
        S["cls"] = S.apply(cls, axis=1)
        hdr = (f'{"класс активов":<16} {"систем":>7} {"доля систем":>13} '
               f'{"подписок":>10} {"доля подписок":>15} {"медиана Sharpe":>15} '
               f'{"медиана maxDD, %":>17}')
        say(hdr)
        say("-" * len(hdr))
        for k, sub in S.groupby("cls"):
            say(f'{k:<16} {len(sub):>7,} {100*len(sub)/len(S):>12.1f}% '
                f'{int(sub["followers"].sum()):>10,} '
                f'{100*sub["followers"].sum()/S["followers"].sum():>14.1f}% '
                f'{sub["sharpe"].median():>15.2f} '
                f'{100*sub["maxdd"].median():>17.1f}')
        say(f'(состав — снимок карточки на момент выкачки, {len(S):,} живых систем '
            f'с непустой structure)')
        w = np.repeat(S["cash"].values, S["followers"].fillna(0).astype(int).values)
        say(f'доля счёта в деньгах на день выгрузки: медиана по системам '
            f'{S["cash"].median():.0f} %, взвешенно по подпискам {np.median(w):.0f} %; '
            f'ровно 100 % у {int((S["cash"] == 100).sum())} систем, '
            f'больше 100 % — у {int((S["cash"] > 100).sum())} (максимум '
            f'{S["cash"].max():.0f} %)')
        # 🔴 Доли structure НЕ нормированы на 100: деньги доходят до 713 %, то есть
        # поле отражает плечо и короткие позиции, а не разбивку счёта на сто частей.
        # Поэтому классы ниже — приблизительные, по доминирующему риску.
    say()

    # ── 9.9 всё сразу ─────────────────────────────────────────────────────────
    say("=" * 104)
    say("9.9 ВСЁ СРАЗУ: РАНГОВАЯ РЕГРЕССИЯ ПОПУЛЯРНОСТИ")
    say("=" * 104)
    say("Парные корреляции путают эффекты (старые системы и доходнее, и известнее).")
    say("Здесь всё в одной модели: ранг числа подписчиков ~ ранги признаков,")
    say("коэффициенты стандартизованы (сопоставимы между собой).")
    say()
    feats = [("витринная год. доходность", "card_annual"),
             ("честный Sharpe (пожизненный)", "sharpe"),
             ("честная просадка (глубина, +)", "abs_dd"),
             ("возраст, лет", "age"),
             ("порог входа", "min_sum"),
             ("ставка от СЧА", "mgmt"),
             ("длина описания", "desc_len"),
             ("рейтинг площадки", "rating"),
             ("ярлык уровня риска", "risk_level")]
    cols = [c for _, c in feats]
    d = LS[cols + ["followers", "owner_id"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) > 100:
        grp = d["owner_id"].to_numpy()
        d = d.drop(columns=["owner_id"])
        R = d.rank()
        Z = (R - R.mean()) / R.std(ddof=1)
        X = np.column_stack([np.ones(len(Z))] + [Z[c].to_numpy() for c in cols])
        y = Z["followers"].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        dof = len(y) - X.shape[1]
        s2 = float(res @ res) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())

        # 🔴 КЛАСТЕРНО-РОБАСТНЫЕ ОШИБКИ ПО АВТОРАМ (добавлено 2026-08-09).
        # Обычные t предполагают независимость наблюдений, а стратегии одного
        # автора зависимы — §9.7 показывает, что личность автора объясняет 67 %
        # разброса популярности. Значит обычные t завышены. Сэндвич-оценка
        # Лианга—Зегера с поправкой на малое число кластеров.
        XtXi = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(grp):
            m = grp == g
            xg, ug = X[m], res[m]
            s = xg.T @ ug
            meat += np.outer(s, s)
        G = len(np.unique(grp))
        c = (G / (G - 1)) * ((len(y) - 1) / dof)
        se_cl = np.sqrt(np.diag(c * XtXi @ meat @ XtXi))

        hdr = (f'{"признак":<34} {"β станд.":>10} {"t обычн.":>10} '
               f'{"t кластер":>11} {"ρ парная":>10}')
        say(hdr)
        say("-" * len(hdr))
        for i, (lab, col) in enumerate(feats, start=1):
            p0, _ = sp(d[col], d["followers"])
            mark = "" if abs(b[i] / se_cl[i]) >= 2 else "  ← незначим"
            say(f'{lab:<34} {b[i]:>+10.3f} {b[i]/se[i]:>+10.2f} '
                f'{b[i]/se_cl[i]:>+11.2f} {p0:>+10.3f}{mark}')
        say("-" * len(hdr))
        say(f'n = {len(d):,} живых систем · R² = {r2:.3f} '
            f'(доля разброса рангов популярности, объяснённая всеми признаками)')
        say(f'кластеров (авторов): {G:,}; максимум систем у одного: '
            f'{int(pd.Series(grp).value_counts().max())}; '
            f'систем у авторов с >= 2 системами: '
            f'{int((pd.Series(grp).map(pd.Series(grp).value_counts()) >= 2).sum()):,}')
        say('🔴 Ориентироваться на «t кластер»: обычные t предполагают независимость '
            'наблюдений,\n   а стратегии одного автора зависимы (§9.7: автор объясняет '
            '67 % разброса популярности).')

        if CHARTS:
            np.savez(CACHE,
                     b=np.array([b[i] for i in range(1, len(feats) + 1)]),
                     se=np.array([se_cl[i] for i in range(1, len(feats) + 1)]),
                     labs=np.array([lab for lab, _ in feats]),
                     r2=r2, n=len(d), n_auth=G)
            say()
            say("── график 30 ────────────────────────────────────────────────────")
            draw_from_cache()
    say()
    say("🔴 Это срез, а не история: причинность отсюда не следует, а истории подписок")
    say("в API нет. Утверждать можно только «системы с таким-то признаком сейчас")
    say("собирают больше подписок».")

    # ── 9.10 сегментация спроса ───────────────────────────────────────────────
    say()
    say("=" * 104)
    say("9.10 МОЖНО ЛИ СЕГМЕНТИРОВАТЬ СПРОС ИЛИ ЕСТЬ ТОЛЬКО СРЕДНЕЕ")
    say("=" * 104)
    say("🔴 Кластеризовать ПОДПИСЧИКОВ нельзя: индивидуальных записей нет, один человек")
    say("может держать несколько подписок, численность аудитории неизвестна. Единица")
    say("наблюдения здесь — ПОДПИСКА, а не человек; экологическая ошибка неустранима.")
    say("Вопрос раздела уже: описывается ли спрос одним числом или у него есть структура.")
    say()
    F = LS[LS["followers"] > 0].copy()
    wf = F["followers"]
    say(f'систем с ≥ 1 подпиской: {len(F):,}; подписок: {int(wf.sum()):,}')
    say()
    say("9.10.1 РАСПРЕДЕЛЕНИЕ ВМЕСТО СРЕДНЕГО (веса — подписки):")
    hdr = (f'{"величина":<26} {"p10":>9} {"p25":>9} {"медиана":>9} {"p75":>9} {"p90":>9}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col, sc in (("Sharpe пожизненный", "sharpe", 1),
                         ("CAGR, % годовых", "cagr", 100),
                         ("максимальная просадка, %", "maxdd", 100),
                         ("волатильность, %", "vol", 100)):
        v = sc * F[col]
        say(f'{lab:<26} {wq(v, wf, .1):>9.2f} {wq(v, wf, .25):>9.2f} '
            f'{wq(v, wf, .5):>9.2f} {wq(v, wf, .75):>9.2f} {wq(v, wf, .9):>9.2f}')
    say()
    say("9.10.2 ПОРОГ ВХОДА = ЖЁСТКИЙ ПОЛ КАПИТАЛА ПОДПИСЧИКА")
    say("Единственная в данных величина, характеризующая не систему, а человека:")
    say("подписавшийся на систему с порогом 1 млн этот миллион имеет.")
    say()
    say(f'взвешенные квантили порога, тыс ₽: p10 {wq(F["min_sum"], wf, .1)/1e3:.0f} · '
        f'p25 {wq(F["min_sum"], wf, .25)/1e3:.0f} · медиана {wq(F["min_sum"], wf, .5)/1e3:.0f} · '
        f'p75 {wq(F["min_sum"], wf, .75)/1e3:.0f} · p90 {wq(F["min_sum"], wf, .9)/1e3:.0f}')
    say()
    CAPB = [0, 50e3, 150e3, 400e3, 1e6, 1e12]
    CAPL = ["< 50к", "50-150к", "150-400к", "400к-1М", "≥ 1М"]
    F["cap"] = pd.cut(F["min_sum"], CAPB, labels=CAPL)
    hdr = (f'{"сегмент по порогу входа":<24} {"систем":>7} {"доля систем":>13} '
           f'{"подписок":>10} {"доля подписок":>15}')
    say(hdr)
    say("-" * len(hdr))
    for k, sub in F.groupby("cap", observed=True):
        say(f'{str(k):<24} {len(sub):>7} {100*len(sub)/len(F):>12.1f}% '
            f'{int(sub["followers"].sum()):>10,} '
            f'{100*sub["followers"].sum()/wf.sum():>14.1f}%')
    say()
    say("9.10.3 УГЛЫ РИСК-ДОХОДНОСТИ: НАСЕЛЕНЫ ЛИ ОНИ ПО-РАЗНОМУ")
    hdr = f'{"сегмент подписок":<48} {"систем":>7} {"подписок":>10} {"доля":>8}'
    say(hdr)
    say("-" * len(hdr))
    tot = wf.sum()
    for lab, m in (
            ("тихие: просадка мельче −20 % И Sharpe ≥ 1",
             (F["maxdd"] > -0.20) & (F["sharpe"] >= 1)),
            ("агрессивные: просадка глубже −50 % И CAGR ≥ 30 %",
             (F["maxdd"] < -0.50) & (F["cagr"] >= 0.30)),
            ("дешёвый вход < 50к И просадка глубже −50 %",
             (F["min_sum"] < 50e3) & (F["maxdd"] < -0.50)),
            ("мусор: Sharpe < 0", F["sharpe"] < 0),
            ("дорогой вход ≥ 1М И Sharpe ≥ 1",
             (F["min_sum"] >= 1e6) & (F["sharpe"] >= 1))):
        m = m.fillna(False)
        say(f'{lab:<48} {int(m.sum()):>7} {int(wf[m].sum()):>10,} '
            f'{100*wf[m].sum()/tot:>7.1f}%')
    say()
    say("9.10.4 🔴 ЧЕГО ДЕЛАТЬ НЕЛЬЗЯ: СРАВНИВАТЬ СЕГМЕНТЫ МЕЖДУ СОБОЙ")
    say("Соблазн — посмотреть, как выбирает каждый капитальный сегмент. Считаем и")
    say("тут же считаем, на скольких системах эта медиана держится.")
    say()
    hdr = (f'{"сегмент":<12} {"подписок":>10} {"Sharpe":>8} {"CAGR,%":>8} '
           f'{"maxDD,%":>9} {"возраст":>8} {"половину держат":>17} {"эфф. N":>8} '
           f'{"топ-1":>7}')
    say(hdr)
    say("-" * len(hdr))
    for k, sub in F.groupby("cap", observed=True):
        ws = sub["followers"]
        f_ = ws.sort_values(ascending=False)
        sh = f_ / f_.sum()
        n50 = int((sh.cumsum() <= 0.5).sum()) + 1
        eff = 1.0 / float((sh ** 2).sum())
        say(f'{str(k):<12} {int(ws.sum()):>10,} {wq(sub["sharpe"], ws, .5):>8.2f} '
            f'{100*wq(sub["cagr"], ws, .5):>8.1f} {100*wq(sub["maxdd"], ws, .5):>9.1f} '
            f'{wq(sub["years"], ws, .5):>8.1f} {n50:>17} {eff:>8.1f} '
            f'{100*sh.iloc[0]:>6.1f}%')
    f_ = wf.sort_values(ascending=False)
    sh = f_ / f_.sum()
    say("-" * len(hdr))
    say(f'{"ВСЕГО":<12} {int(wf.sum()):>10,} {"":>8} {"":>8} {"":>9} {"":>8} '
        f'{int((sh.cumsum() <= 0.5).sum())+1:>17} '
        f'{1.0/float((sh**2).sum()):>8.1f} {100*sh.iloc[0]:>6.1f}%')
    say()
    say("«половину держат» — сколько систем набирают 50 % подписок сегмента; «эфф. N» —")
    say("1/HHI по долям подписок; «топ-1» — доля самой популярной системы сегмента.")
    say("🔴 Медиана каждого сегмента опирается на 5-7 систем → различия между строками")
    say("суть различия между конкретными системами, а не между группами людей.")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(_lines), encoding="utf-8")
    say()
    say(f"лог: {LOG}")


if __name__ == "__main__":
    main()
