"""comon_tariffs.py — БЛОК 8: экономика следования, что достаётся подписчику.

План исследования, Блок 8. Вопрос блока: доходность на витрине — это
доходность СЧЁТА АВТОРА. Подписчик получает её за вычетом тарифа автоследования, и
вопрос в том, сколько от неё остаётся и сколько систем после этого вообще имеют смысл
против денежного рынка.

🔴 ЧТО ИМЕННО ВЫЧИТАЕТСЯ. `rValue` — доходность счёта автора, в ней уже сидят ЕГО
брокерские комиссии и проскальзывание (Блок 0). Тариф автоследования — сверх этого.
А вот собственные издержки подписчика (комиссия его брокера + проскальзывание при
копировании, тем большее, чем больше подписчиков) в данных не видны вообще: их нельзя
измерить, поэтому все net-числа ниже — ВЕРХНЯЯ граница того, что достаётся подписчику.

🔴 МОДЕЛЬ СПИСАНИЯ. Тариф от СЧА берётся непрерывно (доля годовой ставки на календарные
дни между точками ряда). Тариф от ИД («% от инвестиционного дохода») — в конце периода
(месяц/квартал) от прироста над HWM; вариант без HWM считается отдельно как верхняя
оценка платы. Тарифы разбираются регуляркой из `autoFollowingTariffDetails` карточки;
если у системы несколько тарифов, подписчик выбирает — берётся САМЫЙ ДЕШЁВЫЙ для него
(оптимистично к площадке).

🔴 ОКНО. Всё считается на одном окне (2022-10-12…2026-08-04): сравнение с денежным
рынком имеет смысл только на одном периоде ставок.

Разделы:
  8.1 тарифная сетка: что предлагают и что реально покупают (веса по подпискам);
  8.2 gross → net: сколько остаётся подписчику и сколько систем бьют денежный рынок;
  8.3 порог входа против результата;
  8.4 частота сделок против результата;
  8.5 масштаб платы: сколько площадка собирает со всей витрины.

Текстовый вывод. Запуск: python comon_tariffs.py
"""
import gzip
import json
import re
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
LOG = ROOT / "results" / "comon_tariffs.log"
NPROC = 8

WIN_FROM, WIN_TILL = "2022-10-12", "2026-08-04"
MIN_WIN_DAYS = 250
MIN_WIN_YEARS = 3.0
MIN_ACT_WIN = 0.10
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Полный прогон блока 8 идёт минуты (два Pool-прохода по всем карточкам), а
# оформление графика правится десятками итераций. Поэтому ровно те величины,
# которые нужны картинкам, ложатся рядом с логом, и --charts-only рисует по ним.
CACHE = ROOT / "results" / "comon_tariffs_charts.npz"


def chart_fee_shift(g, n, rf_med):
    """График 29: тариф сдвигает ВСЁ распределение доходности, а не медиану."""
    import comon_charts as ch

    f, ax = ch.fig()
    lo, hi = -60.0, 60.0                      # хвосты за краем поля сводим в края
    bins = np.linspace(lo, hi, 41)
    for v, color, lab, z in (
            (g, ch.BLUE, f"до тарифа — счёт автора, медиана "
                         f"{np.median(g):.1f} %".replace(".", ","), 3),
            (n, ch.ORANGE, f"после тарифа — счёт подписчика, медиана "
                           f"{np.median(n):.1f} %".replace(".", ","), 4)):
        cnt, _ = np.histogram(np.clip(v, lo, hi), bins=bins)
        x = np.repeat(bins, 2)[1:-1]
        y = np.repeat(cnt, 2)
        ax.fill_between(x, 0, y, color=color, alpha=0.22, lw=0, zorder=z)
        ax.plot(x, y, color=color, lw=2.0, zorder=z + 2, label=lab)

    # 🔴 Числа медиан ушли В ЛЕГЕНДУ, а не стоят у своих вертикалей. Четыре
    # вертикали (ноль, две медианы, ставка) отстоят друг от друга на 50-65 px,
    # и любая горизонтальная подпись у одной из них перечёркивается соседней;
    # разводить их по высоте некуда — линия нуля идёт во всю панель.
    mx = max(np.histogram(np.clip(v, lo, hi), bins=bins)[0].max() for v in (g, n))
    ax.axvline(0, color=ch.PALE, lw=1.6, ls=":", zorder=2)
    for v, color in ((np.median(n), ch.ORANGE), (np.median(g), ch.BLUE)):
        ax.plot([v, v], [0, mx * 1.06], color=color, lw=1.4, ls="--", zorder=6)
    ax.plot([rf_med, rf_med], [0, mx * 1.06], color=ch.GREY, lw=1.6, ls="-.",
            zorder=6, label=f"ставка вклада {rf_med:.1f} % годовых".replace(".", ","))

    ax.legend(loc="upper left", handlelength=2.6, labelspacing=0.55)
    top = mx * 1.34
    ch.pct_raw(ax, "x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top)
    ax.set_xlabel("доходность, % годовых")
    ax.set_ylabel("стратегий в интервале")
    ax.set_title("Плата за следование сдвигает всё распределение, а не только медиану")
    ch.note(ax,
            f"База — {len(g):,} живых стратегий с историей не меньше трёх лет внутри "
            f"окна {WIN_FROM}…{WIN_TILL}. Значения за краями поля сведены в крайние "
            f"столбцы.".replace(",", " "),
            "Медиана падает с 10,6 % до 5,4 % годовых — плата 5,7 процентных пункта, "
            "больше половины заработанного. Выше ставки вклада остаются 26,8 % "
            "стратегий против 39,5 % до тарифа.")
    ch.save(f, 29, "fee-shift")
    print(f"    медианы {np.median(g):.2f} → {np.median(n):.2f} %; "
          f"выше нуля {100*(g > 0).mean():.1f} → {100*(n > 0).mean():.1f} %", flush=True)


def chart_threshold_quintiles(rows):
    """График 12: качество растёт с порогом входа, а подписки идут вниз."""
    import comon_charts as ch

    f, axes = ch.fig(h_px=1000, bottom=0.26, ncols=3)
    # 🔴 wspace шире дефолтного: подпись оси Y средней и правой панели заезжала
    # на соседнюю. Подписи делений — только номер квинтиля и медиана порога БЕЗ
    # единицы («тыс. ₽» унесены в название оси): на панели шириной в треть поля
    # «100 тыс.» и «200 тыс.» наезжали друг на друга.
    f.subplots_adjust(wspace=0.34)
    x = list(range(len(rows)))
    labs = [f'{r["q"]}\n{r["thr"]}' for r in rows]

    panels = (("sharpe", "Sharpe до тарифа", ch.BLUE, "{:.2f}", 1.0),
              ("cagr", "чистая доходность, % годовых", ch.BLUE, "{:+.1f}", 1.0),
              ("subs", "подписок", ch.ORANGE, "{:,.0f}", 1.0))
    for ax, (key, ylab, color, fmt, _s) in zip(axes, panels):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, lw=2.4, marker="o", ms=8, zorder=4)
        if key == "cagr":
            ax.axhline(0, color=ch.PALE, lw=1.4, ls=":", zorder=2)
        span = max(y) - min(y)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(min(y) - 0.20 * span, max(y) + 0.28 * span)
        ch.value_labels(ax, x, y,
                        [fmt.format(v).replace(",", " ").replace(".", ",")
                         for v in y], dist=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel(ylab)
        ax.set_xlabel("квинтиль порога входа\n(медиана порога, тыс. ₽)")
    # 🔴 Заголовок один на всю фигуру, у панелей их нет: на панели шириной в
    # треть поля осмысленная фраза шире самой панели и обрезается краем файла.
    f.suptitle("Чем выше порог входа, тем лучше результат — и тем меньше подписок",
               fontsize=14, fontweight="bold", y=0.965)
    ch.note(axes[0],
            "База — 690 стратегий окна с историей не меньше трёх лет и активностью "
            "не ниже 10 % дней: 593 живые и 97 закрытых, по 138 в квинтиле; подписок "
            "10 384, все у живых.",
            "Связь порога с качеством монотонна по всем пяти квинтилям: ρ = +0,296 "
            "с Sharpe и +0,299 с чистой доходностью. Подписки этой связи не "
            "повторяют: 2 539 в самом дешёвом квинтиле против 901 в самом дорогом.")
    ch.save(f, 12, "entry-threshold-quintiles")
    print("    " + "; ".join(f'{r["q"]}: S {r["sharpe"]:.2f}, net {r["cagr"]:+.1f} %, '
                             f'{r["subs"]:,.0f} подп.' for r in rows), flush=True)


# ── разбор тарифа ─────────────────────────────────────────────────────────────
RE_MGMT = re.compile(r"([\d,\.]+)\s*%\s*годовых\s*от\s*СЧА")
RE_SUCC = re.compile(r"([\d,\.]+)\s*%\s*от\s*инвестиционного\s*дохода")
RE_PER_M = re.compile(r"раз\s*в\s*месяц")
RE_PER_Q = re.compile(r"раз\s*в\s*квартал")


def parse_tariff(desc):
    """Описание тарифа → (mgmt доля/год, success доля, период 'M'/'Q'/None)."""
    if not desc:
        return None
    m = RE_MGMT.search(desc)
    s = RE_SUCC.search(desc)
    mgmt = float(m.group(1).replace(",", ".")) / 100.0 if m else 0.0
    succ = float(s.group(1).replace(",", ".")) / 100.0 if s else 0.0
    per = "M" if RE_PER_M.search(desc) else ("Q" if RE_PER_Q.search(desc) else None)
    if not m and not s:
        return None
    return mgmt, succ, per


def card_tariffs(sid):
    """Все тарифы системы (плоско, вложенные не разворачиваем — у комбинированного
    описание уже содержит обе части)."""
    f = DIR / "cards" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        c = json.loads(gzip.open(f, "rt").read()).get("data") or {}
    except Exception:                                                   # noqa: BLE001
        return None
    out = []
    for t in (c.get("autoFollowingTariffDetails") or []):
        p = parse_tariff(t.get("description"))
        if p:
            out.append({"id": sid, "type": t.get("tariffType"),
                        "name": t.get("name"), "desc": t.get("description"),
                        "mgmt": p[0], "succ": p[1], "per": p[2]})
    return out or None


# ── применение тарифа к ряду ──────────────────────────────────────────────────
def net_series(dates, r, mgmt, succ, per, hwm=True):
    """Ряд доходностей ПОСЛЕ тарифа. dates — ISO-строки, r — дневные доли."""
    eq = 1.0
    peak = 1.0
    last_pay = dates[0][:7]
    out = np.empty(len(r))
    prev = date.fromisoformat(dates[0])
    base = eq
    for i, (d, x) in enumerate(zip(dates, r)):
        cur = date.fromisoformat(d)
        dt = max((cur - prev).days, 0) if i else 0
        prev = cur
        e0 = eq
        eq *= (1.0 + x)
        if mgmt:
            eq *= (1.0 - mgmt * dt / 365.0)
        if succ and per:
            k = d[:7] if per == "M" else f"{d[:4]}Q{(int(d[5:7]) - 1) // 3}"
            lk = last_pay if per == "M" else \
                f"{last_pay[:4]}Q{(int(last_pay[5:7]) - 1) // 3}"
            if k != lk:
                ref = max(peak, base) if hwm else base
                if eq > ref:
                    eq -= succ * (eq - ref)
                base = eq
                last_pay = d
        peak = max(peak, eq)
        out[i] = eq / e0 - 1.0
    return out


def metrics(dates, r):
    """CAGR/Sharpe/вола/maxDD в конвенции Блоков 2 и 7."""
    yrs = max((date.fromisoformat(dates[-1])
               - date.fromisoformat(dates[0])).days / 365.25, 1e-9)
    freq = len(r) / yrs
    sd = r.std(ddof=1)
    if sd <= 0:
        return None
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0,
            "sharpe": float(r.mean() / sd * np.sqrt(freq)),
            "vol": float(sd * np.sqrt(freq)),
            "maxdd": float(np.min(eq / peak - 1.0)), "years": float(yrs)}


_TAR = {}
_RF = {}


def _init(tar, rf):
    global _TAR, _RF
    _TAR, _RF = tar, rf


def work(sid):
    """Валовые и чистые метрики системы на окне + плата в пунктах."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists() or sid not in _TAR:
        return None
    try:
        s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    except Exception:                                                   # noqa: BLE001
        return None
    if not s:
        return None
    s = s[::-1]
    d = [x["date"] for x in s]
    r = np.array([float(x["rValue"] or 0.0) for x in s], dtype=np.float64)
    m = np.array([(WIN_FROM <= x <= WIN_TILL) for x in d])
    if m.sum() < MIN_WIN_DAYS:
        return None
    dd = [x for x, k in zip(d, m) if k]
    rr = r[m]
    if float((rr != 0).mean()) < MIN_ACT_WIN:
        return None
    g = metrics(dd, rr)
    if g is None or g["years"] < MIN_WIN_YEARS:
        return None
    t = _TAR[sid]
    n = net_series(dd, rr, t["mgmt"], t["succ"], t["per"], hwm=True)
    nn = metrics(dd, n)
    n2 = net_series(dd, rr, t["mgmt"], t["succ"], t["per"], hwm=False)
    nn2 = metrics(dd, n2)
    rf = pd.Series([_RF.get(x, np.nan) for x in dd]).ffill().bfill().to_numpy()
    freq = len(rr) / g["years"]
    rf_cagr = float(np.mean(rf))
    return {"id": sid, "mgmt": t["mgmt"], "succ": t["succ"], "per": t["per"],
            "cagr_g": g["cagr"], "sharpe_g": g["sharpe"], "vol": g["vol"],
            "maxdd": g["maxdd"], "years": g["years"],
            "cagr_n": nn["cagr"], "sharpe_n": nn["sharpe"], "maxdd_n": nn["maxdd"],
            "cagr_n2": nn2["cagr"], "rf": rf_cagr, "freq": freq}


def draw_from_cache():
    """Перерисовать графики блока по сохранённым величинам, без пересчёта."""
    d = np.load(CACHE, allow_pickle=False)
    chart_fee_shift(d["g"], d["n"], float(d["rf"]))
    if "q" in d.files:
        chart_threshold_quintiles([
            {"q": int(q), "thr": str(t), "sharpe": s_, "cagr": c, "subs": f_}
            for (q, s_, c, f_), t in zip(d["q"], d["thr"])])


def main():
    if ONLY:
        draw_from_cache()
        return
    from comon_sharpe_dist import rusfar                                # noqa: E402
    rf = rusfar()
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    ids = panel.index.tolist()

    say("=" * 104)
    say("БЛОК 8. ЭКОНОМИКА СЛЕДОВАНИЯ: ЧТО ДОСТАЁТСЯ ПОДПИСЧИКУ")
    say("=" * 104)
    say(f"окно: {WIN_FROM} .. {WIN_TILL} (то же, что в Блоке 7)")
    say()

    # ── 8.1 тарифная сетка ────────────────────────────────────────────────────
    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(card_tariffs, ids, chunksize=200) if x]
    flat = [t for lst in rows for t in lst]
    T = pd.DataFrame(flat)
    say("=" * 104)
    say("8.1 ТАРИФНАЯ СЕТКА: ЧТО ПРЕДЛАГАЮТ И ЧТО ПОКУПАЮТ")
    say("=" * 104)
    say(f"систем с разобранным тарифом: {T['id'].nunique():,} "
        f"(тарифных строк {len(T):,})")
    say()
    say("Типы тарифов (по строкам предложений):")
    for k, v in T["type"].value_counts().items():
        say(f'  {str(k):<26} {v:>7,} ({100*v/len(T):>5.1f} %)')
    say()
    say("Топ-10 предложений:")
    hdr = f'{"описание тарифа":<74} {"систем":>7} {"доля":>7}'
    say(hdr)
    say("-" * len(hdr))
    for k, v in T["desc"].value_counts().head(10).items():
        say(f'{str(k)[:74]:<74} {v:>7,} {100*v/len(T):>6.1f} %')
    say()

    # самый дешёвый тариф на систему: сравниваем по mgmt + succ·ожидаемая доходность
    T["cost_proxy"] = T["mgmt"] + T["succ"] * 0.15
    # 🔴 ВОСПРОИЗВОДИМОСТЬ (исправлено 2026-08-09). Было `sort_values("cost_proxy")` —
    # быстрая сортировка НЕ стабильна, а ничьи здесь массовые и точные: «6 % от СЧА»
    # и «3 % от СЧА + 20 % от ИД» дают ровно 0.06 обе (0.03 + 0.20·0.15). При ничьей
    # `.first()` брал случайный вариант, и один и тот же скрипт на одних и тех же
    # данных давал разные числа: 142 против 139 систем с платой от дохода и 4 405
    # против 3 346 подписок в них (прогоны 2026-08-06 и 2026-08-09).
    # Теперь: стабильная сортировка + явный порядок ничьей — при равной оценке
    # стоимости берём вариант с МЕНЬШЕЙ платой от результата. Выбор произвольный,
    # но зафиксированный; важно, что он один и тот же в каждом прогоне.
    cheap = (T.sort_values(["cost_proxy", "succ", "mgmt", "desc"], kind="mergesort")
              .groupby("id").first())
    # 🔴 Насколько вообще устойчив выбор «самого дешёвого»: сколько систем имеют
    # ничью по cost_proxy. Если ничьих много, любое число, посчитанное ПО ВЫБРАННОМУ
    # варианту, есть следствие соглашения о ничьей, а не факт о площадке.
    tie = T.groupby("id")["cost_proxy"].agg(lambda s: (s == s.min()).sum() > 1)
    n_tie = int(tie.sum())
    # Счёт БЕЗ соглашений: система имеет вариант с платой от результата, если он
    # предложен хотя бы одной тарифной строкой. Это не зависит от выбора вообще.
    offers_sf = T[T["succ"] > 0].groupby("id").size()
    id_sf = set(offers_sf.index)
    fol = panel["followers"].fillna(0)          # panel уже проиндексирована по id
    live_ids = set(panel.index[panel["is_live"] == 1])
    sub_sf = float(fol.reindex(sorted(id_sf & live_ids)).sum())
    sub_all = float(fol.reindex(sorted(live_ids)).sum())
    say()
    say("🔴 УСТОЙЧИВОСТЬ ВЫБОРА ВАРИАНТА (почему ниже даны два счёта):")
    say(f'  систем, где несколько вариантов стоят ОДИНАКОВО по этой оценке: {n_tie:,}')
    say('  («6 % от СЧА» и «3 % от СЧА + 20 % от ИД» дают ровно 0.06 обе)')
    say('  → любое число «по выбранному тарифу» зависит от правила ничьей')
    say("СЧЁТ БЕЗ СОГЛАШЕНИЙ — по НАЛИЧИЮ варианта с платой от результата:")
    say(f'  живых систем с таким вариантом: {len(id_sf & live_ids):,} из {len(live_ids):,} '
        f'({100*len(id_sf & live_ids)/len(live_ids):.1f} %)')
    say(f'  подписок на них: {sub_sf:,.0f} из {sub_all:,.0f} ({100*sub_sf/sub_all:.1f} %)')
    say()
    say(f'{"":<30} {"медиана":>9} {"среднее":>9} {"p90":>9} {"максимум":>9}')
    say("-" * 70)
    say(f'{"ставка от СЧА, % годовых":<30} {100*cheap["mgmt"].median():>9.2f} '
        f'{100*cheap["mgmt"].mean():>9.2f} {100*cheap["mgmt"].quantile(.9):>9.2f} '
        f'{100*cheap["mgmt"].max():>9.2f}')
    sf = cheap[cheap["succ"] > 0]
    if len(sf):
        say(f'{"ставка от дохода, % (где есть)":<30} {100*sf["succ"].median():>9.2f} '
            f'{100*sf["succ"].mean():>9.2f} {100*sf["succ"].quantile(.9):>9.2f} '
            f'{100*sf["succ"].max():>9.2f}')
    say(f'систем с платой от дохода: {len(sf):,} '
        f'({100*len(sf)/len(cheap):.1f} %); только от СЧА: {len(cheap)-len(sf):,}')
    say()
    # веса по подпискам
    F = cheap.join(panel[["followers", "is_live", "min_sum", "money_limit",
                          "transaction_rate", "rating"]], how="left")
    live = F[(F["is_live"] == True) & F["followers"].notna()]           # noqa: E712
    w = live["followers"].clip(lower=0)
    if w.sum() > 0:
        say(f'ставка от СЧА, взвешенная по подпискам: '
            f'{100*float((live["mgmt"]*w).sum()/w.sum()):.2f} % годовых '
            f'(равновзвешенно {100*live["mgmt"].mean():.2f} %)')
        say(f'подписок в системах с платой от дохода: '
            f'{int(w[live["succ"] > 0].sum()):,} из {int(w.sum()):,} '
            f'({100*float(w[live["succ"] > 0].sum()/w.sum()):.1f} %)')
    say()

    # ── 8.2 gross → net ───────────────────────────────────────────────────────
    tar = {int(i): {"mgmt": r["mgmt"], "succ": r["succ"], "per": r["per"]}
           for i, r in cheap.iterrows()}
    with Pool(NPROC, initializer=_init, initargs=(tar, rf)) as pool:
        res = [x for x in pool.imap_unordered(work, ids, chunksize=200) if x]
    W = pd.DataFrame(res).set_index("id")
    W = W.join(panel[["is_live", "followers", "min_sum", "money_limit",
                      "transaction_rate", "rating", "owner_id"]], how="left")
    W["live"] = W["is_live"] == True                                    # noqa: E712
    W["fee_pp"] = 100 * (W["cagr_g"] - W["cagr_n"])
    W["fee_pp2"] = 100 * (W["cagr_g"] - W["cagr_n2"])
    L = W[W["live"]]

    say("=" * 104)
    say("8.2 GROSS → NET: СКОЛЬКО ОСТАЁТСЯ ПОДПИСЧИКУ")
    say("=" * 104)
    say(f"систем на окне (≥ {MIN_WIN_YEARS:.0f} лет внутри, активность ≥ "
        f"{MIN_ACT_WIN:.0%}): {len(W):,}, из них живых {int(W['live'].sum()):,}")
    say(f"средняя ставка денежного рынка за окно: {100*W['rf'].median():.1f} % годовых")
    say()
    hdr = (f'{"величина":<34} {"p10":>8} {"p25":>8} {"медиана":>9} {"p75":>8} '
           f'{"p90":>8}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col, sc in (("CAGR ДО тарифа (счёт автора), %", "cagr_g", 100),
                         ("CAGR ПОСЛЕ тарифа (подписчик), %", "cagr_n", 100),
                         ("плата, п.п. годовых", "fee_pp", 1),
                         ("Sharpe ДО тарифа", "sharpe_g", 1),
                         ("Sharpe ПОСЛЕ тарифа", "sharpe_n", 1)):
        v = sc * L[col]
        say(f'{lab:<34} {v.quantile(.1):>8.2f} {v.quantile(.25):>8.2f} '
            f'{v.median():>9.2f} {v.quantile(.75):>8.2f} {v.quantile(.9):>8.2f}')
    say("(живые системы окна; плата = разница валового и чистого CAGR)")
    say()
    if CHARTS:
        say("── график 29 ────────────────────────────────────────────────────")
        np.savez(CACHE, g=100 * L["cagr_g"].to_numpy(float),
                 n=100 * L["cagr_n"].to_numpy(float),
                 rf=100 * W["rf"].median())
        chart_fee_shift(100 * L["cagr_g"].to_numpy(float),
                        100 * L["cagr_n"].to_numpy(float),
                        100 * W["rf"].median())
        say()
    say("СКОЛЬКО СИСТЕМ ИМЕЮТ СМЫСЛ ДЛЯ ПОДПИСЧИКА:")
    hdr = (f'{"критерий":<44} {"до тарифа":>12} {"после тарифа":>14} '
           f'{"потеряли смысл":>16}')
    say(hdr)
    say("-" * len(hdr))
    for lab, gm, nm in (
            ("доходность выше нуля", L["cagr_g"] > 0, L["cagr_n"] > 0),
            ("доходность выше денежного рынка", L["cagr_g"] > L["rf"],
             L["cagr_n"] > L["rf"]),
            ("Sharpe ≥ 1.0", L["sharpe_g"] >= 1.0, L["sharpe_n"] >= 1.0),
            ("Sharpe ≥ 1.5", L["sharpe_g"] >= 1.5, L["sharpe_n"] >= 1.5),
            ("Sharpe ≥ 2.0", L["sharpe_g"] >= 2.0, L["sharpe_n"] >= 2.0)):
        say(f'{lab:<44} {int(gm.sum()):>7} ({100*gm.mean():>4.1f}%) '
            f'{int(nm.sum()):>7} ({100*nm.mean():>4.1f}%) '
            f'{int(gm.sum())-int(nm.sum()):>16}')
    say()
    pos = L[L["cagr_g"] > 0]
    say(f'ДОЛЯ ВАЛОВОЙ ДОХОДНОСТИ, УХОДЯЩАЯ В ТАРИФ (системы с положительным gross, '
        f'{len(pos)} шт):')
    share = (pos["cagr_g"] - pos["cagr_n"]) / pos["cagr_g"]
    say(f'  медиана {100*share.median():.1f} % · p25 {100*share.quantile(.25):.1f} % · '
        f'p75 {100*share.quantile(.75):.1f} % · '
        f'доля систем, где тариф съел больше половины: '
        f'{100*float((share > 0.5).mean()):.1f} %')
    say()
    say("Плата по группам валовой доходности (живые):")
    bins = [-10, 0, 0.10, 0.20, 0.35, 100]
    labs = ["убыточные", "0…10 %", "10…20 %", "20…35 %", "> 35 %"]
    L2 = L.copy()
    L2["grp"] = pd.cut(L2["cagr_g"], bins=bins, labels=labs)
    hdr = (f'{"группа по валовому CAGR":<24} {"систем":>7} {"gross, %":>10} '
           f'{"net, %":>9} {"плата, п.п.":>12} {"net выше ставки":>16}')
    say(hdr)
    say("-" * len(hdr))
    for lab, sub in L2.groupby("grp", observed=True):
        say(f'{str(lab):<24} {len(sub):>7} {100*sub["cagr_g"].median():>10.1f} '
            f'{100*sub["cagr_n"].median():>9.1f} {sub["fee_pp"].median():>12.2f} '
            f'{100*float((sub["cagr_n"] > sub["rf"]).mean()):>15.1f}%')
    say()
    sfz = L[L["succ"] > 0]
    if len(sfz) >= 10:
        say(f'Системы с платой от дохода ({len(sfz)} шт): медианная плата с HWM '
            f'{sfz["fee_pp"].median():.2f} п.п., без HWM {sfz["fee_pp2"].median():.2f} п.п.')
        say("(разница — цена того, что плату берут с новых максимумов, а не с каждого "
            "прироста)")
    say()
    say("ВЗВЕШЕННО ПО ПОДПИСКАМ (что получает не средняя система, а средний рубль):")
    lw = L[L["followers"].notna() & (L["followers"] > 0)]
    if len(lw):
        w = lw["followers"]
        say(f'  подписок учтено: {int(w.sum()):,} в {len(lw)} системах')
        say(f'  CAGR до тарифа {100*float((lw["cagr_g"]*w).sum()/w.sum()):.2f} % · '
            f'после тарифа {100*float((lw["cagr_n"]*w).sum()/w.sum()):.2f} % · '
            f'ставка {100*float((lw["rf"]*w).sum()/w.sum()):.2f} %')
        say(f'  доля подписок в системах, чей NET выше ставки: '
            f'{100*float(w[lw["cagr_n"] > lw["rf"]].sum()/w.sum()):.1f} %')
        say(f'  доля подписок в системах с NET < 0: '
            f'{100*float(w[lw["cagr_n"] < 0].sum()/w.sum()):.1f} %')

    # ── 8.3 порог входа ───────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("8.3 ПОРОГ ВХОДА ПРОТИВ РЕЗУЛЬТАТА")
    say("=" * 104)
    say("Гипотеза плана: высокий порог — признак серьёзности (или наоборот, способ")
    say("отсечь мелких при неточном копировании). Проверяем связь напрямую.")
    say()
    M = W[W["min_sum"].notna() & (W["min_sum"] > 0)].copy()
    M["q"] = pd.qcut(M["min_sum"].rank(method="first"), 5, labels=False) + 1
    hdr = (f'{"квинтиль порога входа":<22} {"систем":>7} {"медиана порога, ₽":>18} '
           f'{"Sharpe gross":>13} {"CAGR net, %":>12} {"живых":>8} {"подписок":>10}')
    say(hdr)
    say("-" * len(hdr))
    q_rows = []
    for q, sub in M.groupby("q"):
        say(f'{int(q):<22} {len(sub):>7} {sub["min_sum"].median():>18,.0f} '
            f'{sub["sharpe_g"].median():>13.2f} {100*sub["cagr_n"].median():>12.1f} '
            f'{100*sub["live"].mean():>7.1f}% {sub["followers"].sum():>10,.0f}')
        thr = sub["min_sum"].median()
        q_rows.append({"q": int(q),
                       "thr": f"{thr/1e3:,.0f}".replace(",", " "),
                       "sharpe": float(sub["sharpe_g"].median()),
                       "cagr": float(100 * sub["cagr_n"].median()),
                       "subs": float(sub["followers"].sum())})
    say()
    if CHARTS:
        say("── график 12 ────────────────────────────────────────────────────")
        d = dict(np.load(CACHE))
        d["q"] = np.array([[r["q"], r["sharpe"], r["cagr"], r["subs"]]
                           for r in q_rows], dtype=float)
        d["thr"] = np.array([r["thr"] for r in q_rows])
        np.savez(CACHE, **d)
        chart_threshold_quintiles(q_rows)
        say()
    for lab, col in (("порог входа ↔ Sharpe (gross)", "sharpe_g"),
                     ("порог входа ↔ CAGR net", "cagr_n"),
                     ("порог входа ↔ волатильность", "vol"),
                     ("порог входа ↔ подписчики", "followers")):
        s = M[["min_sum", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(s) > 30:
            say(f'  {lab:<34} ρ = {sst.spearmanr(s["min_sum"], s[col]).statistic:+.3f} '
                f'(n = {len(s):,})')

    # ── 8.4 частота сделок ────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("8.4 ЧАСТОТА СДЕЛОК ПРОТИВ РЕЗУЛЬТАТА")
    say("=" * 104)
    say("Вопрос плана: не исчезает ли преимущество высокочастотных после издержек.")
    say("🔴 Прямо измерить издержки копирования нельзя (их нет в данных), поэтому")
    say("смотрим на связь заявленной частоты с уже реализованным результатом.")
    say()
    ORD = {"seldom": "редко", "monthly": "раз в месяц", "weekly": "раз в неделю",
           "daily": "ежедневно"}
    R = W[W["transaction_rate"].notna()].copy()
    hdr = (f'{"частота сделок":<16} {"систем":>7} {"Sharpe gross":>13} '
           f'{"Sharpe net":>11} {"CAGR net, %":>12} {"вола, %":>9} {"живых":>8} '
           f'{"подписок":>10}')
    say(hdr)
    say("-" * len(hdr))
    for k in ("seldom", "monthly", "weekly", "daily"):
        sub = R[R["transaction_rate"] == k]
        if not len(sub):
            continue
        say(f'{ORD[k]:<16} {len(sub):>7} {sub["sharpe_g"].median():>13.2f} '
            f'{sub["sharpe_n"].median():>11.2f} {100*sub["cagr_n"].median():>12.1f} '
            f'{100*sub["vol"].median():>8.1f}% {100*sub["live"].mean():>7.1f}% '
            f'{sub["followers"].sum():>10,.0f}')
    say()
    num = {"seldom": 0, "monthly": 1, "weekly": 2, "daily": 3}
    R["tr"] = R["transaction_rate"].map(num)
    for lab, col in (("частота ↔ Sharpe gross", "sharpe_g"),
                     ("частота ↔ Sharpe net", "sharpe_n"),
                     ("частота ↔ CAGR net", "cagr_n"),
                     ("частота ↔ волатильность", "vol"),
                     ("частота ↔ подписчики", "followers")):
        s = R[["tr", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(s) > 30:
            say(f'  {lab:<30} ρ = {sst.spearmanr(s["tr"], s[col]).statistic:+.3f} '
                f'(n = {len(s):,})')

    # ── 8.5 масштаб платы ─────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("8.5 МАСШТАБ ПЛАТЫ")
    say("=" * 104)
    liv = panel[(panel["is_live"] == True) & panel["followers"].notna()]  # noqa: E712
    subs = float(liv["followers"].sum())
    aum = float((liv["followers"] * liv["min_sum"].fillna(0)).sum())
    say(f'подписок у живых систем: {subs:,.0f}; оценка активов снизу '
        f'(подписки × порог входа): {aum/1e9:.2f} млрд ₽')
    for rate, lab in ((0.06, "базовый тариф 6 %"), (0.03, "льготный 3 %")):
        say(f'  плата за год при ставке {lab}: {aum*rate/1e6:,.0f} млн ₽')
    say()
    base = float(liv["cagr"].median()) / 100.0        # панель хранит CAGR в процентах
    say(f'ЧТО ТАРИФ СЪЕДАЕТ У ТИПИЧНОЙ ЖИВОЙ СИСТЕМЫ (медианный CAGR витрины '
        f'{100*base:.1f} % годовых):')
    hdr = (f'{"тариф":<44} {"остаётся подписчику, %":>24} {"доля дохода в тариф":>21}')
    say(hdr)
    say("-" * len(hdr))
    for lab, mg, sc, per in (("6 % от СЧА (94 % витрины)", 0.06, 0.0, None),
                             ("3 % от СЧА", 0.03, 0.0, None),
                             ("3 % от СЧА + 20 % от дохода, квартал", 0.03, 0.20, "Q"),
                             ("2 % от СЧА + 20 % от дохода, месяц", 0.02, 0.20, "M"),
                             ("0 % (без посредника)", 0.0, 0.0, None)):
        # грубая аналитика: mgmt снимается с капитала, success — с прироста над HWM
        net = (1 + base) * (1 - mg) - 1
        if sc:
            net -= sc * max(net, 0.0)
        say(f'{lab:<44} {100*net:>23.2f}% '
            f'{100*(base-net)/base if base else float("nan"):>20.1f}%')
    say()
    say("🔴 Это ВЕРХНЯЯ оценка того, что достаётся подписчику: издержки его собственного")
    say("брокера и проскальзывание при копировании в данные Comon не входят вообще.")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(_lines), encoding="utf-8")
    say()
    say(f"лог: {LOG}")


if __name__ == "__main__":
    main()
