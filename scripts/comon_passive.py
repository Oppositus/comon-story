"""comon_passive.py — пассивный эталон против витрины: числа пятого раздела заключения.

Все сравнения серии до сих пор шли ВНУТРИ витрины: стратегия против стратегии, автор
против автора. Здесь появляется внешняя точка отсчёта — «Портфель лежебоки плюс»
(30 % акции + 30 % облигации + 30 % золото + 10 % денежный рынок, ребалансировка раз в
год, четыре биржевых фонда), который читатель может собрать сам, без всякой площадки.

🔴 КЛЮЧЕВОЕ УСЛОВИЕ (указание автора серии). Лежебока считается БЕЗ комиссий Comon:
тариф автоследования к ней конструктивно неприменим, человек собирает её сам, а расходы
фондов уже сидят в цене пая. Витрина показывается в ДВУХ видах — до тарифа (что видит
посетитель) и после (что получает подписчик). Отсюда два перцентиля на каждую метрику.

🔴 ОДНО ОКНО, ОДНА КОНВЕНЦИЯ. Окно — 2022-10-12 .. 2026-08-04, то же, что в главе 5.
Метрики считаются функцией `metrics` из comon_tariffs (импортируется, а не копируется):
Sharpe арифметический mean/std*sqrt(freq) без вычета ставки, freq выводится из самого
ряда. Чистые ряды стратегий строит `net_series` оттуда же с тарифом каждой конкретной
стратегии и защитой от повторной оплаты одного и того же роста. Так числа раздела
совпадают с числами главы 5 по построению, а не по совпадению.

🔴 ДАННЫЕ ПАЁВ ПЕРЕКАЧАНЫ 2026-08-10. Прежний `bpif_daily.json` расходится со свежей
выгрузкой ISS не масштабом, а самими ретёрнами (корреляция 0.81 у фонда акций) — то
есть это другой ценовой ряд, а не тот же в других единицах. Дописывать к нему
недостающие дни было нельзя. Взят `bpif_daily_full.json` — биржевые цены закрытия
рублёвых бордов с ISS, порождённые `fetch_bpif_daily.py`. Чувствительность к выбору
ряда печатается ниже отдельным блоком: вывод раздела не должен зависеть от того, каким
полем ISS мы взяли цену пая.

Запуск: python comon_passive.py
"""
import gzip
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comon_tariffs import (                                    # noqa: E402
    MIN_ACT_WIN, MIN_WIN_DAYS, MIN_WIN_YEARS, WIN_FROM, WIN_TILL,
    card_tariffs, metrics, net_series)

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
SYN = ROOT / "data"
NPROC = 8

WEIGHTS = {"EQMX": 0.30, "OBLG": 0.30, "GOLD": 0.30, "LQDT": 0.10}
FEE_DEFAULT = 0.06        # тариф по умолчанию: 6 % годовых от активов


def say(s=""):
    print(s, flush=True)


def head(t):
    say()
    say("=" * 100)
    say(t)
    say("=" * 100)


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Полный прогон — два Pool-прохода по всем живым карточкам; величины, нужные
# картинке, кладутся рядом, и --charts-only перерисовывает без пересчёта.
CACHE = ROOT / "results" / "comon_passive_charts.npz"

# Четыре панели графика 13: ключ, подпись, множитель, границы поля, формат.
PANELS = (("cagr", "доходность, % годовых", 100, (-60, 60), "{:.1f} %"),
          ("sharpe", "Sharpe", 1, (-2, 3), "{:.2f}"),
          ("maxdd", "максимальная просадка, %", 100, (-100, 0), "{:.1f} %"),
          ("mar", "MAR — доходность на просадку", 1, (-1, 2), "{:.2f}"))


def chart_benchmark(D):
    """График 13: где пассивный эталон стоит в распределении витрины."""
    import comon_charts as ch

    f, axes = ch.fig(h_px=1300, bottom=0.20, nrows=2, ncols=2)
    # hspace больше: заголовок нижней панели в две строки, и при прежнем зазоре
    # он наезжал на сноску верхней; wspace — чтобы подпись оси не лезла к соседу
    f.subplots_adjust(hspace=0.55, wspace=0.24, top=0.88)
    for ax, (key, lab, mul, (lo, hi), fmt) in zip(axes.ravel(), PANELS):
        bins = np.linspace(lo, hi, 37)
        for tag, color, name in (("g", ch.BLUE, "витрина до тарифа"),
                                 ("n", ch.ORANGE, "витрина после тарифа")):
            v = np.clip(mul * D[f"{key}_{tag}"], lo, hi)
            cnt, _ = np.histogram(v, bins=bins)
            xs = np.repeat(bins, 2)[1:-1]
            ys = np.repeat(cnt, 2)
            ax.fill_between(xs, 0, ys, color=color, alpha=0.22, lw=0, zorder=3)
            ax.plot(xs, ys, color=color, lw=1.8, zorder=5, label=name)
        ref = mul * float(D[f"ref_{key}"])
        ax.axvline(ref, color=ch.GREEN, lw=2.2, ls="-.", zorder=6,
                   label="пассивный портфель «Лежебока плюс»")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(lab)
        ax.set_ylabel("стратегий в интервале")
        # Заголовок панели — в ДВЕ строки: одной строкой он шире панели и
        # обрезался краем файла (у левой нижней пропадала первая буква).
        name = lab.split(",")[0]
        ax.set_title(f"{name[0].upper()}{name[1:]}: эталон {fmt.format(ref)}\n"
                     f"обогнали {ch.n_(D[f'better_{key}_g'], 1)} % до тарифа и "
                     f"{ch.n_(D[f'better_{key}_n'], 1)} % после", fontsize=12)
    axes[0][0].legend(loc="upper left", handlelength=2.4, labelspacing=0.5)
    ch.note(axes[1][0],
            f"База витрины — {int(D['n_live'])} живых стратегий с историей не короче "
            f"трёх лет внутри окна {WIN_FROM}…{WIN_TILL} и активностью не ниже 10 % "
            f"дней. Значения за краями поля сведены в крайние столбцы.",
            "В заголовке панели: первая доля — стратегии, обогнавшие эталон до "
            "тарифа, вторая — после тарифа. Эталон считается без тарифа площадки: "
            "человек собирает его сам, платить за следование некому. Сразу по всем "
            "четырём метрикам эталон обогнали 7 стратегий из 593 и 3 после тарифа.")
    # 🔴 Общий заголовок ставится ПОСЛЕ note(): она может ужать панели, и тогда
    # заголовки панелей поднимаются. Раньше общий заголовок оказывался под ними.
    f.suptitle("Пассивный портфель из четырёх фондов против всей витрины",
               fontsize=15, fontweight="bold", y=0.985, va="top")
    ch.save(f, 13, "passive-benchmark")
    print("    " + "; ".join(
        f"{k}: эталон {mul*float(D[f'ref_{k}']):.2f}, обогнали "
        f"{D[f'better_{k}_g']:.1f} / {D[f'better_{k}_n']:.1f} %"
        for k, _l, mul, _b, _f in PANELS), flush=True)


def sortino(r, freq):
    dn = r[r < 0]
    if len(dn) < 5 or dn.std(ddof=1) <= 0:
        return np.nan
    return float(r.mean() / dn.std(ddof=1) * np.sqrt(freq))


def full(dates, r):
    """Метрики + Сортино + MAR в конвенции главы 5."""
    m = metrics(dates, r)
    if m is None:
        return None
    freq = len(r) / m["years"]
    m["sortino"] = sortino(r, freq)
    m["mar"] = m["cagr"] / abs(m["maxdd"]) if m["maxdd"] < -1e-9 else np.nan
    return m


def sluggard(path, lo=WIN_FROM, hi=WIN_TILL):
    """Лежебока-плюс на окне: 30/30/30/10, ребалансировка в первый день года."""
    bp = json.loads(Path(path).read_text())
    px, cal = {}, None
    for tk in WEIGHTS:
        d = {a: b for a, b in bp[tk] if lo <= a <= hi}
        px[tk] = d
        cal = set(d) if cal is None else cal & set(d)
    cal = np.array(sorted(cal))
    units = {tk: WEIGHTS[tk] / px[tk][cal[0]] for tk in WEIGHTS}
    val = [1.0]
    for i in range(1, len(cal)):
        d, prev = cal[i], cal[i - 1]
        v = sum(units[tk] * px[tk][d] for tk in WEIGHTS)
        val.append(v)
        if d[:4] != prev[:4]:                       # первый торговый день года
            units = {tk: WEIGHTS[tk] * v / px[tk][d] for tk in WEIGHTS}
    val = np.array(val)
    return list(cal[1:]), val[1:] / val[:-1] - 1.0


_TAR, _RF = {}, {}


def _init(tar, rf):
    global _TAR, _RF
    _TAR, _RF = tar, rf


def work(sid):
    """Метрики стратегии на окне: валовые и чистые (её собственный тариф)."""
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
    g = full(dd, rr)
    if g is None or g["years"] < MIN_WIN_YEARS:
        return None
    t = _TAR[sid]
    n = full(dd, net_series(dd, rr, t["mgmt"], t["succ"], t["per"], hwm=True))
    if n is None:
        return None
    rfv = pd.Series([_RF.get(x, np.nan) for x in dd]).ffill().bfill().to_numpy()
    return {"id": sid, "rf_avg": float(np.mean(rfv)),
            **{f"{k}_g": v for k, v in g.items()},
            **{f"{k}_n": v for k, v in n.items()}}


def pct_better(vals, ref, higher_is_better=True):
    """Доля стратегий витрины, которые ЛУЧШЕ эталона по этой метрике."""
    v = np.asarray(pd.Series(vals).dropna(), dtype=float)
    m = v > ref if higher_is_better else v > ref     # для просадки «больше» = мельче
    return 100.0 * m.mean(), len(v)


def main():
    if ONLY:
        chart_benchmark(dict(np.load(CACHE, allow_pickle=False)))
        return
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False).set_index("id")
    panel["is_live"] = pd.to_numeric(panel["is_live"], errors="coerce")
    live_ids = sorted(panel.index[panel["is_live"] == 1])
    say(f"окно: {WIN_FROM} .. {WIN_TILL} (то же, что в главе 5)")
    say(f"живых стратегий: {len(live_ids):,}; фильтр окна — не меньше {MIN_WIN_DAYS} дней,")
    say(f"активность не ниже {MIN_ACT_WIN:.0%}, история на окне не короче {MIN_WIN_YEARS} лет")

    # ── тарифы: тот же выбор «самого дешёвого варианта», что в главе 5 ─────────
    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(card_tariffs, live_ids, chunksize=200) if x]
    T = pd.DataFrame([t for lst in rows for t in lst])
    T["cost_proxy"] = T["mgmt"] + T["succ"] * 0.15
    cheap = (T.sort_values(["cost_proxy", "succ", "mgmt", "desc"], kind="mergesort")
              .groupby("id").first())
    tar = {int(i): {"mgmt": r["mgmt"], "succ": r["succ"], "per": r["per"]}
           for i, r in cheap.iterrows()}
    rf = json.loads((DIR / "rusfar.json").read_text())
    with Pool(NPROC, initializer=_init, initargs=(tar, rf)) as pool:
        res = [x for x in pool.imap_unordered(work, live_ids, chunksize=50) if x]
    W = pd.DataFrame(res)
    say(f"стратегий, прошедших фильтр окна: {len(W):,}")

    # ── 1. Эталон ─────────────────────────────────────────────────────────────
    head("1. ПАССИВНЫЙ ЭТАЛОН НА ОКНЕ ВИТРИНЫ")
    dts, r = sluggard(SYN / "bpif_daily_full.json")
    sl = full(dts, r)
    say(f'«Лежебока плюс» 30/30/30/10, ребалансировка раз в год, {len(r):,} торговых дней')
    say(f'{dts[0]} .. {dts[-1]}, {sl["years"]:.2f} года')
    say()
    hdr = f'{"величина":<34}{"Лежебока плюс":>16}{"медиана витрины (gross)":>26}{"медиана витрины (net)":>24}'
    say(hdr)
    say("-" * len(hdr))
    for key, lab, mul, fmt in (("cagr", "CAGR, % годовых", 100, "{:.1f}"),
                               ("vol", "волатильность, %", 100, "{:.1f}"),
                               ("maxdd", "максимальная просадка, %", 100, "{:.1f}"),
                               ("sharpe", "Sharpe (арифметический)", 1, "{:.2f}"),
                               ("sortino", "Сортино", 1, "{:.2f}"),
                               ("mar", "MAR (CAGR / просадка)", 1, "{:.2f}")):
        say(f'{lab:<34}{fmt.format(sl[key] * mul):>16}'
            f'{fmt.format(W[f"{key}_g"].median() * mul):>26}'
            f'{fmt.format(W[f"{key}_n"].median() * mul):>24}')
    say()
    say("Контроль: медианы витрины на окне должны совпасть с записанными в служебном")
    say("плане (CAGR 10.6 / вола 26.2 / просадка -41.5 / Sharpe 0.58 / Сортино 0.72 / MAR 0.28).")

    # ── 1б. Чувствительность к ценовому ряду ──────────────────────────────────
    head("1б. ЧУВСТВИТЕЛЬНОСТЬ: два ценовых ряда паёв дают одно и то же?")
    say("Прежний файл bpif_daily.json и свежая выгрузка ISS расходятся по дневным")
    say("ретёрнам (корреляция 0.81 у фонда акций). Считаем эталон на обоих и смотрим,")
    say("меняется ли вывод. Общее окно у прежнего файла короче справа на три дня.")
    say()
    old_dts, old_r = sluggard(SYN / "bpif_daily.json")
    so = full(old_dts, old_r)
    hdr = f'{"величина":<34}{"ISS, свежая":>14}{"прежний файл":>15}{"разница":>12}'
    say(hdr)
    say("-" * len(hdr))
    for key, lab, mul in (("cagr", "CAGR, %", 100), ("vol", "волатильность, %", 100),
                          ("maxdd", "просадка, %", 100), ("sharpe", "Sharpe", 1),
                          ("sortino", "Сортино", 1), ("mar", "MAR", 1)):
        a, b = sl[key] * mul, so[key] * mul
        say(f'{lab:<34}{a:>14.2f}{b:>15.2f}{a - b:>+12.2f}')
    say()
    say(f'   прежний файл: {old_dts[0]} .. {old_dts[-1]}, {len(old_r):,} дней')

    # ── 2. Перцентили ─────────────────────────────────────────────────────────
    head("2. ГДЕ ЭТАЛОН СТОИТ СРЕДИ СТРАТЕГИЙ ВИТРИНЫ")
    say("Для каждой метрики: сколько стратегий витрины ЛУЧШЕ пассивного портфеля.")
    say("Слева — до тарифа (что показано посетителю), справа — после тарифа каждой")
    say("конкретной стратегии (что получает подписчик). Лежебока в обоих случаях без")
    say("тарифа площадки: платить за неё некому.")
    say()
    hdr = (f'{"метрика":<28}{"эталон":>10}{"лучше него (gross)":>22}'
           f'{"лучше него (net)":>20}{"перцентиль gross":>19}{"перцентиль net":>17}')
    say(hdr)
    say("-" * len(hdr))
    for key, lab, mul, fmt in (("cagr", "CAGR, % годовых", 100, "{:.1f}"),
                               ("sharpe", "Sharpe", 1, "{:.2f}"),
                               ("maxdd", "макс. просадка, %", 100, "{:.1f}"),
                               ("mar", "MAR", 1, "{:.2f}"),
                               ("sortino", "Сортино", 1, "{:.2f}")):
        ref = sl[key]
        pg, ng = pct_better(W[f"{key}_g"], ref)
        pn, nn = pct_better(W[f"{key}_n"], ref)
        say(f'{lab:<28}{fmt.format(ref * mul):>10}'
            f'{f"{pg:.1f} % ({int(round(pg * ng / 100))} из {ng})":>22}'
            f'{f"{pn:.1f} % ({int(round(pn * nn / 100))})":>20}'
            f'{f"{100 - pg:.1f}-й":>19}{f"{100 - pn:.1f}-й":>17}')
    say()
    if CHARTS:
        d = {"n_live": len(W)}
        for key, _l, _m, _b, _f in PANELS:
            d[f"{key}_g"] = W[f"{key}_g"].to_numpy(float)
            d[f"{key}_n"] = W[f"{key}_n"].to_numpy(float)
            d[f"ref_{key}"] = sl[key]
            d[f"better_{key}_g"] = pct_better(W[f"{key}_g"], sl[key])[0]
            d[f"better_{key}_n"] = pct_better(W[f"{key}_n"], sl[key])[0]
        np.savez(CACHE, **d)
        say("── график 13 ────────────────────────────────────────────────────")
        chart_benchmark(d)
        say()

    say("Читается так: «перцентиль gross 94-й» = пассивный портфель выше 94 % того,")
    say("что витрина показывает; «лучше него 6 %» = столько стратегий его обогнали.")
    say()
    say("Сколько стратегий обогнали эталон СРАЗУ ПО ВСЕМ четырём метрикам:")
    for tag, lab in (("g", "до тарифа"), ("n", "после тарифа")):
        m = ((W[f"cagr_{tag}"] > sl["cagr"]) & (W[f"sharpe_{tag}"] > sl["sharpe"])
             & (W[f"maxdd_{tag}"] > sl["maxdd"]) & (W[f"mar_{tag}"] > sl["mar"]))
        say(f'   {lab:<16}{int(m.sum()):>5} из {len(W):,} ({100 * m.mean():.1f} %)')

    # ── 3. А если бы эталон продавался на витрине ─────────────────────────────
    head("3. ЧТО СТАЛО БЫ С САМИМ ЭТАЛОНОМ ПОД ТАРИФОМ ПЛОЩАДКИ")
    rfv = pd.Series([rf.get(x, np.nan) for x in dts]).ffill().bfill().to_numpy()
    rate = float(np.mean(rfv))
    say(f'ставка денежного рынка на окне (RUSFAR, среднее): {100 * rate:.1f} % годовых')
    say()
    net_sl = full(dts, r - FEE_DEFAULT / (len(r) / sl["years"]))
    say(f'{"":<44}{"CAGR, %":>10}{"Sharpe":>9}{"выше ставки?":>15}')
    say("-" * 78)
    say(f'{"Лежебока плюс как есть (платить некому)":<44}{100 * sl["cagr"]:>10.1f}'
        f'{sl["sharpe"]:>9.2f}{"да" if sl["cagr"] > rate else "нет":>15}')
    say(f'{"она же под 6 % годовых от активов":<44}{100 * net_sl["cagr"]:>10.1f}'
        f'{net_sl["sharpe"]:>9.2f}{"да" if net_sl["cagr"] > rate else "нет":>15}')
    say()
    say("Доля стратегий витрины, обогнавших ставку денежного рынка:")
    # 🔴 У каждой стратегии своя средняя ставка за её дни окна — так считала глава 5.
    # С общей медианой вместо неё выходит 39.6 / 27.3 против опубликованных 39.5 / 26.8.
    say(f'   до тарифа  {100 * (W["cagr_g"] > W["rf_avg"]).mean():.1f} %')
    say(f'   после      {100 * (W["cagr_n"] > W["rf_avg"]).mean():.1f} %')

    # ── 4. В своём классе риска ───────────────────────────────────────────────
    head("4. В СВОЁМ КЛАССЕ РИСКА: децили волатильности, ПЕРЕСЧИТАННЫЕ НА ОКНЕ")
    say("В служебном плане стояло сравнение с децилями Блока 4, но они считались за")
    say("жизнь систем на РАЗНЫХ окнах. Пересчитываем на общем окне — иначе сравнение")
    say("некорректно.")
    say()
    W["dec"] = pd.qcut(W["vol_g"], 10, labels=False, duplicates="drop")
    hdr = (f'{"дециль волатильности":<22}{"систем":>8}{"вола, %":>10}{"CAGR gross, %":>15}'
           f'{"Sharpe gross":>14}{"CAGR net, %":>13}')
    say(hdr)
    say("-" * len(hdr))
    for d_, g in W.groupby("dec"):
        say(f'{f"{int(d_) + 1}-й":<22}{len(g):>8}{100 * g["vol_g"].median():>10.1f}'
            f'{100 * g["cagr_g"].median():>15.1f}{g["sharpe_g"].median():>14.2f}'
            f'{100 * g["cagr_n"].median():>13.1f}')
    say()
    d1 = W[W["dec"] == 0]
    say(f'Первый дециль (самые спокойные): вола {100 * d1["vol_g"].median():.1f} % — это')
    say(f'уровень эталона ({100 * sl["vol"]:.1f} %). Медианный CAGR у них '
        f'{100 * d1["cagr_g"].median():.1f} % до тарифа и {100 * d1["cagr_n"].median():.1f} % после,')
    say(f'Sharpe {d1["sharpe_g"].median():.2f} — против {100 * sl["cagr"]:.1f} % и '
        f'{sl["sharpe"]:.2f} у эталона.')
    say()
    n_same = W[(W["vol_g"] >= sl["vol"] * 0.75) & (W["vol_g"] <= sl["vol"] * 1.25)]
    say(f'Строже — стратегии с волатильностью в пределах ±25 % от эталона '
        f'({100 * sl["vol"] * 0.75:.1f}–{100 * sl["vol"] * 1.25:.1f} %): {len(n_same)} штук,')
    if len(n_same):
        say(f'   медианный CAGR {100 * n_same["cagr_g"].median():.1f} % gross / '
            f'{100 * n_same["cagr_n"].median():.1f} % net, Sharpe {n_same["sharpe_g"].median():.2f};')
        say(f'   обогнали эталон по CAGR {int((n_same["cagr_g"] > sl["cagr"]).sum())} до тарифа '
            f'и {int((n_same["cagr_n"] > sl["cagr"]).sum())} после.')

    # ── 5. Честно против собственного вывода ──────────────────────────────────
    head("5. ЧЕСТНО ПРОТИВ СОБСТВЕННОГО ВЫВОДА: эталон на всей доступной истории")
    say("Окно витрины (конец 2022 — середина 2026) щедро к пассиву: высокая ставка")
    say("денежного рынка и ралли золота. Считаем тот же портфель на ВСЕЙ истории, какая")
    say("есть у четырёх фондов (её ограничивает золотой фонд, торгуется с 2020-07-15).")
    say()
    ldts, lr = sluggard(SYN / "bpif_daily_full.json", lo="2000-01-01", hi=WIN_TILL)
    lo_ = full(ldts, lr)
    lfreq = len(lr) / lo_["years"]
    lrf = pd.Series([rf.get(x, np.nan) for x in ldts]).ffill().bfill().to_numpy()
    ex = lr - lrf / lfreq
    sh_ex = float(ex.mean() / ex.std(ddof=1) * np.sqrt(lfreq))
    ex_w = r - pd.Series([rf.get(x, np.nan) for x in dts]).ffill().bfill().to_numpy() \
        / (len(r) / sl["years"])
    sh_ex_w = float(ex_w.mean() / ex_w.std(ddof=1) * np.sqrt(len(r) / sl["years"]))
    hdr = f'{"величина":<40}{"окно витрины":>16}{"вся история фондов":>22}'
    say(hdr)
    say("-" * len(hdr))
    say(f'{"период":<40}{f"{dts[0]}..":>16}{f"{ldts[0]}..":>22}')
    say(f'{"лет":<40}{sl["years"]:>16.2f}{lo_["years"]:>22.2f}')
    for key, lab, mul in (("cagr", "CAGR, % годовых", 100),
                          ("vol", "волатильность, %", 100),
                          ("maxdd", "максимальная просадка, %", 100),
                          ("sharpe", "Sharpe", 1), ("mar", "MAR", 1)):
        say(f'{lab:<40}{sl[key] * mul:>16.2f}{lo_[key] * mul:>22.2f}')
    say(f'{"Sharpe СВЕРХ денежного рынка":<40}{sh_ex_w:>16.2f}{sh_ex:>22.2f}')
    say(f'{"средняя ставка денежного рынка, %":<40}{100 * rate:>16.1f}'
        f'{100 * float(np.mean(lrf)):>22.1f}')
    say()
    say("🔴 Это и есть главная оговорка раздела: на длинной истории пассивный портфель")
    say("   проигрывает собственной денежной ноге. Его преимущество на окне витрины —")
    say("   свойство окна, а не вечная истина.")

    say()
    say("готово")


if __name__ == "__main__":
    main()
