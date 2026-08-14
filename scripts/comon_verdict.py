"""comon_verdict.py — выгодно ли автоследование клиенту: числа шестого раздела заключения.

Два вопроса, оба числовые.

1) ЧТО РЕАЛЬНО КУПЛЕНО. Медиана витрины описывает предложение, а не покупку: две
   трети живых стратегий не выбрал никто. Поэтому все метрики считаются ещё и
   ВЗВЕШЕННЫМИ ПО ПОДПИСКАМ — так видно не то, что лежит на прилавке, а то, что унесли.

2) ТРИ УСЛОВИЯ ВЫИГРЫША и сколько стратегий проходят их сразу. Условия заданы
   заранее и все три проверяемы на витрине ДО подписки:
     а) экономика — чистая (после тарифа) доходность выше ставки денежного рынка;
     б) не рынок под другим именем — бета к индексу мала (бету покупают индексным
        фондом за копейки, платить за неё 6 % годовых незачем);
     в) достоверность — результат отличим от случайности, t-статистика Sharpe >= 2.

🔴 ОДНО ОКНО, ОДНА КОНВЕНЦИЯ. Окно 2022-10-12 .. 2026-08-04 и метрики — те же, что в
главе 5 и в пятом разделе заключения: функции `metrics` и `net_series` импортируются
из comon_tariffs, тариф берётся у каждой стратегии свой. Бета считается ЗДЕСЬ и на
этом же окне (готовый beta.csv.gz посчитан за жизнь систем на разных окнах — для
сравнения условий он не годится). t-статистика — формула Lo, как в первом разделе.

Запуск: python comon_verdict.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comon_tariffs import (                                    # noqa: E402
    MIN_ACT_WIN, MIN_WIN_DAYS, MIN_WIN_YEARS, WIN_FROM, WIN_TILL,
    card_tariffs, metrics, net_series)

from multiprocessing import Pool                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
NPROC = 8

BETA_MAX = 0.30           # порог «не рынок под другим именем»
T_MIN = 2.0               # порог достоверности
MIN_BETA_DAYS = 200       # общих торговых дней с индексом для оценки беты


def say(s=""):
    print(s, flush=True)


def head(t):
    say()
    say("=" * 100)
    say(t)
    say("=" * 100)


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Прогон — два Pool-прохода по всем живым карточкам; величины для картинок
# сохраняются рядом, чтобы оформление правилось без пересчёта.
CACHE = ROOT / "results" / "comon_verdict_charts.npz"


def chart_bought(cagr_n, followers, med_all, med_subs):
    """График 15: спрос отфильтровал плохое — распределение сдвинуто вправо."""
    import comon_charts as ch

    f, ax = ch.fig()
    v = 100 * np.asarray(cagr_n, float)
    w = np.asarray(followers, float)
    lo, hi = -60.0, 80.0
    bins = np.linspace(lo, hi, 36)
    vc = np.clip(v, lo, hi)
    xs = np.repeat(bins, 2)[1:-1]
    for weights, color, lab, med in (
            (None, ch.BLUE, "все стратегии, каждая с весом 1", med_all),
            (w, ch.ORANGE, "они же с весом по числу подписок", med_subs)):
        cnt, _ = np.histogram(vc, bins=bins, weights=weights)
        ys = np.repeat(cnt / cnt.sum(), 2)
        ax.fill_between(xs, 0, ys, color=color, alpha=0.22, lw=0, zorder=3)
        ax.plot(xs, ys, color=color, lw=2.2, zorder=5,
                label=f"{lab}: медиана {ch.n_(med, 1)} %")
    top = ax.get_ylim()[1]
    for med, color in ((med_all, ch.BLUE), (med_subs, ch.ORANGE)):
        ax.plot([med, med], [0, top * 0.82], color=color, lw=1.6, ls="--", zorder=6)
    ax.axvline(0, color=ch.PALE, lw=1.6, ls=":", zorder=2)

    ax.legend(loc="upper left", handlelength=2.6, labelspacing=0.55)
    ch.pct_raw(ax, "x")
    ch.pct(ax, "y", decimals=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top)
    ax.set_xlabel("доходность после тарифа, % годовых")
    ax.set_ylabel("доля своей выборки")
    ax.set_title("Покупают не медиану: спрос смещён в сторону лучшей части витрины")
    ch.note(ax,
            f"База — {ch.n_(len(v))} живых стратегий с историей не короче трёх лет "
            f"на окне {WIN_FROM}…{WIN_TILL}; в них {ch.n_(w.sum())} подписок. Каждая "
            f"кривая нормирована на свою выборку; значения за краями поля сведены "
            f"в крайние столбцы.",
            "Сдвиг вправо — это не разница двух чисел, а разное распределение: доля "
            "подписок в стратегиях выше ставки денежного рынка 61,4 % при 26,8 % "
            "таких стратегий. Важная оговорка: фильтрация идёт по ПРОШЛОЙ "
            "доходности, а она предсказывает будущую слабо.")
    ch.save(f, 15, "bought-vs-shelf")
    print(f"    медиана по стратегиям {med_all:.2f} %, по подпискам {med_subs:.2f} %",
          flush=True)


def chart_funnel(steps):
    """График 37: воронка трёх условий — 593 стратегии превращаются в 35."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=980, bottom=0.30)
    f.subplots_adjust(left=0.30)
    n0 = steps[0]["n"]
    ys = list(range(len(steps)))[::-1]
    cols = [ch.SEQ[2], ch.SEQ[4], ch.SEQ[6], ch.SEQ[8]]
    for y, st, c in zip(ys, steps, cols):
        ax.barh(y, st["n"], height=0.62, color=c, zorder=3)
        # Подпись в две строки: в одну она не влезает — на 1600 px при кегле 11
        # строка в 48 знаков занимает 600 px, а справа от длинной полосы свободно
        # чуть больше 400.
        ax.annotate(f"{ch.n_(st['n'])} — {ch.n_(100*st['n']/n0, 1)} % витрины\n"
                    f"{ch.n_(st['subs_share'], 1)} % всех подписок",
                    xy=(st["n"], y), xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color=ch.INK, ha="left", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([st["lab"] for st in steps], fontsize=11)
    ax.set_xlim(0, n0 * 1.62)
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ax.set_xlabel("стратегий, штук")
    ax.grid(axis="y", visible=False)
    ax.set_title("Три условия выигрыша проходят 35 стратегий из 593")
    ch.note(ax,
            f"База — {ch.n_(n0)} живых стратегий с историей не короче трёх лет на "
            f"окне {WIN_FROM}…{WIN_TILL}; в них 10 384 подписки, 77,3 % всех "
            f"подписок площадки. Условия накапливаются сверху вниз.",
            "Все три условия проверяемы на витрине до подписки. На прошедшие их "
            "стратегии приходится каждая пятая подписка — вчетверо больше, чем "
            "следовало бы из их доли: спрос смещён в правильную сторону.")
    ch.save(f, 37, "three-conditions-funnel")
    print("    " + " → ".join(f"{st['n']}" for st in steps), flush=True)


def t_sharpe(s_ar, years):
    return s_ar * np.sqrt(max(years, 0)) / np.sqrt(1 + s_ar * s_ar / 2)


def wmedian(v, w):
    """Медиана, взвешенная весами (подписками)."""
    d = pd.DataFrame({"v": v, "w": w}).dropna()
    d = d[d["w"] > 0].sort_values("v")
    if d.empty:
        return np.nan
    c = d["w"].cumsum() / d["w"].sum()
    return float(d.loc[c >= 0.5, "v"].iloc[0])


_TAR, _RF, _IDX = {}, {}, {}


def _init(tar, rf, idx):
    global _TAR, _RF, _IDX
    _TAR, _RF, _IDX = tar, rf, idx


def work(sid):
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
    n = metrics(dd, net_series(dd, rr, t["mgmt"], t["succ"], t["per"], hwm=True))
    if n is None:
        return None
    # бета к индексу на этом же окне, по общим торговым дням
    pair = [(v, _IDX[x]) for v, x in zip(rr, dd) if x in _IDX]
    beta = r2 = np.nan
    if len(pair) >= MIN_BETA_DAYS:
        a = np.array([p[0] for p in pair])
        b = np.array([p[1] for p in pair])
        if b.std(ddof=1) > 0 and a.std(ddof=1) > 0:
            beta = float(np.cov(a, b, ddof=1)[0, 1] / b.var(ddof=1))
            r2 = float(np.corrcoef(a, b)[0, 1] ** 2)
    rf = pd.Series([_RF.get(x, np.nan) for x in dd]).ffill().bfill().to_numpy()
    return {"id": sid, "mgmt": t["mgmt"], "succ": t["succ"],
            "cagr_g": g["cagr"], "sharpe_g": g["sharpe"], "vol": g["vol"],
            "maxdd_g": g["maxdd"], "years": g["years"],
            "cagr_n": n["cagr"], "sharpe_n": n["sharpe"], "maxdd_n": n["maxdd"],
            "beta": beta, "r2": r2, "rf": float(np.mean(rf)),
            "t_g": float(t_sharpe(g["sharpe"], g["years"]))}


def draw_from_cache():
    d = np.load(CACHE, allow_pickle=False)
    chart_bought(d["cagr_n"], d["followers"], float(d["med_all"]),
                 float(d["med_subs"]))
    chart_funnel([{"n": int(n), "subs_share": float(s), "lab": str(l)}
                  for n, s, l in zip(d["n"], d["subs"], d["labs"])])


def main():
    if ONLY:
        draw_from_cache()
        return
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False).set_index("id")
    panel["is_live"] = pd.to_numeric(panel["is_live"], errors="coerce")
    live_ids = sorted(panel.index[panel["is_live"] == 1])

    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(card_tariffs, live_ids, chunksize=200) if x]
    T = pd.DataFrame([t for lst in rows for t in lst])
    T["cost_proxy"] = T["mgmt"] + T["succ"] * 0.15
    cheap = (T.sort_values(["cost_proxy", "succ", "mgmt", "desc"], kind="mergesort")
              .groupby("id").first())
    tar = {int(i): {"mgmt": r["mgmt"], "succ": r["succ"], "per": r["per"]}
           for i, r in cheap.iterrows()}
    rf = json.loads((DIR / "rusfar.json").read_text())
    px = json.loads((DIR / "idx_IMOEX.json").read_text())
    days = sorted(px)
    idx = {days[i]: px[days[i]] / px[days[i - 1]] - 1.0 for i in range(1, len(days))}

    with Pool(NPROC, initializer=_init, initargs=(tar, rf, idx)) as pool:
        res = [x for x in pool.imap_unordered(work, live_ids, chunksize=50) if x]
    W = pd.DataFrame(res).set_index("id")
    W["followers"] = panel["followers"].reindex(W.index).fillna(0)
    W["min_sum"] = panel["min_sum"].reindex(W.index)
    W["rate"] = panel["transaction_rate"].reindex(W.index)
    rate = float(W["rf"].median())
    subs_all = float(panel.loc[panel["is_live"] == 1, "followers"].fillna(0).sum())

    say(f"окно: {WIN_FROM} .. {WIN_TILL}; живых стратегий {len(live_ids):,}")
    say(f"прошли фильтр окна: {len(W):,} стратегий")
    say(f"подписок в них: {int(W['followers'].sum()):,} из {int(subs_all):,} "
        f"по всем живым ({100 * W['followers'].sum() / subs_all:.1f} %)")
    say(f"ставка денежного рынка на окне: {100 * rate:.1f} % годовых")
    say(f"бета посчитана у {int(W['beta'].notna().sum()):,} из {len(W):,}")

    # ------------------------------------------------------------ расчёт 5а ----
    head("РАСЧЁТ 5а. ЧТО ЛЕЖИТ НА ПРИЛАВКЕ И ЧТО РЕАЛЬНО КУПЛЕНО")
    say("Слева — медиана по стратегиям (описывает предложение), справа — медиана,")
    say("взвешенная по подпискам (описывает покупку). Две трети живых стратегий не")
    say("выбрал никто, поэтому эти два столбца — про разное.")
    say()
    w = W["followers"]
    hdr = (f'{"величина":<36}{"медиана по стратегиям":>24}{"медиана по подпискам":>23}')
    say(hdr)
    say("-" * len(hdr))
    for col, lab, mul, fmt in (("cagr_g", "CAGR до тарифа, % годовых", 100, "{:.1f}"),
                               ("cagr_n", "CAGR после тарифа, % годовых", 100, "{:.1f}"),
                               ("sharpe_g", "Sharpe до тарифа", 1, "{:.2f}"),
                               ("sharpe_n", "Sharpe после тарифа", 1, "{:.2f}"),
                               ("maxdd_n", "макс. просадка после тарифа, %", 100, "{:.1f}"),
                               ("vol", "волатильность, %", 100, "{:.1f}"),
                               ("beta", "бета к индексу", 1, "{:.2f}"),
                               ("years", "длина истории на окне, лет", 1, "{:.2f}")):
        say(f'{lab:<36}{fmt.format(W[col].median() * mul):>24}'
            f'{fmt.format(wmedian(W[col], w) * mul):>23}')
    say()
    say("Доля стратегий и доля подписок, у которых чистая доходность выше ставки:")
    # 🔴 Сравниваем со СВОЕЙ ставкой каждой стратегии (среднее RUSFAR по её дням окна),
    # а не с общей медианой: так считала глава 5, и иначе получаются 27.3 % вместо
    # опубликованных 26.8 % — одна величина двумя способами внутри одного текста.
    ok_rate = W["cagr_n"] > W["rf"]
    say(f'   стратегий {100 * ok_rate.mean():.1f} %   подписок '
        f'{100 * w[ok_rate].sum() / w.sum():.1f} %')
    say("Доля стратегий и подписок с чистой доходностью ниже нуля:")
    neg = W["cagr_n"] < 0
    say(f'   стратегий {100 * neg.mean():.1f} %   подписок '
        f'{100 * w[neg].sum() / w.sum():.1f} %')

    # ------------------------------------------------------------ расчёт 5б ----
    head("РАСЧЁТ 5б. ТРИ УСЛОВИЯ, ПРИ КОТОРЫХ АВТОСЛЕДОВАНИЕ ВЫИГРЫВАЕТ")
    say("а) экономика:    чистая доходность выше ставки денежного рынка "
        f"({100 * rate:.1f} % годовых)")
    say(f"б) не рынок:     |бета| к индексу не больше {BETA_MAX:.2f}")
    say(f"в) достоверность: t-статистика Sharpe не меньше {T_MIN:.0f}")
    say()
    A = W["cagr_n"] > W["rf"]                  # своя ставка каждой стратегии (см. выше)
    B = W["beta"].abs() <= BETA_MAX
    C = W["t_g"] >= T_MIN
    hdr = f'{"условие":<44}{"стратегий":>12}{"доля":>9}{"подписок":>12}{"доля подписок":>16}'
    say(hdr)
    say("-" * len(hdr))
    for m, lab in ((A, "а) чистая доходность выше ставки"),
                   (B, f"б) |бета| <= {BETA_MAX:.2f}"),
                   (C, f"в) t-статистика >= {T_MIN:.0f}"),
                   (A & B, "а и б"), (A & C, "а и в"), (B & C, "б и в"),
                   (A & B & C, "🔴 ВСЕ ТРИ СРАЗУ")):
        m = m.fillna(False)
        say(f'{lab:<44}{int(m.sum()):>12,}{f"{100 * m.mean():.1f} %":>9}'
            f'{int(w[m].sum()):>12,}{f"{100 * w[m].sum() / w.sum():.1f} %":>16}')
    say()
    say(f'База: {len(W):,} живых стратегий с историей не короче {MIN_WIN_YEARS:.0f} лет')
    say(f'на окне; в них {int(w.sum()):,} подписок ({100 * w.sum() / subs_all:.1f} % всех).')

    ok = (A & B & C).fillna(False)

    if CHARTS:
        A_, B_, C_ = A.fillna(False), B.fillna(False), C.fillna(False)
        cuts = [(pd.Series(True, index=W.index), "все стратегии окна"),
                (A_, "и чистая доходность\nвыше ставки вклада"),
                (A_ & B_, f"и |бета| к индексу\nне больше {BETA_MAX:.2f}"
                          .replace("0.30", "0,30")),
                (A_ & C_ & B_, "и результат отличим\nот случайности")]
        d = {"cagr_n": W["cagr_n"].to_numpy(float),
             "followers": w.to_numpy(float),
             "med_all": 100 * float(W["cagr_n"].median()),
             "med_subs": 100 * float(wmedian(W["cagr_n"], w)),
             "n": np.array([int(m.sum()) for m, _ in cuts], float),
             "subs": np.array([100 * float(w[m].sum() / w.sum()) for m, _ in cuts]),
             "labs": np.array([lab for _, lab in cuts])}
        np.savez(CACHE, **d)
        head("ГРАФИКИ 15 и 37")
        draw_from_cache()

    say()
    say("Что представляют собой прошедшие все три (медианы):")
    hdr = f'{"величина":<36}{"прошедшие":>14}{"все на окне":>14}'
    say(hdr)
    say("-" * len(hdr))
    for col, lab, mul, fmt in (("cagr_n", "CAGR после тарифа, %", 100, "{:.1f}"),
                               ("sharpe_n", "Sharpe после тарифа", 1, "{:.2f}"),
                               ("maxdd_n", "просадка после тарифа, %", 100, "{:.1f}"),
                               ("vol", "волатильность, %", 100, "{:.1f}"),
                               ("beta", "бета к индексу", 1, "{:.2f}"),
                               ("years", "история на окне, лет", 1, "{:.2f}"),
                               ("min_sum", "порог входа, тыс ₽", 0.001, "{:.0f}"),
                               ("followers", "подписчиков", 1, "{:.0f}")):
        say(f'{lab:<36}{fmt.format(W.loc[ok, col].median() * mul):>14}'
            f'{fmt.format(W[col].median() * mul):>14}')
    say()
    say("🔴 Можно ли было найти их заранее — по видимым на витрине признакам:")
    say(f'   медианный порог входа у прошедших {W.loc[ok, "min_sum"].median() / 1000:.0f} тыс ₽ '
        f'против {W["min_sum"].median() / 1000:.0f} тыс ₽ у всех;')
    frq = W.loc[ok, "rate"].value_counts()
    say(f'   частота сделок у прошедших: ' + ", ".join(f"{k} {v}" for k, v in frq.items()))
    say(f'   медианная длина истории {W.loc[ok, "years"].median():.2f} против '
        f'{W["years"].median():.2f} года.')

    # ------------------------------------------------------- чувствительность --
    head("ЧУВСТВИТЕЛЬНОСТЬ: пороги выбраны нами, проверяем устойчивость")
    hdr = f'{"порог беты":<14}' + "".join(f'{f"t >= {t:.1f}":>14}' for t in (1.5, 2.0, 3.0))
    say(hdr)
    say("-" * len(hdr))
    for bmax in (0.20, 0.30, 0.50, 9.99):
        row = f'{("любая" if bmax > 9 else f"<= {bmax:.2f}"):<14}'
        for tmin in (1.5, 2.0, 3.0):
            m = (A & (W["beta"].abs() <= bmax) & (W["t_g"] >= tmin)).fillna(False)
            row += f'{f"{int(m.sum())} ({100 * m.mean():.1f} %)":>14}'
        say(row)
    say()
    say("Число прошедших меняется, порядок величины — нет: это единицы процентов")
    say("витрины при любом разумном выборе порогов.")

    # ------------------------------------------------------------- рыночность --
    head("СКОЛЬКО НА ВИТРИНЕ РЫНКА ПОД ДРУГИМ ИМЕНЕМ")
    say("Бета к индексу Мосбиржи, посчитанная на окне по общим торговым дням.")
    say()
    hdr = f'{"группа по бете":<26}{"стратегий":>12}{"доля":>9}{"подписок":>12}{"доля подписок":>16}'
    say(hdr)
    say("-" * len(hdr))
    b = W["beta"]
    for lo, hi, lab in ((-99, -0.3, "меньше -0.30 (обратная)"), (-0.3, 0.3, "от -0.30 до +0.30"),
                        (0.3, 0.7, "от +0.30 до +0.70"), (0.7, 99, "больше +0.70")):
        m = ((b > lo) & (b <= hi)).fillna(False)
        say(f'{lab:<26}{int(m.sum()):>12,}{f"{100 * m.mean():.1f} %":>9}'
            f'{int(w[m].sum()):>12,}{f"{100 * w[m].sum() / w.sum():.1f} %":>16}')
    say()
    say(f'медианный R² к индексу: {W["r2"].median():.3f}; '
        f'у стратегий с бетой больше 0.70 — {W.loc[b > 0.7, "r2"].median():.3f}')

    say()
    say("готово")


if __name__ == "__main__":
    main()
