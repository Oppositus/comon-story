"""comon_multiplicity.py — БЛОК 5: эмпирическая плата за перебор.

План исследования, Блок 5. Мы считаем поправку на множественность
гипотез теоретически (DSR, Bailey-López de Prado) и спорим о правилах учёта N
. Здесь перебор виден в дикой природе: автор с
k системами на витрине — это k попыток, из которых он показывает лучшую.

Что можно измерить прямо:
  • сколько попыток делает типичный автор;
  • насколько лучшая попытка лучше медианной — это и есть плата за перебор в пунктах
    Sharpe, наблюдаемая, а не выведенная;
  • какому ЭФФЕКТИВНОМУ числу независимых попыток соответствует наблюдаемый прирост
    (обратная задача к E[max] N нормальных) — и совпадает ли это с оценкой через
    корреляцию между системами автора. Два независимых пути к одному числу;
  • переоценена ли лучшая система: чем из большего числа попыток она отобрана, тем
    сильнее должна регрессировать и тем чаще умирать. Прямой аналог вопроса
    «сколько стоит отбор champion из сетки».

🔴 ОГРАНИЧЕНИЕ ЧЕСТНОСТИ. Автор публикует не все попытки: неудачные тесты не доходят
до витрины. Значит наблюдаемое k — НИЖНЯЯ граница реального перебора, а измеренная
плата — нижняя граница настоящей. Это делает вывод консервативным в нужную сторону:
если даже видимая часть перебора стоит столько-то, полная стоит больше.

Разделы:
  5.1 сколько систем на автора;
  5.2 насколько лучшая попытка лучше медианной;
  5.3 эффективное число независимых попыток — два независимых способа;
  5.4 выживают ли многопопыточные авторы и их лучшие системы;
  5.5 что это значит для учёта числа попыток.

Текстовый вывод. Запуск: python comon_multiplicity.py
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
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "author_pairs.csv.gz"
NPROC = 8

MIN_TRADE_DAYS = 100        # фильтр измеримости — тот же, что в Блоках 2–4
MIN_ACTIVITY = 0.10
MIN_OVERLAP = 250           # общих дней, иначе корреляция пары — шум
# группы авторов по числу стратегий: нужны и таблицам, и графикам 10 и 23
GRP = [(2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 10, "6–10"),
       (11, 20, "11–20"), (21, 10**9, "> 20")]
MAX_PAIRS_AUTHOR = 200      # у фабрик пар тысячи; берём первые 200 по id
EULER = 0.5772156649
# 🔴 owner 2215 «comon» — СЛУЖЕБНЫЙ аккаунт площадки, а не автор: на него переписаны
# системы при единовременной чистке 2014 года (4 058 систем, 80.6 % архиваций того года,
# ни одной живой; см. comon-block1-survival.md и comon-admin-closures.md). Считать их
# «попытками одного автора» нельзя — это чужие системы разных людей в одной корзине.
SERVICE_OWNERS = {2215}
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def emax(n):
    """E[max] n независимых стандартных нормальных (Bailey-López de Prado, как в DSR)."""
    n = max(float(n), 1.0 + 1e-9)
    return ((1 - EULER) * stats.norm.ppf(1 - 1 / n)
            + EULER * stats.norm.ppf(1 - 1 / (n * np.e)))


# 🔴 ПОЧЕМУ НЕ E[max] НАПРЯМУЮ. Наблюдаемая статистика — не максимум, а РАЗНОСТЬ
# (max − median) выборки из k систем, делённая на оценку σ. Её нулевое распределение
# отличается от E[max]: медиана смещена относительно среднего, а при малых k выборочные
# max и s связаны жёстким неравенством (max−mean)/s ≤ (k−1)/√k — при k = 3 это 1.155,
# и статистика упирается в потолок независимо от данных. Поэтому эталон берём симуляцией
# ровно той же статистики на k НЕЗАВИСИМЫХ нормальных.
_NULL_K = np.unique(np.round(np.geomspace(2, 5000, 90)).astype(int))
_NULL_V = None


def _null_table(B=20000, seed=12345):
    """Медиана (max − median) для выборки k независимых N(0,1); σ известна и равна 1."""
    global _NULL_V
    if _NULL_V is None:
        rng = np.random.default_rng(seed)
        _NULL_V = np.array([float(np.median(x.max(1) - np.median(x, 1)))
                            for x in (rng.standard_normal((B, int(k))) for k in _NULL_K)])
    return _NULL_V


def null_gain(k):
    """Ожидаемый прирост (в σ) при k независимых попытках — по симуляции."""
    v = _null_table()
    return float(np.interp(np.log(max(k, 2)), np.log(_NULL_K), v))


def n_from_gain(g):
    """Обратная задача: какому числу НЕЗАВИСИМЫХ попыток отвечает прирост g (в σ)."""
    if not np.isfinite(g) or g <= 0:
        return np.nan
    v = _null_table()
    if g <= v[0]:
        return 2.0
    if g >= v[-1]:
        return float(_NULL_K[-1])
    return float(np.exp(np.interp(g, v, np.log(_NULL_K))))


def series(sid):
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


CHARTS = "--charts" in sys.argv


def chart_best_of_n():
    """График 9: Sharpe лучшего из N случайных попыток за год."""
    import comon_charts as ch

    f, ax = ch.fig()
    n = np.geomspace(2, 300, 200)
    ax.plot(n, [emax(x) for x in n], color=ch.BLUE, lw=2.4, zorder=3)
    ax.axhline(1.94, color=ch.ORANGE, lw=1.8, ls="--", zorder=2)
    ax.annotate("1,94 — лучшая настоящая стратегия площадки\nза пять лет истории",
                xy=(2.15, 1.94), xytext=(0, 8), textcoords="offset points",
                fontsize=11, color=ch.ORANGE, ha="left")
    marks = [(2, "двое"), (5, "пятеро"), (10, "десять"), (20, "двадцать"),
             (100, "сто")]
    for k, lab in marks:
        v = emax(k)
        ax.plot([k], [v], marker="o", ms=9, color=ch.BLUE, zorder=4,
                markeredgecolor=ch.SURFACE, markeredgewidth=1.6)
        # кривая монотонно растёт: свободно справа-снизу от точки, а у последней
        # отметки — слева-сверху, иначе подпись уходит за край поля
        up, dx, ha = (False, 9, "left") if k < 100 else (True, -10, "right")
        ch.stack_label(ax, (k, v), f"{v:.2f}".replace(".", ","), lab,
                       dx=dx, up=up, ha=ha, color=ch.BLUE)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100, 300])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", "300"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(1.8, 340)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("сколько человек пробует (никто из них не умеет торговать)")
    ax.set_ylabel("Sharpe лучшего из них за год")
    ax.set_title("Двадцать бездарностей за год дают то же, что лучшая стратегия за пять лет")
    ch.note(ax, "Ожидаемый максимум независимых случайных попыток при годовой истории. "
                "Чистая арифметика случайных чисел: никаких предположений о рынке.")
    p = ch.save(f, 9, "best-of-n")
    print("    " + "; ".join(f"N={k} → {emax(k):.2f}" for k, _ in marks), flush=True)
    return p


def chart_penalty_heat():
    """График 22: сколько пунктов Sharpe добавить к планке — поправка N × T."""
    import comon_charts as ch
    from matplotlib.colors import LinearSegmentedColormap

    ns = [2, 3, 10, 30, 100, 1000]
    ts = [1, 2, 3, 5, 10]
    Z = np.array([[emax(n) / np.sqrt(t) for t in ts] for n in ns])
    f, ax = ch.fig(h_px=950, bottom=0.26)
    cmap = LinearSegmentedColormap.from_list("seq", ch.SEQ)
    im = ax.imshow(Z, cmap=cmap, aspect="auto", vmin=0, vmax=Z.max())
    for i in range(len(ns)):
        for j in range(len(ts)):
            v = Z[i, j]
            ax.annotate(f"{v:.2f}".replace(".", ","), xy=(j, i), ha="center",
                        va="center", fontsize=13, fontweight="bold",
                        color="#ffffff" if v > 0.55 * Z.max() else ch.INK)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t} год" if t == 1 else f"{t} года" if t < 5 else f"{t} лет"
                        for t in ts])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"{n:,}".replace(",", " ") for n in ns])
    ax.set_xlabel("длина истории")
    ax.set_ylabel("сколько было попыток")
    ax.grid(False)
    ax.set_title("Надбавка к планке Sharpe за перебор вариантов")
    f.colorbar(im, ax=ax, pad=0.02).set_label("пунктов Sharpe", fontsize=11)
    ch.note(ax, "Столько нужно прибавить к требуемому Sharpe, чтобы лучший из N попыток "
                "что-то значил. Время работает на честность: та же тридцатка попыток "
                "стоит 1,20 пункта на трёх годах и 0,66 на десяти.")
    p = ch.save(f, 22, "overfit-penalty-heatmap")
    print(f"    сетка {len(ns)}×{len(ts)}, от {Z.min():.2f} до {Z.max():.2f} пунктов",
          flush=True)
    return p


def chart_gain_vs_null(A):
    """График 10: наблюдаемый прирост лучшей над медианной против эталона."""
    import comon_charts as ch

    f, ax = ch.fig()
    xs, obs, null, ns = [], [], [], []
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5:
            continue
        kk = float(sub["k"].median())
        sd_g = float(np.sqrt(np.nanmean(sub["sd"] ** 2)))
        xs.append(kk)
        obs.append(float(sub["gain"].median()) / sd_g)
        null.append(null_gain(kk))
        ns.append(len(sub))
    ax.plot(xs, null, color=ch.ORANGE, lw=2.2, ls="--", marker="s", ms=8, zorder=3,
            label="если бы попытки были независимы")
    ax.plot(xs, obs, color=ch.BLUE, lw=2.4, marker="o", ms=9, zorder=4,
            label="наблюдается у авторов")
    ax.fill_between(xs, obs, null, color=ch.BAND, alpha=0.45, lw=0, zorder=2)
    ax.legend(loc="upper left", handlelength=3.0, labelspacing=0.6)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0f}" for x in xs])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("сколько стратегий у автора")
    ax.set_ylabel("насколько лучшая выше типичной, в мерах разброса")
    ax.set_title("Попытки одного автора — не независимые попытки")
    ch.note(ax,
            "Прирост считается по каждому автору отдельно (его лучшая минус его "
            "типичная) и лишь потом усредняется по группе.",
            "Зазор между линиями и есть мера связанности: чем ниже наблюдаемая линия, "
            "тем сильнее стратегии автора повторяют друг друга.",
            "Авторов в группах: " + ", ".join(f"{int(k)} — {n}"
                                              for k, n in zip(xs, ns)) + ".")
    p = ch.save(f, 10, "gain-vs-independence")
    print("    " + "; ".join(f"k={k:.0f}: набл {o:.2f} / эталон {z:.2f}"
                             for k, o, z in zip(xs, obs, null)), flush=True)
    return p


def chart_neff(A, PR, rho_bar):
    """График 23: во сколько независимых попыток превращаются k связанных."""
    import comon_charts as ch

    f, ax = ch.fig()
    xs, na_v, nb_v = [], [], []
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5 or lo == 2 or hi > 20:      # k=2 вырожден, >20 неоднородна
            continue
        kk = float(sub["k"].median())
        sd_g = float(np.sqrt(np.nanmean(sub["sd"] ** 2)))
        pr_g = PR[PR["owner"].isin(sub["owner"])]
        rb = float(pr_g["rho"].mean()) if len(pr_g) >= 10 else rho_bar
        xs.append(kk)
        na_v.append(n_from_gain(float(sub["gain"].median()) / sd_g))
        nb_v.append(kk / (1 + (kk - 1) * max(rb, 0.0)))
    dia = np.linspace(min(xs) * 0.9, max(xs) * 1.1, 50)
    ax.plot(dia, dia, color=ch.PALE, lw=1.6, ls=":", zorder=2,
            label="если бы все попытки были разными")
    ax.plot(xs, na_v, color=ch.BLUE, lw=2.4, marker="o", ms=9, zorder=4,
            label="оценка по приросту лучшей")
    ax.plot(xs, nb_v, color=ch.AQUA, lw=2.4, marker="^", ms=9, zorder=4,
            label="оценка по сходству кривых")
    ax.legend(loc="upper left", handlelength=3.0, labelspacing=0.6)
    ax.set_xlabel("сколько стратегий у автора")
    ax.set_ylabel("сколько среди них по-настоящему разных")
    ax.set_title("Полтора десятка стратегий работают как две-пять")
    ch.note(ax,
            "Две независимые оценки: обратная задача к эталону случайного перебора и "
            "прямая корреляция дневных кривых. Совпадение оценок — сильный аргумент.",
            f"Средняя корреляция между системами одного автора — {rho_bar:.2f}"
            .replace(".", ",") + f". Пар в расчёте: {len(PR):,}".replace(",", " ") + ".")
    p = ch.save(f, 23, "effective-independent-tries")
    print("    " + "; ".join(f"k={k:.0f}: по приросту {a:.1f}, по сходству {b:.1f}"
                             for k, a, b in zip(xs, na_v, nb_v)), flush=True)
    return p


def chart_authors_vs_systems(rows, n_auth, n_sys):
    """График 24: одиночек большинство среди авторов, но не среди стратегий."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=1000, bottom=0.26)
    x = np.arange(len(rows))
    a = [r["sh_a"] for r in rows]
    s_ = [r["sh_s"] for r in rows]
    ax.bar(x - 0.19, a, width=0.36, color=ch.BLUE, zorder=3,
           label=f"доля авторов (всего {ch.n_(n_auth)})")
    ax.bar(x + 0.19, s_, width=0.36, color=ch.ORANGE, zorder=3,
           label=f"доля стратегий (всего {ch.n_(n_sys)})")
    for xi, r in zip(x, rows):
        for dx, v, n, c in ((-0.19, r["sh_a"], r["authors"], ch.BLUE),
                            (0.19, r["sh_s"], r["sys"], ch.ORANGE)):
            ax.annotate(f"{ch.n_(v, 1)} %\n{ch.n_(n)}", xy=(xi + dx, v),
                        xytext=(0, 6), textcoords="offset points", fontsize=11,
                        color=c, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([r["lab"] for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(max(a), max(s_)) * 1.30)
    ch.pct_raw(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handlelength=2.2, labelspacing=0.55)
    ax.set_xlabel("сколько измеримых стратегий у автора")
    ax.set_ylabel("доля своей популяции")
    ax.set_title("Одиночек большинство среди авторов, но не среди стратегий")
    ch.note(ax,
            f"База — {ch.n_(n_sys)} стратегий, прошедших фильтр измеримости, "
            f"у {ch.n_(n_auth)} авторов; служебный аккаунт площадки исключён. "
            f"Под каждой долей — абсолютное число.",
            "Две единицы счёта отвечают на разные вопросы. Людей с единственной "
            "попыткой — большинство, но написанного ими на витрине меньше "
            "трети: почти две трети того, что видит посетитель, сделано теми, "
            "у кого попытка не одна.")
    ch.save(f, 24, "authors-vs-systems")
    print("    " + "; ".join(f"k={r['lab']}: авторов {r['sh_a']:.1f} % / систем "
                             f"{r['sh_s']:.1f} %" for r in rows), flush=True)


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    S = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    M = S.join(panel[["cagr", "maxdd", "owner_id", "author", "is_live", "years",
                      "archived_at"]], how="inner", rsuffix="_p")
    good = (M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)
    G = M[good].copy()
    G["live"] = G["is_live"] == True                                    # noqa: E712
    n_svc = int(M["owner_id"].isin(SERVICE_OWNERS).sum())
    n_svc_g = int(G["owner_id"].isin(SERVICE_OWNERS).sum())
    panel = panel[~panel["owner_id"].isin(SERVICE_OWNERS)]
    M = M[~M["owner_id"].isin(SERVICE_OWNERS)]
    G = G[~G["owner_id"].isin(SERVICE_OWNERS)]

    # ── 5.1 ────────────────────────────────────────────────────────────────────
    say("=" * 104)
    say("5.1 СКОЛЬКО СИСТЕМ НА ОДНОГО АВТОРА")
    say("=" * 104)
    say(f"🔴 Исключён служебный аккаунт площадки owner 2215 «comon»: {n_svc:,} систем"
        f" с рядом от 30 точек")
    say(f"(из них {n_svc_g:,} прошли бы фильтр измеримости). Это не автор — на него переписаны")
    say("системы при единовременной чистке 2014 года (ни одной живой, медианный Sharpe −0.65).")
    say("Без исключения он один давал бы 78 % группы «> 20 попыток» и ломал бы весь блок.")
    say()
    say("Три выборки: вся популяция (включая системы без ряда), ряд от 30 точек,")
    say("и измеримые (фильтр Блоков 2–4). Разница показывает, сколько попыток")
    say("автора вообще не дожили до состояния «есть что мерить».")
    say()
    hdr = (f'{"выборка":<26} {"авторов":>9} {"систем":>9} {"медиана":>9} {"p90":>7} '
           f'{"p99":>7} {"макс":>7} {"доля у топ-1 %":>16}')
    say(hdr)
    say("-" * len(hdr))
    # 🔴 M построена из sharpe.csv.gz, куда попадают ТОЛЬКО ряды от 30 точек
    # (фильтр в comon_sharpe_dist.py) — это 11 657 из 18 354 систем с рядом.
    # Подпись «системы с рядом» читалась как все 18 354 и однажды утекла в текст
    # главы 3 («54.2 % от тех, у кого есть ряд» вместо 34.4 %). Подпись — точная.
    for lab, sub in (("вся популяция", panel[panel["owner_id"].notna()]),
                     ("ряд от 30 точек", M), ("измеримые", G)):
        vc = sub["owner_id"].value_counts()
        top = int(np.ceil(len(vc) * 0.01))
        say(f'{lab:<26} {len(vc):>9,} {len(sub):>9,} {vc.median():>9.0f} '
            f'{vc.quantile(.90):>7.0f} {vc.quantile(.99):>7.0f} {vc.max():>7.0f} '
            f'{100*vc.nlargest(top).sum()/len(sub):>15.1f}%')
    say()
    vc = G["owner_id"].value_counts()
    hdr = (f'{"попыток у автора (k)":<24} {"авторов":>9} {"систем":>9} '
           f'{"доля систем":>13}')
    say(hdr)
    say("-" * len(hdr))
    kbins = [(1, 1, "1"), (2, 3, "2–3"), (4, 10, "4–10"),
             (11, 30, "11–30"), (31, 10**9, "> 30")]
    krows = []
    for lo, hi, lab in kbins:
        m = (vc >= lo) & (vc <= hi)
        say(f'{lab:<24} {int(m.sum()):>9,} {int(vc[m].sum()):>9,} '
            f'{100*vc[m].sum()/len(G):>12.1f}%')
        krows.append({"lab": lab, "authors": int(m.sum()), "sys": int(vc[m].sum()),
                      "sh_a": 100 * float(m.sum() / len(vc)),
                      "sh_s": 100 * float(vc[m].sum() / len(G))})
    if CHARTS:
        say()
        say("── график 24 ────────────────────────────────────────────────────")
        chart_authors_vs_systems(krows, len(vc), len(G))

    # ── 5.2 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("5.2 НАСКОЛЬКО ЛУЧШАЯ ПОПЫТКА ЛУЧШЕ МЕДИАННОЙ")
    say("=" * 104)
    say("Внутри автора: S лучшей системы минус S медианной. Это плата за перебор,")
    say("наблюдаемая напрямую: столько Sharpe добавляет право показать лучшее из k.")
    say()
    rows = []
    for oid, sub in G.groupby("owner_id"):
        k = len(sub)
        if k < 2:
            continue
        s = sub["sharpe_ar"].to_numpy()
        rows.append({"owner": oid, "k": k, "s_max": float(s.max()),
                     "s_med": float(np.median(s)), "s_min": float(s.min()),
                     "sd": float(s.std(ddof=1)),
                     "best_live": bool(sub.loc[sub["sharpe_ar"].idxmax(), "live"]),
                     "any_live": bool(sub["live"].any()),
                     "live_share": float(sub["live"].mean())})
    A = pd.DataFrame(rows)
    A["gain"] = A["s_max"] - A["s_med"]
    say(f"авторов с ≥ 2 измеримыми системами: {len(A):,}")
    # общий масштаб разброса внутри автора — пул по авторам с k ≥ 3
    sd_pool = float(np.sqrt(np.nanmean(A.loc[A["k"] >= 3, "sd"] ** 2)))
    say(f"пулированный разброс Sharpe ВНУТРИ автора (k ≥ 3): σ = {sd_pool:.3f}")
    say(f"для сравнения, разброс Sharpe МЕЖДУ всеми системами: "
        f"σ = {G['sharpe_ar'].std(ddof=1):.3f}")
    say()
    say("🔴 Нормировка — на разброс ВНУТРИ СВОЕЙ группы k, а не на общий: авторы с")
    say("десятками систем имеют втрое больший собственный разброс, и общий σ дал бы")
    say("им фиктивно огромный прирост в сигмах.")
    say()
    A.to_csv(DIR / "authors.csv.gz", index=False, compression="gzip")
    hdr = (f'{"k (попыток)":<14} {"k от–до":>10} {"авторов":>9} {"S лучшей":>10} '
           f'{"S медианной":>13} {"прирост":>9} {"σ группы":>10} {"прирост/σ":>11} '
           f'{"эталон незав.":>14} {"E[max] (для справки)":>21}')
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5:
            continue
        g = sub["gain"].median()
        kk = sub["k"].median()
        sd_g = float(np.sqrt(np.nanmean(sub["sd"] ** 2)))
        # при k = 2 медиана двух точек = их среднее, и (max−med)/σ ≡ const:
        # величина вырождена и о переборе ничего не сообщает
        rel = "вырожд." if lo == 2 else f"{g/sd_g:.2f}"
        rng = f'{int(sub["k"].min())}–{int(sub["k"].max())}'
        say(f'{lab:<14} {rng:>10} {len(sub):>9,} {sub["s_max"].median():>10.2f} '
            f'{sub["s_med"].median():>13.2f} {g:>9.2f} {sd_g:>10.2f} {rel:>11} '
            f'{null_gain(kk):>14.2f} {emax(kk):>21.2f}')
    say()
    say("«прирост/σ» — во сколько сигм внутриавторского разброса обходится право выбрать")
    say("лучшее. «эталон незав.» — та же статистика (max − median)/σ, посчитанная СИМУЛЯЦИЕЙ")
    say("на k НЕЗАВИСИМЫХ нормальных попытках (20 000 повторов). Если наблюдаемое НИЖЕ")
    say("эталона — попытки автора коррелированы (что и ждём: это варианты одной идеи).")
    say()
    say("🔴 Колонка E[max] дана СПРАВОЧНО и как эталон НЕ годится: она описывает ожидание")
    say("максимума при известной σ, а у нас статистика — разность (max − median), и σ")
    say("оценена по той же выборке. При малых k выборочные max и s связаны неравенством")
    say("(max−mean)/s ≤ (k−1)/√k — при k = 3 потолок 1.155, и статистика упирается в него")
    say("независимо от данных. Симуляция воспроизводит эти эффекты, аналитика — нет.")
    say("k = 2 помечено «вырожд.»: там медиана совпадает со средним двух точек, отношение")
    say("(max−median)/σ математически постоянно и о переборе не сообщает ничего.")

    # ── 5.3 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("5.3 ЭФФЕКТИВНОЕ ЧИСЛО НЕЗАВИСИМЫХ ПОПЫТОК — ДВА НЕЗАВИСИМЫХ СПОСОБА")
    say("=" * 104)
    say("Способ A: обратная задача к СИМУЛИРОВАННОМУ эталону. Наблюдаемый прирост/σ →")
    say("при каком числе независимых попыток та же статистика имеет то же значение.")
    say("Способ B: прямая корреляция дневных рядов систем одного автора. Средняя ρ̄ →")
    say("эффективное N по Kish: N_eff = k / (1 + (k−1)·ρ̄).")
    say("Совпадение двух оценок — сильный аргумент; расхождение тоже информативно.")
    say()
    multi = vc[vc >= 2].index.tolist()
    cand = G[G["owner_id"].isin(multi)].index.tolist()
    with Pool(NPROC) as pool:
        ser = {x[0]: (x[1], x[2]) for x in
               pool.imap_unordered(series, cand, chunksize=100) if x}
    say(f"рядов загружено: {len(ser):,}")
    prs = []
    for oid in multi:
        ids = [i for i in G.index[G["owner_id"] == oid] if i in ser]
        if len(ids) < 2:
            continue
        for j, (a, b) in enumerate(combinations(sorted(ids), 2)):
            if j >= MAX_PAIRS_AUTHOR:
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
            if np.isfinite(rho):
                prs.append({"owner": oid, "a": a, "b": b, "n": len(common), "rho": rho})
    PR = pd.DataFrame(prs)
    PR.to_csv(OUT, index=False, compression="gzip")
    say(f"пар систем одного автора с общим окном ≥ {MIN_OVERLAP} дней: {len(PR):,} "
        f"у {PR['owner'].nunique():,} авторов")
    say()
    hdr = (f'{"квантиль ρ":<14} {"p5":>8} {"p25":>8} {"медиана":>9} {"p75":>8} '
           f'{"p95":>8} {"доля ρ > 0.8":>14}')
    say(hdr)
    say("-" * len(hdr))
    say(f'{"":<14}' + "".join(f'{PR["rho"].quantile(p):>8.2f}'
                              for p in (.05, .25, .50, .75, .95))
        + f'{100*(PR["rho"] > 0.8).mean():>13.1f}%')
    say()
    rho_bar = float(PR["rho"].mean())
    say(f"средняя корреляция между системами одного автора: ρ̄ = {rho_bar:.3f}")
    say()
    hdr = (f'{"k (попыток)":<14} {"k от–до":>10} {"авторов":>9} {"прирост/σ":>11} '
           f'{"N_eff (A)":>11} {"N_eff (B)":>11} {"N_eff/k (A)":>13} {"N_eff/k (B)":>13}')
    say(hdr)
    say("-" * len(hdr))
    shares = []
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5:
            continue
        kk = float(sub["k"].median())
        sd_g = float(np.sqrt(np.nanmean(sub["sd"] ** 2)))
        g = float(sub["gain"].median()) / sd_g
        na = np.nan if lo == 2 else n_from_gain(g)          # k=2 вырожден, см. 5.2
        # ρ̄ по авторам этой группы, если пар хватает
        pr_g = PR[PR["owner"].isin(sub["owner"])]
        rb = float(pr_g["rho"].mean()) if len(pr_g) >= 10 else rho_bar
        nb = kk / (1 + (kk - 1) * max(rb, 0.0))
        if np.isfinite(na) and hi <= 20:        # группа > 20 неоднородна, см. ниже
            shares.append((kk, na / kk, nb / kk))
        sa_s = "—" if not np.isfinite(na) else f"{na:.1f}"
        ra_s = "—" if not np.isfinite(na) else f"{na/kk:.2f}"
        rng = f'{int(sub["k"].min())}–{int(sub["k"].max())}'
        say(f'{lab:<14} {rng:>10} {len(sub):>9,} {g:>11.2f} {sa_s:>11} {nb:>11.1f} '
            f'{ra_s:>13} {nb/kk:>13.2f}')
    say()
    say("⚠️ Последняя строка ненадёжна: 5 авторов, k разнится втрое (23–69), медиана k по")
    say("группе не соответствует медиане прироста → способ A даёт завышенную оценку. В выводах")
    say("используются строки k = 3…20, где группы однородны.")
    say("N_eff (A) — из наблюдаемого прироста лучшей над медианной.")
    say("N_eff (B) — из корреляции рядов, формула Kish.")
    say("N_eff/k — доля попыток, которые «считаются» как независимые.")

    if CHARTS:
        say()
        say("── графики 9, 10, 22, 23 ─────────────────────────────────────────")
        # графики 10 и 23 рисует comon_leverage_clones.py: главa 4 считает эти
        # таблицы по ПОЛНОМУ перебору пар, а здесь стоит ограничение
        # MAX_PAIRS_AUTHOR — оно смещает ρ̄ вверх (0.374 против 0.325)
        chart_best_of_n()
        chart_penalty_heat()

    # ── 5.4 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("5.4 ПЕРЕОЦЕНЕНА ЛИ ЛУЧШАЯ СИСТЕМА: ВЫЖИВАНИЕ ПО ЧИСЛУ ПОПЫТОК")
    say("=" * 104)
    say("Если лучшая система автора — результат отбора из k попыток, то чем больше k,")
    say("тем сильнее в её Sharpe вклад везения и тем хуже она должна жить дальше.")
    say("Это прямой аналог вопроса «сколько стоит отбор лучшего варианта из сетки».")
    say()
    hdr = (f'{"k (попыток)":<14} {"авторов":>9} {"S лучшей":>10} {"лучшая жива":>13} '
           f'{"жива любая":>12} {"доля живых систем":>19} {"лет жизни":>11}')
    say(hdr)
    say("-" * len(hdr))
    yrs = G.groupby("owner_id")["years"].median()
    for lo, hi, lab in [(2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 10, "6–10"),
                        (11, 20, "11–20"), (21, 10**9, "> 20")]:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5:
            continue
        yy = yrs.reindex(sub["owner"]).median()
        say(f'{lab:<14} {len(sub):>9,} {sub["s_max"].median():>10.2f} '
            f'{100*sub["best_live"].mean():>12.1f}% {100*sub["any_live"].mean():>11.1f}% '
            f'{100*sub["live_share"].mean():>18.1f}% {yy:>11.1f}')
    say()
    # авторы-одиночки как контрольная группа
    solo = G[G["owner_id"].isin(vc[vc == 1].index)]
    say(f'контроль — авторы с ОДНОЙ системой: {len(solo):,} систем, '
        f'медиана S {solo["sharpe_ar"].median():.2f}, живых {100*solo["live"].mean():.1f} %, '
        f'медиана лет жизни {solo["years"].median():.1f}')
    say()
    for lab, a, b in (("k ↔ S лучшей системы", "k", "s_max"),
                      ("k ↔ прирост лучшей над медианной", "k", "gain"),
                      ("k ↔ доля живых систем автора", "k", "live_share")):
        s = A[[a, b]].dropna()
        rho = stats.spearmanr(s[a], s[b]).statistic
        say(f'{lab:<40} Spearman ρ = {rho:+.3f}')

    # ── 5.5 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("5.5 ЧТО ЭТО ЗНАЧИТ ДЛЯ УЧЁТА ЧИСЛА ПОПЫТОК")
    say("=" * 104)
    say("Правило учёта: N считается по ДАННЫМ и ЦЕЛИ,")
    say("единица — подглядывание с правом отбора, корреляцию кредитует V в формуле DSR.")
    if shares:
        sa = np.nanmedian([x[1] for x in shares])
        sb = np.nanmedian([x[2] for x in shares])
        say(f"Здесь измерено: доля попыток автора, работающих как независимые, ≈ "
            f"{sa:.2f} (через прирост) и {sb:.2f} (через корреляцию рядов).")
    say()
    say("Ориентир: сколько Sharpe нужно ДОБАВИТЬ к порогу, чтобы скомпенсировать перебор")
    say("N независимых попыток при годовой длине ряда T (формула порога DSR):")
    say()
    hdr = f'{"N попыток":<14}' + "".join(f'{f"T={t} лет":>12}' for t in (1, 2, 3, 5, 10))
    say(hdr)
    say("-" * len(hdr))
    for n in (2, 3, 10, 30, 100, 1000, 10000):
        row = f'{n:<14,}'
        for T in (1, 2, 3, 5, 10):
            # порог отбора: SR* = σ_SR · E[max_N], σ_SR ≈ sqrt(1/T) при S≈0
            row += f'{emax(n) / np.sqrt(T):>12.2f}'
        say(row)
    say()
    say("Последняя строка — масштаб машинного перебора: десять тысяч конфигураций при")
    say(f"трёхлетнем ряде поднимают планку на {emax(10000)/np.sqrt(3):.2f} пункта Sharpe.")

    (ROOT / "results" / "comon_multiplicity.log").write_text(
        "\n".join(_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
