"""comon_leverage_clones.py — ПОПЫТКА ИЛИ ТА ЖЕ СИСТЕМА ПОД ДРУГИМ ПЛЕЧОМ.

Пункт «Глава 4, п.3» ручной вычитки серии: у одного
автора часто стоят «система А», «система А × 2» и «система Б». Считать это за три
попытки нельзя, и доля пар с корреляцией выше 0.8 — это во многом такие клоны, а не
разные ставки. Вопрос: если клоны схлопнуть в одну запись, изменятся ли выводы 4.3–4.4.

КАК ОПРЕДЕЛЯЕТСЯ КЛОН. Две системы одного автора считаются одной попыткой, если их
дневные ряды на общем окне (≥ 250 торговых дней) движутся практически одинаково:
ρ ≥ порога. Плечо на корреляцию не влияет вовсе — оно меняет только масштаб, — поэтому
высокая ρ ловит и «та же система под плечом», и «та же система с другими параметрами».
Наклон регрессии b на a через ноль (λ) показывает само плечо: λ ≈ 1 — копия, λ ≠ 1 —
масштабированная версия. Порог по ρ — соглашение, поэтому считается ТРИ (0.90/0.95/0.99).

СХЛОПЫВАНИЕ. Внутри автора клоны образуют граф; в каждой связной компоненте остаётся
ОДНА система — с самой длинной историей (больше всего торговых дней). Выбор нейтрален
к результату: по длине ряда, а не по Sharpe.

Считается «до» и «после» на одних и тех же данных:
  1 сколько систем и авторов теряется при схлопывании (три порога);
  2 корреляции пар: медиана, среднее, доля выше 0.8;
  3 плата за перебор по группам k (таблица 4.3);
  4 независимые попытки двумя способами (таблица 4.4);
  5 «гипотеза, которая не подтвердилась» (Sharpe лучшей и живучесть по k).

Запуск: python comon_leverage_clones.py
"""
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comon_multiplicity import (                                       # noqa: E402
    DIR, MIN_ACTIVITY, MIN_TRADE_DAYS, MIN_OVERLAP, SERVICE_OWNERS,
    null_gain, n_from_gain, series)

NPROC = 8
TAUS = (0.90, 0.95, 0.99)
TAU_MAIN = 0.95
GRP = [(3, 3, "3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 20, "11–20"),
       (21, 10**9, "> 20")]
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def author_table(G):
    """Таблица авторов: k, лучший/медианный Sharpe, разброс, живучесть."""
    rows = []
    for oid, sub in G.groupby("owner_id"):
        if len(sub) < 2:
            continue
        s = sub["sharpe_ar"].to_numpy()
        rows.append({"owner": oid, "k": len(sub), "s_max": float(s.max()),
                     "s_med": float(np.median(s)), "sd": float(s.std(ddof=1)),
                     "best_live": bool(sub.loc[sub["sharpe_ar"].idxmax(), "live"]),
                     "any_live": bool(sub["live"].any())})
    A = pd.DataFrame(rows)
    A["gain"] = A["s_max"] - A["s_med"]
    return A


def gain_table(A, rho_bar, title):
    """Плата за перебор + независимые попытки двумя способами."""
    say(title)
    hdr = (f'{"k (попыток)":<12} {"авторов":>9} {"мед. k":>7} {"S лучшей":>10}'
           f' {"прирост":>9} {"σ группы":>10} {"прирост/σ":>11} {"эталон":>8}'
           f' {"незав. (эталон)":>16} {"незав. (ρ)":>12}'
           f' {"доля (эталон)":>14} {"доля (ρ)":>10}')
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5:
            continue
        g = float(sub["gain"].median())
        kk = float(sub["k"].median())
        # 🔴 разброс группы — взвешенный по числу стратегий, как в главе 4
        # (скрипт comon_multiplicity считает невзвешенный, и числа расходятся)
        sd_g = float(np.sqrt(np.average(sub["sd"] ** 2, weights=sub["k"])))
        rel = g / sd_g
        n_eta = n_from_gain(rel)
        n_rho = kk / (1 + (kk - 1) * rho_bar)
        say(f'{lab:<12} {len(sub):>9,} {kk:>7.0f} {sub["s_max"].median():>10.2f}'
            f' {g:>9.2f} {sd_g:>10.2f} {rel:>11.2f} {null_gain(kk):>8.2f}'
            f' {n_eta:>16.1f} {n_rho:>12.1f}'
            f' {n_eta/kk:>14.2f} {n_rho/kk:>10.2f}')
    say()


def pairs_of(G, ser, max_pairs=None):
    """Пары систем одного автора с общим окном ≥ MIN_OVERLAP.

    max_pairs воспроизводит ограничение скрипта главы (первые 200 пар по порядку id);
    None — все пары без ограничения.
    """
    out = []
    for oid, sub in G.groupby("owner_id"):
        ids = [i for i in sub.index if i in ser]
        if len(ids) < 2:
            continue
        for j, (a, b) in enumerate(combinations(sorted(ids), 2)):
            if max_pairs is not None and j >= max_pairs:
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
            if not np.isfinite(rho):
                continue
            lam = float(xa @ xb / (xa @ xa)) if xa @ xa > 0 else np.nan
            out.append({"owner": oid, "a": a, "b": b, "n": len(common),
                        "rho": rho, "lam": lam,
                        "vol_ratio": float(xb.std(ddof=1) / xa.std(ddof=1))})
    return pd.DataFrame(out)


def collapse(G, PR, tau):
    """Схлопнуть клоны (ρ ≥ tau) в одну систему на компоненту связности."""
    keep = set(G.index)
    drop = set()
    for oid, sub in PR[PR["rho"] >= tau].groupby("owner"):
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        for _, r in sub.iterrows():
            ra, rb = find(r["a"]), find(r["b"])
            if ra != rb:
                parent[ra] = rb
        comp = {}
        for i in set(sub["a"]) | set(sub["b"]):
            comp.setdefault(find(i), []).append(i)
        for members in comp.values():
            if len(members) < 2:
                continue
            best = max(members, key=lambda i: G.at[i, "n_trade"])
            drop |= {i for i in members if i != best}
    return G.loc[sorted(keep - drop)], drop


CHARTS = "--charts" in sys.argv


def _rows(A, rho_bar):
    """Строки таблицы 4.4 в виде чисел — тот же расчёт, что печатает gain_table."""
    out = []
    for lo, hi, lab in GRP:
        sub = A[(A["k"] >= lo) & (A["k"] <= hi)]
        if len(sub) < 5 or hi > 20:          # группа «> 20» неоднородна (k от 23 до 69)
            continue
        kk = float(sub["k"].median())
        sd_g = float(np.sqrt(np.average(sub["sd"] ** 2, weights=sub["k"])))
        rel = float(sub["gain"].median()) / sd_g
        out.append({"k": kk, "lab": lab, "n": len(sub), "rel": rel,
                    "null": null_gain(kk), "n_eta": n_from_gain(rel),
                    "n_rho": kk / (1 + (kk - 1) * rho_bar)})
    return out


def chart_gain_vs_null(A, rho_bar):
    """График 10: наблюдаемый прирост лучшей над типичной против эталона."""
    import comon_charts as ch

    r = _rows(A, rho_bar)
    # ось КАТЕГОРИАЛЬНАЯ: точки — это группы авторов, а не отдельные значения.
    # Числовая ось получалась логарифмической, шаг «3 → 4» занимал столько же,
    # сколько «7 → 12», а пропущенные 5, 6, 8…11 читались как потерянные данные.
    xs = list(range(len(r)))
    f, ax = ch.fig()
    ax.plot(xs, [x["null"] for x in r], color=ch.ORANGE, lw=2.2, ls="--", marker="s",
            ms=8, zorder=3, label="если бы попытки были независимы")
    ax.plot(xs, [x["rel"] for x in r], color=ch.BLUE, lw=2.6, marker="o", ms=9,
            zorder=4, label="наблюдается у авторов")
    ax.fill_between(xs, [x["rel"] for x in r], [x["null"] for x in r],
                    color=ch.BAND, alpha=0.40, lw=0, zorder=2,
                    label="разрыв — то, что съедает повторяемость")
    ax.legend(loc="upper left", handlelength=3.0, labelspacing=0.6)
    ax.set_xlim(-0.35, len(r) - 0.65)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x['lab']}\n{x['n']} авторов" for x in r])
    ax.set_ylim(0, max(x["null"] for x in r) * 1.35)
    ax.set_xlabel("группа авторов по числу их стратегий")
    # в две строки: одной строкой подпись длиннее панели по высоте и обрезается
    ax.set_ylabel("насколько лучшая выше типичной,\nв мерах разброса")
    ax.set_title("Попытки одного автора — не независимые попытки")
    ch.note(ax,
            "Прирост считается по каждому автору отдельно (его лучшая минус его "
            "типичная) и лишь потом усредняется по группе.",
            "Зазор между линиями и есть мера связанности: чем ниже наблюдаемая "
            "линия, тем сильнее стратегии автора повторяют друг друга.",
            "Авторов с двумя стратегиями на графике нет: там статистика вырождена. "
            "Группы больше двадцати нет тоже — в ней пять авторов с числом "
            "стратегий от 23 до 69, и общее для группы значение бессмысленно.")
    pth = ch.save(f, 10, "gain-vs-independence")
    print("    " + "; ".join(f"{x['lab']} ({x['n']} авт.): набл {x['rel']:.2f} / "
                             f"эталон {x['null']:.2f}" for x in r), flush=True)
    return pth


def chart_neff(A, rho_bar, npairs):
    """График 23: во сколько независимых попыток превращаются k связанных."""
    import comon_charts as ch

    r = _rows(A, rho_bar)
    xs = list(range(len(r)))
    ks = [x["k"] for x in r]
    f, ax = ch.fig()
    ax.plot(xs, ks, color=ch.PALE, lw=1.8, ls=":", marker="D", ms=7, zorder=2,
            label="если бы все попытки были разными")
    ax.plot(xs, [x["n_eta"] for x in r], color=ch.BLUE, lw=2.6, marker="o", ms=9,
            zorder=4, label="оценка по приросту лучшей")
    ax.plot(xs, [x["n_rho"] for x in r], color=ch.AQUA, lw=2.6, marker="^", ms=9,
            zorder=4, label="оценка по сходству кривых")
    ax.legend(loc="upper left", handlelength=3.0, labelspacing=0.6)
    ax.set_xlim(-0.35, len(r) - 0.65)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x['lab']}\n{x['n']} авторов" for x in r])
    ax.set_ylim(0, max(ks) * 1.15)
    ax.set_xlabel("группа авторов по числу их стратегий")
    ax.set_ylabel("сколько среди них по-настоящему разных")
    ax.set_title("Полтора десятка стратегий работают как две-пять")
    ch.note(ax,
            "Две независимые оценки: обратная задача к эталону случайного перебора "
            "и прямая корреляция дневных кривых. Совпадение оценок — сильный "
            "аргумент.",
            f"Средняя корреляция между системами одного автора — "
            f"{rho_bar:.3f}".replace(".", ",")
            + f"; пар в расчёте {npairs:,}".replace(",", " ")
            + " (полный перебор, без ограничения на автора).")
    pth = ch.save(f, 23, "effective-independent-tries")
    print("    " + "; ".join(f"{x['lab']} (медиана {x['k']:.0f}): по приросту "
                             f"{x['n_eta']:.1f}, по сходству {x['n_rho']:.1f}"
                             for x in r), flush=True)
    return pth


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    S = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    M = S.join(panel[["owner_id", "author", "is_live"]], how="inner", rsuffix="_p")
    G = M[(M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)].copy()
    G["live"] = G["is_live"] == True                                    # noqa: E712
    G = G[~G["owner_id"].isin(SERVICE_OWNERS)]

    say("=" * 118)
    say("ПОПЫТКА ИЛИ ТА ЖЕ СИСТЕМА ПОД ДРУГИМ ПЛЕЧОМ: что будет, если клоны схлопнуть")
    say("=" * 118)
    say(f"измеримых систем: {len(G):,} у {G['owner_id'].nunique():,} авторов")
    say("Клон = пара систем одного автора с ρ ≥ порога на окне ≥ 250 общих дней.")
    say("Плечо корреляцию не меняет, поэтому ρ ловит и масштабированные копии.")
    say()

    vc = G["owner_id"].value_counts()
    multi = set(vc[vc >= 2].index)
    cand = G.index[G["owner_id"].isin(multi)].tolist()
    with Pool(NPROC) as pool:
        ser = {x[0]: (x[1], x[2]) for x in
               pool.imap_unordered(series, cand, chunksize=100) if x}
    say(f"рядов загружено: {len(ser):,}")
    GS = G[G.index.isin(ser)]
    PR = pairs_of(GS, ser)
    PRC = pairs_of(GS, ser, max_pairs=200)      # конвенция главы: первые 200 пар автора
    say(f"пар с общим окном ≥ {MIN_OVERLAP} дней: {len(PR):,} "
        f"у {PR['owner'].nunique():,} авторов")
    say(f"из них в конвенции главы (первые 200 пар автора): {len(PRC):,} — "
        f"контроль воспроизводимости")
    say()

    # ── 1. сколько клонов ─────────────────────────────────────────────────────
    say("=" * 118)
    say("1. СКОЛЬКО ЭТО КЛОНОВ (три порога)")
    say("=" * 118)
    hdr = (f'{"порог ρ":<10} {"пар-клонов":>12} {"доля пар":>10} {"систем схлопнуто":>18}'
           f' {"доля систем":>13} {"авторов задето":>16} {"медиана λ":>11}'
           f' {"доля λ вне 0.9–1.1":>20}')
    say(hdr)
    say("-" * len(hdr))
    for tau in TAUS:
        cl = PR[PR["rho"] >= tau]
        _, drop = collapse(G, PR, tau)
        lam = cl["lam"].abs()
        say(f'{tau:<10.2f} {len(cl):>12,} {100*len(cl)/len(PR):>9.1f}%'
            f' {len(drop):>18,} {100*len(drop)/len(G):>12.1f}%'
            f' {cl["owner"].nunique():>16,} {lam.median():>11.2f}'
            f' {100*((lam < 0.9) | (lam > 1.1)).mean():>19.1f}%')
    say()
    say("λ — наклон регрессии одной системы на другую через ноль: 1.0 значит копию")
    say("один в один, 2.0 — ту же систему с удвоенным размером позиции.")
    say()

    # ── 2. корреляции до и после ──────────────────────────────────────────────
    say("=" * 118)
    say(f"2. КОРРЕЛЯЦИИ ПАР ДО И ПОСЛЕ СХЛОПЫВАНИЯ (порог {TAU_MAIN})")
    say("=" * 118)
    G2, drop = collapse(G, PR, TAU_MAIN)
    PR2 = PR[PR["a"].isin(G2.index) & PR["b"].isin(G2.index)]
    hdr = (f'{"выборка":<22} {"пар":>8} {"p5":>7} {"p25":>7} {"медиана":>9} {"p75":>7}'
           f' {"p95":>7} {"среднее ρ":>11} {"доля ρ > 0.8":>14}')
    say(hdr)
    say("-" * len(hdr))
    for lab, P in (("Конвенция главы (200)", PRC), ("Все пары", PR),
                   ("Все пары без клонов", PR2)):
        q = P["rho"].quantile([.05, .25, .5, .75, .95]).to_numpy()
        say(f'{lab:<22} {len(P):>8,} {q[0]:>7.2f} {q[1]:>7.2f} {q[2]:>9.2f}'
            f' {q[3]:>7.2f} {q[4]:>7.2f} {P["rho"].mean():>11.3f}'
            f' {100*(P["rho"] > 0.8).mean():>13.1f}%')
    say()

    # ── 3–4. плата за перебор и независимые попытки ───────────────────────────
    say("=" * 118)
    say("3. ПЛАТА ЗА ПЕРЕБОР И НЕЗАВИСИМЫЕ ПОПЫТКИ")
    say("=" * 118)
    A1, A2 = author_table(G), author_table(G2)
    say(f'авторов с ≥ 2 измеримыми: было {len(A1):,}, стало {len(A2):,}')
    say()
    gain_table(A1, float(PRC["rho"].mean()),
               f'КОНТРОЛЬ — конвенция главы (ρ̄ = {PRC["rho"].mean():.3f} '
               f'по первым 200 парам автора):')
    if CHARTS:
        chart_gain_vs_null(A1, float(PR["rho"].mean()))
        chart_neff(A1, float(PR["rho"].mean()), len(PR))
    gain_table(A1, float(PR["rho"].mean()),
               f'ДО схлопывания, все пары (ρ̄ = {PR["rho"].mean():.3f}):')
    gain_table(A2, float(PR2["rho"].mean()),
               f'ПОСЛЕ схлопывания, ρ ≥ {TAU_MAIN} (ρ̄ = {PR2["rho"].mean():.3f}):')

    # ── 5. гипотеза о живучести лучшей ────────────────────────────────────────
    say("=" * 118)
    say("4. «ГИПОТЕЗА, КОТОРАЯ НЕ ПОДТВЕРДИЛАСЬ»: Sharpe лучшей и живучесть по k")
    say("=" * 118)
    hdr = (f'{"k":<10} {"авторов до":>11} {"S лучшей до":>13} {"лучшая жива до":>16}'
           f' {"авторов после":>14} {"S лучшей после":>16} {"лучшая жива после":>19}')
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in [(2, 2, "2")] + GRP:
        s1 = A1[(A1["k"] >= lo) & (A1["k"] <= hi)]
        s2 = A2[(A2["k"] >= lo) & (A2["k"] <= hi)]
        if len(s1) < 5:
            continue
        say(f'{lab:<10} {len(s1):>11,} {s1["s_max"].median():>13.2f}'
            f' {100*s1["best_live"].mean():>15.1f}% {len(s2):>14,}'
            f' {s2["s_max"].median():>16.2f} {100*s2["best_live"].mean():>18.1f}%')
    say()

    out = Path(__file__).resolve().parents[1] / "results" / "comon_leverage_clones.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    say(f"[записано] {out}")


if __name__ == "__main__":
    main()
