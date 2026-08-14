"""comon_subs_clusters.py — кластеризуются ли подписки СВЕРХ уже посчитанного.

Блок 9 §9.10 закрыл вопрос о кластеризации ПОДПИСЧИКОВ (нельзя: индивидуальных
записей нет) и посчитал концентрацию подписок (Джини 0.931, эфф. N по HHI 57.9).
Здесь — два разреза, которых там не было.

РАЗРЕЗ 1. Сколько НЕЗАВИСИМЫХ СТАВОК стоит за подписками.
  Концентрация по HHI считает, во скольких стратегиях лежат деньги, и молчит о том,
  насколько эти стратегии похожи. Если популярные системы ходят вместе, выбор из
  полутора тысяч вариантов оказывается одной ставкой. Считаем корреляции дневных
  доходностей, взвешенные по подпискам, и эффективное число независимых
  направлений через participation ratio собственных чисел:
      N_eff = (Σλ)² / Σλ²      — сколько «настоящих» степеней свободы в портфеле рынка.

РАЗРЕЗ 2. Есть ли у подписочной массы кластеры по ПОВЕДЕНИЮ (не по признакам карточки).
  Иерархическая кластеризация по 1−ρ дневных рядов + силуэт + бутстрап-устойчивость
  (как в разделе 4.5). Дальше смотрим, ЧЕМ кластеры оказались: класс активов, автор,
  риск, — и сколько подписок в каждом.

🔴 Правило то же, что в 4.5: кластеризуем по ПОВЕДЕНИЮ (форма дневного ряда),
описываем ИСХОДОМ и признаками. Если устойчивости нет — так и говорим.

Текстовый вывод, без графиков. Запуск: python comon_subs_clusters.py
"""
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md

WIN_FROM = "2023-08-01"      # общее окно: три года до среза, чтобы хватило пересечения
MIN_DAYS = 250               # минимум совместных торговых дней для корреляции
NPROC = 8


def say(s=""):
    print(s, flush=True)


def load_series(sid):
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8")).get("data") or {}
    except Exception:                                                   # noqa: BLE001
        return None
    pts = d.get("strategy") or []
    if not pts:
        return None
    s = pd.Series({p["date"]: p.get("rValue") for p in pts}, dtype="float64")
    s = s[s.index >= WIN_FROM].sort_index()
    if len(s) < MIN_DAYS + 1:
        return None
    r = s.diff().dropna() / 100.0        # rValue в процентах от вложенного
    return sid, r


def eff_n(w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return float(1.0 / (w ** 2).sum())


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False)
    L = panel[(panel["is_live"] == 1) & (panel["followers"].fillna(0) > 0)].copy()
    L["followers"] = L["followers"].fillna(0)
    say(f"живых стратегий с подписчиками: {len(L)}, подписок {int(L['followers'].sum()):,}")

    with Pool(NPROC) as pool:
        got = [x for x in pool.imap_unordered(load_series, list(L["id"]), chunksize=20) if x]
    R = pd.DataFrame({sid: r for sid, r in got})
    say(f"с рядом не короче {MIN_DAYS} дней в окне с {WIN_FROM}: {R.shape[1]} стратегий")

    R = R.dropna(axis=0, how="all")
    cnt = R.notna().sum()
    R = R[cnt[cnt >= MIN_DAYS].index]
    C = R.corr(min_periods=MIN_DAYS)
    keep = C.notna().sum() >= len(C) * 0.5      # у кого есть корреляция хотя бы с половиной
    C = C.loc[keep, keep]
    C = C.fillna(0.0)
    ids = list(C.index)
    w = L.set_index("id").loc[ids, "followers"].values
    say(f"в корреляционной матрице: {len(ids)} стратегий, подписок {int(w.sum()):,} "
        f"({100*w.sum()/L['followers'].sum():.1f} % всех)")

    # ------------------------------------------------------------------ разрез 1
    say()
    say("=" * 96)
    say("1. СКОЛЬКО НЕЗАВИСИМЫХ СТАВОК СТОИТ ЗА ПОДПИСКАМИ")
    say("=" * 96)
    say(f"{'мера':<52} {'значение':>12}")
    say("-" * 66)
    say(f"{'стратегий в матрице':<52} {len(ids):>12}")
    say(f"{'эфф. N по концентрации подписок (1/HHI)':<52} {eff_n(w):>12.1f}")
    med = np.median(C.values[np.triu_indices(len(C), 1)])
    say(f"{'медианная попарная корреляция':<52} {med:>12.3f}")
    ww = w / w.sum()
    # взвешенная по подпискам корреляционная матрица направлений
    D = np.diag(np.sqrt(ww))
    M = D @ C.values @ D
    lam = np.linalg.eigvalsh(M)
    lam = lam[lam > 0]
    say(f"{'эфф. число независимых ставок (participation ratio)':<52} "
        f"{lam.sum() ** 2 / (lam ** 2).sum():>12.1f}")
    top = np.sort(lam)[::-1]
    say(f"{'доля дисперсии в 1-й компоненте':<52} {100*top[0]/top.sum():>11.1f} %")
    say(f"{'в первых трёх':<52} {100*top[:3].sum()/top.sum():>11.1f} %")
    # равновзвешенный контроль
    Du = np.diag(np.sqrt(np.full(len(ids), 1 / len(ids))))
    lu = np.linalg.eigvalsh(Du @ C.values @ Du)
    lu = lu[lu > 0]
    say(f"{'то же равновзвешенно (для сравнения)':<52} "
        f"{lu.sum() ** 2 / (lu ** 2).sum():>12.1f}")

    # ------------------------------------------------------------------ разрез 2
    say()
    say("=" * 96)
    say("2. КЛАСТЕРЫ ПО ПОВЕДЕНИЮ (иерархическая, 1−ρ, Ward)")
    say("=" * 96)
    Dm = 1.0 - C.values
    np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2
    Z = linkage(squareform(Dm, checks=False), method="ward")

    def silh(lab):
        s = []
        for i in range(len(lab)):
            same = (lab == lab[i]) & (np.arange(len(lab)) != i)
            other = lab != lab[i]
            if not same.any() or not other.any():
                continue
            a = Dm[i, same].mean()
            b = min(Dm[i, lab == k].mean() for k in set(lab[other]))
            s.append((b - a) / max(a, b))
        return float(np.mean(s)) if s else np.nan

    say(f"{'k':>3} {'силуэт':>8} {'устойчивость':>13} {'размеры кластеров':<34} {'доли подписок'}")
    say("-" * 96)
    rng = np.random.default_rng(0)
    for k in (2, 3, 4, 5, 6):
        lab = fcluster(Z, k, criterion="maxclust")
        # бутстрап-устойчивость: доля пар, остающихся вместе при пересборке
        agree = []
        for _ in range(20):
            idx = rng.choice(len(ids), size=int(len(ids) * 0.8), replace=False)
            sub = Dm[np.ix_(idx, idx)]
            zb = linkage(squareform(sub, checks=False), method="ward")
            lb = fcluster(zb, k, criterion="maxclust")
            a = lab[idx]
            same_o = a[:, None] == a[None, :]
            same_b = lb[:, None] == lb[None, :]
            iu = np.triu_indices(len(idx), 1)
            agree.append((same_o[iu] == same_b[iu]).mean())
        sizes = [int((lab == c).sum()) for c in range(1, k + 1)]
        subs = [100 * w[lab == c].sum() / w.sum() for c in range(1, k + 1)]
        say(f"{k:>3} {silh(lab):>8.3f} {np.mean(agree):>13.3f} "
            f"{str(sizes):<34} {' '.join(f'{x:.0f}%' for x in subs)}")

    # ------------------------------------------------------- чем кластеры оказались
    say()
    say("=" * 96)
    say("3. ЧЕМ ОКАЗАЛИСЬ КЛАСТЕРЫ (описание исходом и признаками, k=3)")
    say("=" * 96)
    lab = fcluster(Z, 3, criterion="maxclust")
    P = L.set_index("id").loc[ids].copy()
    P["cl"] = lab
    P["w"] = w
    say(f"{'кластер':>8} {'систем':>7} {'подписок':>10} {'доля':>7} {'медиана ρ внутри':>17} "
        f"{'Sharpe':>8} {'CAGR %':>8} {'maxDD %':>9} {'вола %':>8}")
    for c in sorted(set(lab)):
        m = lab == c
        sub = C.values[np.ix_(m, m)]
        iu = np.triu_indices(m.sum(), 1)
        g = P[P["cl"] == c]
        say(f"{c:>8} {int(m.sum()):>7} {int(g['w'].sum()):>10,} "
            f"{100*g['w'].sum()/w.sum():>6.1f}% {np.median(sub[iu]) if m.sum() > 1 else np.nan:>17.3f} "
            f"{g['sharpe'].median():>8.2f} {100*g['cagr'].median():>8.1f} "
            f"{100*g['maxdd'].median():>9.1f} {100*g['vol'].median():>8.1f}")
    # чем кластеры оказались по существу: бета к рынку и состав
    B = pd.read_csv(DIR / "beta.csv.gz")[["id", "beta", "alpha", "r2", "corr"]]
    P = P.reset_index().merge(B, on="id", how="left").set_index("id")
    say()
    say("связь с индексом МосБиржи и состав кластеров:")
    say(f"{'кластер':>8} {'медиана беты':>13} {'медиана R²':>11} {'бета > 0.5':>11} "
        f"{'ежедневных':>11} {'топ-автор кластера (подписок)':<34}")
    for c in sorted(set(lab)):
        g = P[P["cl"] == c]
        ta = g.groupby("author")["w"].sum().sort_values(ascending=False)
        say(f"{c:>8} {g['beta'].median():>13.2f} {g['r2'].median():>11.2f} "
            f"{100*(g['beta'] > 0.5).mean():>10.0f}% "
            f"{100*(g['transaction_rate'] == 'daily').mean():>10.0f}% "
            f"{ta.index[0] + ' (' + str(int(ta.iloc[0])) + ')':<34}")

    say()
    say("межкластерные корреляции (медианы):")
    for a in sorted(set(lab)):
        row = []
        for b in sorted(set(lab)):
            sub = C.values[np.ix_(lab == a, lab == b)]
            row.append(f"{np.median(sub):>7.3f}")
        say(f"  кластер {a}: " + " ".join(row))


if __name__ == "__main__":
    main()
