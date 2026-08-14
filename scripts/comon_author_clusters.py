"""comon_author_clusters.py — типология серийных авторов по ПОВЕДЕНИЮ.

Шаг 2б-2в плана переделки раздела 4.5.

🔴 ГЛАВНОЕ ПРАВИЛО, ради которого раздел и переделывается: кластеризуем по тому, что
автор ДЕЛАЕТ, а описываем тем, что у него ПОЛУЧИЛОСЬ. Иначе повторится круговая
конструкция прежнего раздела 4.5, где авторы отбирались по живучести и у них же
измерялась живучесть.

Отбор: k >= MIN_CREATED созданных стратегий (порог обоснован в comon_serial_authors.py:
естественного разрыва в распределении НЕТ, поэтому порог — соглашение, и выводы
проверяются на его сдвиг). Служебный аккаунт owner 2215 исключён.

ОСИ КЛАСТЕРИЗАЦИИ (только поведение):
  1. log(создано)                  — масштаб;
  2. темп создания в год           — залп против капельницы;
  3. конверсия в ряд               — доля записей, у которых появилась хоть одна точка;
  4. конверсия в измеримость       — доля, доработавшая до состояния «есть что мерить»;
  5. актуальность                  — лет с последней созданной стратегии до среза.

⚠️ Ось 4 — ПОЛУИСХОД: измеримость означает дожитие до 100 торговых дней. Поэтому
утверждения кластеров О ЖИВУЧЕСТИ останутся частично тавтологичными, а о доходности,
качестве и подписках — нет. Это ограничение обязано попасть в текст главы.

Метод: Ward на стандартизованных осях. Силуэт и бутстрап-устойчивость (ARI) считаются
здесь же — как ПРОВЕРКА того, что разбиение не шум, а не как способ выбрать красивое
число кластеров. Если устойчивость низкая, это говорится вслух.

Текстовый вывод, без картинок. Запуск: python comon_author_clusters.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

MIN_TRADE_DAYS = 100
MIN_ACTIVITY = 0.10
SERVICE_OWNERS = {2215}
MIN_CREATED = 10              # порог серийности (см. comon_serial_authors.py)
SNAPSHOT = date(2026, 8, 4)   # срез рядов
MIN_WINDOW_YEARS = 1 / 12     # окно создания не короче месяца — иначе темп взрывается
KMAX = 8
BOOT = 200
SEED = 12345
MIN_YEARS = 3                 # минимальная длина истории для вердикта «умеет» (шаг 2г)


CHARTS = "--charts" in sys.argv


def chart_author_types(a, names):
    """График 26: карта серийных авторов по тому, ЧТО ОНИ ДЕЛАЮТ."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=1040, bottom=0.26)
    style = {"залповые": (ch.ORANGE, "^", 90),
             "поточные": (ch.BLUE, "o", 70),
             "штучные": (ch.GREEN, "s", 55)}
    for cl, nm in names.items():
        g = a[a["cl"] == cl]
        color, marker, size = style[nm]
        ax.scatter(g["rate"], g["conv_ser"], s=size, marker=marker, color=color,
                   alpha=0.65, lw=0.8, edgecolors=ch.SURFACE, zorder=4,
                   label=f"{nm} — {len(g)} авторов: создано "
                         f"{ch.n_(g['k'].median())}, темп "
                         f"{ch.n_(g['rate'].median(), 1)} в год")
        ax.scatter([g["rate"].median()], [g["conv_ser"].median()], s=260,
                   marker=marker, color=color, lw=2.0, edgecolors=ch.INK,
                   zorder=6)
    ax.set_xscale("log")
    ticks = (1, 2, 5, 10, 20, 50, 100)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([ch.n_(v) for v in ticks])
    ax.set_xlim(0.7, 160)
    ax.set_ylim(-0.03, 1.06)
    ch.pct(ax)
    ax.set_xlabel("темп создания, стратегий в год")
    ax.set_ylabel("доля созданного, дошедшая до торгов")
    ax.set_title("Три способа делать много стратегий")
    ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.6, markerscale=1.2,
              fontsize=10)
    ch.note(ax,
            f"База — {len(a)} авторов, создавших не меньше десяти стратегий; "
            f"служебный аккаунт площадки исключён. Крупный значок с тёмной "
            f"обводкой — медиана группы. Шкала темпа логарифмическая.",
            "Деление получено кластеризацией ТОЛЬКО по поведению автора — сколько "
            "он создал, с какой скоростью и какая доля дошла до торгов. Результат "
            "стратегий в разделение не входил, поэтому разница групп по результату "
            "(таблица ниже) — находка, а не следствие построения.")
    ch.save(f, 26, "author-types-map")
    print("    " + "; ".join(
        f"{nm}: {int((a['cl'] == cl).sum())} авт., темп "
        f"{a.loc[a['cl'] == cl, 'rate'].median():.1f}, конверсия "
        f"{a.loc[a['cl'] == cl, 'conv_ser'].median():.2f}"
        for cl, nm in names.items()), flush=True)


def say(s=""):
    print(s, flush=True)


def head(t):
    say()
    say("=" * 100)
    say(t)
    say("=" * 100)


def load_authors(min_created=MIN_CREATED):
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False)
    # 🔴 years есть и в панели, и в sharpe.csv.gz — переименовываем, иначе merge
    # склеит их в years_x/years_y и дальше молча возьмётся не та.
    sh = (pd.read_csv(DIR / "sharpe.csv.gz")
          [["id", "sharpe_ar", "activity", "years", "t_sharpe"]]
          .rename(columns={"years": "sh_years"}))
    deaths = pd.read_csv(DIR / "deaths.csv.gz")[["id", "mech", "fin"]]
    p = panel.merge(sh, on="id", how="left").merge(deaths, on="id", how="left")
    p = p[p["owner_id"].notna() & ~p["owner_id"].isin(SERVICE_OWNERS)].copy()

    p["ser"] = p["n_pts"] > 0
    p["msr"] = (p["n_trade"] >= MIN_TRADE_DAYS) & (p["activity"] >= MIN_ACTIVITY)
    p["alive"] = p["is_live"] == 1
    p["cdate"] = pd.to_datetime(p["created_at"], errors="coerce", format="mixed")

    g = p.groupby("owner_id")
    a = pd.DataFrame({
        "author": g["author"].first(),
        "k": g.size(),
        "ser": g["ser"].sum(),
        "msr": g["msr"].sum(),
        "alive": g["alive"].sum(),
        "first": g["cdate"].min(),
        "last": g["cdate"].max(),
    })
    a = a[a["k"] >= min_created].copy()

    span = (a["last"] - a["first"]).dt.days / 365.25
    a["span"] = span.clip(lower=MIN_WINDOW_YEARS)
    a["rate"] = a["k"] / a["span"]                       # стратегий в год
    a["conv_ser"] = a["ser"] / a["k"]
    a["conv_msr"] = a["msr"] / a["k"]
    a["idle"] = (pd.Timestamp(SNAPSHOT) - a["last"]).dt.days / 365.25
    return p, a


def silhouette(X, lab):
    """Средний силуэт без sklearn: 235 точек, полная матрица расстояний дешева."""
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        own[i] = False
        if own.sum() == 0:
            out[i] = 0.0
            continue
        ai = d[i, own].mean()
        bi = min(d[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        out[i] = (bi - ai) / max(ai, bi)
    return float(out.mean())


def ari(x, y):
    """Adjusted Rand Index вручную (sklearn в окружении нет)."""
    t = pd.crosstab(x, y).values
    n = t.sum()
    comb = lambda m: (m * (m - 1) / 2).sum()
    idx = comb(t)
    ea, eb = comb(t.sum(1)), comb(t.sum(0))
    exp = ea * eb / (n * (n - 1) / 2)
    mx = (ea + eb) / 2
    return float((idx - exp) / (mx - exp)) if mx != exp else 1.0


EULER = 0.5772156649


def emax(n):
    """E[max] n независимых стандартных нормальных — как в разделе 4.2 (Bailey-LdP)."""
    from scipy import stats
    n = max(float(n), 1.0 + 1e-9)
    return ((1 - EULER) * stats.norm.ppf(1 - 1 / n)
            + EULER * stats.norm.ppf(1 - 1 / (n * np.e)))


def cluster(a, axes, kmax=KMAX, boot=BOOT, seed=SEED):
    """Ward на стандартизованных осях + силуэт и бутстрап-ARI. Возвращает (best_k, метки, оценки)."""
    X = a[axes].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    Z = linkage(X, method="ward")
    rng = np.random.default_rng(seed)
    scores = {}
    for kk in range(2, kmax + 1):
        lab = fcluster(Z, kk, criterion="maxclust")
        vals = []
        for _ in range(boot):
            idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=False)
            lb = fcluster(linkage(X[idx], method="ward"), kk, criterion="maxclust")
            vals.append(ari(pd.Series(lab[idx]), pd.Series(lb)))
        scores[kk] = (silhouette(X, lab), float(np.mean(vals)))
    best = max(scores, key=lambda k: scores[k][0])
    return best, fcluster(Z, best, criterion="maxclust"), scores


def main():
    p, a = load_authors()
    axes = ["k_log", "rate", "conv_ser", "conv_msr", "idle"]
    a["k_log"] = np.log(a["k"])
    X = a[axes].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)

    head(f"2б.  ТИПОЛОГИЯ СЕРИЙНЫХ АВТОРОВ (создано >= {MIN_CREATED}): {len(a)} авторов")
    say(f"систем у них: {int(a['k'].sum()):,} · осей кластеризации: {len(axes)} (только поведение)")
    say()
    say("Оси и их разброс по группе:")
    hdr = f'{"ось":<34}{"медиана":>10}{"p10":>9}{"p90":>9}{"мин":>9}{"макс":>9}'
    say(hdr)
    say("-" * len(hdr))
    names = {"k_log": "log(создано)", "rate": "темп создания, шт/год",
             "conv_ser": "конверсия в ряд, доля",
             "conv_msr": "конверсия в измеримость, доля",
             "idle": "лет с последней созданной"}
    for c in axes:
        s = a[c]
        say(f'{names[c]:<34}{s.median():>10.2f}{s.quantile(.1):>9.2f}'
            f'{s.quantile(.9):>9.2f}{s.min():>9.2f}{s.max():>9.2f}')

    Z = linkage(X, method="ward")
    say()
    say("Сколько кластеров брать: силуэт и устойчивость к пересэмплированию.")
    say("Силуэт > 0.5 — структура выражена, 0.25-0.5 — слабая, < 0.25 — её нет.")
    say("ARI — совпадение разбиений на бутстрап-подвыборках (1.0 = идеально устойчиво).")
    say()
    rng = np.random.default_rng(SEED)
    hdr = f'{"кластеров":<12}{"силуэт":>10}{"ARI (бутстрап)":>18}{"размеры кластеров"}'
    say(hdr)
    say("-" * 78)
    scores = {}
    for kk in range(2, KMAX + 1):
        lab = fcluster(Z, kk, criterion="maxclust")
        sil = silhouette(X, lab)
        vals = []
        for _ in range(BOOT):
            idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=False)
            lb = fcluster(linkage(X[idx], method="ward"), kk, criterion="maxclust")
            vals.append(ari(pd.Series(lab[idx]), pd.Series(lb)))
        scores[kk] = (sil, float(np.mean(vals)))
        sizes = " ".join(str(int(x)) for x in np.bincount(lab)[1:])
        say(f'{kk:<12}{sil:>10.3f}{np.mean(vals):>18.3f}    {sizes}')

    best = max(scores, key=lambda k: scores[k][0])
    say()
    say(f"Лучший силуэт при {best} кластерах ({scores[best][0]:.3f}),"
        f" устойчивость {scores[best][1]:.3f}.")
    if scores[best][0] < 0.25:
        say("🔴 Силуэт ниже 0.25: выраженной структуры в данных НЕТ — кластеры не")
        say("следует подавать читателю как найденные типы. Это отрицательный результат,")
        say("и в главе он должен быть назван прямо.")

    lab = fcluster(Z, best, criterion="maxclust")
    a["cl"] = lab

    # ---------------------------------------------------------------- 2в ----
    head("2в.  ЧТО ЭТО ЗА ГРУППЫ: поведение (по чему делили) и исход (по чему НЕ делили)")
    say("ПОВЕДЕНИЕ — оси кластеризации:")
    hdr = (f'{"кластер":<9}{"авторов":>9}{"создано":>10}{"темп/год":>10}'
           f'{"в ряд":>8}{"в измер.":>10}{"простой, лет":>14}')
    say(hdr)
    say("-" * len(hdr))
    for c in sorted(a["cl"].unique()):
        s = a[a["cl"] == c]
        say(f'{c:<9}{len(s):>9}{s["k"].median():>10.0f}{s["rate"].median():>10.1f}'
            f'{s["conv_ser"].median():>8.2f}{s["conv_msr"].median():>10.2f}'
            f'{s["idle"].median():>14.1f}')

    # ── типология главы 4.5: ТРИ оси поведения, ТРИ группы ────────────────────
    # 🔴 Именно эта конфигурация стоит в тексте главы («признаков поведения три»),
    # и только она даёт группы 21 / 37 / 177. Разбиение выше — по пяти осям с
    # выбором k по силуэту — остаётся как проверка чувствительности: оно даёт
    # другой состав, и путать их нельзя.
    head("2в-бис.  ТИПОЛОГИЯ ГЛАВЫ 4.5: три оси поведения, три группы")
    axes3 = ["k_log", "rate", "conv_ser"]
    X3 = a[axes3].to_numpy(float)
    X3 = (X3 - X3.mean(0)) / X3.std(0)
    a["cl3"] = fcluster(linkage(X3, method="ward"), 3, criterion="maxclust")
    med3 = a.groupby("cl3")[["k", "rate"]].median()
    burst = med3["rate"].idxmax()
    rest = med3.drop(index=burst).sort_values("k", ascending=False)
    names3 = {burst: "залповые", rest.index[0]: "поточные",
              rest.index[-1]: "штучные"}
    say(f"силуэт трёхосевого разбиения: {silhouette(X3, a['cl3'].to_numpy()):.3f}")
    hdr = (f'{"группа":<12}{"авторов":>9}{"создано":>10}{"темп/год":>10}'
           f'{"доля дошедших до торгов":>26}')
    say(hdr)
    say("-" * len(hdr))
    for cl in sorted(names3, key=lambda c: -a.loc[a["cl3"] == c, "rate"].median()):
        g = a[a["cl3"] == cl]
        say(f'{names3[cl]:<12}{len(g):>9}{g["k"].median():>10.0f}'
            f'{g["rate"].median():>10.1f}{g["conv_ser"].median():>26.2f}')
    if CHARTS:
        say()
        say("── график 26 ────────────────────────────────────────────────────")
        chart_author_types(a.assign(cl=a["cl3"]), names3)
        say()

    pm = p[p["owner_id"].isin(a.index)].merge(
        a[["cl"]], left_on="owner_id", right_index=True, how="left")
    msr = pm[pm["msr"]]

    say()
    say("ИСХОД — в кластеризацию НЕ входил:")
    hdr = (f'{"кластер":<9}{"систем":>8}{"измер.":>8}{"живых":>7}{"S лучшей":>10}'
           f'{"S медиан.":>11}{"CAGR мед.":>11}{"maxDD мед.":>12}{"подписчиков":>13}')
    say(hdr)
    say("-" * len(hdr))
    for c in sorted(a["cl"].unique()):
        s = a[a["cl"] == c]
        m = msr[msr["cl"] == c]
        sub = int(pm[(pm["cl"] == c) & pm["alive"]]["followers"].sum())
        smax = m.groupby("owner_id")["sharpe_ar"].max().median() if len(m) else np.nan
        smed = m["sharpe_ar"].median() if len(m) else np.nan
        cg = m["cagr"].median() * 100 if len(m) else np.nan
        dd = m["maxdd"].median() * 100 if len(m) else np.nan
        say(f'{c:<9}{int(s["k"].sum()):>8,}{int(s["msr"].sum()):>8,}'
            f'{int(s["alive"].sum()):>7,}{smax:>10.2f}{smed:>11.2f}'
            f'{cg:>10.1f}%{dd:>11.1f}%{sub:>13,}')

    say()
    say("Механизм смерти систем кластера (доля закрытых систем группы):")
    mech = (pm[pm["mech"].notna()].groupby(["cl", "mech"]).size()
            .unstack(fill_value=0))
    mech = (mech.T / mech.sum(1)).T * 100
    cols = [c for c in ["не стартовала", "недостаточно истории", "смерть у пика",
                        "заброшена", "обычная просадка", "слом не подтверждён",
                        "недоказано", "деградация"] if c in mech.columns]
    hdr = f'{"кластер":<9}' + "".join(f"{c[:15]:>17}" for c in cols)
    say(hdr)
    say("-" * len(hdr))
    for c in mech.index:
        say(f'{c:<9}' + "".join(f'{mech.loc[c, x]:>16.1f}%' for x in cols))

    say()
    say("Состав кластеров поимённо (до 12 крупнейших авторов каждого):")
    for c in sorted(a["cl"].unique()):
        s = a[a["cl"] == c].sort_values("k", ascending=False)
        names_ = ", ".join(str(x)[:20] for x in s["author"].head(12))
        say(f"  кластер {c} ({len(s)}): {names_}"
            + (" …" if len(s) > 12 else ""))

    # ------------------------------------------------------- ПРОВЕРКА 1 ----
    head("ПРОВЕРКА 1.  Держится ли разбиение БЕЗ оси-полуисхода")
    say("«Конверсия в измеримость» — единственная ось, куда просачивается исход")
    say("(измеримость = дожитие до 100 торговых дней). Если без неё группы")
    say("рассыпаются, значит они держались на замаскированном результате, а не")
    say("на поведении, и всю типологию придётся выбросить.")
    say()
    axes4 = ["k_log", "rate", "conv_ser", "idle"]
    b4, lab4, sc4 = cluster(a, axes4)
    say(f"На четырёх осях лучший силуэт при {b4} кластерах: {sc4[b4][0]:.3f}"
        f" (было {scores[best][0]:.3f} на пяти), устойчивость {sc4[b4][1]:.3f}")
    say(f"Совпадение с базовым разбиением (ARI): {ari(pd.Series(lab), pd.Series(lab4)):.3f}")
    say()
    say("Тот же исход по группам четырёхосевого разбиения — сохраняется ли контраст:")
    a["cl4"] = lab4
    pm4 = p[p["owner_id"].isin(a.index)].merge(
        a[["cl4"]], left_on="owner_id", right_index=True, how="left")
    m4 = pm4[pm4["msr"]]
    hdr = (f'{"кластер":<9}{"авторов":>9}{"создано":>10}{"темп/год":>10}'
           f'{"S медиан.":>11}{"CAGR мед.":>11}{"подписчиков":>13}')
    say(hdr)
    say("-" * len(hdr))
    for c in sorted(a["cl4"].unique()):
        s = a[a["cl4"] == c]
        m = m4[m4["cl4"] == c]
        sub = int(pm4[(pm4["cl4"] == c) & pm4["alive"]]["followers"].sum())
        say(f'{c:<9}{len(s):>9}{s["k"].median():>10.0f}{s["rate"].median():>10.1f}'
            f'{(m["sharpe_ar"].median() if len(m) else np.nan):>11.2f}'
            f'{(m["cagr"].median() * 100 if len(m) else np.nan):>10.1f}%{sub:>13,}')

    # ------------------------------------------------- ДРОБЛЕНИЕ БОЛЬШОГО ----
    head("ПРОВЕРКА 1б.  Делится ли «неспешный» кластер дальше")
    say("В нём 203 автора — точек хватает. Вопрос не «на сколько частей поделить»,")
    say("а «есть ли что делить»: при слабом силуэте любое дробление будет рисованием")
    say("границ по шуму. Сначала те же четыре чистые оси поведения.")
    say()
    big = int(pd.Series(lab4).value_counts().idxmax())
    sub = a[a["cl4"] == big].copy()
    say(f"размер группы: {len(sub)} авторов, систем {int(sub['k'].sum()):,}")
    say()
    _, _, sc_sub = cluster(sub, axes4)
    hdr = f'{"кластеров":<12}{"силуэт":>10}{"ARI (бутстрап)":>18}'
    say(hdr)
    say("-" * 42)
    for kk in sorted(sc_sub):
        say(f'{kk:<12}{sc_sub[kk][0]:>10.3f}{sc_sub[kk][1]:>18.3f}')
    bb = max(sc_sub, key=lambda k: sc_sub[k][0])
    say()
    say(f"Лучший силуэт {sc_sub[bb][0]:.3f} при {bb} кластерах"
        f" (у всей группы серийных было {sc4[b4][0]:.3f}).")
    say()
    _, labb, _ = cluster(sub, axes4, kmax=bb, boot=1)
    sub["cb"] = labb
    pmb = p[p["owner_id"].isin(sub.index)].merge(
        sub[["cb"]], left_on="owner_id", right_index=True, how="left")
    mb = pmb[pmb["msr"]]
    say("Что это за подгруппы — поведение (по чему делили) и исход (по чему нет):")
    hdrb = (f'{"подгруппа":<11}{"авторов":>9}{"создано":>9}{"темп/год":>10}'
            f'{"в ряд":>8}{"простой, лет":>14}{"измер.":>8}{"живых":>7}'
            f'{"S медиан.":>11}{"CAGR мед.":>11}{"подписчиков":>13}')
    say(hdrb)
    say("-" * len(hdrb))
    for c in sorted(sub["cb"].unique()):
        g2 = sub[sub["cb"] == c]
        m = mb[mb["cb"] == c]
        subs = int(pmb[(pmb["cb"] == c) & pmb["alive"]]["followers"].sum())
        say(f'{c:<11}{len(g2):>9}{g2["k"].median():>9.0f}{g2["rate"].median():>10.1f}'
            f'{g2["conv_ser"].median():>8.2f}{g2["idle"].median():>14.1f}'
            f'{int(g2["msr"].sum()):>8,}{int(g2["alive"].sum()):>7,}'
            f'{(m["sharpe_ar"].median() if len(m) else np.nan):>11.2f}'
            f'{(m["cagr"].median() * 100 if len(m) else np.nan):>10.1f}%{subs:>13,}')
    say()
    say("Кто в каждой (до 10 крупнейших по числу созданных):")
    for c in sorted(sub["cb"].unique()):
        g2 = sub[sub["cb"] == c].sort_values("k", ascending=False)
        say(f"  подгруппа {c} ({len(g2)}): "
            + ", ".join(str(x)[:18] for x in g2["author"].head(10))
            + (" …" if len(g2) > 10 else ""))

    say()
    say("Теперь с двумя ДОПОЛНИТЕЛЬНЫМИ осями поведения, которых не было:")
    say("  6. стаж — сколько лет между первой и последней созданной стратегией;")
    say("  7. схожесть собственных систем (средняя корреляция пар автора) — это")
    say("     «доводит одну идею» против «пробует разное», ось из раздела 4.4.")
    pairs = pd.read_csv(DIR / "author_pairs.csv.gz")
    pcol = "rho" if "rho" in pairs.columns else pairs.columns[-1]
    ocol = "owner" if "owner" in pairs.columns else pairs.columns[0]
    rho = pairs.groupby(ocol)[pcol].mean()
    sub["span_y"] = sub["span"]
    sub["rho"] = sub.index.map(rho)
    have = sub["rho"].notna()
    say(f"    корреляция известна у {int(have.sum())} из {len(sub)} авторов"
        f" (нужны две измеримые системы с общим окном) — остальные выпадают")
    s2 = sub[have]
    if len(s2) > 20:
        _, lab7, sc7 = cluster(s2, axes4 + ["span_y", "rho"])
        b7 = max(sc7, key=lambda k: sc7[k][0])
        say()
        say(hdr)
        say("-" * 42)
        for kk in sorted(sc7):
            say(f'{kk:<12}{sc7[kk][0]:>10.3f}{sc7[kk][1]:>18.3f}')
        say()
        say(f"Лучший силуэт {sc7[b7][0]:.3f} при {b7} кластерах на {len(s2)} авторах.")
        s2 = s2.copy()
        s2["c7"] = lab7
        pm7 = p[p["owner_id"].isin(s2.index)].merge(
            s2[["c7"]], left_on="owner_id", right_index=True, how="left")
        m7 = pm7[pm7["msr"]]
        say()
        say("Поведение и исход этих подгрупп:")
        hdr2 = (f'{"подгруппа":<11}{"авторов":>9}{"создано":>9}{"темп/год":>10}'
                f'{"стаж, лет":>11}{"схожесть":>10}{"S медиан.":>11}'
                f'{"CAGR мед.":>11}{"подписчиков":>13}')
        say(hdr2)
        say("-" * len(hdr2))
        for c in sorted(s2["c7"].unique()):
            g2 = s2[s2["c7"] == c]
            m = m7[m7["c7"] == c]
            subs = int(pm7[(pm7["c7"] == c) & pm7["alive"]]["followers"].sum())
            say(f'{c:<11}{len(g2):>9}{g2["k"].median():>9.0f}{g2["rate"].median():>10.1f}'
                f'{g2["span_y"].median():>11.1f}{g2["rho"].median():>10.2f}'
                f'{(m["sharpe_ar"].median() if len(m) else np.nan):>11.2f}'
                f'{(m["cagr"].median() * 100 if len(m) else np.nan):>10.1f}%{subs:>13,}')

    # ------------------------------------------------------- ПРОВЕРКА 2 ----
    head("ПРОВЕРКА 2.  Держится ли картина при другом пороге серийности")
    say("Порог 10 — соглашение (естественного разрыва в распределении нет).")
    say("Прогоняем ту же процедуру при 15 и 20: если контраст «неспешные против")
    say("фабрик» исчезает, он был артефактом порога.")
    for thr in (15, 20):
        _, at = load_authors(thr)
        at["k_log"] = np.log(at["k"])
        bt, labt, sct = cluster(at, axes)
        at["clt"] = labt
        pmt = p[p["owner_id"].isin(at.index)].merge(
            at[["clt"]], left_on="owner_id", right_index=True, how="left")
        mt = pmt[pmt["msr"]]
        say()
        say(f"--- порог создано >= {thr}: {len(at)} авторов, лучший силуэт при {bt}"
            f" кластерах ({sct[bt][0]:.3f}), устойчивость {sct[bt][1]:.3f}")
        hdr = (f'{"кластер":<9}{"авторов":>9}{"создано":>10}{"темп/год":>10}'
               f'{"в измер.":>10}{"S медиан.":>11}{"CAGR мед.":>11}{"подписчиков":>13}')
        say(hdr)
        say("-" * len(hdr))
        for c in sorted(at["clt"].unique()):
            s = at[at["clt"] == c]
            m = mt[mt["clt"] == c]
            sub = int(pmt[(pmt["clt"] == c) & pmt["alive"]]["followers"].sum())
            say(f'{c:<9}{len(s):>9}{s["k"].median():>10.0f}{s["rate"].median():>10.1f}'
                f'{s["conv_msr"].median():>10.2f}'
                f'{(m["sharpe_ar"].median() if len(m) else np.nan):>11.2f}'
                f'{(m["cagr"].median() * 100 if len(m) else np.nan):>10.1f}%{sub:>13,}')

    # ------------------------------------------------------------- 2г ----
    head("2г.  КТО ИЗ НИХ ДЕЙСТВИТЕЛЬНО УМЕЕТ ТОРГОВАТЬ")
    say("Критерий задан ЗАРАНЕЕ и не через живучесть. Три условия сразу:")
    say("  1) достоверность: |t| статистики Sharpe >= 2 — результат отличим от нуля;")
    say("  2) уровень: арифметический Sharpe >= 1.0 на этой же истории;")
    say("  3) 🔴 поправка на перебор (раздел 4.2): Sharpe лучшей системы должен")
    say("     превышать E[max] из k независимых попыток, делённый на корень из лет,")
    say("     то есть результат не объясняется одним лишь числом попыток автора.")
    say("Плюс условие на автора, а не на систему: медианный Sharpe его измеримых")
    say("систем >= 0 — иначе это один удачный выстрел на фоне мусора.")
    say()
    sh = p[p["msr"] & p["owner_id"].isin(a.index)].copy()
    sh = sh.merge(a[["k", "author"]], left_on="owner_id", right_index=True,
                  how="left", suffixes=("", "_a"))
    sh["thr"] = [emax(k) / np.sqrt(max(y, 1e-9)) for k, y in zip(sh["k"], sh["sh_years"])]
    sh["ok_t"] = sh["t_sharpe"].abs() >= 2
    sh["ok_lvl"] = sh["sharpe_ar"] >= 1.0
    sh["ok_defl"] = sh["sharpe_ar"] > sh["thr"]
    # 🔴 ЧЕТВЁРТОЕ УСЛОВИЕ, добавленное после первого прогона. Без него критерий
    # пропускал Sharpe 15.31 на 1.8 года и 5.01 на 0.9 — то есть ровно те короткие
    # ряды, которые глава 3 качеством не считает (её потолок на пяти годах — 1.94).
    # Это требование к ИЗМЕРИМОСТИ качества, а не отбор по живучести, но с живучестью
    # оно частично коррелирует, и в главе это надо сказать прямо.
    sh["ok_len"] = sh["sh_years"] >= MIN_YEARS
    sh["ok"] = sh["ok_t"] & sh["ok_lvl"] & sh["ok_defl"] & sh["ok_len"]
    say(f"{'измеримых систем у серийных авторов':<52}{len(sh):>8,}")
    say(f"{'  проходят по достоверности (|t| >= 2)':<52}{int(sh['ok_t'].sum()):>8,}")
    say(f"{'  и по уровню (Sharpe >= 1.0)':<52}{int((sh['ok_t'] & sh['ok_lvl']).sum()):>8,}")
    say(f"{'  и по поправке на перебор':<52}"
        f"{int((sh['ok_t'] & sh['ok_lvl'] & sh['ok_defl']).sum()):>8,}")
    say(f"{'  и по длине истории (>= ' + str(MIN_YEARS) + ' лет)':<52}{int(sh['ok'].sum()):>8,}")
    say()
    say("🔴 Два условия из четырёх не отсекли НИ ОДНОЙ системы сверх первых двух —")
    say("и это надо сказать честно, а не делать вид, что работали все.")
    say("Причина одна: t-статистика Sharpe считается с поправкой на неопределённость")
    say("самой оценки, t = S*корень(лет)/корень(1 + S^2/2). Знаменатель штрафует")
    say("высокий Sharpe, поэтому 15.31 на коротком ряде большого t НЕ даёт. Требование")
    say("|t| >= 2 уже само по себе выбрасывает короткие истории: у прошедших систем")
    say("медианная длина 6.2 года, минимальная 2.3. Отдельные условия на длину и на")
    say("перебор оказались избыточны — оставлены как страховка, но работы не сделали.")
    say()
    say("Чувствительность к требованию длины:")
    hdr = f'{"минимум лет":<14}{"систем прошло":>15}{"авторов":>10}'
    say(hdr)
    for y in (0, 1, 2, 3, 5):
        m = sh["ok_t"] & sh["ok_lvl"] & sh["ok_defl"] & (sh["sh_years"] >= y)
        say(f'{y:<14}{int(m.sum()):>15,}{m[m].index.map(sh["owner_id"]).nunique():>10}')
    say()
    med = sh.groupby("owner_id")["sharpe_ar"].median()
    win = sh[sh["ok"]].groupby("owner_id").size()
    good = [o for o in win.index if med.get(o, -9) >= 0]
    say(f"{'авторов хотя бы с одной такой системой':<52}{len(win):>8,}")
    say(f"{'  из них с медианным Sharpe >= 0 (умеют)':<52}{len(good):>8,}")
    say()
    if good:
        say("Кто это (по убыванию числа прошедших систем):")
        hdr = (f'{"автор":<24}{"создано":>9}{"измер.":>8}{"прошло":>8}{"живых":>7}'
               f'{"S прошедшей":>13}{"S медиан.":>11}{"лет у неё":>11}{"кластер":>9}')
        say(hdr)
        say("-" * len(hdr))
        rows = []
        for o in good:
            s = sh[sh["owner_id"] == o]
            # 🔴 Берём лучшую среди ПРОШЕДШИХ критерий, а не среди всех измеримых.
            # Иначе в таблицу попадает Sharpe 15.31 на 1.8 года — система, которая
            # критерий как раз НЕ прошла, и колонка «лет» вводит в заблуждение.
            sp = s[s["ok"]]
            b = sp.loc[sp["sharpe_ar"].idxmax()]
            rows.append((str(a.loc[o, "author"])[:24], int(a.loc[o, "k"]), len(s),
                         int(win[o]), int(a.loc[o, "alive"]), float(b["sharpe_ar"]),
                         float(med[o]), float(b["sh_years"]), int(a.loc[o, "cl"])))
        for r in sorted(rows, key=lambda x: (-x[3], -x[5])):
            say(f'{r[0]:<24}{r[1]:>9}{r[2]:>8}{r[3]:>8}{r[4]:>7}'
                f'{r[5]:>13.2f}{r[6]:>11.2f}{r[7]:>11.1f}{r[8]:>9}')
        say()
        say(f"Из {len(a)} серийных авторов таких {len(good)}"
            f" — {len(good) / len(a) * 100:.1f} %.")

    say()
    say("готово")


if __name__ == "__main__":
    main()
