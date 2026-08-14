"""comon_author_profiles.py — Блок 5, приложение: подробные профили пяти авторов.

План исследования, Блок 5. Пять авторов с более чем 20 ИЗМЕРИМЫМИ
системами (см. comon-block5-multiplicity.md, 5.1) — единственные, у кого выборка
собственных попыток достаточна, чтобы говорить об авторе, а не об одной системе.

Профили строятся по ВСЕМ системам автора, включая архивные и не стартовавшие: смотреть
только на живые значит повторить survivorship bias, который Блок 1 измерил (доживает
11.7 % за 3 года). Метрики качества при этом считаются по системам, прошедшим фильтр измеримости, — Sharpe на 30 торговых днях не значит ничего.

Скрипт ГЕНЕРИРУЕТ markdown-файл results/comon-block5-authors.md (таблицы + рейтинг), чтобы
числа не переносились в документ руками. Текстовые выводы дописываются поверх вручную —
между маркерами <!-- ВЫВОД: ... --> (при перегенерации их надо вписать заново).

Запуск: python comon_author_profiles.py
"""
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comon_survival import km, km_at, km_median                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "cards")       # первичных данных нет в репозитории — см. DATA.md
OUT = ROOT / "results" / "comon-block5-authors.md"

ASOF = date(2026, 8, 5)
MIN_TRADE_DAYS = 100          # фильтр измеримости — как в Блоках 2–5
MIN_ACTIVITY = 0.10
AUTHORS = [109657, 297246, 178285, 45461, 324225]   # пятёрка из 5.1, порядок задаст сортировка
_md = []


def w(s=""):
    _md.append(str(s))


def structure(sid):
    f = DIR / "cards" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        c = json.loads(gzip.open(f, "rt").read()).get("data") or {}
    except Exception:                                                    # noqa: BLE001
        return None
    st = c.get("structure") or []
    if not st:
        return None
    d = {x["id"]: x["value"] for x in st}
    return {"fut": d.get("Fut", 0.0), "eq": d.get("Micex", 0.0),
            "bond": d.get("Gko", 0.0), "cash": d.get("Money", 0.0)}


def pct(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100*x:.1f} %".replace("-", "\u2212")


def num(x, d=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}".replace("-", "\u2212")


def yrs(x):
    """Срок в годах; у KM-медианы её может не быть — кривая не опустилась до 0.5."""
    return "не достигнута" if x is None or not np.isfinite(x) else f"{x:.1f} г"


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    S = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    deaths = pd.read_csv(DIR / "deaths.csv.gz", index_col="id")
    panel["live"] = panel["is_live"] == True                             # noqa: E712

    # срок жизни: от первой даты ряда до смерти (последняя сделка) либо до даты выкачки
    def lifetime(r):
        if not isinstance(r["first"], str):
            return np.nan, 0
        d0 = date.fromisoformat(r["first"])
        if r["live"]:
            return (ASOF - d0).days / 365.25, 0
        end = r["last_trade"] if isinstance(r["last_trade"], str) else r["last"]
        if not isinstance(end, str):
            return np.nan, 0
        return (date.fromisoformat(end) - d0).days / 365.25, 1

    lt = panel.apply(lifetime, axis=1, result_type="expand")
    panel["life_yrs"], panel["dead"] = lt[0], lt[1]

    prof = {}
    for oid in AUTHORS:
        A = panel[panel["owner_id"] == oid].copy()
        name = str(A["author"].iloc[0])
        Q = A.join(S[["sharpe_ar", "activity", "n_trade"]], how="left", rsuffix="_s")
        good = (Q["n_trade_s"] >= MIN_TRADE_DAYS) & (Q["activity"] >= MIN_ACTIVITY)
        G = Q[good]
        traded = A[A["n_trade"].fillna(0) > 0]              # хоть одна сделка
        # Каплан-Мейер по торговавшим системам автора
        m = traded["life_yrs"].notna()
        t, e = traded.loc[m, "life_yrs"].to_numpy(), traded.loc[m, "dead"].to_numpy()
        if len(t) >= 5:
            ti, Sv, se = km(t, e)
            km1 = km_at(ti, Sv, se, 1.0)[0]
            km3 = km_at(ti, Sv, se, 3.0)[0]
            kmm = km_median(ti, Sv)
        else:
            km1 = km3 = kmm = np.nan
        st = [structure(i) for i in A.index]
        st = pd.DataFrame([x for x in st if x])
        D = deaths.reindex(A.index).dropna(subset=["mech"])
        prof[oid] = {
            "name": name, "A": A, "G": G, "traded": traded, "D": D, "st": st,
            "n_all": len(A), "n_ser": int(A["n_pts"].notna().sum()),
            "n_traded": len(traded), "n_qual": len(G), "n_live": int(A["live"].sum()),
            "first": str(A["created_at"].min())[:7], "last": str(A["created_at"].max())[:7],
            "km1": km1, "km3": km3, "kmm": kmm,
            "s_max": G["sharpe_ar"].max() if len(G) else np.nan,
            "s_med": G["sharpe_ar"].median() if len(G) else np.nan,
            "cagr_med": G["cagr"].median() if len(G) else np.nan,
            "dd_med": G["maxdd"].median() if len(G) else np.nan,
            "vol_med": G["vol"].median() if len(G) else np.nan,
            "life_med": traded["life_yrs"].median(),
            "live_share": A["live"].mean(),
            "prof_share": (G["cagr"] > 0).mean() if len(G) else np.nan,
            "foll": A["followers"].sum(), "minsum": A["min_sum"].median(),
            "ruin": (D["fin"] == "разорение").mean() if len(D) else np.nan,
        }

    order = sorted(AUTHORS, key=lambda o: -prof[o]["n_all"])

    # ── шапка ──────────────────────────────────────────────────────────────────
    w("# Comon, Блок 5 (приложение): профили пяти многопопыточных авторов")
    w()
    w("**Статус:** составлено 2026-08-06. **Родительский документ:** "
      "[`comon-block5-multiplicity.md`](comon-block5-multiplicity.md) → 5.1.")
    w("**Скрипт:** [`comon_author_profiles.py`](../scripts/comon_author_profiles.py) "
      "· данные — панель Блока 0, `sharpe.csv.gz` (Блок 2), `deaths.csv.gz` (Блок 1).")
    w()
    w("---")
    w()
    w("## Зачем и как читать")
    w()
    w("Это пять авторов, у которых больше 20 систем прошли фильтр измеримости, — "
      "единственные на площадке, у кого выборка собственных попыток достаточна, чтобы "
      "говорить об **авторе**, а не об отдельной системе.")
    w()
    w("🔴 **Считается по всем системам, включая архивные и не стартовавшие.** Смотреть "
      "только на живые значит повторить survivorship bias, измеренный в "
      "[Блоке 1](comon-block1-survival.md): до трёх лет доживает 11.7 % систем площадки. "
      "Метрики качества (Sharpe, CAGR, просадка) при этом берутся по системам, прошедшим "
      "фильтр (≥ 100 торговых дней, активность ≥ 10 %) — Sharpe на тридцати днях не значит "
      "ничего.")
    w()
    w("Выживаемость — Каплан–Мейер по **торговавшим** системам автора (хотя бы одна сделка); "
      "срок жизни считается от первой даты ряда до последней сделки, живые цензурируются "
      "датой выкачки 2026-08-05. Это те же определения, что в Блоке 1.")
    w()
    w("⚠️ `followers` у архивных систем обнулён площадкой (см. ловушки Блока 0), поэтому "
      "подписчики отражают только живые системы и для сравнения авторов почти бесполезны.")
    w()
    w("---")
    w()
    w("## Сводка: пять авторов рядом")
    w()
    w("| автор | систем всего | с рядом | торговали | измеримых | живых | период создания |")
    w("|---|---:|---:|---:|---:|---:|---|")
    for o in order:
        p = prof[o]
        w(f'| **{p["name"]}** | {p["n_all"]} | {p["n_ser"]} | {p["n_traded"]} | '
          f'{p["n_qual"]} | {p["n_live"]} | {p["first"]} … {p["last"]} |')
    w()
    w("| автор | S лучшей | S медианной | CAGR мед. | вола мед. | maxDD мед. | "
      "доля прибыльных | подписчиков |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for o in order:
        p = prof[o]
        w(f'| **{p["name"]}** | {num(p["s_max"])} | {num(p["s_med"])} | '
          f'{pct(p["cagr_med"])} | {pct(p["vol_med"])} | {pct(p["dd_med"])} | '
          f'{pct(p["prof_share"])} | {int(p["foll"])} |')
    w()
    w("| автор | доля живых | медиана срока жизни | KM: доживает 1 год | KM: 3 года | "
      "медиана KM | разорений среди мёртвых |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for o in order:
        p = prof[o]
        w(f'| **{p["name"]}** | {pct(p["live_share"])} | {num(p["life_med"], 1)} г | '
          f'{pct(p["km1"])} | {pct(p["km3"])} | {yrs(p["kmm"])} | {pct(p["ruin"])} |')
    w()
    w("Для сравнения — вся площадка (Блок 1): медиана срока жизни торгующей системы "
      "**0.47 года** (173 дня), доживает до 3 лет **11.7 %**, разорение — исход у "
      "**12.5 %** мёртвых.")
    w()
    w("<!-- ВЫВОД: сводка -->")
    w()
    w("---")

    # ── по авторам ─────────────────────────────────────────────────────────────
    for o in order:
        p = prof[o]
        A, G, D, st = p["A"], p["G"], p["D"], p["st"]
        w()
        w(f'## {p["name"]} (owner {o}) — {p["n_all"]} систем')
        w()
        ms = f'{p["minsum"]:,.0f}'.replace(",", "\u00a0")
        w(f'**Профиль.** Систем всего {p["n_all"]}, с дневным рядом {p["n_ser"]}, '
          f'торговали {p["n_traded"]}, прошли фильтр измеримости {p["n_qual"]}, '
          f'живых {p["n_live"]}. Создание систем: {p["first"]} … {p["last"]}. '
          f'Медианный порог входа `minSum` — {ms} ₽.')
        if len(st):
            w()
            w(f'Состав портфелей (медианные доли по {len(st)} карточкам): фьючерсы '
              f'{st["fut"].median():.0f} %, акции {st["eq"].median():.0f} %, облигации '
              f'{st["bond"].median():.0f} %, деньги {st["cash"].median():.0f} %.')
        w()
        w("**Метрики измеримых систем** (квантили по "
          f'{p["n_qual"]} системам):')
        w()
        w("| метрика | p5 | p25 | медиана | p75 | p95 | макс |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for lab, col, sc in (("Sharpe (арифм.)", "sharpe_ar", 1),
                             ("CAGR, %", "cagr", 100),
                             ("волатильность, %", "vol", 100),
                             ("макс. просадка, %", "maxdd", 100),
                             ("скос дневных", "skew", 1),
                             ("длина истории, лет", "years", 1)):
            if col not in G.columns or not len(G):
                continue
            v = G[col].dropna() * sc
            if not len(v):
                continue
            cells = [f"{v.quantile(.05):.2f}", f"{v.quantile(.25):.2f}",
                     f"**{v.median():.2f}**", f"{v.quantile(.75):.2f}",
                     f"{v.quantile(.95):.2f}", f"{v.max():.2f}"]
            w(f"| {lab} | " + " | ".join(c.replace("-", "\u2212") for c in cells) + " |")
        w()
        # причины смерти
        if len(D):
            w(f'**Как умирали** (по {len(D)} архивным системам с классификацией Блока 1):')
            w()
            w("| механизм смерти | систем | доля | | финансовый исход | систем | доля |")
            w("|---|---:|---:|---|---|---:|---:|")
            mv = D["mech"].value_counts()
            fv = D["fin"].value_counts()
            keys = list(range(max(len(mv), len(fv))))
            for i in keys:
                a = (f'{mv.index[i]} | {mv.iloc[i]} | {100*mv.iloc[i]/len(D):.0f} %'
                     if i < len(mv) else " |  | ")
                b = (f'{fv.index[i]} | {fv.iloc[i]} | {100*fv.iloc[i]/len(D):.0f} %'
                     if i < len(fv) else " |  | ")
                w(f'| {a} | | {b} |')
            w()
        # создание по годам
        yr = pd.to_datetime(A["created_at"], errors="coerce").dt.year.value_counts().sort_index()
        if len(yr):
            w("**Когда создавались системы:** " +
              ", ".join(f"{int(y)} — {int(n)}" for y, n in yr.items()) + ".")
            w()
        # топ и антитоп
        if len(G):
            w("**Лучшие пять систем** (по Sharpe, только измеримые):")
            w()
            w("| id | Sharpe | CAGR | maxDD | торг. дней | лет | статус |")
            w("|---:|---:|---:|---:|---:|---:|---|")
            for i, r in G.nlargest(5, "sharpe_ar").iterrows():
                w(f'| {i} | {r["sharpe_ar"]:.2f} | {pct(r["cagr"])} | {pct(r["maxdd"])} | '
                  f'{int(r["n_trade"])} | {r["years"]:.1f} | '
                  f'{"жива" if r["live"] else "архив"} |')
            w()
            w("**Худшие пять систем:**")
            w()
            w("| id | Sharpe | CAGR | maxDD | торг. дней | лет | статус |")
            w("|---:|---:|---:|---:|---:|---:|---|")
            for i, r in G.nsmallest(5, "sharpe_ar").iterrows():
                w(f'| {i} | {r["sharpe_ar"]:.2f} | {pct(r["cagr"])} | {pct(r["maxdd"])} | '
                  f'{int(r["n_trade"])} | {r["years"]:.1f} | '
                  f'{"жива" if r["live"] else "архив"} |')
            w()
        w(f'<!-- ВЫВОД: {p["name"]} -->')
        w()
        w("---")

    # ── рейтинг ────────────────────────────────────────────────────────────────
    w()
    w("## Кто из авторов лучше")
    w()
    w("Единственного «лучшего» назвать нельзя: параметры расходятся, и каждый описывает "
      "свою сторону. Ниже — ранги по семи независимым параметрам (1 = лучший).")
    w()
    crit = [
        ("качество сигнала", "s_med", True, "медианный Sharpe измеримых систем"),
        ("надёжность", "live_share", True, "доля живых систем от всех созданных"),
        ("выживаемость", "km3", True, "KM: доля доживших до 3 лет"),
        ("долговечность", "life_med", True, "медиана срока жизни торговавшей системы"),
        ("риск", "dd_med", True, "медианная макс. просадка (ближе к нулю — лучше)"),
        ("доходность", "cagr_med", True, "медианный CAGR измеримых систем"),
        ("доля прибыльных", "prof_share", True, "доля измеримых систем с CAGR > 0"),
        ("дисциплина запуска", "startrate", True,
         "доля созданных систем, дошедших до измеримого ряда"),
    ]
    for o in order:
        prof[o]["startrate"] = prof[o]["n_qual"] / prof[o]["n_all"]
    w("| параметр | " + " | ".join(prof[o]["name"] for o in order) + " |")
    w("|---|" + "---:|" * len(order))
    ranks = {o: [] for o in order}
    for lab, key, bigger, expl in crit:
        vals = {o: prof[o][key] for o in order}
        fin = {o: v for o, v in vals.items() if v is not None and np.isfinite(v)}
        srt = sorted(fin, key=lambda o: -fin[o] if bigger else fin[o])
        rk = {o: i + 1 for i, o in enumerate(srt)}
        for o in order:
            if o in rk:
                ranks[o].append(rk[o])
        cells = []
        for o in order:
            v = vals[o]
            if key in ("live_share", "km3", "dd_med", "cagr_med", "prof_share", "startrate"):
                sv = pct(v)
            elif key == "life_med":
                sv = yrs(v)
            else:
                sv = num(v)
            cells.append(f'{sv} ({rk.get(o, "—")})')
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("| **средний ранг** | " +
      " | ".join(f"**{np.mean(ranks[o]):.1f}**" for o in order) + " |")
    w()
    w("В скобках — ранг среди пяти. Расшифровка параметров: " +
      "; ".join(f"**{lab}** — {expl}" for lab, _, _, expl in crit) + ".")
    w()
    w("<!-- ВЫВОД: рейтинг -->")

    OUT.write_text("\n".join(_md) + "\n", encoding="utf-8")
    print(f"записано: {OUT} ({len(_md)} строк)")


if __name__ == "__main__":
    main()
