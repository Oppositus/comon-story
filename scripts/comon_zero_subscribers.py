"""comon_zero_subscribers.py — есть ли качество среди живых стратегий, которых не выбрал НИКТО.

Вопрос юзера к разделу 5.1 главы 5: две трети живых стратегий (66.1 %, 957 из 1 448) не
имеют ни одного подписчика. Есть ли среди них такие, что не хуже топовых? Сколько их?
И, если есть, — почему их не выбрали.

Логика счёта:
  1. базы и распределение подписок;
  2. качество по группам популярности (0 / 1-5 / 6-20 / 21-100 / 101+);
  3. «не хуже топовых» — сколько безподписочных выше МЕДИАНЫ топ-группы по каждой метрике
     и сколько выше одновременно по трём (Sharpe, CAGR, просадка);
  4. строгий критерий «проверяемое качество» из раздела 4.5 (достоверность + уровень +
     длина) — сколько таких среди безподписочных и среди выбранных;
  5. 🔴 поправка на множественность: 957 попыток — среди них хорошие будут и случайно.
     Считаем, сколько ожидается при нулевой альфе (t ~ N(0,1)), и сравниваем с наблюдённым;
  6. почему не нашли: возраст, порог входа, ставка тарифа, рейтинг, длина описания, теги,
     аватар, известность автора (подписки на ДРУГИЕ его системы).

Единица счёта — стратегия; followers = подписки, а не люди (Блок 9 §9.10).
Sharpe здесь геометрический (cagr/vol из панели) — та же конвенция, что в Блоке 9 §9.5
и в интерлюдии; арифметический sharpe_ar печатается рядом для сверки с главой 3.

Текстовый вывод, без графиков. Запуск: python comon_zero_subscribers.py
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "cards")       # первичных данных нет в репозитории — см. DATA.md

MIN_TRADE_DAYS = 100          # фильтр измеримости — тот же, что в главах 3-4
MIN_ACTIVITY = 0.10
SERVICE_OWNERS = {2215}       # служебный аккаунт площадки, не автор
TOP_CUT = 101                 # «топовые» — группа 101+ подписчиков (Блок 9, первый проход)


def say(s=""):
    print(s, flush=True)


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.percentile(x, p))


def load():
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False)
    sh = pd.read_csv(DIR / "sharpe.csv.gz")[["id", "sharpe_ar", "activity", "t_sharpe"]]
    p = panel.merge(sh, on="id", how="left")
    p["msr"] = (p["n_trade"] >= MIN_TRADE_DAYS) & (p["activity"] >= MIN_ACTIVITY)
    return p


def cards_extra(ids):
    """Длина описания, число тегов, наличие аватара — их нет в панели, берём из карточек."""
    rows = []
    for i in ids:
        f = DIR / "cards" / f"{i}.json.gz"
        if not f.exists():
            continue
        try:
            d = json.load(gzip.open(f, "rt", encoding="utf-8")).get("data") or {}
        except Exception:
            continue
        txt = d.get("textDescription") or d.get("description") or ""
        rows.append({
            "id": i,
            "desc_len": len(txt),
            "n_tags": len(d.get("tags") or []),
            "has_avatar": bool(d.get("authorAvatarUrl")),
            "acc_follow": d.get("accuracyFollowingPercent"),
        })
    return pd.DataFrame(rows)


def main():
    p = load()
    live = p[p["is_live"] == 1].copy()
    say("=" * 78)
    say("1. БАЗЫ")
    say("=" * 78)
    say(f"живых стратегий                : {len(live)}")
    say(f"  из них с рядом (n_pts>0)     : {int((live['n_pts'] > 0).sum())}")
    say(f"  торговали хотя бы день       : {int((live['n_trade'] > 0).sum())}")
    say(f"  измеримых (>=100 дн, >=10 %) : {int(live['msr'].sum())}")
    zero = live["followers"].fillna(0) == 0
    say(f"без единого подписчика         : {int(zero.sum())} ({100 * zero.mean():.1f} %)")
    say(f"  из них измеримых             : {int((zero & live['msr']).sum())}")
    say(f"с подписчиками                 : {int((~zero).sum())}, подписок {int(live['followers'].sum())}")
    say(f"  из них измеримых             : {int((~zero & live['msr']).sum())}")

    # ---------------------------------------------------------------- 2
    live["grp"] = pd.cut(live["followers"].fillna(0),
                         [-1, 0, 5, 20, 100, 10 ** 9],
                         labels=["0", "1-5", "6-20", "21-100", "101+"])
    say()
    say("=" * 78)
    say("2. КАЧЕСТВО ПО ГРУППАМ ПОПУЛЯРНОСТИ (только измеримые; медианы)")
    say("=" * 78)
    say(f"{'группа':>8} {'систем':>7} {'подписок':>9} {'Sharpe':>7} {'Sh_ar':>7} "
        f"{'CAGR %':>8} {'вола %':>7} {'maxDD %':>8} {'лет':>6} {'порог тыс':>10}")
    m = live[live["msr"]]
    for g, gg in m.groupby("grp", observed=True):
        say(f"{str(g):>8} {len(gg):>7} {int(gg['followers'].sum()):>9} "
            f"{gg['sharpe'].median():>7.2f} {gg['sharpe_ar'].median():>7.2f} "
            f"{100 * gg['cagr'].median():>8.1f} {100 * gg['vol'].median():>7.1f} "
            f"{100 * gg['maxdd'].median():>8.1f} {gg['years'].median():>6.2f} "
            f"{gg['min_sum'].median() / 1000:>10.0f}")

    # ---------------------------------------------------------------- 3
    top = m[m["followers"] >= TOP_CUT]
    z = m[m["followers"].fillna(0) == 0]
    subs = m[m["followers"].fillna(0) > 0]
    say()
    say("=" * 78)
    say(f"3. «НЕ ХУЖЕ ТОПОВЫХ»: планка — медиана группы {TOP_CUT}+ ({len(top)} систем)")
    say("=" * 78)
    bars = {"sharpe": top["sharpe"].median(), "cagr": top["cagr"].median(),
            "maxdd": top["maxdd"].median()}
    say(f"планка Sharpe {bars['sharpe']:.2f} · CAGR {100 * bars['cagr']:.1f} % · "
        f"просадка {100 * bars['maxdd']:.1f} %")
    say()
    say(f"{'критерий':<44} {'безподп.':>10} {'доля':>7} {'выбранные':>10} {'доля':>7}")
    ok_z = {}
    ok_s = {}
    tests = {
        "Sharpe выше медианы топ-группы": lambda d: d["sharpe"] >= bars["sharpe"],
        "CAGR выше медианы топ-группы": lambda d: d["cagr"] >= bars["cagr"],
        "просадка мельче медианы топ-группы": lambda d: d["maxdd"] >= bars["maxdd"],
        "все три сразу": lambda d: (d["sharpe"] >= bars["sharpe"]) & (d["cagr"] >= bars["cagr"]) & (d["maxdd"] >= bars["maxdd"]),
    }
    for name, f in tests.items():
        a, b = f(z), f(subs)
        ok_z[name], ok_s[name] = int(a.sum()), int(b.sum())
        say(f"{name:<44} {int(a.sum()):>10} {100 * a.mean():>6.1f} % "
            f"{int(b.sum()):>10} {100 * b.mean():>6.1f} %")

    # ---------------------------------------------------------------- 4
    say()
    say("=" * 78)
    say("4. СТРОГИЙ КРИТЕРИЙ ПРОВЕРЯЕМОГО КАЧЕСТВА (как в разделе 4.5)")
    say("=" * 78)
    say("достоверность t>=2 · уровень Sharpe_ar>=1 · длина >=3 лет · измеримость")
    for nm, d in (("безподписочные", z), ("выбранные", subs)):
        c1 = d["t_sharpe"] >= 2
        c2 = d["sharpe_ar"] >= 1
        c3 = d["years"] >= 3
        allc = c1 & c2 & c3
        say(f"  {nm:<16} n={len(d):>4} · t>=2: {int(c1.sum()):>4} · Sh>=1: {int(c2.sum()):>4} "
            f"· >=3 лет: {int(c3.sum()):>4} · ВСЁ: {int(allc.sum()):>4} ({100 * allc.mean():.1f} %)")
    z_strict = z[(z["t_sharpe"] >= 2) & (z["sharpe_ar"] >= 1) & (z["years"] >= 3)]
    s_strict = subs[(subs["t_sharpe"] >= 2) & (subs["sharpe_ar"] >= 1) & (subs["years"] >= 3)]

    # ---------------------------------------------------------------- 5
    say()
    say("=" * 78)
    say("5. ПОПРАВКА НА МНОЖЕСТВЕННОСТЬ: сколько таких было бы случайно")
    say("=" * 78)
    from math import erf, sqrt
    surv = lambda t: 0.5 * (1 - erf(t / sqrt(2)))
    for nm, d in (("безподписочные", z), ("выбранные", subs)):
        n = len(d)
        exp2 = n * surv(2.0)
        obs2 = int((d["t_sharpe"] >= 2).sum())
        exp3 = n * surv(3.0)
        obs3 = int((d["t_sharpe"] >= 3).sum())
        say(f"  {nm:<16} n={n:>4} · t>=2 набл. {obs2:>4} против {exp2:>5.1f} ожидаемых "
            f"· t>=3 набл. {obs3:>4} против {exp3:>5.1f}")
    say("  (ожидание при нулевой альфе; t независимыми не являются — оценка грубая)")
    say()
    say("  🔴 контроль длины: безподписочные моложе, а t растёт с длиной ряда.")
    say("  Тот же счёт только на историях >= 3 лет:")
    for nm, d in (("безподписочные", z[z["years"] >= 3]), ("выбранные", subs[subs["years"] >= 3])):
        n = len(d)
        obs2, obs3 = int((d["t_sharpe"] >= 2).sum()), int((d["t_sharpe"] >= 3).sum())
        say(f"  {nm:<16} n={n:>4} · t>=2 набл. {obs2:>4} против {n * surv(2.0):>5.1f} ожидаемых "
            f"· t>=3 набл. {obs3:>4} против {n * surv(3.0):>5.1f}")

    # ------------------------------------------------------- 5б. доступность
    say()
    say("=" * 78)
    say("5б. ДОСТУПНОСТЬ ПО КАПИТАЛУ: по карману ли публике то, что она не выбрала")
    say("=" * 78)
    good = z[(z["sharpe"] >= bars["sharpe"]) & (z["cagr"] >= bars["cagr"]) & (z["maxdd"] >= bars["maxdd"])]
    say("взвешенная по подпискам медиана порога входа (Блок 9 §9.10.2) — 180 тыс ₽")
    say(f"{'группа':<38} {'n':>5} {'медиана порога':>15} {'<=180к':>8} {'<=400к':>8} {'>=1 млн':>9}")
    for nm, d in (("безподписочные не хуже топовых", good),
                  ("безподписочные строгого критерия", z_strict),
                  ("выбранные строгого критерия", s_strict),
                  ("все измеримые живые", m)):
        ms = d["min_sum"]
        say(f"{nm:<38} {len(d):>5} {ms.median() / 1000:>13.0f} к "
            f"{100 * (ms <= 180000).mean():>7.0f} % {100 * (ms <= 400000).mean():>7.0f} % "
            f"{100 * (ms >= 1000000).mean():>8.0f} %")
    say()
    say("авторы 24 строгих безподписочных (сколько систем у каждого в этой группе):")
    vc = live[live["id"].isin(z_strict["id"])]["author"].value_counts()
    say("  " + " · ".join(f"{a} — {n}" for a, n in vc.items()))

    # ---------------------------------------------------------------- 6
    say()
    say("=" * 78)
    say("6. ПОЧЕМУ ИХ НЕ НАШЛИ: чем сильные безподписочные отличаются от сильных выбранных")
    say("=" * 78)
    ex = cards_extra(list(live["id"]))
    live2 = live.merge(ex, on="id", how="left")
    # известность автора: подписки на ДРУГИЕ живые системы того же автора
    tot = live2.groupby("owner_id")["followers"].sum()
    live2["auth_other"] = live2["owner_id"].map(tot) - live2["followers"].fillna(0)
    key = lambda ids: live2[live2["id"].isin(ids)]
    say(f"{'признак':<34} {'сильные безподп.':>17} {'сильные выбранные':>18}")
    A, B = key(z_strict["id"]), key(s_strict["id"])
    say(f"{'систем':<34} {len(A):>17} {len(B):>18}")
    for lbl, col, mul, fmt in [
        ("возраст ряда, лет", "years", 1, "{:.2f}"),
        ("порог входа, тыс ₽", "min_sum", 1e-3, "{:.0f}"),
        ("рейтинг площадки", "rating", 1, "{:.2f}"),
        ("длина описания, симв.", "desc_len", 1, "{:.0f}"),
        ("число тегов", "n_tags", 1, "{:.1f}"),
        ("подписки на др. системы автора", "auth_other", 1, "{:.0f}"),
        ("Sharpe (геом.)", "sharpe", 1, "{:.2f}"),
        ("CAGR, %", "cagr", 100, "{:.1f}"),
        ("макс. просадка, %", "maxdd", 100, "{:.1f}"),
    ]:
        va = A[col].median() * mul if len(A) else np.nan
        vb = B[col].median() * mul if len(B) else np.nan
        say(f"{lbl:<34} {fmt.format(va):>17} {fmt.format(vb):>18}")
    for lbl, col in [("есть аватар автора", "has_avatar")]:
        say(f"{lbl:<34} {100 * A[col].mean():>16.0f} % {100 * B[col].mean():>17.0f} %")
    say(f"{'автор без подписок вообще':<34} {100 * (A['auth_other'] == 0).mean():>16.0f} % "
        f"{100 * (B['auth_other'] == 0).mean():>17.0f} %")

    say()
    say("сильные безподписочные поштучно (Sharpe_ar по убыванию):")
    say(f"{'id':>7} {'автор':<22} {'Sh_ar':>6} {'t':>5} {'CAGR%':>7} {'maxDD%':>7} "
        f"{'лет':>5} {'порог тыс':>9} {'опис.':>6} {'др.подп':>8}")
    for _, r in A.sort_values("sharpe_ar", ascending=False).iterrows():
        say(f"{int(r['id']):>7} {str(r['author'])[:22]:<22} {r['sharpe_ar']:>6.2f} "
            f"{r['t_sharpe']:>5.1f} {100 * r['cagr']:>7.1f} {100 * r['maxdd']:>7.1f} "
            f"{r['years']:>5.1f} {r['min_sum'] / 1000:>9.0f} {r['desc_len']:>6.0f} "
            f"{r['auth_other']:>8.0f}")


if __name__ == "__main__":
    main()
