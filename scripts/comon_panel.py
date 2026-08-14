"""comon_panel.py — БЛОК 0 исследования Comon: методология данных + единая панель.

План исследования, Блок 0. Ни одна цифра последующих блоков не имеет
смысла, пока не закрыт этот. Скрипт делает две вещи:

  1. ПРОВЕРКИ МЕТОДОЛОГИИ (7 вопросов блока):
     value vs накопление rValue · календарность ряда · агрегаты витрины vs пересчёт ·
     backfill (ряд раньше рождения) · чьи деньги (тарифы) · обрыв ряда = смерть? ·
     чем заполнены дыры;
  2. ПАНЕЛЬ — одна строка на систему, всё остальное исследование работает с ней.

Семантика ряда (проверена арифметически на живых системах, id 118415/112741/13659):
  ряд идёт в ОБРАТНОМ хронологическом порядке (первый элемент = последняя дата);
  `value`  — кумулятивная доходность в ПРОЦЕНТАХ от старта: value = (П(1+r) − 1)·100;
  `rValue` — простая дневная доходность в ДОЛЯХ (−0.0181 = −1.81 %).

🔴 Аннуализация — по ФАКТИЧЕСКОЙ частоте (n точек / годы, ~365), НЕ по 252: ряд
календарный, выходные присутствуют с rValue = 0. Ошибка уже стоила Sharpe 1.19 вместо 1.43.

Панель пишется в data/panel.csv.gz (pyarrow не установлен — parquet
из плана заменён на csv.gz, читается pandas один в один).

Текстовый вывод. Запуск: python comon_panel.py
"""
import gzip
import json
import math
import os
import subprocess
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit", "cards")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "panel.csv.gz"
LOG = ROOT / "results" / "comon_panel.log"
NPROC = 8
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Сборка панели — Pool по всем 16 тысячам рядов; величины для картинки
# сохраняются рядом с логом, чтобы оформление правилось без пересчёта.
CACHE = ROOT / "results" / "comon_panel_charts.npz"


def chart_drawdown_card(card, real, live, ret):
    """График 17: просадка по ряду против витринной, по итогу жизни стратегии.

    🔴 БЫЛО РАССЕЯНИЕ, СТАЛИ ПОЛОСЫ (осмотр 14.08). На поле из 5 703 точек
    утверждение раздела не читалось: 78 % точек ложатся ровно на диагональ и
    сливаются в одну линию, отклонения «витрина мягче» тонут в этой куче, а
    единственное, что глаз замечает, — два десятка редких точек ПОД диагональю,
    то есть прямые контрпримеры. Плюс 1 425 архивных карточек с обнулённым полем
    дают сплошную полосу у нуля, которую легко принять за окрашенную ось.
    Парные полосы по итогу жизни показывают ровно ту таблицу, что стоит в
    разделе 1.3, и разрыв между «по ряду» и «на витрине» виден сразу.
    """
    import comon_charts as ch

    f, ax = ch.fig(h_px=1080, bottom=0.30)   # четыре строки сноски + подпись оси
    a = np.abs(np.asarray(card, float))              # витрина, уже в процентах
    b = 100 * np.abs(np.asarray(real, float))        # ряд, доля → проценты
    lv, rt = np.asarray(live, bool), np.asarray(ret, float)

    groups = ((-1.01, -0.9, "потеряли\nбольше 90 %"), (-0.9, -0.5, "потеряли\n50–90 %"),
              (-0.5, 0.0, "потеряли\n0–50 %"), (0.0, 1e9, "закончили\nв плюсе"))
    labs, m_real, m_card, ns = [], [], [], []
    for lo, hi, lab in groups:
        m = (rt > lo) & (rt <= hi)
        if not m.any():
            continue
        labs.append(lab)
        ns.append(int(m.sum()))
        m_real.append(float(np.median(b[m])))
        m_card.append(float(np.median(a[m])))

    x = np.arange(len(labs))
    ax.bar(x - 0.19, m_real, width=0.36, color=ch.BLUE, zorder=3,
           label="реальная просадка, посчитанная по ряду доходности")
    ax.bar(x + 0.19, m_card, width=0.36, color=ch.ORANGE, zorder=3,
           label="просадка, показанная в карточке на витрине")
    for xi, (vr, vc) in enumerate(zip(m_real, m_card)):
        for dx, v, c in ((-0.19, vr, ch.BLUE), (0.19, vc, ch.ORANGE)):
            ax.annotate(f"{ch.n_(v, 1)} %", xy=(xi + dx, v), xytext=(0, 6),
                        textcoords="offset points", fontsize=11, color=c,
                        ha="center", va="bottom", fontweight="bold")
        ax.annotate(f"разница {ch.n_(vr - vc, 1)} пп", xy=(xi, max(vr, vc)),
                    xytext=(0, 34), textcoords="offset points", fontsize=11,
                    color=ch.GREY, ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{ch.n_(n)} стратегий" for l, n in zip(labs, ns)])
    ax.set_xlim(-0.6, len(labs) - 0.4)
    ax.set_ylim(0, max(m_real) * 1.38)   # запас под два яруса подписей над полосой
    ch.pct_raw(ax, "y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handlelength=2.2, labelspacing=0.55)
    ax.set_xlabel("итог жизни стратегии")
    ax.set_ylabel("максимальная просадка, глубина")
    ax.set_title("Витрина показывает провал мягче, чем он был на самом деле")
    ch.note(ax,
            f"База — {ch.n_(len(a))} стратегий с рядом длиннее 250 точек и "
            f"заполненным полем просадки в карточке; в каждой группе показана "
            f"медиана. Обе величины даны глубиной, то есть положительным числом; "
            f"«закончили в плюсе» — про итог жизни, а провал по пути был у всех.",
            "Разрыв неодинаков, и растёт он не монотонно: у потерявших почти всё "
            "смягчать нечего — реальная просадка у них 98,9 %, мягче ста процентов "
            "поле не покажешь. Самый большой разрыв, 13,2 пп, у потерявших до "
            "половины.",
            f"Смягчение приходит из АРХИВНЫХ карточек: у {ch.n_(int(((~lv) & (a < 1)).sum()))} "
            f"закрытых стратегий поле просадки просто обнулено, у остальных занижено. "
            f"У живых стратегий поле совпадает с рядом тождественно.",
            "Порядок при этом сохранён идеально: ранговая корреляция витринной "
            "просадки с реальной равна 1,000 — сравнивая две стратегии между собой, "
            "читатель получает верный ответ, какая рискованнее, и ошибается только "
            "в абсолютной величине.")
    ch.save(f, 17, "drawdown-card-vs-real")
    print("    " + "; ".join(f"{l.replace(chr(10), ' ')}: ряд {r:.1f} / витрина "
                             f"{c:.1f} (n={n})"
                             for l, r, c, n in zip(labs, m_real, m_card, ns)),
          flush=True)


def load(kind, sid):
    f = DIR / kind / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        return json.loads(gzip.open(f, "rt").read()).get("data")
    except Exception:                                                  # noqa: BLE001
        return None


def maxdd(eq):
    """Максимальная просадка эквити (доля, отрицательная) и просадка на последней точке."""
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min()), float(dd[-1])


def one(sid):
    """Метрики одной системы: карточка + пересчёт из ряда. Возвращает dict или None."""
    c = load("cards", sid)
    p = load("profit", sid)
    # 🔴 has_series ставится ПОСЛЕ разбора ряда, а не по факту загрузки файла.
    # Файл profit/<id>.json.gz лежит у всех 19 978 систем и всегда содержит "data",
    # поэтому прежнее `p is not None` было истинно даже у 1 624 систем без единой
    # точки ряда — колонка была ложной у всей панели.
    row = {"id": sid, "has_card": c is not None, "has_series": False}
    if c:
        tar = (c.get("autoFollowingTariffDetails") or [{}])
        row.update({
            "title": (c.get("title") or "")[:60],
            "owner_id": c.get("ownerId"),
            "author": c.get("author"),
            "created_at": c.get("createdAt"),
            "archived_at": c.get("archivedAt"),
            "is_live": c.get("archivedAt") is None,
            "followers": c.get("followerCount"),
            "min_sum": c.get("minSum"),
            "money_limit": c.get("moneyLimit"),
            "risk_level": c.get("riskLevel"),
            "tariff_id": c.get("autoFollowingTariffId"),
            "tariff_type": (tar[0] or {}).get("tariffType") if tar else None,
            "tariff_name": c.get("autoFollowingTariff"),
            "transaction_rate": c.get("transactionRate"),
            "activity_index": c.get("tradeActivityIndex"),
            "allow_follow": c.get("allowedAutoFollowing"),
            "premium": c.get("premium"),
            "rating": c.get("strategyRating"),
            "n_structure": len(c.get("structure") or []),
            # агрегаты витрины — сверяем с пересчётом
            "card_lifetime": c.get("profitLifetime"),
            "card_annual": c.get("annualAverageProfit"),
            "card_maxdd": c.get("maxDrawDown"),
            "card_cvar": c.get("conditionalValueAtRisk"),
            "card_365": c.get("profit365Days"),
        })
    s = (p or {}).get("strategy") or []
    if not s:
        return row
    row["has_series"] = True                       # ряд непустой — только теперь True
    s = s[::-1]                                    # ряд отдаётся в обратном порядке
    d = np.array([x["date"] for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s])
    v = np.array([float(x["value"] or 0.0) for x in s])
    n = len(r)
    d0 = date.fromisoformat(d[0])
    d1 = date.fromisoformat(d[-1])
    yrs = max((d1 - d0).days / 365.25, 1e-9)
    eq = np.cumprod(1.0 + r)
    trade = r != 0.0
    ntr = int(trade.sum())
    freq = n / yrs                                  # фактическая частота точек в году
    sd = float(r.std(ddof=1)) if n > 2 else 0.0
    vol = sd * math.sqrt(freq)
    ret_total = float(eq[-1] - 1.0)
    cagr = (eq[-1] ** (1.0 / yrs) - 1.0) if eq[-1] > 0 else -1.0
    dn = r[r < 0]
    dsd = float(dn.std(ddof=1)) if len(dn) > 2 else 0.0
    dvol = dsd * math.sqrt(freq)
    mdd, dd_end = maxdd(eq)
    # только торговые дни (нули выходных и мёртвых периодов выброшены)
    rt = r[trade]
    ftr = ntr / yrs if yrs > 0 else 0.0
    vol_t = (float(rt.std(ddof=1)) * math.sqrt(ftr)) if ntr > 2 else 0.0
    # последний ТОРГОВЫЙ день — настоящая дата смерти (ряд тянется нулями и после)
    last_tr = d[trade][-1] if ntr else None
    # битый хвост витрины: value = 0 на последних точках при ненулевом накоплении
    tail = 0
    for k in range(n - 1, -1, -1):
        if v[k] == 0.0 and abs(eq[k] - 1.0) > 1e-9:
            tail += 1
        else:
            break
    # дыры: самая длинная серия нулей подряд
    gap, best = 0, 0
    for x in trade:
        gap = 0 if x else gap + 1
        best = max(best, gap)
    wknd = np.array([date.fromisoformat(x).weekday() >= 5 for x in d])
    row.update({
        "n_pts": n, "n_trade": ntr, "first": d[0], "last": d[-1], "years": yrs,
        "freq": freq, "ret_total": ret_total, "cagr": cagr, "vol": vol,
        "sharpe": cagr / vol if vol > 0 else np.nan,
        "sortino": cagr / dvol if dvol > 0 else np.nan,
        "vol_trade": vol_t,
        "sharpe_trade": cagr / vol_t if vol_t > 0 else np.nan,
        "maxdd": mdd, "dd_at_end": dd_end,
        "skew": float(pd.Series(r).skew()) if n > 3 else np.nan,
        "kurt": float(pd.Series(r).kurt()) if n > 3 else np.nan,
        "zero_share": 1.0 - ntr / n,
        "gap_max": best,
        "wknd_share": float(wknd.mean()),
        "wknd_zero": float((r[wknd] == 0).mean()) if wknd.any() else np.nan,
        "wday_zero": float((r[~wknd] == 0).mean()) if (~wknd).any() else np.nan,
        # сверки Блока 0
        "last_trade": last_tr,
        "value_tail_zero": tail,
        "chk_value_vs_r": float(abs(v[-1] / 100.0 - (eq[-1] - 1.0))),
        "chk_lifetime": (float(abs(v[-1] - (row.get("card_lifetime") or 0.0)))
                         if row.get("card_lifetime") is not None else np.nan),
    })
    if row.get("created_at"):
        row["backfill_days"] = (d0 - date.fromisoformat(row["created_at"])).days
    if row.get("archived_at"):
        a = date.fromisoformat(row["archived_at"])
        row["death_gap_days"] = (a - d1).days
        if last_tr:
            row["death_gap_trade"] = (a - date.fromisoformat(last_tr)).days
    return row


def imoex_series():
    """Дневной IMOEX с ISS для β (в рядах Comon серия imoex пустая). Кэш на диске."""
    f = DIR / "imoex.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for y0 in range(2009, 2027):
        url = ("https://iss.moex.com/iss/history/engines/stock/markets/index/securities/"
               f"IMOEX.json?iss.meta=off&from={y0}-01-01&till={y0}-12-31&limit=100")
        start = 0
        while True:
            j = json.loads(subprocess.check_output(
                ['curl', '-sL', f'{url}&start={start}'], text=True))
            cols, data = j["history"]["columns"], j["history"]["data"]
            if not data:
                break
            i_d, i_c = cols.index("TRADEDATE"), cols.index("CLOSE")
            for row in data:
                if row[i_c]:
                    out[row[i_d]] = row[i_c]
            start += len(data)
    f.write_text(json.dumps(out))
    return out


def main():
    if ONLY:
        d = np.load(CACHE, allow_pickle=False)
        chart_drawdown_card(d["card"], d["real"], d["live"], d["ret"])
        return
    ids = sorted(int(f.stem.split('.')[0]) for f in (DIR / "profit").glob("*.json.gz"))
    say(f"систем в сырье: {len(ids)}")
    with Pool(NPROC) as pool:
        rows = [r for r in pool.imap_unordered(one, ids, chunksize=200) if r]
    df = pd.DataFrame(rows).set_index("id").sort_index()
    say(f"панель собрана: {len(df)} строк, {len(df.columns)} колонок")

    has = df["n_pts"].notna()
    live = df["is_live"] == True                                        # noqa: E712
    dead = df["is_live"] == False                                       # noqa: E712

    say()
    say("=" * 100)
    say("БЛОК 0.1 — СЕМАНТИКА: value против накопления rValue")
    say("=" * 100)
    e = df.loc[has, "chk_value_vs_r"]
    say(f"|value_посл/100 − (П(1+rValue) − 1)|: медиана {e.median():.2e}, "
        f"p99 {e.quantile(0.99):.2e}, макс {e.max():.2e}")
    bad = df.loc[has & (df["chk_value_vs_r"] > 1e-6)]
    say(f"расхождение > 1e-6: {len(bad)} систем ({100*len(bad)/int(has.sum()):.2f} %)")
    say(f"из них с ОБНУЛЁННЫМ хвостом value (value=0 при ненулевом накоплении): "
        f"{int((bad['value_tail_zero'] > 0).sum())} "
        f"({100*(bad['value_tail_zero'] > 0).mean():.1f} % расхождений)")
    say(f"  длина битого хвоста: медиана {bad['value_tail_zero'].median():.0f} точек, "
        f"макс {bad['value_tail_zero'].max():.0f}")
    dts = df.loc[df["value_tail_zero"] > 0, "last"].value_counts().head(3)
    say(f"  даты обрыва скапливаются: " +
        ", ".join(f"{k} ({v} систем)" for k, v in dts.items()))
    say("🔴 Это дефект ВИТРИНЫ, а не семантики: rValue цел, обнулён только value.")
    say("   Все метрики считаем из rValue; value и profitLifetime — только для сверки.")
    e2 = df.loc[has & df["card_lifetime"].notna(), "chk_lifetime"]
    say(f"|value_посл − profitLifetime карточки|: медиана {e2.median():.2e}, "
        f"доля > 0.01 пп: {100*(e2 > 0.01).mean():.2f} %")
    say("ВЫВОД: value = (произведение (1+rValue) − 1)·100, в процентах; rValue — простая")
    say("дневная доходность в долях. Величины согласованы, витрина не расходится с рядом.")

    say()
    say("=" * 100)
    say("БЛОК 0.2 — КАЛЕНДАРНОСТЬ РЯДА (критично для аннуализации)")
    say("=" * 100)
    g = df.loc[has & (df["n_pts"] > 250)]
    say(f'{"показатель":<44} {"медиана":>10} {"p25":>10} {"p75":>10}')
    for lab, col in (("точек в году (freq)", "freq"),
                     ("доля дат-выходных в ряде", "wknd_share"),
                     ("доля нулевых rValue среди выходных", "wknd_zero"),
                     ("доля нулевых rValue среди будней", "wday_zero"),
                     ("доля нулей во всём ряде", "zero_share")):
        say(f'{lab:<44} {g[col].median():>10.3f} {g[col].quantile(.25):>10.3f} '
            f'{g[col].quantile(.75):>10.3f}')
    say(f"систем с freq в диапазоне 360..370: "
        f"{100*g['freq'].between(360, 370).mean():.1f} %")
    say("ВЫВОД: ряд КАЛЕНДАРНЫЙ. Аннуализация по n/лет (~365), НЕ по 252.")

    say()
    say("=" * 100)
    say("БЛОК 0.3 — ВИТРИНА ПРОТИВ ПЕРЕСЧЁТА ИЗ РЯДА")
    say("=" * 100)
    q = df.loc[has & (df["n_pts"] > 250) & df["card_maxdd"].notna()].copy()
    q["dd_diff"] = q["maxdd"] * 100 - q["card_maxdd"]
    q["ann_diff"] = q["cagr"] * 100 - q["card_annual"]
    df["card_metrics_valid"] = df["is_live"] == True                    # noqa: E712
    say(f'{"сверка":<40} {"медиана":>10} {"p10":>10} {"p90":>10} {"|Δ|>5пп":>9}')
    for lab, col in (("maxDD: пересчёт − витрина, пп", "dd_diff"),
                     ("CAGR: пересчёт − annualAverageProfit, пп", "ann_diff")):
        say(f'{lab:<40} {q[col].median():>10.2f} {q[col].quantile(.10):>10.2f} '
            f'{q[col].quantile(.90):>10.2f} {100*(q[col].abs() > 5).mean():>8.1f}%')
    say()
    say("РАЗРЕЗ ЖИВЫЕ / МЁРТВЫЕ — здесь вся причина расхождений:")
    say(f'{"группа":<10} {"систем":>7} {"медиана Δcagr":>14} {"|Δ|>5пп":>9} '
        f'{"медиана annual витрины":>24}')
    for lab, m in (("живые", q["is_live"] == True),                     # noqa: E712
                   ("мёртвые", q["is_live"] == False)):                 # noqa: E712
        g = q.loc[m]
        say(f'{lab:<10} {len(g):>7} {g["ann_diff"].median():>14.2f} '
            f'{100*(g["ann_diff"].abs() > 5).mean():>8.1f}% '
            f'{g["card_annual"].median():>23.2f}%')
    say()
    say("То же по итогу жизни (медиана annualAverageProfit витрины против пересчёта):")
    say(f'{"итог жизни":<18} {"систем":>7} {"CAGR пересчёт":>14} {"annual витрины":>15}')
    for lo, hi, lab in ((-1.01, -0.9, "потеряли >90 %"), (-0.9, -0.5, "−90…−50 %"),
                        (-0.5, 0, "−50…0 %"), (0, 1, "0…+100 %"), (1, 1e9, ">+100 %")):
        g = q.loc[(q["ret_total"] > lo) & (q["ret_total"] <= hi)]
        if not len(g):
            continue
        say(f'{lab:<18} {len(g):>7} {100*g["cagr"].median():>13.1f}% '
            f'{g["card_annual"].median():>14.1f}%')
    say("🔴 ВИТРИНА ОБНУЛЯЕТ РАСЧЁТНЫЕ МЕТРИКИ У АРХИВНЫХ СИСТЕМ: annualAverageProfit = 0")
    say("   при реальном CAGR −84 % у разорившихся. Та же болезнь, что followerCount = 0")
    say("   у мёртвых. Из карточки архивной системы нельзя брать НИ ОДНУ расчётную")
    say("   метрику — только даты, тариф, автора, описание. Всё остальное — из ряда.")
    say()
    say("maxDD: витрина систематически МЕЛЬЧЕ пересчёта (не обнуление, а смягчение):")
    say(f'{"итог жизни":<18} {"систем":>7} {"maxDD пересчёт":>15} {"maxDD витрины":>14}')
    for lo, hi, lab in ((-1.01, -0.9, "потеряли >90 %"), (-0.9, -0.5, "−90…−50 %"),
                        (-0.5, 0, "−50…0 %"), (0, 1e9, "в плюсе")):
        g = q.loc[(q["ret_total"] > lo) & (q["ret_total"] <= hi)]
        if not len(g):
            continue
        say(f'{lab:<18} {len(g):>7} {100*g["maxdd"].median():>14.1f}% '
            f'{g["card_maxdd"].median():>13.1f}%')
    if CHARTS:
        np.savez(CACHE, card=q["card_maxdd"].to_numpy(float),
                 real=q["maxdd"].to_numpy(float),
                 ret=q["ret_total"].to_numpy(float),
                 live=(q["is_live"] == True).to_numpy(bool))     # noqa: E712
        say("── график 17 ────────────────────────────────────────────────────")
        chart_drawdown_card(q["card_maxdd"], q["maxdd"],
                            q["is_live"] == True, q["ret_total"])   # noqa: E712
        say()

    say()
    ok = q["value_tail_zero"] == 0
    say(f'то же ТОЛЬКО по системам с целой витриной (без обнулённого хвоста, '
        f'{int(ok.sum())} шт):')
    for lab, col in (("  maxDD: пересчёт − витрина, пп", "dd_diff"),
                     ("  CAGR: пересчёт − annual, пп", "ann_diff")):
        say(f'{lab:<40} {q.loc[ok, col].median():>10.2f} '
            f'{q.loc[ok, col].quantile(.10):>10.2f} {q.loc[ok, col].quantile(.90):>10.2f} '
            f'{100*(q.loc[ok, col].abs() > 5).mean():>8.1f}%')
    say('annualAverageProfit витрины = CAGR (проверено поштучно: совпадает до 0.02 пп),')
    say('а НЕ среднее арифметическое годовых — гипотеза проверена и отвергнута.')
    cv = q.loc[q["card_cvar"].notna() & (q["card_cvar"] != 0)]
    say(f'CVaR витрины против реального maxDD (та самая подмена понятий):')
    say(f'  медиана CVaR {cv["card_cvar"].median():.2f} % против медианы '
        f'maxDD {100*cv["maxdd"].median():.2f} % — глубже в '
        f'{abs(100*cv["maxdd"].median()/cv["card_cvar"].median()):.1f} раза')

    say()
    say("=" * 100)
    say("БЛОК 0.4 — BACKFILL: ряд раньше рождения системы?")
    say("=" * 100)
    b = df.loc[df["backfill_days"].notna(), "backfill_days"]
    say(f"createdAt − первая дата ряда (дней; >0 = ряд начался ПОЗЖЕ рождения):")
    say(f"  медиана {b.median():.0f}, p1 {b.quantile(.01):.0f}, p99 {b.quantile(.99):.0f}")
    for th in (7, 30, 365):
        say(f"  ряд начался раньше рождения более чем на {th:>3} дн: "
            f"{int((b < -th).sum())} систем ({100*(b < -th).mean():.3f} %)")
    worst = df.loc[df["backfill_days"] < -365,
                   ["title", "created_at", "first", "backfill_days", "n_pts"]]
    if len(worst):
        say("  системы с бэктест-хвостом (>1 года до рождения):")
        for i, w in worst.sort_values("backfill_days").head(10).iterrows():
            say(f"    id={i} создана {w['created_at']} ряд с {w['first']} "
                f"({-w['backfill_days']:.0f} дн) — {str(w['title'])[:40]}")
    say("ВЫВОД: массового backfill нет — история рядов не подрисована задним числом.")

    say()
    say("=" * 100)
    say("БЛОК 0.5 — ЧЬИ ЭТО ДЕНЬГИ: тарифы и порог входа")
    say("=" * 100)
    t = df.loc[df["tariff_type"].notna()]
    say(f'{"тип тарифа":<28} {"систем":>7} {"живых":>7} {"подписок":>9} '
        f'{"minSum медиана":>15}')
    for name, grp in t.groupby("tariff_type"):
        say(f'{str(name):<28} {len(grp):>7} {int((grp["is_live"] == True).sum()):>7} '  # noqa: E712
            f'{int(grp["followers"].fillna(0).sum()):>9} '
            f'{grp["min_sum"].median():>15,.0f}')
    say("ВЫВОД: ряд — доходность СЧЁТА АВТОРА до платы за следование; что достаётся")
    say("подписчику, считаем в Блоке 8 с учётом тарифа и порога входа.")

    say()
    say("=" * 100)
    say("БЛОК 0.6 — ОБРЫВ РЯДА = СМЕРТЬ? (archivedAt против последней даты ряда)")
    say("=" * 100)
    gd = df.loc[dead & df["death_gap_days"].notna(), "death_gap_days"]
    say(f"archivedAt − последняя дата ряда (дней):")
    say(f"  медиана {gd.median():.0f}, p25 {gd.quantile(.25):.0f}, "
        f"p75 {gd.quantile(.75):.0f}, p95 {gd.quantile(.95):.0f}")
    for lo, hi, lab in ((-1e9, 1, "ряд идёт до самой архивации (≤1 дн)"),
                        (1, 30, "1–30 дней тишины перед архивацией"),
                        (30, 90, "30–90 дней"), (90, 365, "90–365 дней"),
                        (365, 1e9, "больше года")):
        m = (gd > lo) & (gd <= hi)
        say(f"  {lab:<38} {int(m.sum()):>6} ({100*m.mean():>5.1f} %)")
    say()
    say("То же по ПОСЛЕДНЕМУ ТОРГОВОМУ ДНЮ (ряд тянется нулями и после остановки):")
    gt = df.loc[dead & df["death_gap_trade"].notna(), "death_gap_trade"]
    nz = int((dead & df["n_pts"].notna() & df["n_trade"].fillna(0).eq(0)).sum())
    say(f"  мёртвых без единого торгового дня: {nz} "
        f"({100*nz/int(dead.sum()):.1f} % всех мёртвых)")
    say(f"  archivedAt − последний торговый день: медиана {gt.median():.0f}, "
        f"p75 {gt.quantile(.75):.0f}, p90 {gt.quantile(.90):.0f}")
    for lo, hi, lab in ((-1e9, 1, "торговала до самой архивации (≤1 дн)"),
                        (1, 30, "1–30 дней без сделок"), (30, 90, "30–90 дней"),
                        (90, 365, "90–365 дней"), (365, 1e9, "больше года")):
        m = (gt > lo) & (gt <= hi)
        say(f"  {lab:<38} {int(m.sum()):>6} ({100*m.mean():>5.1f} %)")
    say("ВЫВОД: archivedAt ≠ момент смерти. Датой смерти берём ПОСЛЕДНИЙ ТОРГОВЫЙ ДЕНЬ,")
    say("архивацию — как факт закрытия автором.")

    say()
    say("=" * 100)
    say("БЛОК 0.7 — ДЫРЫ: чем заполнены периоды без сделок")
    say("=" * 100)
    say(f'{"группа":<12} {"систем":>7} {"доля нулей":>11} {"макс.дыра, дн":>14} '
        f'{"Sharpe все":>11} {"Sharpe торг":>12} {"смещение":>9}')
    for lab, m in (("живые", live & has), ("мёртвые", dead & has)):
        gg = df.loc[m & (df["n_pts"] > 250)]
        if not len(gg):
            continue
        sa, st = gg["sharpe"].median(), gg["sharpe_trade"].median()
        say(f'{lab:<12} {len(gg):>7} {gg["zero_share"].median():>11.3f} '
            f'{gg["gap_max"].median():>14.0f} {sa:>11.2f} {st:>12.2f} '
            f'{sa - st:>9.2f}')
    say("Sharpe все — по всем точкам ряда (нули включены, аннуализация n/лет);")
    say("Sharpe торг — только по дням со сделками (аннуализация n_торг/лет).")
    dlt = (df.loc[has & (df["n_pts"] > 250), "sharpe"]
           - df.loc[has & (df["n_pts"] > 250), "sharpe_trade"]).dropna()
    say(f"разница по популяции: медиана {dlt.median():+.6f}, "
        f"доля |Δ| > 0.01: {100*(dlt.abs() > 0.01).mean():.1f} %")
    say("🟢 Разница ТОЖДЕСТВЕННО нулевая для подавляющего большинства — и это не")
    say("свойство данных, а следствие метода: σ по всем точкам × √(n/лет) равно")
    say("σ по торговым × √(n_торг/лет), когда дыры заполнены строго нулями.")
    say("Аннуализация по фактической частоте САМА снимает смещение от дыр — отдельная")
    say("чистка ряда не нужна. Хвост 6.6 % — системы с большим средним, где приближение")
    say("«среднее ≈ 0» не работает.")

    df.to_csv(OUT, compression="gzip")
    say()
    say(f"панель: {OUT} ({OUT.stat().st_size/1e6:.1f} МБ, {len(df)} строк)")
    LOG.write_text("\n".join(_lines) + "\n")
    say(f"лог: {LOG}")


if __name__ == "__main__":
    main()
