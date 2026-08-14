"""comon_hidden.py — чего не видно по кривой доходности: числа третьего раздела заключения.

Подписчик видит одну линию. Раздел показывает, чего по ней не видно, — на тех же
данных. Здесь считаются три вещи, которых в сериях ещё не было:

  2) ХРУПКОСТЬ — какая доля итога сделана лучшими 5 днями и лучшим календарным
     месяцем. Стратегия, у которой полугодовая прибыль сделана за неделю, на графике
     неотличима от ровной;
  3) ВРЕМЯ ПОД ВОДОЙ — доля дней ниже прежнего пика и длина самого долгого такого
     отрезка; сопоставление с глубиной просадки (глубину витрина показывает,
     длительность — нет);
  6) АРИФМЕТИКА РАЗМЕРА ВЫБОРКИ — сколько лет нужно, чтобы Sharpe отличался от нуля,
     и сколько стратегий столько живут.

🔴 ДОЛЯ ИТОГА СЧИТАЕТСЯ В ЛОГАРИФМАХ. Дневные доходности перемножаются, а не
складываются, поэтому «доля итога, сделанная лучшими днями» имеет смысл только как
доля суммы log(1+r): она аддитивна и в сумме даёт ровно итог. Доля считается лишь у
стратегий с положительным итогом — у убыточной «доля итога» не определена (знаменатель
отрицательный, и отношение переворачивается).

🔴 БАЗА. Живые измеримые стратегии Comon: is_live == 1, не меньше 100 торговых дней,
активность (доля торговых дней в ряде) не ниже 10 % — те же 1 262 системы, что в
интерлюдии и в первом разделе заключения.

Ряды: data/profit/<id>.json.gz, поле rValue (дневная доходность).

Запуск: python comon_hidden.py
"""
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md

MIN_TRADE_DAYS = 100
MIN_ACTIVITY = 0.10
TOP_DAYS = 5


def say(s=""):
    print(s, flush=True)


def head(t):
    say()
    say("=" * 96)
    say(t)
    say("=" * 96)


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Прогон читает 1 262 сжатых ряда; величины для картинок кладутся рядом с логом,
# чтобы оформление правилось без повторного чтения файлов.
CACHE = ROOT / "results" / "comon_hidden_charts.npz"


def chart_fragility(sh5, n_days):
    """График 14: какая доля итога сделана пятью лучшими днями."""
    import comon_charts as ch

    f, ax = ch.fig()
    v = 100 * np.asarray(sh5, float)
    lo, hi = 0.0, 600.0
    bins = np.linspace(lo, hi, 41)
    cnt, _ = np.histogram(np.clip(v, lo, hi), bins=bins)
    xs = np.repeat(bins, 2)[1:-1]
    ys = np.repeat(cnt, 2)
    ax.fill_between(xs, 0, ys, color=ch.BLUE, alpha=0.25, lw=0, zorder=3)
    ax.plot(xs, ys, color=ch.BLUE, lw=2.0, zorder=5,
            label=f"{ch.n_(len(v))} стратегий с положительным итогом")
    med, p90 = np.median(v), np.percentile(v, 90)
    over = 100.0 * (v > 100).mean()
    ax.plot([med, med], [0, cnt.max() * 1.06], color=ch.BLUE, lw=1.6, ls="--",
            zorder=6, label=f"медиана {ch.n_(med, 1)} %")
    ax.plot([100, 100], [0, cnt.max() * 1.06], color=ch.ORANGE, lw=1.8, ls="-.",
            zorder=6, label="лучшие пять дней = весь итог (100 %)")
    ax.plot([p90, p90], [0, cnt.max() * 1.06], color=ch.GREY, lw=1.4, ls=":",
            zorder=6, label=f"90-й перцентиль {ch.n_(p90)} %")
    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.55)
    ch.pct_raw(ax, "x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, cnt.max() * 1.30)
    ax.set_xlabel("доля итога, сделанная пятью лучшими днями")
    ax.set_ylabel("стратегий в интервале")
    ax.set_title("У медианной стратегии пять дней дают четыре пятых всего результата")
    ch.note(ax,
            f"База — {ch.n_(len(v))} живых измеримых стратегий с положительным "
            f"итогом из 1 262 (у убыточной доля итога не определена: знаменатель "
            f"отрицательный). Доля считается в логарифмах доходностей; медианная "
            f"стратегия имеет {ch.n_(n_days)} дней истории.",
            f"Значения выше 100 % — не ошибка: у {ch.n_(over)} % стратегий пять "
            f"лучших дней дали больше, чем весь итог, то есть остальная история в "
            f"сумме убыточна. Значения за краем поля сведены в крайний столбец.")
    ch.save(f, 14, "fragility-best-days")
    print(f"    медиана {med:.1f} %, p90 {p90:.1f} %, выше 100 % — {over:.1f} %",
          flush=True)


def chart_underwater(months, idx_win, idx_all):
    """График 28: самый долгий непрерывный отрезок ниже прежнего пика."""
    import comon_charts as ch

    f, ax = ch.fig()
    v = np.asarray(months, float)
    lo, hi = 0.0, 84.0
    bins = np.linspace(lo, hi, 43)
    cnt, _ = np.histogram(np.clip(v, lo, hi), bins=bins)
    xs = np.repeat(bins, 2)[1:-1]
    ys = np.repeat(cnt, 2)
    ax.fill_between(xs, 0, ys, color=ch.BLUE, alpha=0.25, lw=0, zorder=3)
    ax.plot(xs, ys, color=ch.BLUE, lw=2.0, zorder=5,
            label=f"{ch.n_(len(v))} живых измеримых стратегий")
    med, p90 = np.median(v), np.percentile(v, 90)
    top = cnt.max()
    ax.plot([med, med], [0, top * 1.06], color=ch.BLUE, lw=1.6, ls="--", zorder=6,
            label=f"медиана {ch.n_(med, 1)} месяца")
    ax.plot([p90, p90], [0, top * 1.06], color=ch.GREY, lw=1.4, ls=":", zorder=6,
            label=f"90-й перцентиль {ch.n_(p90, 1)} месяца — "
                  f"больше четырёх с половиной лет")
    ax.plot([idx_win, idx_win], [0, top * 1.06], color=ch.GREEN, lw=2.0, ls="-.",
            zorder=6,
            label=f"индекс Мосбиржи, окно витрины (с 12 октября 2022): "
                  f"{ch.n_(idx_win, 1)} месяца")
    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.55)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top * 1.30)
    ax.set_xlabel("самый долгий непрерывный отрезок ниже прежнего пика, месяцев")
    ax.set_ylabel("стратегий в интервале")
    ax.set_title("Глубину ямы витрина показывает, длительность — нет")
    ch.note(ax,
            f"База — {ch.n_(len(v))} живых измеримых стратегий: не меньше 100 "
            f"торговых дней и активность не ниже 10 % дней. Значения за краем поля "
            f"сведены в крайний столбец.",
            f"Медианная стратегия провела в худшем эпизоде почти шестнадцать месяцев "
            f"подряд ниже прежнего пика, каждая десятая — больше четырёх с половиной "
            f"лет. Зелёная черта — тот же счёт по индексу Мосбиржи на окне витрины, "
            f"с 12 октября 2022 года: {ch.n_(idx_win, 1)} месяца. На всей истории "
            f"индекса с 2005 года его худший эпизод длиннее — {ch.n_(idx_all, 1)} "
            f"месяца, но это другое окно, и сравнивать с ним стратегии нельзя.")
    ch.save(f, 28, "underwater-duration")
    print(f"    медиана {med:.1f} мес, p90 {p90:.1f} мес, индекс {idx_win:.1f}",
          flush=True)


def chart_years_needed(years):
    """График 36: сколько лет истории нужно, чтобы отличить Sharpe от нуля."""
    import comon_charts as ch

    f, ax = ch.fig()
    S = np.linspace(0.4, 2.5, 200)
    y2 = 4 * (1 + S * S / 2) / (S * S)
    y3 = 9 * (1 + S * S / 2) / (S * S)
    ax.plot(S, y3, color=ch.ORANGE, lw=2.0, ls="--", zorder=4,
            label="высокая достоверность (t ≥ 3)")
    ax.plot(S, y2, color=ch.BLUE, lw=2.6, zorder=5,
            label="достоверность (t ≥ 2)")

    med_life = float(np.median(years))
    ax.axhline(med_life, color=ch.GREY, lw=1.6, ls=":", zorder=3,
               label=f"медианная живая стратегия живёт {ch.n_(med_life, 2)} года")
    marks = [1.0, 1.5, 2.0]
    my = [4 * (1 + s * s / 2) / (s * s) for s in marks]
    ax.plot(marks, my, marker="o", ms=9, lw=0, color=ch.BLUE, zorder=6)
    ch.value_labels(ax, marks, my,
                    [f"{ch.n_(s, 1)} → {ch.n_(y, 1)} года, таких историй "
                     f"{ch.n_(100*np.mean(np.asarray(years) >= y), 1)} %"
                     for s, y in zip(marks, my)], dist=26)

    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.55)
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(0, 30)
    ax.set_xlabel("истинный Sharpe стратегии")
    ax.set_ylabel("нужная длина истории, лет")
    ax.set_title("Чтобы доказать хороший результат, нужно шесть лет истории")
    ch.note(ax,
            "Из t-статистики Sharpe (Lo, 2002): лет = t² · (1 + S²/2) / S². "
            f"Доли считаны по {ch.n_(len(years))} живым измеримым стратегиям — "
            f"столько из них имеют историю нужной длины.",
            "Кривая гиперболическая: результат тем труднее доказать, чем он скромнее. "
            "Sharpe 0,5 не доказуем в принципе — восемнадцать лет больше, чем "
            "существует сама площадка.")
    ch.save(f, 36, "years-to-prove")
    print("    " + "; ".join(f"S {s:.1f} → {y:.1f} года "
                             f"({100*np.mean(np.asarray(years) >= y):.1f} %)"
                             for s, y in zip(marks, my)), flush=True)


def series(sid):
    """Дневной ряд стратегии: (даты, доходности). Порядок в файле обратный."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None, None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if len(s) < 30:
        return None, None
    s = s[::-1]
    d = np.array([x["date"] for x in s])
    r = np.array([float(x["rValue"] or 0.0) for x in s])
    return d, r


def one(sid):
    d, r = series(sid)
    if r is None:
        return None
    # 🔴 Дневная доходность −1 (полная потеря за день) ломает логарифм. Такие точки
    # в данных есть; клипуем снизу, иначе весь ряд превращается в −inf.
    lg = np.log1p(np.clip(r, -0.999, None))
    tot = lg.sum()
    eq = np.exp(np.cumsum(lg))
    peak = np.maximum.accumulate(eq)
    under = eq < peak * (1 - 1e-12)
    # самый долгий непрерывный отрезок под водой
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    dt = pd.to_datetime(d)
    months = pd.Series(lg).groupby(dt.to_period("M").astype(str).to_numpy()).sum()
    top5 = np.sort(lg)[-TOP_DAYS:].sum()
    out = {"id": sid, "n": len(r), "tot_log": tot,
           "ret_total": float(np.expm1(tot)),
           "ret_ex_top5": float(np.expm1(tot - top5)),
           "top5_log": float(top5),
           "best_month_log": float(months.max()),
           "ret_ex_best_month": float(np.expm1(tot - months.max())),
           "uw_share": float(under.mean()),
           "uw_longest_days": int(longest * (
               (date.fromisoformat(d[-1]) - date.fromisoformat(d[0])).days / max(len(d) - 1, 1))),
           "maxdd": float((eq / peak - 1).min())}
    return out


def draw_from_cache():
    d = np.load(CACHE, allow_pickle=False)
    chart_fragility(d["sh5"], float(d["n_days"]))
    chart_underwater(d["uw_months"], float(d["idx_win"]), float(d["idx_all"]))
    chart_years_needed(d["years"])


def main():
    if ONLY:
        draw_from_cache()
        return
    p = pd.read_csv(DIR / "panel.csv.gz", low_memory=False)
    p["is_live"] = pd.to_numeric(p["is_live"], errors="coerce")
    p["activity"] = p["n_trade"] / p["n_pts"]
    meas = p[(p["is_live"] == 1) & (p["n_trade"] >= MIN_TRADE_DAYS)
             & (p["activity"] >= MIN_ACTIVITY)]
    say(f"база: {len(meas):,} живых измеримых стратегий (>=100 торговых дней, активность >=10 %)")

    rows = [x for x in (one(int(i)) for i in meas["id"]) if x is not None]
    d = pd.DataFrame(rows).merge(
        meas[["id", "years", "sharpe", "cagr", "maxdd", "followers"]].rename(
            columns={"maxdd": "maxdd_panel"}), on="id", how="left")
    say(f"ряды прочитаны у {len(d):,} из {len(meas):,}")
    say(f"контроль просадки против панели: медиана расхождения "
        f"{(d['maxdd'] - d['maxdd_panel']).abs().median():.2e}")

    # ------------------------------------------------------------ расчёт 2 ----
    head("РАСЧЁТ 2. ХРУПКОСТЬ: сколько итога сделано лучшими днями")
    say("Доля считается в логарифмах (они складываются, а доходности перемножаются) и")
    say("только у стратегий с положительным итогом — у убыточной доля итога не")
    say("определена. Второй способ, годный для всех: пересчитать итог БЕЗ лучших дней.")
    say()
    pos = d[d["tot_log"] > 0].copy()
    pos["sh5"] = pos["top5_log"] / pos["tot_log"]
    pos["shm"] = pos["best_month_log"] / pos["tot_log"]
    say(f'стратегий с положительным итогом: {len(pos):,} из {len(d):,} '
        f'({100 * len(pos) / len(d):.1f} %)')
    say()
    hdr = f'{"доля итога, сделанная...":<34}{"медиана":>10}{"p25":>9}{"p75":>9}{"p90":>9}'
    say(hdr)
    say("-" * len(hdr))
    for col, lab in (("sh5", f"лучшими {TOP_DAYS} днями"),
                     ("shm", "лучшим календарным месяцем")):
        v = pos[col]
        say(f'{lab:<34}{f"{100 * v.median():.1f} %":>10}{f"{100 * v.quantile(.25):.1f} %":>9}'
            f'{f"{100 * v.quantile(.75):.1f} %":>9}{f"{100 * v.quantile(.9):.1f} %":>9}')
    say()
    say(f'   у медианной стратегии {TOP_DAYS} лучших дней из '
        f'{int(pos["n"].median()):,} дают {100 * pos["sh5"].median():.0f} % всего итога')
    say(f'   у каждой четвёртой (p75) — {100 * pos["sh5"].quantile(.75):.0f} % и больше')
    say(f'   у каждой десятой (p90)   — {100 * pos["sh5"].quantile(.9):.0f} % и больше')
    say()
    say("Проверка на всей базе (в том числе убыточных) — что остаётся без лучших дней:")
    hdr = f'{"итог":<40}{"медиана, %":>13}{"доля стратегий в плюсе":>26}'
    say(hdr)
    say("-" * len(hdr))
    for col, lab in (("ret_total", "как есть"),
                     ("ret_ex_top5", f"без {TOP_DAYS} лучших дней"),
                     ("ret_ex_best_month", "без лучшего месяца")):
        say(f'{lab:<40}{100 * d[col].median():>13.1f}'
            f'{f"{100 * (d[col] > 0).mean():.1f} %":>26}')
    say()
    flip5 = ((d["ret_total"] > 0) & (d["ret_ex_top5"] <= 0)).mean()
    flipm = ((d["ret_total"] > 0) & (d["ret_ex_best_month"] <= 0)).mean()
    say(f'🔴 Изъятие {TOP_DAYS} лучших дней переводит из прибыли в убыток '
        f'{100 * flip5:.1f} % всех стратегий базы')
    say(f'   ({100 * flip5 / (d["ret_total"] > 0).mean():.1f} % от прибыльных);')
    say(f'   изъятие лучшего месяца — {100 * flipm:.1f} % базы '
        f'({100 * flipm / (d["ret_total"] > 0).mean():.1f} % прибыльных).')
    say()
    say("Связь хрупкости с тем, что видно на витрине:")
    for col, lab in (("sharpe", "Sharpe"), ("years", "длина истории"),
                     ("followers", "число подписчиков")):
        v = pos[["sh5", col]].dropna()
        say(f'   ро(доля лучших {TOP_DAYS} дней, {lab}) = '
            f'{v["sh5"].corr(v[col], method="spearman"):+.3f}')
    say("   (ранговая корреляция; хрупкость сама по себе на витрине не видна ничем)")

    # ------------------------------------------------------------ расчёт 3 ----
    head("РАСЧЁТ 3. ВРЕМЯ ПОД ВОДОЙ: длительность, которую витрина не показывает")
    say("Под водой = стоимость ниже прежнего максимума. Глубину ямы витрина")
    say("показывает (поле максимальной просадки), длительность — нет.")
    say()
    hdr = f'{"величина":<40}{"медиана":>12}{"p25":>10}{"p75":>10}{"p90":>10}'
    say(hdr)
    say("-" * len(hdr))
    say(f'{"доля дней ниже прежнего пика":<40}'
        f'{f"{100 * d['uw_share'].median():.1f} %":>12}{f"{100 * d['uw_share'].quantile(.25):.1f} %":>10}'
        f'{f"{100 * d['uw_share'].quantile(.75):.1f} %":>10}{f"{100 * d['uw_share'].quantile(.9):.1f} %":>10}')
    v = d["uw_longest_days"]
    say(f'{"самый долгий отрезок под водой, дней":<40}{v.median():>12,.0f}{v.quantile(.25):>10,.0f}'
        f'{v.quantile(.75):>10,.0f}{v.quantile(.9):>10,.0f}')
    say(f'{"  то же в месяцах":<40}{v.median() / 30.4:>12,.1f}{v.quantile(.25) / 30.4:>10,.1f}'
        f'{v.quantile(.75) / 30.4:>10,.1f}{v.quantile(.9) / 30.4:>10,.1f}')
    say()
    say("Как это соотносится с глубиной (её витрина показывает):")
    hdr = f'{"группа по глубине просадки":<30}{"систем":>9}{"доля дней под водой":>22}{"самый долгий, мес":>20}'
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in ((-1.01, -0.5, "глубже -50 %"), (-0.5, -0.3, "от -50 до -30 %"),
                        (-0.3, -0.15, "от -30 до -15 %"), (-0.15, 0.01, "мельче -15 %")):
        g = d[(d["maxdd"] > lo) & (d["maxdd"] <= hi)]
        if not len(g):
            continue
        say(f'{lab:<30}{len(g):>9,}{f"{100 * g["uw_share"].median():.1f} %":>22}'
            f'{g["uw_longest_days"].median() / 30.4:>20,.1f}')
    say()
    say("🔴 ЭТАЛОН, без которого эти числа читаются неверно. Быть ниже прежнего пика —")
    say("   нормальное состояние ЛЮБОЙ растущей кривой: новый максимум по определению")
    say("   редкое событие. Считаем то же самое по индексу Мосбиржи на том же окне:")
    idx_longest = {}
    idx = json.load(open(DIR / "idx_IMOEX.json"))
    dts = sorted(idx)
    px = np.array([float(idx[k]) for k in dts])
    for lab, mask in (("вся история индекса с 2005 года", np.ones(len(px), bool)),
                      ("окно витрины, с 2022-10-12",
                       np.array([k >= "2022-10-12" for k in dts]))):
        e = px[mask]
        peak = np.maximum.accumulate(e)
        under = e < peak * (1 - 1e-12)
        longest, cur = 0, 0
        for u in under:
            cur = cur + 1 if u else 0
            longest = max(longest, cur)
        idx_longest[lab] = longest / 21
        say(f'   {lab:<38}доля дней под водой {100 * under.mean():.1f} %, '
            f'самый долгий отрезок {longest / 21:.1f} мес')
    say("   То есть высокая доля дней под водой — свойство рынка, а не автоследования.")
    say("   Информативна ДЛИТЕЛЬНОСТЬ худшего отрезка, и вот она у стратегий велика.")
    say()
    rho = d[["uw_share", "maxdd"]].dropna()
    say(f'ро(доля дней под водой, глубина просадки) = '
        f'{rho["uw_share"].corr(rho["maxdd"], method="spearman"):+.3f} '
        f'(знак: чем глубже, тем дольше)')
    say()
    say("🔴 Ограничение: считаем по ЖИВЫМ стратегиям — это те, кто из ямы так или иначе")
    say("   вышел или ещё в ней сидит. Утверждать, что подписчик уходит по длительности,")
    say("   а не по глубине, мы НЕ можем: истории подписок в данных нет (глава 5).")

    # ------------------------------------------------------------ расчёт 6 ----
    head("РАСЧЁТ 6. СКОЛЬКО НАДО НАБЛЮДЕНИЙ, ЧТОБЫ ОТЛИЧИТЬ УМЕНИЕ ОТ СЛУЧАЯ")
    say("t-статистика Sharpe: t = S*sqrt(лет)/sqrt(1 + S^2/2) (Lo, 2002). Отсюда")
    say("нужная длина истории: лет = t^2 * (1 + S^2/2) / S^2.")
    say()
    say("🔴 Знаменатель обязателен. Без него (наивное t = S*sqrt(лет)) для Sharpe 1.0")
    say("   получается 4 года вместо 6 — то есть требование к длине занижается в полтора")
    say("   раза, а доля «достоверных» стратегий завышается вдвое (первый раздел).")
    say()
    hdr = (f'{"истинный Sharpe":<18}{"лет для t>=2":>14}{"лет для t>=3":>14}'
           f'{"живых столько (t>=2)":>22}{"их доля":>10}')
    say(hdr)
    say("-" * len(hdr))
    for S in (0.5, 0.75, 1.0, 1.5, 2.0):
        y2 = 4 * (1 + S * S / 2) / (S * S)
        y3 = 9 * (1 + S * S / 2) / (S * S)
        n = int((d["years"] >= y2).sum())
        say(f'{S:<18.2f}{y2:>14.1f}{y3:>14.1f}{n:>22,}{f"{100 * n / len(d):.1f} %":>10}')
    say()
    say(f'Для сравнения: медианная длина истории живой измеримой стратегии '
        f'{d["years"].median():.2f} года,')
    say(f'а всей популяции — 173 дня (глава 2). Стратегий с историей от 6 лет среди')
    say(f'живых измеримых {int((d["years"] >= 6).sum()):,} из {len(d):,} '
        f'({100 * (d["years"] >= 6).mean():.1f} %).')
    say()
    say("То же с другой стороны — сколько стратегий витрины вообще имеют длину,")
    say("на которой их собственный показанный Sharpe отличим от нуля:")
    say("   см. первый раздел заключения: t >= 2 у 9.0 % живых измеримых (113 из 1 262).")

    if CHARTS:
        head("ГРАФИКИ 14, 28, 36")
        np.savez(CACHE, sh5=pos["sh5"].to_numpy(float),
                 n_days=float(pos["n"].median()),
                 uw_months=(d["uw_longest_days"] / 30.4).to_numpy(float),
                 idx_win=idx_longest["окно витрины, с 2022-10-12"],
                 idx_all=idx_longest["вся история индекса с 2005 года"],
                 years=d["years"].to_numpy(float))
        draw_from_cache()

    say()
    say("готово")


if __name__ == "__main__":
    main()
