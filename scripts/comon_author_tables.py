"""comon_author_tables.py — поимённые таблицы авторов Блока 5 (порождение вместо ручной вставки).

Зачем: таблицы §5.1 «все авторы, создавшие больше 20 стратегий» и §5.4 «сводка по
пятёрке» были вставлены в comon-block5-multiplicity.md РУКАМИ — значения верны
(пересчёт 2026-08-07 совпал построчно), но воспроизводимости не было: перепроверить
их можно было только вручную. Этот скрипт печатает те же таблицы из панели.

🔴 Три базы, которые здесь легко перепутать (в блоке на этом уже спотыкались):
  • «всего систем»  — все записи популяции, включая пустые заготовки;
  • «с рядом»       — есть хотя бы ОДНА точка ряда (n_pts > 0), это 18 354 системы
                      во всей популяции. НЕ путать с «ряд от 30 точек» (11 657) —
                      так подписана строка сводной таблицы §5.1, потому что Sharpe
                      на более коротком ряде не считается;
  • «измеримых»     — фильтр измеримости: >= 100 торговых дней И активность >= 10 %.

🔴 Отсечка «больше 20» (строго), как в поимённой таблице блока. Отсечка «20 и больше»
даёт другой состав — на этом разошлись число 12 и число 11 в соседних абзацах блока.

Sharpe — арифметический (sharpe_ar из sharpe.csv.gz), та же конвенция, что в блоке.
Служебный аккаунт площадки owner 2215 «comon» исключён.

Текстовый вывод. Запуск: python comon_author_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

MIN_TRADE_DAYS = 100          # фильтр измеримости — тот же, что в Блоках 2-4
MIN_ACTIVITY = 0.10
SERVICE_OWNERS = {2215}       # служебный аккаунт площадки, не автор
CUTOFF = 20                   # «больше 20» — строго, как в поимённой таблице блока
# 🔴 Пятёрка задаётся owner_id, а НЕ именем: имена авторов на площадке не уникальны
# (например, «Сергей» — это и owner 259732, и owner 388224), отбор по имени падает.
FIVE = {109657: "Finam InvestLAB", 297246: "Moex15", 178285: "camry759",
        45461: "howtotrade", 324225: "TradeCenter"}


def say(s=""):
    print(s, flush=True)


CHARTS = "--charts" in sys.argv


def chart_attempt_funnel(steps):
    """График 34: воронка попыток — из трёх созданных стратегий мерить можно одну."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=980, bottom=0.28)
    f.subplots_adjust(left=0.26)
    n0 = steps[0]["n"]
    ys = list(range(len(steps)))[::-1]
    cols = [ch.SEQ[2], ch.SEQ[4], ch.SEQ[6], ch.SEQ[8]]
    for y, st, c in zip(ys, steps, cols):
        ax.barh(y, st["n"], height=0.62, color=c, zorder=3)
        ax.annotate(f"{ch.n_(st['n'])} — {ch.n_(100*st['n']/n0, 1)} % созданного\n"
                    f"{ch.n_(st['authors'])} авторов",
                    xy=(st["n"], y), xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color=ch.INK, ha="left", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([st["lab"] for st in steps], fontsize=11)
    ax.set_xlim(0, n0 * 1.55)
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ch.num(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("стратегий, штук")
    # заголовок центрован по панели, а она сдвинута вправо подписями ступеней —
    # длинная фраза обрезается краем файла, поэтому она короткая
    ax.set_title("Из трёх созданных стратегий измерить можно одну")
    ch.note(ax,
            "База — вся популяция без служебного аккаунта площадки. Фильтр "
            "измеримости тот же, что в главе 3: не меньше 100 торговых дней и "
            "активность не ниже 10 % дней.",
            "Медиана на каждой ступени одна и та же — одна стратегия на автора. "
            "Отсев идёт не по смертности, а раньше: большинство записей никогда "
            "не становятся работающими системами.")
    ch.save(f, 34, "attempt-funnel")
    print("    " + " → ".join(f"{st['n']} ({st['authors']} авт.)" for st in steps),
          flush=True)


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False)
    sh = pd.read_csv(DIR / "sharpe.csv.gz")[["id", "sharpe_ar", "activity", "n_trade"]]
    p = panel.merge(sh.rename(columns={"n_trade": "n_trade_sh"}), on="id", how="left")
    p = p[p["owner_id"].notna() & ~p["owner_id"].isin(SERVICE_OWNERS)]

    p["ser"] = p["n_pts"] > 0                       # хотя бы одна точка ряда
    p["trd"] = p["n_trade"] > 0                     # хотя бы один торговый день
    p["msr"] = (p["n_trade"] >= MIN_TRADE_DAYS) & (p["activity"] >= MIN_ACTIVITY)
    p["alive"] = p["is_live"] == 1
    p["month"] = p["created_at"].astype(str).str[:7]

    g = p.groupby("owner_id")
    a = pd.DataFrame({
        "author": g["author"].first(),
        "total": g.size(),
        "ser": g["ser"].sum(),
        "trd": g["trd"].sum(),
        "msr": g["msr"].sum(),
        "alive": g["alive"].sum(),
        "first": g["month"].min(),
        "last": g["month"].max(),
    })
    m = p[p["msr"]]
    a["s_max"] = m.groupby("owner_id")["sharpe_ar"].max()
    a["s_med"] = m.groupby("owner_id")["sharpe_ar"].median()
    a["cagr_med"] = m.groupby("owner_id")["cagr"].median()
    a["vol_med"] = m.groupby("owner_id")["vol"].median()
    a["dd_med"] = m.groupby("owner_id")["maxdd"].median()
    a["prof"] = m.groupby("owner_id")["ret_total"].apply(lambda s: (s > 0).mean())
    a["subs"] = p[p["alive"]].groupby("owner_id")["followers"].sum()

    # ── воронка попыток (таблица 4.1 главы 4) ────────────────────────────────
    say("=" * 104)
    say("4.1  ВОРОНКА ПОПЫТОК: от созданной записи до измеримой стратегии")
    say("=" * 104)
    hdr = (f'{"ступень":<44} {"стратегий":>10} {"авторов":>9} '
           f'{"медиана на автора":>18} {"максимум":>9}')
    say(hdr)
    say("-" * len(hdr))
    funnel = []
    for lab, m in (("создано всего", pd.Series(True, index=p.index)),
                   ("есть хотя бы одна точка ряда доходности", p["ser"]),
                   ("торговали хотя бы один день", p["trd"]),
                   ("прошли фильтр измеримости", p["msr"])):
        d = p[m]
        cnt = d.groupby("owner_id").size()
        say(f'{lab:<44} {len(d):>10,} {d["owner_id"].nunique():>9,} '
            f'{cnt.median():>18.0f} {int(cnt.max()):>9}')
        funnel.append({"lab": lab.replace("есть хотя бы одна точку", "")
                       .replace("есть хотя бы одна точка ряда доходности",
                                "есть хотя бы одна точка\nряда доходности")
                       .replace("торговали хотя бы один день",
                                "торговали хотя бы\nодин день")
                       .replace("прошли фильтр измеримости",
                                "прошли фильтр\nизмеримости"),
                       "n": len(d), "authors": int(d["owner_id"].nunique())})
    say()
    if CHARTS:
        say("── график 34 ────────────────────────────────────────────────────")
        chart_attempt_funnel(funnel)
        say()

    big = a[a["total"] > CUTOFF].sort_values("total", ascending=False)

    say("=" * 104)
    say(f"5.1  ВСЕ АВТОРЫ, СОЗДАВШИЕ БОЛЬШЕ {CUTOFF} СТРАТЕГИЙ (служебный аккаунт исключён)")
    say("=" * 104)
    say(f"авторов: {len(big):,} · систем у них: {int(big['total'].sum()):,}"
        f" · с рядом: {int(big['ser'].sum()):,} · измеримых: {int(big['msr'].sum()):,}"
        f" ({big['msr'].sum() / big['total'].sum() * 100:.0f} %)"
        f" · живых: {int(big['alive'].sum()):,}"
        f" ({big['alive'].sum() / big['total'].sum() * 100:.0f} %)")
    say(f"без единой измеримой системы: {int((big['msr'] == 0).sum())} авторов"
        f" (при {int(big.loc[big['msr'] == 0, 'total'].min())}"
        f"-{int(big.loc[big['msr'] == 0, 'total'].max())} созданных)")
    say()
    hdr = (f'{"owner":>8} {"автор":<22} {"всего":>7} {"с рядом":>8} {"измер.":>7}'
           f' {"живых":>6} {"S лучшей":>9} {"S медиан.":>10}  период создания')
    say(hdr)
    say("-" * len(hdr))
    for oid, r in big.iterrows():
        sx = f'{r["s_max"]:.2f}' if pd.notna(r["s_max"]) else "—"
        sm = f'{r["s_med"]:.2f}' if pd.notna(r["s_med"]) else "—"
        say(f'{int(oid):>8} {str(r["author"])[:22]:<22} {int(r["total"]):>7}'
            f' {int(r["ser"]):>8} {int(r["msr"]):>7} {int(r["alive"]):>6}'
            f' {sx:>9} {sm:>10}  {r["first"]} … {r["last"]}')

    say()
    say("=" * 104)
    say("5.4  СВОДКА ПО ПЯТЁРКЕ")
    say("=" * 104)
    say("🔴 Пятёрка отобрана по числу ИЗМЕРИМЫХ систем — тому самому признаку, который")
    say("в ней потом и меряется. Раздел 4.5 главы 4 по этой причине переписывается;")
    say("таблица оставлена для сверки с блоком, а не как самостоятельный результат.")
    say()
    f5 = a.reindex(list(FIVE))
    f5.index = [FIVE[i] for i in f5.index]
    hdr = (f'{"автор":<18} {"всего":>7} {"с рядом":>8} {"торговали":>10}'
           f' {"измер.":>7} {"живых":>6}  период создания')
    say(hdr)
    say("-" * len(hdr))
    for name, r in f5.iterrows():
        say(f'{name:<18} {int(r["total"]):>7} {int(r["ser"]):>8} {int(r["trd"]):>10}'
            f' {int(r["msr"]):>7} {int(r["alive"]):>6}  {r["first"]} … {r["last"]}')
    say()
    hdr = (f'{"автор":<18} {"S лучшей":>9} {"S медиан.":>10} {"CAGR мед.":>10}'
           f' {"вола мед.":>10} {"maxDD мед.":>11} {"прибыльных":>11} {"подписчиков":>12}')
    say(hdr)
    say("-" * len(hdr))
    for name, r in f5.iterrows():
        say(f'{name:<18} {r["s_max"]:>9.2f} {r["s_med"]:>10.2f}'
            f' {r["cagr_med"] * 100:>9.1f}% {r["vol_med"] * 100:>9.1f}%'
            f' {r["dd_med"] * 100:>10.1f}% {r["prof"] * 100:>10.1f}%'
            f' {int(r["subs"]):>12,}')
    say()
    say("⚠️ followers у архивных систем обнулён площадкой — подписчики считаются"
        " только по живым системам автора.")
    say()
    say("готово")


if __name__ == "__main__":
    main()
