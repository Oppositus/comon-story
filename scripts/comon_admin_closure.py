"""comon_admin_closure.py — сколько стратегий закрыла АДМИНИСТРАЦИЯ, а не автор.

Повод: правила площадки https://docs.comon.ru/trader-information/restrictions-and-requirements/
дают закрытый список причин, по которым администрация сама переносит стратегию в архив
(с отключением подписчиков и блокировкой учётной записи автора):

  ① полный вывод денежных средств со счёта стратегии;
  ② оценка клиентского счёта в течение месяца не превышает 30 000 ₽;
  ③ доходность не меняется на протяжении трёх месяцев или более;
  ④ доходность стратегии составляет менее −95 %;
  ⑤ по стратегии нет никаких операций на протяжении трёх месяцев или более.

Почему это важно для Блока 1: класс «заброшена» там трактовался как «автор потерял
интерес». Критерии ③ и ⑤ означают, что тот же самый признак (нет сделок 90+ дней) —
это ТРИГГЕР ПРИНУДИТЕЛЬНОГО ЗАКРЫТИЯ. То есть архивация могла быть не решением автора,
а действием площадки — и тогда это смерть не стратегии, а учётной записи.

Что измеримо по данным:
  ③⑤ — ПРЯМО: разрыв между последним торговым днём и archivedAt ≥ 90 дней;
  ④  — ПРЯМО: итог жизни ≤ −95 % (порог правил — от старта, не от пика);
  ②  — ОЦЕНКА СВЕРХУ через minSum·(1+итог) < 30 000 ₽: minSum не равен счёту автора,
        но связан с ним (это минимум, который автор объявляет достаточным);
  ①  — не наблюдаем напрямую; проявляется как обрыв сделок, то есть попадает в ③⑤.

Дополнительно: правила грозят блокировкой аккаунта за «многократные попытки создания
стратегий с короткими сроками существования» — это проверяется по ownerId.

Текстовый вывод. Запуск: python comon_admin_closure.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comon_survival import km, km_at, km_median                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"
LOG = ROOT / "results" / "comon_admin_closure.log"

GAP_DAYS = 90          # «три месяца» из критериев ③⑤
RUIN = -0.95           # порог критерия ④
ACCOUNT_MIN = 30_000   # порог критерия ②
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


CHARTS = "--charts" in sys.argv


def chart_three_curves(curves):
    """График 18: кривая дожития не держится на мусоре — три выборки, одна форма."""
    import comon_charts as ch

    f, ax = ch.fig()
    styles = ((ch.BLUE, "-", 2.8), (ch.ORANGE, "--", 2.2), (ch.GREEN, "-.", 2.2))
    xmax = 8.0
    for (lab, n, med, ti, S), (color, ls, lw) in zip(curves, styles):
        x = np.concatenate([[0.0], np.asarray(ti, float) / 365.25])
        y = np.concatenate([[1.0], np.asarray(S, float)])
        m = x <= xmax
        ax.step(x[m], y[m], where="post", color=color, ls=ls, lw=lw, zorder=4,
                label=f"{lab} ({ch.n_(n)}): медиана {ch.n_(med)} дней")
    ax.axhline(0.5, color=ch.PALE, lw=1.4, ls=":", zorder=2)
    ax.annotate("половина популяции", xy=(xmax, 0.5), xytext=(-6, 8),
                textcoords="offset points", fontsize=10, color=ch.GREY,
                ha="right", va="bottom")
    ax.legend(loc="upper right", handlelength=3.0, labelspacing=0.6)
    ch.pct(ax)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_xlim(-0.15, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("возраст стратегии, лет")
    ax.set_ylabel("доля доживших")
    ax.set_title("Цифры сдвигаются, картина остаётся")
    ch.note(ax,
            "Оценка Каплана—Мейера по трём выборкам: вся популяция торговавших "
            "стратегий; она же без партии служебного аккаунта площадки, куда в "
            "2014 году разом ушли 4 058 отвязанных заготовок; и она же без всех "
            "авторов, у которых 20 и более стратегий.",
            "Исключение мусора поднимает кривую, но не меняет ни её формы, ни "
            "порядка величин: к пяти годам доживает от 6,9 до 8,9 % при любой из "
            "трёх границ выборки. Ось обрезана на восьми годах.")
    ch.save(f, 18, "survival-robustness")
    print("    " + "; ".join(f"{lab}: n={n}, медиана {med:.0f} дн"
                             for lab, n, med, _t, _s in curves), flush=True)


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    deaths = pd.read_csv(DIR / "deaths.csv.gz", index_col="id", low_memory=False)
    df = panel.join(deaths[["mech", "fin"]], how="left")
    dead = df[df["is_live"] == False].copy()                            # noqa: E712
    n = len(dead)
    say(f"архивных систем: {n}")
    say(f"пороги правил: тишина ≥ {GAP_DAYS} дн · итог ≤ {100*RUIN:.0f} % · "
        f"счёт < {ACCOUNT_MIN:,} ₽")

    # ── признаки критериев ───────────────────────────────────────────────────
    started = dead["n_trade"].fillna(0) > 0
    gap = dead["death_gap_trade"]                       # archivedAt − последняя сделка
    life = (pd.to_datetime(dead["archived_at"]) - pd.to_datetime(dead["created_at"])).dt.days

    c35 = (gap >= GAP_DAYS) & started                   # ③⑤ тишина у ТОРГОВАВШЕЙ системы
    c35_never = (~started) & (life >= GAP_DAYS)         # ③⑤ вообще ни одной операции
    c4 = dead["ret_total"] <= RUIN                      # ④ итог ≤ −95 %
    acct = dead["min_sum"] * (1.0 + dead["ret_total"].fillna(0))
    c2 = (acct < ACCOUNT_MIN) & started                 # ② оценка счёта (прокси)

    say()
    say("=" * 96)
    say("КАЖДЫЙ КРИТЕРИЙ ПО ОТДЕЛЬНОСТИ (пересечения не вычтены)")
    say("=" * 96)
    say(f'{"критерий":<58} {"систем":>8} {"доля архива":>13}')
    say("-" * 96)
    for lab, m in (
        ("③⑤ нет сделок ≥ 3 мес до архивации (торговавшие)", c35),
        ("③⑤ ни одной операции за всю жизнь, прожила ≥ 3 мес", c35_never),
        ("④ итог ≤ −95 %", c4),
        ("② оценка счёта < 30 000 ₽ — minSum·(1+итог), ОЦЕНКА СВЕРХУ", c2),
    ):
        say(f'{lab:<58} {int(m.sum()):>8,} {100*m.mean():>12.1f}%')

    # ── пересечения и итоговые оценки ────────────────────────────────────────
    hard = (c35 | c35_never | c4).fillna(False)         # то, что видно ПРЯМО
    soft = (hard | c2).fillna(False)                    # плюс оценка по счёту
    say()
    say("=" * 96)
    say("СВОДНАЯ ОЦЕНКА: сколько закрыто администрацией")
    say("=" * 96)
    say(f'{"оценка":<52} {"систем":>8} {"доля архива":>13}')
    say("-" * 96)
    say(f'{"НИЖНЯЯ  — только прямо наблюдаемые ③⑤④":<52} '
        f'{int(hard.sum()):>8,} {100*hard.mean():>12.1f}%')
    say(f'{"ВЕРХНЯЯ — плюс оценка критерия ② по minSum":<52} '
        f'{int(soft.sum()):>8,} {100*soft.mean():>12.1f}%')
    say(f'{"НЕ подпадают ни под один наблюдаемый критерий":<52} '
        f'{int((~soft).sum()):>8,} {100*(~soft).mean():>12.1f}%')

    say()
    say("Пересечения прямых критериев (систем):")
    say(f'  только тишина ③⑤ ......... {int(((c35 | c35_never) & ~c4).sum()):>7,}')
    say(f'  только итог ≤ −95 % ④ .... {int((c4 & ~(c35 | c35_never)).sum()):>7,}')
    say(f'  и то, и другое ........... {int((c4 & (c35 | c35_never)).sum()):>7,}')

    # ── как это меняет классы Блока 1 ────────────────────────────────────────
    say()
    say("=" * 96)
    say("ЧТО ЭТО ДЕЛАЕТ С КЛАССАМИ БЛОКА 1")
    say("=" * 96)
    say(f'{"класс механизма (Блок 1)":<26} {"всего":>8} {"под ③⑤④":>10} {"доля":>8} '
        f'{"+ ② сверху":>12} {"доля":>8}')
    say("-" * 96)
    for m in ("не стартовала", "заброшена", "смерть у пика", "обычная просадка",
              "деградация", "недоказано", "недостаточно истории"):
        g = dead["mech"] == m
        if not g.sum():
            continue
        say(f'{m:<26} {int(g.sum()):>8,} {int((g & hard).sum()):>10,} '
            f'{100*(hard[g]).mean():>7.1f}% {int((g & soft).sum()):>12,} '
            f'{100*(soft[g]).mean():>7.1f}%')

    say()
    say("=" * 96)
    say("ПЕРЕСМОТР ПРИЧИН СМЕРТИ: три стороны вместо двух")
    say("=" * 96)
    fin = dead["fin"].fillna("нет данных")
    mech = dead["mech"].fillna("прочее")
    # приоритет: принудительное закрытие -> рыночная причина -> решение автора
    side = np.where(
        hard, "администрация (наблюдаемо)",
        np.where(fin == "разорение", "рынок: разорение",
                 np.where(mech == "деградация", "рынок: деградация",
                          np.where(fin == "убыток", "рынок: убыток",
                                   np.where(fin == "прибыль", "автор: ушёл в прибыли",
                                            "автор: прочее")))))
    say(f'{"сторона":<30} {"систем":>8} {"доля архива":>13} {"медиана итога":>15}')
    say("-" * 96)
    for s in ("администрация (наблюдаемо)", "рынок: разорение", "рынок: деградация",
              "рынок: убыток", "автор: ушёл в прибыли", "автор: прочее"):
        m = side == s
        if not m.sum():
            continue
        med = dead.loc[m, "ret_total"].median()
        say(f'{s:<30} {int(m.sum()):>8,} {100*m.mean():>12.1f}% '
            f'{100*med if pd.notna(med) else float("nan"):>14.1f}%')
    say("-" * 96)
    adm = (side == "администрация (наблюдаемо)").mean()
    mkt = np.isin(side, ["рынок: разорение", "рынок: деградация", "рынок: убыток"]).mean()
    aut = np.isin(side, ["автор: ушёл в прибыли", "автор: прочее"]).mean()
    say(f'ИТОГО: администрация {100*adm:.1f} % · рынок {100*mkt:.1f} % · '
        f'автор {100*aut:.1f} %')

    # ── прямой маркер: отвязка стратегии от аккаунта автора ──────────────────
    say()
    say("=" * 96)
    say("ПРЯМОЙ МАРКЕР: служебный аккаунт `comon` (owner_id 2215)")
    say("=" * 96)
    say("Правила: администрация переносит стратегию в архив «в том числе ПУТЁМ ОТКЛЮЧЕНИЯ")
    say("СЧЁТА СТРАТЕГИИ ОТ АККАУНТА» с последующей блокировкой учётной записи автора.")
    say("Отвязанные стратегии числятся за служебным аккаунтом — это наблюдаемый след.")
    say()
    cm = df["owner_id"] == 2215
    c, o = df[cm], df[(~cm) & (df["is_live"] == False)]                 # noqa: E712
    say(f'систем за аккаунтом `comon`: {len(c):,} — {100*len(c)/len(dead):.1f} % архива, '
        f'живых среди них {int((c["is_live"] == True).sum())}')          # noqa: E712
    say()
    say(f'{"признак":<34} {"comon":>12} {"прочие архивные":>17}')
    say("-" * 66)
    for lab, f in (("без единой сделки, %", lambda x: 100*(x["n_trade"].fillna(0) == 0).mean()),
                   ("медиана торговых дней", lambda x: x["n_trade"].median()),
                   ("медиана итога, %", lambda x: 100*x["ret_total"].median()),
                   ("итог ≤ −95 %, %", lambda x: 100*(x["ret_total"] <= RUIN).mean()),
                   ("медиана minSum, ₽", lambda x: x["min_sum"].median()),
                   ("подписчиков всего", lambda x: x["followers"].fillna(0).sum()),
                   ("тишина ≥ 90 дн, %", lambda x: 100*(x["death_gap_trade"] >= GAP_DAYS).mean())):
        say(f'{lab:<34} {f(c):>12,.1f} {f(o):>17,.1f}')
    say()
    yr_c = pd.to_datetime(c["archived_at"]).dt.year.value_counts().sort_index()
    yr_o = pd.to_datetime(o["archived_at"]).dt.year.value_counts().sort_index()
    say("Архивации по годам — это РАЗОВОЕ СОБЫТИЕ, а не текущая практика:")
    say(f'{"год":<8} {"архив comon":>13} {"архив прочие":>14} {"доля comon":>12}')
    for y in sorted(set(yr_c.index) | set(yr_o.index)):
        a, b = int(yr_c.get(y, 0)), int(yr_o.get(y, 0))
        if a == 0 and y not in (2013, 2014, 2015):
            continue
        say(f'{int(y):<8} {a:>13,} {b:>14,} {100*a/(a+b) if a+b else 0:>11.1f}%')
    say(f'создано: 2012 — {int((pd.to_datetime(c["created_at"]).dt.year == 2012).sum()):,}, '
        f'2013 — {int((pd.to_datetime(c["created_at"]).dt.year == 2013).sum()):,}, '
        f'2014 — {int((pd.to_datetime(c["created_at"]).dt.year == 2014).sum()):,}')
    say("🔴 Партия целиком создана в 2012–2014 и закрыта в 2014–2016. Это единовременная")
    say("чистка, а не постоянный поток санкций. Когорта 2013 в Блоке 1 искажена именно ею.")

    # ── устойчивость Блока 1 к этой партии ───────────────────────────────────
    say()
    say("=" * 96)
    say("УСТОЙЧИВОСТЬ БЛОКА 1: та же выживаемость БЕЗ партии 2014 года")
    say("=" * 96)
    born = pd.to_datetime(df["created_at"], errors="coerce")
    last_tr = pd.to_datetime(df["last_trade"], errors="coerce")
    alive = df["is_live"] == True                                       # noqa: E712
    started = df["n_trade"].fillna(0) > 0
    end = last_tr.where(~alive, pd.Timestamp("2026-08-05"))
    life = (end - born).dt.days
    base = started & born.notna() & life.notna() & (life >= 0)
    say(f'{"выборка":<28} {"систем":>8} {"медиана жизни":>15} {"1 год":>9} '
        f'{"3 года":>9} {"5 лет":>9}')
    say("-" * 96)
    curves = []
    for lab, m in (("вся популяция (Блок 1)", base),
                   ("без партии `comon`", base & ~cm),
                   ("без всех авторов с 20+", base & ~df["owner_id"].isin(
                       df.groupby("owner_id").size().pipe(lambda s: s[s >= 20]).index))):
        t_ = life[m].to_numpy(float)
        e_ = (~alive[m]).to_numpy(int)
        ti_, S_, se_ = km(t_, e_)
        vals = [km_at(ti_, S_, se_, y * 365.25)[0] for y in (1, 3, 5)]
        say(f'{lab:<28} {int(m.sum()):>8,} {km_median(ti_, S_):>15.0f} '
            + "".join(f'{100*v:>8.1f}%' for v in vals))
        curves.append((lab.replace(" (Блок 1)", "").replace("`comon`",
                                   "служебного аккаунта"),
                       int(m.sum()), km_median(ti_, S_), ti_, S_))
    if CHARTS:
        say("── график 18 ────────────────────────────────────────────────────")
        chart_three_curves(curves)
    say("Вывод устойчив: исключение партии удлиняет медиану жизни, но порядок")
    say("величин и форма кривой сохраняются.")

    # ── авторы-рецидивисты ───────────────────────────────────────────────────
    say()
    say("=" * 96)
    say("МНОГОКРАТНЫЕ ПОПЫТКИ НА ОДНОМ СЧЁТЕ (правила грозят блокировкой аккаунта)")
    say("=" * 96)
    own = df[df["owner_id"].notna()].groupby("owner_id")
    stats = pd.DataFrame({
        "n": own.size(),
        "n_dead": own.apply(lambda g: int((g["is_live"] == False).sum())),  # noqa: E712
        "med_life": own.apply(
            lambda g: (pd.to_datetime(g["last"]) - pd.to_datetime(g["created_at"])
                       ).dt.days.median()),
    })
    say(f"уникальных авторов: {len(stats):,}, систем на автора: "
        f"медиана {stats['n'].median():.0f}, p90 {stats['n'].quantile(.9):.0f}, "
        f"макс {stats['n'].max():.0f}")
    say()
    say(f'{"стратегий у автора":<22} {"авторов":>9} {"систем":>9} '
        f'{"медиана жизни, дн":>19}')
    for lo, hi, lab in ((1, 1, "1"), (2, 3, "2–3"), (4, 9, "4–9"), (10, 19, "10–19"),
                        (20, 10**9, "20 и больше")):
        m = (stats["n"] >= lo) & (stats["n"] <= hi)
        if not m.sum():
            continue
        say(f'{lab:<22} {int(m.sum()):>9,} {int(stats.loc[m, "n"].sum()):>9,} '
            f'{stats.loc[m, "med_life"].median():>19.0f}')
    say()
    say("Чем больше стратегий на одного автора, тем короче их жизнь — это ровно то")
    say("поведение, против которого написано правило о блокировке аккаунта.")

    LOG.write_text("\n".join(_lines) + "\n")
    say(f"\nлог: {LOG}")


if __name__ == "__main__":
    main()
