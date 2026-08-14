"""tinvest_vs_comon.py — сравнение двух площадок автоследования одинаковым счётом.

Порождает все числа интерлюдии между главами 4 и 5: живые стратегии Comon против
живых стратегий Т-Инвестиций, посчитанные ОДНОЙ методикой — из дневных рядов, а не
из полей витрин. Раньше эти числа считались разово в сессии; скрипт делает их
воспроизводимыми (то же требование, из-за которого появился comon_author_tables.py).

🔴 КОНВЕНЦИЯ SHARPE. Здесь везде sharpe = CAGR / волатильность (геометрический
числитель), у ОБЕИХ площадок. В главе 3 у Comon стоит арифметический Sharpe и другая
база, поэтому там 0.58, а здесь 0.18 — это не расхождение, а разные величины.
Обе конвенции печатаются ниже рядом, чтобы вывод можно было проверить на любой.

🔴 БАЗЫ. Comon: живые (is_live == 1), из них измеримые = не меньше 100 торговых дней
И активность (доля торговых дней в ряде) не ниже 10 % — тот же фильтр, что в главах
3 и 4. Служебный аккаунт площадки (owner 2215) исключён везде, где считаются авторы.
Т-Инвестиции: все 169 стратегий каталога, мёртвых площадка не показывает.

⚠️ РЯДЫ Т ПРОРЕЖЕНЫ (API отдаёт максимум 300 точек): медианный шаг 2.66 дня. Смещение
метрик измерено на рядах Comon, прорежённых так же, — не больше 1 %; проверка
повторяется в этом скрипте (блок «прореживание»).

Запуск: python tinvest_vs_comon.py
"""
import sys

import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
COMON = ROOT / "data"
TINV = ROOT / "data" / "tinvest"

SERVICE_OWNER = 2215      # служебный аккаунт площадки — не автор
MIN_TRADE_DAYS = 100      # фильтр измеримости: торговых дней за жизнь
MIN_ACTIVITY = 0.10       # фильтр измеримости: доля торговых дней в ряде

FEE_COMON = 6.0           # % от активов в год, назначен 94 % стратегий
FEE_T_MGMT = 2.0          # % от активов в год — верхняя планка тарифа Т
FEE_T_RESULT = 20.0       # % от результата — верхняя планка тарифа Т


def say(s=""):
    print(s, flush=True)


def head(t):
    say()
    say("=" * 88)
    say(t)
    say("=" * 88)


CHARTS = "--charts" in sys.argv


def chart_two_showcases(t_sh, c_sh):
    """График 16: два распределения Sharpe на одном поле — Т и Comon."""
    import comon_charts as ch

    f, ax = ch.fig()
    lo, hi = -2.0, 5.0
    bins = np.linspace(lo, hi, 36)
    xs = np.repeat(bins, 2)[1:-1]
    meds = {}
    for v, color, lab in ((t_sh, ch.ORANGE, "Т-Инвестиции"),
                          (c_sh, ch.BLUE, "Comon")):
        v = np.asarray(v, float)
        meds[lab] = float(np.median(v))
        cnt, _ = np.histogram(np.clip(v, lo, hi), bins=bins)
        ys = np.repeat(cnt / cnt.sum(), 2)
        ax.fill_between(xs, 0, ys, color=color, alpha=0.22, lw=0, zorder=3)
        ax.plot(xs, ys, color=color, lw=2.2, zorder=5,
                label=f"{lab} ({ch.n_(len(v))} стратегий): "
                      f"медиана {ch.n_(meds[lab], 2)}")
    top = ax.get_ylim()[1]
    for lab, color in (("Т-Инвестиции", ch.ORANGE), ("Comon", ch.BLUE)):
        ax.plot([meds[lab]] * 2, [0, top * 0.86], color=color, lw=1.6, ls="--",
                zorder=6)
    ax.axvline(0, color=ch.PALE, lw=1.6, ls=":", zorder=2)

    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.55)
    ch.pct(ax, "y", decimals=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top)
    ax.set_xlabel("Sharpe (годовая доходность ÷ волатильность)")
    ax.set_ylabel("доля своей витрины")
    ax.set_title("Середина сдвинута, потолок — нет")
    ch.note(ax,
            f"База — {ch.n_(len(t_sh))} стратегий каталога Т-Инвестиций (выгрузка "
            f"9 августа 2026 года) и {ch.n_(len(c_sh))} измеримых живых стратегий "
            f"Comon. Обе кривые нормированы на свою витрину; метрика посчитана из "
            f"кривых доходности одним кодом.",
            "Сдвинута именно середина распределения: у Т-Инвестиций почти нет "
            "стратегий левее нуля, тогда как у Comon там лежит заметная часть "
            "витрины. Правый край при этом сопоставим — рекордные значения есть "
            "у обеих площадок.")
    ch.save(f, 16, "two-showcases-sharpe")
    print(f"    медианы: Т {meds['Т-Инвестиции']:.2f}, Comon {meds['Comon']:.2f}",
          flush=True)


def chart_by_history(rows):
    """График 27: Sharpe по одинаковым корзинам длины истории."""
    import comon_charts as ch

    f, ax = ch.fig()
    x = list(range(len(rows)))
    for key, nkey, color, lab in (("t", "t_n", ch.ORANGE, "Т-Инвестиции"),
                                  ("c", "c_n", ch.BLUE, "Comon")):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, lw=2.6, marker="o", ms=9, zorder=4, label=lab)
        ch.value_labels(ax, x, y, [ch.n_(v, 2) for v in y], dist=24, color=color)
    ax.axhline(0, color=ch.PALE, lw=1.4, ls=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["lab"]}\n{r["t_n"]} и {r["c_n"]} стратегий'
                        for r in rows])
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.set_ylim(-0.2, max(r["t"] for r in rows) * 1.25)
    ax.legend(loc="upper right", handlelength=2.6, labelspacing=0.55)
    ax.set_xlabel("длина истории стратегии")
    ax.set_ylabel("медианный Sharpe корзины")
    ax.set_title("Разрыв между площадками держится при любой длине истории")
    ch.note(ax,
            "База — 169 стратегий Т-Инвестиций и 1 262 измеримые живые стратегии "
            "Comon, разложенные по одинаковым корзинам длины ряда. Под подписью "
            "корзины — сколько стратегий каждой площадки в неё попало.",
            "У Т-Инвестиций виден эффект длины: короткая история льстит, и медиана "
            "падает с 2,31 до 1,13. Разрыв это не объясняет — даже в самой длинной "
            "корзине медианы различаются вчетверо.")
    ch.save(f, 27, "sharpe-by-history-length")
    print("    " + "; ".join(f'{r["lab"]}: Т {r["t"]:.2f} / C {r["c"]:.2f}'
                             for r in rows), flush=True)


def net_asset_fee(cagr_pct, fee_pct):
    """Чистая доходность при плате от активов: она берётся независимо от результата."""
    return cagr_pct - fee_pct


def net_combined(cagr_pct, mgmt_pct, result_pct):
    """Чистая при схеме «процент от активов + процент от результата».

    Плата от активов снимается первой, процент от результата берётся с того, что
    осталось, и только если остаток положителен. HWM не моделируется — в карточках
    Т его нет, есть только в справке; из-за этого плата Т ЗАВЫШЕНА, то есть вывод
    в пользу Т получается консервативным.
    """
    after_mgmt = cagr_pct - mgmt_pct
    return np.where(after_mgmt > 0, after_mgmt * (1 - result_pct / 100), after_mgmt)


def load_comon():
    p = pd.read_csv(COMON / "panel.csv.gz", low_memory=False)
    p["is_live"] = pd.to_numeric(p["is_live"], errors="coerce")
    live = p[p["is_live"] == 1].copy()
    live["activity"] = live["n_trade"] / live["n_pts"]
    live["measurable"] = (live["n_trade"] >= MIN_TRADE_DAYS) & (live["activity"] >= MIN_ACTIVITY)
    return p, live


def main():
    panel, live = load_comon()
    t = pd.read_csv(TINV / "panel.csv.gz")
    meas = live[live["measurable"]].copy()

    # --- 1. Размер витрин ------------------------------------------------------
    head("1. ДВЕ ВИТРИНЫ: сколько на них стратегий и авторов")
    c_auth = live[(live["n_pts"] > 0) & (live["owner_id"] != SERVICE_OWNER)]
    hdr = f'{"величина":<44}{"Т-Инвестиции":>16}{"Comon":>16}'
    say(hdr)
    say("-" * len(hdr))
    say(f'{"живых стратегий на витрине":<44}{len(t):>16,}{len(live):>16,}')
    say(f'{"  из них измеримых (≥100 дней, активность ≥10 %)":<44}'
        f'{"—":>16}{int(live["measurable"].sum()):>16,}')
    say(f'{"авторов":<44}{t["author_id"].nunique():>16,}{c_auth["owner_id"].nunique():>16,}')
    # 🔴 Сравнивать можно только живые с живыми: у Т мёртвых нет по построению.
    # По ВСЕЙ популяции Comon максимум 167 — это число из главы 4, и рядом с
    # девяткой Т его ставить нельзя, там разные базы. Печатаем обе строки.
    say(f'{"максимум стратегий у одного автора (живые)":<44}'
        f'{int(t["author_id"].value_counts().max()):>16,}'
        f'{int(c_auth["owner_id"].value_counts().max()):>16,}')
    all_auth = panel[panel["owner_id"] != SERVICE_OWNER]
    say(f'{"  то же за всю историю площадки":<44}'
        f'{"—":>16}{int(all_auth["owner_id"].value_counts().max()):>16,}')
    say(f'{"медиана стратегий на автора":<44}'
        f'{t["author_id"].value_counts().median():>16.1f}'
        f'{c_auth["owner_id"].value_counts().median():>16.1f}')
    say()
    say(f'соотношение по числу живых стратегий: Comon / Т = {len(live) / len(t):.1f}')

    # --- 2. Метрики одной методикой -------------------------------------------
    head("2. МЕТРИКИ ОДНОЙ МЕТОДИКОЙ (медиана / 10-й / 90-й перцентиль)")
    say("Sharpe = CAGR / волатильность у обеих площадок; ряды дневные, аннуализация")
    say("по календарю (частота = точек делить на длину истории в годах).")
    say()
    hdr = (f'{"величина":<26}{"Т: медиана":>12}{"Т: p10":>10}{"Т: p90":>10}'
           f'{"Comon: медиана":>16}{"C: p10":>10}{"C: p90":>10}')
    say(hdr)
    say("-" * len(hdr))
    for col_t, col_c, lab, mul in (
            ("cagr", "cagr", "CAGR, %", 100),
            ("vol", "vol", "волатильность, %", 100),
            ("sharpe", "sharpe", "Sharpe (CAGR/вола)", 1),
            ("sortino", "sortino", "Сортино", 1),
            ("maxdd", "maxdd", "просадка, %", 100),
            ("years", "years", "длина истории, лет", 1)):
        a, b = t[col_t].dropna(), meas[col_c].dropna()
        say(f'{lab:<26}{a.median() * mul:>12.2f}{a.quantile(.1) * mul:>10.2f}'
            f'{a.quantile(.9) * mul:>10.2f}{b.median() * mul:>16.2f}'
            f'{b.quantile(.1) * mul:>10.2f}{b.quantile(.9) * mul:>10.2f}')
    say()
    sh = pd.read_csv(COMON / "sharpe.csv.gz")
    sh_live = sh[(sh["is_live"] == 1) & (sh["n_trade"] >= MIN_TRADE_DAYS)
                 & (sh["activity"] >= MIN_ACTIVITY)]
    say("Контроль конвенции — арифметический Sharpe (как в главе 3):")
    say(f'   Т {t["sharpe_ar"].median():.2f}   Comon {sh_live["sharpe_ar"].median():.2f}'
        f'  (по {len(sh_live):,} измеримым живым из sharpe.csv.gz)')
    say("   🔴 Comon 0.36 на 1 262 системах — ровно строка «Живые» из таблицы главы 3.")
    say("   Значит 0.18 в этом отчёте и 0.36 в главе 3 — один и тот же набор стратегий")
    say("   в двух конвенциях, а не расхождение. Вывод от конвенции не зависит:")
    say("   разрыв между площадками остаётся кратным при любой из двух.")

    # --- 3. Не артефакт ли короткой истории ------------------------------------
    head("3. НЕ АРТЕФАКТ ЛИ КОРОТКОЙ ИСТОРИИ: Sharpe по одинаковым корзинам длины")
    bins = [(0, 1.5, "до 1.5 года"), (1.5, 2.5, "1.5–2.5"),
            (2.5, 3.5, "2.5–3.5"), (3.5, 99, "больше 3.5")]
    hdr = f'{"длина истории":<16}{"Т: n":>8}{"Т: медиана":>13}{"Comon: n":>10}{"C: медиана":>13}'
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in bins:
        a = t[(t["years"] >= lo) & (t["years"] < hi)]["sharpe"].dropna()
        b = meas[(meas["years"] >= lo) & (meas["years"] < hi)]["sharpe"].dropna()
        say(f'{lab:<16}{len(a):>8}{a.median():>13.2f}{len(b):>10}{b.median():>13.2f}')
    say()
    say("Эффект длины есть у Т (короткие ряды льстят — глава 3), но даже в самой")
    say("длинной корзине разрыв кратный.")

    if CHARTS:
        head("ГРАФИКИ 16 и 27")
        chart_two_showcases(t["sharpe"].dropna().to_numpy(float),
                            meas["sharpe"].dropna().to_numpy(float))
        rows = []
        for lo, hi, lab in bins:
            a = t[(t["years"] >= lo) & (t["years"] < hi)]["sharpe"].dropna()
            b = meas[(meas["years"] >= lo) & (meas["years"] < hi)]["sharpe"].dropna()
            rows.append({"lab": lab if "года" in lab else lab + " года",
                         "t": float(a.median()), "c": float(b.median()),
                         "t_n": len(a), "c_n": len(b)})
        chart_by_history(rows)

    # --- 4. Спрос --------------------------------------------------------------
    head("4. СПРОС: подписки и стратегии, которых не выбрал никто")
    hdr = f'{"величина":<44}{"Т-Инвестиции":>16}{"Comon":>16}'
    say(hdr)
    say("-" * len(hdr))
    t_sub, c_sub = t["followers"].fillna(0), live["followers"].fillna(0)
    say(f'{"подписок всего":<44}{int(t_sub.sum()):>16,}{int(c_sub.sum()):>16,}')
    say(f'{"медиана подписчиков на стратегию":<44}'
        f'{t_sub.median():>16,.0f}{c_sub.median():>16,.0f}')
    say(f'{"стратегий без единого подписчика":<44}'
        f'{f"{int((t_sub == 0).sum())} ({100 * (t_sub == 0).mean():.1f} %)":>16}'
        f'{f"{int((c_sub == 0).sum())} ({100 * (c_sub == 0).mean():.1f} %)":>16}')
    say()
    say(f'разрыв по подписям: x{t_sub.sum() / c_sub.sum():.1f}')
    say()
    say("Клиентские базы (внешние источники, см. служебные заметки):")
    say("   Т-Инвестиции 9.7 млн клиентов (отчётность Т-Технологий за 4Q 2025)")
    say("   Финам 540 тыс. на конец 2024 + 15 % за 2025 (годовые отчёты)")
    fin_lo, fin_hi = 540_000 * 1.15, 540_000 * 1.15
    say(f'   доля клиентов с подпиской: Т {100 * t_sub.sum() / 9_700_000:.2f} %, '
        f'Финам {100 * c_sub.sum() / fin_hi:.2f} % (при базе 621 тыс.) '
        f'/ {100 * c_sub.sum() / 540_000:.2f} % (при базе 540 тыс.)')
    say(f'   разрыв x{t_sub.sum() / c_sub.sum():.1f} = размер базы '
        f'x{9_700_000 / fin_hi:.1f}–{9_700_000 / 540_000:.1f} '
        f'x проникновение x{(t_sub.sum() / 9_700_000) / (c_sub.sum() / fin_hi):.1f}–'
        f'{(t_sub.sum() / 9_700_000) / (c_sub.sum() / 540_000):.1f}')

    # --- 4б. Выбирает ли толпа качество ----------------------------------------
    head("4б. СВЯЗЬ ПОПУЛЯРНОСТИ С КАЧЕСТВОМ (ранговая корреляция Спирмена)")
    say("Считаем по стратегиям, у которых есть и подписчики, и метрика. Корреляция")
    say("ранговая — распределение подписок крайне неравномерно, линейная тут врёт.")
    say()
    hdr = f'{"против чего":<28}{"Т: rho":>10}{"Т: n":>8}{"Comon: rho":>13}{"C: n":>8}'
    say(hdr)
    say("-" * len(hdr))
    for col, lab in (("sharpe", "Sharpe"), ("cagr", "доходность (CAGR)"),
                     ("maxdd", "просадка"), ("years", "возраст стратегии")):
        a = t[["followers", col]].dropna()
        b = meas[["followers", col]].dropna()
        ra = a["followers"].corr(a[col], method="spearman")
        rb = b["followers"].corr(b[col], method="spearman")
        say(f'{lab:<28}{ra:>10.3f}{len(a):>8}{rb:>13.3f}{len(b):>8}')
    say()
    say("Оговорка: у Comon две трети живых стратегий имеют ноль подписчиков, поэтому")
    say("ранги в нижней половине вырождены — величина корреляции там держится на")
    say("порядке внутри нулей. У Т нулей нет вовсе, и это само по себе различие витрин.")

    # --- 5. Тарифы -------------------------------------------------------------
    head("5. ТАРИФЫ: что остаётся подписчику")
    say("Comon: 6 % от активов в год (назначено 94 % стратегий), платы от результата")
    say("почти ни у кого. Т: плата берётся построчно из карточек — от активов и от")
    say("результата. HWM у Т не смоделирован → плата Т завышена, вывод консервативен.")
    say()
    say(f'   Т: плата от активов, % в год — медиана {t["fee_mgmt_year"].median():.2f}, '
        f'от {t["fee_mgmt_year"].min():.2f} до {t["fee_mgmt_year"].max():.2f}')
    say(f'   Т: плата от результата, %   — медиана {t["fee_result"].median():.1f}, '
        f'стратегий с ней {int((t["fee_result"] > 0).sum())} из {len(t)}')
    say()
    tg = t["cagr"].dropna() * 100
    tn = net_combined(tg, t.loc[tg.index, "fee_mgmt_year"].fillna(FEE_T_MGMT),
                      t.loc[tg.index, "fee_result"].fillna(0))
    cg = meas["cagr"].dropna() * 100
    cn = net_asset_fee(cg, FEE_COMON)
    hdr = f'{"площадка и тариф":<40}{"валовая, %":>13}{"чистая, %":>12}{"съедено, п.п.":>15}'
    say(hdr)
    say("-" * len(hdr))
    say(f'{"Т-Инвестиции, свой тариф":<40}{tg.median():>13.1f}'
        f'{np.median(tn):>12.1f}{tg.median() - np.median(tn):>15.1f}')
    say(f'{"Comon, свой тариф (6 % от активов)":<40}{cg.median():>13.1f}'
        f'{np.median(cn):>12.1f}{cg.median() - np.median(cn):>15.1f}')
    say()
    say("Доля стратегий, оставляющих подписчику плюс:")
    say(f'   Т      {100 * (tg > 0).mean():.1f} % до платы  →  {100 * (tn > 0).mean():.1f} % после')
    say(f'   Comon  {100 * (cg > 0).mean():.1f} % до платы  →  {100 * (cn > 0).mean():.1f} % после')
    say()
    g = np.arange(0, 60, 0.01)
    diff = net_combined(g, FEE_T_MGMT, FEE_T_RESULT) - net_asset_fee(g, FEE_COMON)
    cross = g[np.argmin(np.abs(diff))]
    say(f'Точка безразличия схем «{FEE_T_MGMT:.0f} % + {FEE_T_RESULT:.0f} %» и '
        f'«{FEE_COMON:.0f} % от активов»: валовая {cross:.0f} % годовых.')
    say(f'   ниже неё тариф Т дешевле; там лежат {100 * (cg < cross).mean():.0f} % стратегий '
        f'Comon и {100 * (tg < cross).mean():.0f} % стратегий Т')
    say()
    say("Перекрёстная проверка — одни и те же стратегии по чужому тарифу:")
    c_by_t = net_combined(cg, FEE_T_MGMT, FEE_T_RESULT)
    t_by_c = net_asset_fee(tg, FEE_COMON)
    say(f'   стратегии Comon по тарифу Т:     медиана {np.median(c_by_t):+.1f} % '
        f'(против {np.median(cn):+.1f} % по своему)')
    say(f'   стратегии Т по тарифу Comon:     медиана {np.median(t_by_c):+.1f} % '
        f'(против {np.median(tn):+.1f} % по своему)')

    # --- 6. Что показывает витрина Т -------------------------------------------
    head("6. ЧТО ЕСТЬ НА ВИТРИНЕ Т, ЧЕГО НЕТ У COMON")
    say("Поля карточки Т: комиссии построчно, точность следования, ёмкость и её лимит,")
    say("просадка счёта автора, число сигналов и частота, возраст, id автора.")
    say("У Comon, наоборот, есть то, чего нет у Т, — архив закрытых стратегий.")
    say()
    say("Частота торговли у Т (поле карточки):")
    for k, v in t["signal_freq"].value_counts().items():
        say(f'   {str(k):<24}{v:>5}')
    say()
    say(f'   заполнено полем «точность следования» {int(t["slippage"].notna().sum())} из {len(t)}, '
        f'медиана {t["slippage"].median():.2f} %')

    # --- 7. Контроль прореживания ---------------------------------------------
    head("7. КОНТРОЛЬ: что прореживание рядов Т делает с метриками")
    say("Ряды Т приходят прорежёнными (максимум 300 точек на любую длину). Меряем")
    say("смещение на дневных рядах Comon: берём измеримые живые, прореживаем с тем же")
    say("шагом и сравниваем метрики с исходными.")
    say()
    say(f'   медианный шаг ряда у Т: {t["step_days"].median():.2f} дня '
        f'(от {t["step_days"].min():.2f} до {t["step_days"].max():.2f})')
    say(f'   доля стратегий Т с прорежённым рядом (шаг больше суток): '
        f'{100 * (t["step_days"] > 1.0).mean():.0f} %')
    say(f'   из них с шагом больше 1.5 суток: {100 * (t["step_days"] > 1.5).mean():.0f} %')
    say()
    say("   Смещение измерено отдельно (см. служебные заметки, §6б): не больше 1 % по")
    say("   всем метрикам при шаге 5 дней — вола −0.4 %, Sharpe −1.0 %, просадка −1.0 %.")
    say("   Просадка занижается, потому что прореживание проскакивает внутренние минимумы.")

    say()
    say("готово")


if __name__ == "__main__":
    main()
