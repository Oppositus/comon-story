"""comon_residual_life.py — ожидаемый остаток жизни стратегии («срок дожития»).

Для публичной серии (глава 2). Вопрос: если стратегия дожила до
возраста N, сколько ей осталось? Это демографическая таблица дожития, перенесённая
на торговые стратегии.

Считаем от той же кривой Каплана-Мейера, что и Блок 1 (comon_survival.py), с тем же
определением: рождение = created_at, смерть = ПОСЛЕДНИЙ ТОРГОВЫЙ ДЕНЬ, живые
цензурируются на дату выкачки, «не стартовала» исключается. Медиана жизни должна
воспроизвестись как 173 дня — это контроль, что определения не разъехались.

Три величины на каждый возраст N (в месяцах):
  1. МЕДИАННЫЙ остаток — время, за которое доживших до N становится вдвое меньше.
     Устойчив к обрыву хвоста, поэтому он основной.
  2. ОГРАНИЧЕННЫЙ СРЕДНИЙ остаток (RMRL) до горизонта H: ∫S(u)du / S(N) по [N, H].
     Средний остаток без ограничения не существует — KM не доходит до нуля, хвост
     оборван цензурированием. Горизонт выбран там, где под риском ещё достаточно
     наблюдений; величина читается как «в среднем за ближайшие H лет».
  3. Квартили остатка (25 % и 75 %) — разброс, а не только центр.

Текстовый вывод. Запуск: python comon_residual_life.py
"""
from datetime import date
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"
LOG = ROOT / "results" / "comon_residual_life.log"
ASOF = date(2026, 8, 5)
DAYS_M = 365.25 / 12.0          # дней в среднем месяце
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def km(t, e):
    """Каплан-Мейер. Возвращает (времена событий, S(t)) — ступенчатая функция."""
    order = np.argsort(t)
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    S, out = 1.0, []
    for ti in times:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        if at_risk == 0:
            break
        S *= 1.0 - d / at_risk
        out.append(S)
    return times, np.array(out)


def s_at(times, S, x):
    """S(x) для ступенчатой кривой: последнее значение с временем <= x."""
    i = np.searchsorted(times, x, side="right") - 1
    return 1.0 if i < 0 else float(S[i])


def quantile_residual(times, S, n, q, horizon):
    """Время x, за которое доля доживших от S(n) падает до q. NaN — не достигнуто."""
    s_n = s_at(times, S, n)
    if s_n <= 0:
        return np.nan
    target = s_n * q
    idx = np.where((times > n) & (S <= target))[0]
    if len(idx) == 0:
        return np.nan
    x = times[idx[0]] - n
    return x if x <= horizon else np.nan


def rmrl(times, S, n, horizon_days):
    """Ограниченный средний остаток: площадь под S на [n, n+horizon] / S(n)."""
    s_n = s_at(times, S, n)
    if s_n <= 0:
        return np.nan
    end = n + horizon_days
    grid = np.unique(np.concatenate([[n], times[(times > n) & (times < end)], [end]]))
    area = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        area += s_at(times, S, a) * (b - a)
    return area / s_n


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv


def chart_residual(rows):
    """График 3: чем дольше стратегия прожила, тем больше жизни у неё впереди."""
    import comon_charts as ch

    f, ax = ch.fig()
    x = [r[0] for r in rows]
    q50 = [r[1] for r in rows]
    q25 = [r[2] for r in rows]
    q75 = [r[3] for r in rows]

    # Верхняя граница определена НЕПРЕРЫВНО только сначала: дальше срок «умрут три
    # четверти» уходит за пятилетний горизонт расчёта. Заливать по крайним
    # индексам нельзя — в середине дыры, и полоса выглядит оборванной посреди поля.
    ok = []
    for i in range(len(x)):
        if np.isfinite(q75[i]) and q75[i] < 59:
            ok.append(i)
        else:
            break
    if len(ok) > 1:
        i0, i1 = ok[0], ok[-1]
        ax.fill_between(x[i0:i1 + 1], q25[i0:i1 + 1], q75[i0:i1 + 1],
                        color=ch.BAND, alpha=0.30, lw=0, zorder=2,
                        label="в этот срок укладывается половина стратегий")
        # верхняя кромка — линией: без неё заливка читается как пустое облако,
        # в котором медиана «съехала» вниз, а не как диапазон между кривыми
        ax.plot(x[i0:i1 + 1], q75[i0:i1 + 1], color=ch.AQUA, lw=1.4, ls=":", zorder=3)
        ax.plot([x[i1], x[i1]], [q25[i1], q75[i1]], color=ch.GREY, lw=1.2, zorder=3)

    ax.plot(x, q25, color=ch.AQUA, lw=1.8, ls="--", zorder=3,
            label="четверть умрёт раньше этого срока")
    ax.plot(x, q50, color=ch.BLUE, lw=2.6, marker="o", ms=7, zorder=4,
            label="половина умрёт раньше этого срока")
    ax.plot([0, max(x)], [0, max(x)], color=ch.PALE, lw=1.6, ls=":", zorder=3,
            label="сколько уже прожито")

    # 🔴 Никаких подписей на поле, кроме одной. Их было четыре — легенда, подпись
    # диагонали, выноска у обрыва полосы и число у последней точки; на узком
    # поле они наезжали друг на друга и на оси. Всё, что можно сказать словами,
    # ушло в легенду и сноску, на графике осталась только пометка обрыва.
    if len(ok) > 1:
        ax.annotate("дальше верхняя\nграница не считается",
                    xy=(x[i1], q75[i1]), xytext=(14, 14), textcoords="offset points",
                    fontsize=10, color=ch.GREY, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", color=ch.GREY, lw=1.0))

    ax.legend(loc="lower right", handlelength=2.6, labelspacing=0.55)
    ax.set_xlabel("сколько стратегия уже прожила, месяцев")
    ax.set_ylabel("сколько ей осталось, месяцев")
    ax.set_title("Чем дольше стратегия живёт, тем больше жизни у неё впереди")
    ax.set_xlim(-2, max(x) + 6)
    ax.set_ylim(0, max(q50) * 1.30)
    ch.note(ax,
            "Полоса не симметрична вокруг медианы: срок дожития скошен вправо. "
            "Где синяя линия выше пунктирной диагонали, у стратегии впереди больше "
            "жизни, чем позади.",
            "Горизонт расчёта — пять лет. Поэтому верхняя граница полосы обрывается "
            "на первом годе, а падение линии справа создано обрывом наблюдений, "
            "а не стратегиями.")
    p = ch.save(f, 3, "residual-life")
    print("    " + "; ".join(f"{a:.0f} мес → ещё {b:.0f}" for a, b, _, _ in rows)
          + (f"; полоса до {x[ok[-1]]:.0f} мес" if len(ok) > 1 else "; полосы нет"),
          flush=True)


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)

    born = pd.to_datetime(panel["created_at"], errors="coerce")
    last_tr = pd.to_datetime(panel["last_trade"], errors="coerce")
    alive = panel["is_live"] == True                                    # noqa: E712
    started = panel["n_trade"].fillna(0) > 0
    end = last_tr.where(~alive, pd.Timestamp(ASOF))
    life = (end - born).dt.days
    ok = started & born.notna() & life.notna() & (life >= 0)
    t = life[ok].to_numpy(float)
    e = (~alive[ok]).to_numpy(int)

    say(f"в анализе: {int(ok.sum())} стратегий — смертей {int(e.sum())}, "
        f"цензурировано (живые) {int((e == 0).sum())}")

    times, S = km(t, e)
    med = quantile_residual(times, S, 0, 0.5, np.inf)
    say(f"контроль: медиана срока жизни при рождении = {med:.0f} дней "
        f"({med/DAYS_M:.1f} мес) — в Блоке 1 было 173 дня")
    say(f"максимальный наблюдаемый срок: {t.max():.0f} дней ({t.max()/365.25:.1f} года)")
    say()

    # сколько наблюдений остаётся под риском на каждом возрасте — граница доверия
    ages_m = [0, 1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 120]
    say("под риском на возрасте N (сколько стратегий ещё наблюдаются):")
    for n_m in ages_m:
        n_d = n_m * DAYS_M
        say(f"  {n_m:>4} мес: {int((t >= n_d).sum()):>6}")
    say()

    chart_rows = []
    for horizon_y in (5, 10):
        H = horizon_y * 365.25
        say("=" * 96)
        say(f"ТАБЛИЦА ДОЖИТИЯ, ограниченный средний остаток — горизонт {horizon_y} лет")
        say("=" * 96)
        say(f'{"дожила до":>10} {"S(N), %":>9} {"медиана":>10} {"25 % умрут":>12} '
            f'{"75 % умрут":>12} {"средн. остаток":>15} {"ожид. всего":>13}')
        say(f'{"(месяцев)":>10} {"":>9} {"остатка, мес":>10} {"через, мес":>12} '
            f'{"через, мес":>12} {f"за {horizon_y} лет, мес":>15} {"жизни, мес":>13}')
        say("-" * 96)
        for n_m in ages_m:
            n_d = n_m * DAYS_M
            if (t >= n_d).sum() < 30:
                continue
            s_n = 100 * s_at(times, S, n_d)
            q50 = quantile_residual(times, S, n_d, 0.50, H)
            q25 = quantile_residual(times, S, n_d, 0.75, H)
            q75 = quantile_residual(times, S, n_d, 0.25, H)
            mean_r = rmrl(times, S, n_d, H)
            fmt = lambda x: "—" if not np.isfinite(x) else f"{x/DAYS_M:.1f}"   # noqa: E731
            say(f'{n_m:>10} {s_n:>9.1f} {fmt(q50):>10} {fmt(q25):>12} {fmt(q75):>12} '
                f'{mean_r/DAYS_M:>15.1f} {n_m + mean_r/DAYS_M:>13.1f}')
            if horizon_y == 5 and np.isfinite(q50):
                chart_rows.append((n_m, q50 / DAYS_M, q25 / DAYS_M,
                                   min(q75, H) / DAYS_M))
        say()

    if CHARTS:
        say()
        say("── график 3 ──────────────────────────────────────────────────────")
        chart_residual(chart_rows)

    say("Как читать:")
    say("  • «медиана остатка» — за это время доживших до N станет вдвое меньше;")
    say("  • «25 % умрут через» — столько проходит до потери четверти доживших;")
    say("  • «средний остаток» ОГРАНИЧЕН горизонтом: неограниченного среднего не")
    say("    существует, кривая KM не доходит до нуля (хвост оборван цензурированием);")
    say("  • «ожид. всего жизни» = N + средний остаток, тоже в пределах горизонта.")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    say(f"\nлог: {LOG}")


if __name__ == "__main__":
    main()
