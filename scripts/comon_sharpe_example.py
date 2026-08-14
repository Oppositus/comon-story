"""comon_sharpe_example.py — учебная картинка к словарю введения (график 35 списка §4).

Единственный график серии, который рисуется НЕ по данным площадки: это условный
пример из словаря введения [comon-story-0-intro.md] — почему доходность сама по себе
ничего не говорит о качестве.

  A: 40 % годовых при волатильности 40 % — Sharpe 1,0
  B: 20 % годовых при волатильности 10 % — Sharpe 2,0
  B с двойным размером позиции: 40 % при 20 % — та же доходность, что у A, вдвое
  спокойнее. Обратное преобразование невозможно, и в этом весь смысл примера.

🔴 ЧИСЛА ЗАДАНЫ, А НЕ ИЗМЕРЕНЫ. Параметры взяты из таблицы словаря; дневные
доходности порождены нормальным шумом с этими параметрами и ФИКСИРОВАННЫМ зерном,
поэтому картинка воспроизводится до пикселя. Плечо применяется к тем же самым дневным
доходностям (умножением на два) — это ровно «торговать вдвое большим объёмом», а не
вторая независимая случайность.

⚠️ Реализованные просадки — свойство конкретной траектории, а не обещание: в словаре
у A стоит −45 %, у B −12 %, и зерно выбрано так, чтобы реализация была близка к этим
величинам. Соответствие проверяется печатью ниже.

Текстовый вывод + график. Запуск: python comon_sharpe_example.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

YEARS = 5
DAYS = 252
STEPS = DAYS * YEARS
SEED = 2629              # отобрано перебором (--pick-seed): просадки −44,7 и −12,0
A = {"mu": 0.40, "sig": 0.40, "lab": "A", "sharpe": 1.0}
B = {"mu": 0.20, "sig": 0.10, "lab": "B", "sharpe": 2.0}


def rets(mu, sig, rng):
    """Дневные доходности с заданной годовой доходностью и волатильностью."""
    return rng.normal(mu / DAYS, sig / np.sqrt(DAYS), STEPS)


def equity(r):
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def maxdd(eq):
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def build(seed):
    rng = np.random.default_rng(seed)
    ra, rb = rets(A["mu"], A["sig"], rng), rets(B["mu"], B["sig"], rng)
    return ra, rb, 2.0 * rb          # плечо = тот же ряд, вдвое больший объём


def main():
    import comon_charts as ch

    ra, rb, rl = build(SEED)
    ea, eb, el = equity(ra), equity(rb), equity(rl)
    x = np.arange(len(ea)) / DAYS

    f, ax = ch.fig(h_px=1040, bottom=0.28)
    series = ((ea, ch.ORANGE, "-", 2.8,
               f"A: {ch.n_(100*A['mu'])} % годовых при волатильности "
               f"{ch.n_(100*A['sig'])} % — Sharpe {ch.n_(A['sharpe'], 1)}, "
               f"просадка по пути {ch.n_(100*maxdd(ea))} %"),
              (eb, ch.BLUE, "--", 2.4,
               f"B: {ch.n_(100*B['mu'])} % годовых при волатильности "
               f"{ch.n_(100*B['sig'])} % — Sharpe {ch.n_(B['sharpe'], 1)}, "
               f"просадка по пути {ch.n_(100*maxdd(eb))} %"),
              (el, ch.GREEN, "-.", 2.4,
               f"B вдвое большим объёмом: {ch.n_(200*B['mu'])} % при "
               f"{ch.n_(200*B['sig'])} % — Sharpe тот же, просадка "
               f"{ch.n_(100*maxdd(el))} %"))
    for eq, color, ls, lw, lab in series:
        ax.plot(x, eq, color=color, lw=lw, ls=ls, zorder=4, label=lab)
    ax.axhline(1.0, color=ch.PALE, lw=1.4, ls=":", zorder=2)

    ax.set_xlim(0, YEARS)
    ax.set_ylim(0, max(ea.max(), el.max()) * 1.12)
    ax.set_xlabel("годы")
    ax.set_ylabel("капитал, начальный принят за единицу")
    ax.set_title("Доходность выбирают размером позиции, качество — нет")
    ax.legend(loc="upper left", handlelength=2.8, labelspacing=0.6, fontsize=10)
    ch.note(ax,
            "Условный пример, а не данные площадки: доходность и волатильность "
            "заданы, дневные колебания порождены случайным шумом с этими "
            "параметрами и фиксированным зерном. Просадки — свойство конкретной "
            "траектории.",
            "У A доходность вдвое выше, чем у B, и трясёт её вчетверо сильнее. "
            "Чтобы получить из B ту же доходность, достаточно торговать вдвое "
            "большим объёмом — зелёная линия: 40 % годовых при волатильности "
            "20 %, вдвое спокойнее A.",
            "Обратное преобразование невозможно: убавив риск у A вдвое, получите "
            "20 % при волатильности 20 % — вдвое хуже B. Поэтому качество меряют "
            "не доходностью, а доходностью на единицу риска: Sharpe у A равен "
            "1,0, у B — 2,0, и плечо его почти не меняет.")
    ch.save(f, 35, "sharpe-example")
    print(f"    A: итог ×{ea[-1]:.2f}, просадка {100*maxdd(ea):.0f} % (в словаре −45); "
          f"B: ×{eb[-1]:.2f}, {100*maxdd(eb):.0f} % (в словаре −12); "
          f"B с плечом: ×{el[-1]:.2f}, {100*maxdd(el):.0f} %", flush=True)


def pick_seed(n=4000):
    """Служебное: перебрать зёрна и найти траекторию, близкую к словарным числам."""
    best = None
    for s in range(n):
        ra, rb, _ = build(s)
        d = (abs(maxdd(equity(ra)) + 0.45) / 0.45
             + abs(maxdd(equity(rb)) + 0.12) / 0.12)
        if best is None or d < best[0]:
            best = (d, s, maxdd(equity(ra)), maxdd(equity(rb)))
    print(f"seed={best[1]}  A dd {100*best[2]:.1f} %  B dd {100*best[3]:.1f} %")


if __name__ == "__main__":
    if "--pick-seed" in sys.argv:
        pick_seed()
    else:
        main()
