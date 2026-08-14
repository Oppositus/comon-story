"""comon_peak_entry.py — ЧТО ПОЛУЧИЛ БЫ ПОДПИСЧИК, ВОШЕДШИЙ НЕ В ПЕРВЫЙ ДЕНЬ.

Пункт «Глава 2, п.1» ручной вычитки серии: итог
стратегии — это результат того, кто подписался в ПЕРВЫЙ день её жизни и не выходил.
Таких подписчиков не бывает: на витрину смотрят, когда там уже есть что показать.
Здесь считаются два других входа, оба — на тех же 13 395 умерших стратегиях:

  1) ВХОД НА ПИКЕ  — человек увидел хорошую историю и подписался на максимуме кривой.
     Результат = eq_конец / eq_пик − 1. По построению он никогда не положителен: это
     ровно та просадка от вершины, которой закончилась жизнь стратегии.
  2) СЛУЧАЙНЫЙ ВХОД — подписка в произвольный день жизни (равномерно). Медиана по всем
     дням входа. Это нейтральная середина между «первым днём» и «пиком».

🔴 ЧЕГО ЭТИ ЧИСЛА НЕ ЗНАЧАТ. Пик определяется задним числом; подписчик в момент входа
не знает, что стоит на вершине. Поэтому вход на пике — не «типичный подписчик», а
верхняя граница пессимизма, а случайный вход — середина. Вместе они дают вилку, внутри
которой лежит реальный опыт. Ни один из трёх сценариев не учитывает плату за следование
(глава 2 везде считает валовые числа) и комиссии брокера подписчика.

Отдельно считается «витринный» подвыбор: стратегии, у которых ДО пика было не меньше
100 торговых дней и накопленный рост не меньше +50 %, — то есть те, чья история
действительно выглядела привлекательной в момент, когда на неё смотрели.

Разделы:
  1 три сценария: распределение по всем умершим;
  2 по механизму смерти (классы таблицы 2.3);
  3 витринный подвыбор;
  4 главное расхождение: стратегии, закончившие В ПЛЮСЕ.

Запуск: python comon_peak_entry.py
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
NPROC = 8

MIN_HIST_BEFORE = 100      # торговых дней до пика — «история, на которую смотрят»
MIN_GROWTH_PEAK = 1.50     # рост к пику, при котором витрина выглядит привлекательно
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def work(sid):
    """Три сценария входа для одной стратегии."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    try:
        s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    except Exception:                                                   # noqa: BLE001
        return None
    if len(s) < 2:
        return None
    s = s[::-1]
    r = np.array([float(x["rValue"] or 0.0) for x in s], dtype=np.float64)
    if not np.any(r != 0):
        return None
    eq = np.cumprod(1.0 + r)
    if eq[-1] <= 0 or np.any(~np.isfinite(eq)):
        return None
    p = int(np.argmax(eq))
    # вход «в конце дня t» — доступны дни 0..len-2, иначе держать нечего
    tail = eq[-1] / eq[:-1] - 1.0
    return {"id": sid,
            "total": float(eq[-1] - 1.0),
            "peak_eq": float(eq[p]),
            "from_peak": float(eq[-1] / eq[p] - 1.0),
            "rand_med": float(np.median(tail)),
            "rand_neg": float((tail < 0).mean()),
            "trade_before": int((r[:p + 1] != 0).sum()),
            "trade_after": int((r[p + 1:] != 0).sum()),
            "pts_after": int(len(r) - p - 1)}


def dist(name, v):
    q = np.percentile(v, [10, 25, 50, 75, 90])
    say(f'{name:<34} {100*q[0]:>9.1f} {100*q[1]:>9.1f} {100*q[2]:>9.1f}'
        f' {100*q[3]:>9.1f} {100*q[4]:>9.1f} {100*(v < 0).mean():>10.1f}')


def by_group(df, col, order=None):
    hdr = (f'{"Класс":<26} {"Систем":>7} {"Итог":>9} {"На пике":>9} {"Случайный":>10}'
           f' {"В минусе на пике":>17} {"Дней после пика":>16}')
    say(hdr)
    say("-" * len(hdr))
    keys = order or list(df[col].value_counts().index)
    for k in keys:
        d = df[df[col] == k]
        if not len(d):
            continue
        say(f'{str(k):<26} {len(d):>7,} {100*d.total.median():>8.1f}%'
            f' {100*d.from_peak.median():>8.1f}% {100*d.rand_med.median():>9.1f}%'
            f' {100*(d.from_peak < 0).mean():>16.1f}% {int(d.pts_after.median()):>16,}')
    say("-" * len(hdr))
    say(f'{"ВСЕГО":<26} {len(df):>7,} {100*df.total.median():>8.1f}%'
        f' {100*df.from_peak.median():>8.1f}% {100*df.rand_med.median():>9.1f}%'
        f' {100*(df.from_peak < 0).mean():>16.1f}% {int(df.pts_after.median()):>16,}')
    say()


def main():
    deaths = pd.read_csv(DIR / "deaths.csv.gz").set_index("id")
    ids = deaths.index[deaths["n_trade"] > 0].tolist()

    say("=" * 112)
    say("ЧТО ПОЛУЧИЛ БЫ ПОДПИСЧИК, ВОШЕДШИЙ НЕ В ПЕРВЫЙ ДЕНЬ")
    say("=" * 112)
    say("Итог стратегии = результат подписавшегося в первый день и не выходившего.")
    say("Вход на пике = верхняя граница пессимизма (пик виден только задним числом).")
    say("Случайный вход = медиана по всем возможным дням подписки.")
    say("Все числа валовые: плата за следование и комиссии брокера НЕ вычтены.")
    say()

    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(work, ids, chunksize=200) if x]
    W = pd.DataFrame(rows).set_index("id").join(deaths[["mech", "fin", "n_trade"]])
    say(f'умерших стратегий, торговавших хотя бы день: {len(ids):,}; '
        f'посчитано: {len(W):,}')
    say()

    # ── 1. три сценария ───────────────────────────────────────────────────────
    say("=" * 112)
    say("1. ТРИ СЦЕНАРИЯ ВХОДА: РАСПРЕДЕЛЕНИЕ ИТОГОВОГО РЕЗУЛЬТАТА, %")
    say("=" * 112)
    hdr = (f'{"Сценарий":<34} {"p10":>9} {"p25":>9} {"медиана":>9} {"p75":>9}'
           f' {"p90":>9} {"в минусе":>10}')
    say(hdr)
    say("-" * len(hdr))
    dist("Подписался в первый день", W.total.to_numpy())
    dist("Подписался в случайный день", W.rand_med.to_numpy())
    dist("Подписался на пике", W.from_peak.to_numpy())
    say()

    # ── 2. по механизму смерти ────────────────────────────────────────────────
    say("=" * 112)
    say("2. ПО МЕХАНИЗМУ СМЕРТИ (классы таблицы 2.3), медианы")
    say("=" * 112)
    by_group(W, "mech", ["смерть у пика", "обычная просадка", "деградация",
                         "недоказано", "заброшена", "недостаточно истории"])

    # ── 3. витринный подвыбор ─────────────────────────────────────────────────
    say("=" * 112)
    say(f"3. «ВИТРИННЫЙ» ПОДВЫБОР: до пика ≥ {MIN_HIST_BEFORE} торговых дней "
        f"и рост ≥ +{100*(MIN_GROWTH_PEAK-1):.0f} %")
    say("=" * 112)
    V = W[(W.trade_before >= MIN_HIST_BEFORE) & (W.peak_eq >= MIN_GROWTH_PEAK)]
    say(f'таких стратегий: {len(V):,} ({100*len(V)/len(W):.1f} % умерших с торгами)')
    say(f'медианный рост к пику: {100*(V.peak_eq.median()-1):.0f} %; '
        f'медианная история до пика: {int(V.trade_before.median()):,} торговых дней')
    say()
    by_group(V, "mech", ["смерть у пика", "обычная просадка", "деградация",
                         "недоказано", "заброшена", "недостаточно истории"])

    # ── 4. стратегии, закончившие в плюсе ─────────────────────────────────────
    say("=" * 112)
    say("4. ГЛАВНОЕ РАСХОЖДЕНИЕ: СТРАТЕГИИ, ЗАКОНЧИВШИЕ В ПЛЮСЕ")
    say("=" * 112)
    P = W[W.total > 0]
    say(f'закончили жизнь в плюсе: {len(P):,} из {len(W):,} ({100*len(P)/len(W):.1f} %)')
    say(f'из них вошедший на пике оказался бы в минусе: '
        f'{int((P.from_peak < 0).sum()):,} ({100*(P.from_peak < 0).mean():.1f} %)')
    say(f'медиана: итог стратегии {100*P.total.median():+.1f} %, '
        f'вход на пике {100*P.from_peak.median():+.1f} %, '
        f'случайный вход {100*P.rand_med.median():+.1f} %')
    say()
    hdr = (f'{"Порог итога стратегии":<26} {"Систем":>8} {"Медиана итога":>14}'
           f' {"На пике":>10} {"Случайный":>11} {"В минусе на пике":>18}')
    say(hdr)
    say("-" * len(hdr))
    for lo, hi, lab in ((0, .25, "0…+25 %"), (.25, 1.0, "+25…+100 %"),
                        (1.0, 5.0, "+100…+500 %"), (5.0, np.inf, "больше +500 %")):
        d = P[(P.total >= lo) & (P.total < hi)]
        if not len(d):
            continue
        say(f'{lab:<26} {len(d):>8,} {100*d.total.median():>13.1f}%'
            f' {100*d.from_peak.median():>9.1f}% {100*d.rand_med.median():>10.1f}%'
            f' {100*(d.from_peak < 0).mean():>17.1f}%')
    say()

    out = Path(__file__).resolve().parents[1] / "results" / "comon_peak_entry.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    say(f"[записано] {out}")


if __name__ == "__main__":
    main()
