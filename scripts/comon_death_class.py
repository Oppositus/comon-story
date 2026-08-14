"""comon_death_class.py — БЛОК 1.2–1.3: классификация смерти стратегий Comon.

План исследования, Блок 1. Опирается на правила методологии (Блок 0): метрики только
из rValue, дата смерти = ПОСЛЕДНИЙ ТОРГОВЫЙ ДЕНЬ, карточка архивной системы
недостоверна.

ДВЕ НЕЗАВИСИМЫЕ ОСИ (система может закрыться в плюсе, будучи год как сломанной):

  Ось I — финансовый исход:  разорение (maxDD ≥ 90 % или эквити ≤ 0.1) / убыток / прибыль.
  Ось II — механизм смерти:  не стартовала (0 торговых дней) · заброшена (нет сделок
    ≥ 90 дн до архивации) · смерть у пика (просадка ≤ 5 %) · обычная просадка ·
    ДЕГРАДАЦИЯ · недостаточно истории (< 250 торговых дней до последнего пика).

ОТЛИЧИТЬ ОБЫЧНУЮ ПРОСАДКУ ОТ ДЕГРАДАЦИИ (1.3) — два теста, вердикт только при согласии:

  ① блочный bootstrap СОБСТВЕННОГО распределения: история до последнего пика →
    circular block bootstrap (блок сохраняет автокорреляцию и кластеризацию волы) →
    B траекторий длиной с финальный отрезок → распределение МАКСИМАЛЬНОЙ просадки на
    отрезке такой длины. Сравниваем именно с максимумом, а не с типичной просадкой:
    финальную просадку мы разглядываем потому, что она последняя, — без этой поправки
    протащим ошибку отбора. Фактическая глубже 95-го перцентиля → своим риском не
    объясняется.
  ② точка разлома (CUSUM + перестановочный тест): есть ли момент сдвига средней
    доходности вниз, значим ли он с поправкой на поиск точки, и прожил ли новый режим
    ≥ 60 торговых дней. Деградация — устойчивый новый режим, а не одна плохая неделя.

Согласие обоих → «деградация»; один из двух → «недоказано».

Текстовый вывод + deaths.csv.gz (вход Блока 1.4). Запуск: python comon_death_class.py
"""
import gzip
import json
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "deaths.csv.gz"
LOG = ROOT / "results" / "comon_death_class.log"

NPROC = 8
B_BOOT = 2000          # траекторий бутстрапа (95-й перцентиль оценивается устойчиво)
BLOCK = 10             # длина блока, торговых дней
MIN_HIST = 250         # торговых дней до пика — иначе мощности на вердикт нет
MIN_REGIME = 60        # торговых дней жизни нового режима, иначе не деградация
ABANDON_DAYS = 90      # календарных дней без сделок до архивации = заброшена
PEAK_TOL = -0.05       # просадка не глубже 5 % = «смерть у пика»
N_PERM = 500           # перестановок для значимости точки разлома
ALPHA = 0.05
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def maxdd_rows(eq):
    """Максимальная просадка для каждой строки матрицы эквити."""
    peak = np.maximum.accumulate(eq, axis=1)
    return (eq / peak - 1.0).min(axis=1)


def boot_pvalue(hist, length, rng):
    """Доля бутстрап-траекторий, чья МАКСИМАЛЬНАЯ просадка глубже фактической.

    Возвращает (p, порог 5 %). p мал → фактическая просадка вне собственного опыта.
    """
    n = len(hist)
    nb = int(np.ceil(length / BLOCK))
    starts = rng.integers(0, n, size=(B_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n   # circular
    sample = hist[idx.reshape(B_BOOT, -1)[:, :length]]
    eq = np.cumprod(1.0 + sample, axis=1)
    mdd = maxdd_rows(eq)
    return mdd, float(np.percentile(mdd, 5))


def cusum_break(r, rng):
    """Точка сдвига среднего вниз + значимость с поправкой на поиск точки.

    Возвращает (индекс точки, p-value перестановочного теста, длина нового режима,
    среднее до, среднее после).
    """
    n = len(r)
    c = np.cumsum(r - r.mean())
    # 🔴 точка сдвига ВНИЗ = МАКСИМУМ накопленного отклонения: до неё доходность выше
    # средней (c растёт), после — ниже (c падает к нулю, куда cumsum приходит всегда).
    # argmin дал бы обратное — начало восходящего участка.
    k = int(np.argmax(c))
    stat = float(c[k])
    if k < 1 or n - k < 2:
        return k, 1.0, n - k, np.nan, np.nan
    # перестановочный тест: та же статистика на случайных перестановках того же ряда —
    # поправка на то, что точку разлома мы ИСКАЛИ, а не задали заранее
    perm = rng.permuted(np.tile(r, (N_PERM, 1)), axis=1)
    cc = np.cumsum(perm - perm.mean(axis=1, keepdims=True), axis=1)
    null = cc.max(axis=1)
    p = float((null >= stat).mean())
    return k, p, n - k, float(r[:k].mean()), float(r[k:].mean())


def load_returns(sid):
    """Дневные доходности ТОРГОВЫХ дней + их даты (ряд отдаётся в обратном порядке)."""
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None, None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if not s:
        return None, None
    s = s[::-1]
    r = np.array([float(x["rValue"] or 0.0) for x in s])
    d = np.array([x["date"] for x in s])
    m = r != 0.0
    return r[m], d[m]


def classify(args):
    """Классификация одной архивной системы по обеим осям."""
    sid, archived = args
    rng = np.random.default_rng(sid)
    out = {"id": sid}
    r, d = load_returns(sid)
    if r is None or len(r) == 0:
        out.update({"mech": "не стартовала", "fin": "нет данных", "n_trade": 0})
        return out
    eq = np.cumprod(1.0 + r)
    ret_total = float(eq[-1] - 1.0)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    death = d[-1]
    gap = (date.fromisoformat(archived) - date.fromisoformat(death)).days
    out.update({"n_trade": len(r), "death_date": death, "ret_total": ret_total,
                "maxdd": mdd, "abandon_gap": gap})
    # ── ось I: финансовый исход ──
    if mdd <= -0.90 or eq[-1] <= 0.1:
        out["fin"] = "разорение"
    elif ret_total < 0:
        out["fin"] = "убыток"
    else:
        out["fin"] = "прибыль"
    # ── ось II: механизм ──
    pk = int(np.argmax(eq))
    dd_final = float(eq[-1] / eq[pk] - 1.0)
    hist = r[:pk + 1]
    length = len(r) - 1 - pk
    out.update({"dd_final": dd_final, "n_hist": len(hist), "len_final": length})
    if gap >= ABANDON_DAYS:
        out["mech"] = "заброшена"
        return out
    if dd_final >= PEAK_TOL:
        out["mech"] = "смерть у пика"
        return out
    if len(hist) < MIN_HIST or length < 2:
        out["mech"] = "недостаточно истории"
        return out
    # ① собственное распределение максимальной просадки
    mdd_boot, thr5 = boot_pvalue(hist, length, rng)
    p_boot = float((mdd_boot <= dd_final).mean())
    # ② точка разлома на всём ряде
    k, p_perm, n_reg, m0, m1 = cusum_break(r, rng)
    ok_boot = p_boot < ALPHA
    ok_cpd = (p_perm < ALPHA) and (n_reg >= MIN_REGIME) and (m1 < m0)
    out.update({"p_boot": p_boot, "boot_thr5": thr5, "p_cpd": p_perm,
                "regime_len": n_reg, "mean_before": m0, "mean_after": m1,
                "break_at": d[k] if 0 <= k < len(d) else None})
    if ok_boot and ok_cpd:
        out["mech"] = "деградация"
    elif ok_boot or ok_cpd:
        out["mech"] = "недоказано"
    else:
        out["mech"] = "обычная просадка"
    return out


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    dead = panel[(panel["is_live"] == False) & panel["archived_at"].notna()]  # noqa: E712
    say(f"архивных систем: {len(dead)} (живые — цензурированные, идут в Блок 1.4)")
    say(f"бутстрап: B={B_BOOT}, блок {BLOCK} дн; порог истории {MIN_HIST} торговых дней;")
    say(f"новый режим ≥ {MIN_REGIME} дн; заброшена = ≥ {ABANDON_DAYS} календарных дней "
        f"без сделок до архивации")
    args = [(int(i), a) for i, a in dead["archived_at"].items()]
    with Pool(NPROC) as pool:
        rows = list(pool.imap_unordered(classify, args, chunksize=100))
    df = pd.DataFrame(rows).set_index("id").sort_index()
    df.to_csv(OUT, compression="gzip")

    MECH = ["не стартовала", "заброшена", "смерть у пика", "обычная просадка",
            "деградация", "недоказано", "недостаточно истории"]
    FIN = ["разорение", "убыток", "прибыль", "нет данных"]

    say()
    say("=" * 96)
    say("ОСЬ II — МЕХАНИЗМ СМЕРТИ")
    say("=" * 96)
    say(f'{"класс":<24} {"систем":>7} {"доля":>7} {"медиана дней жизни":>19} '
        f'{"медиана итога, %":>17}')
    life = (pd.to_datetime(df["death_date"]) - pd.to_datetime(
        panel.loc[df.index, "created_at"])).dt.days
    for m in MECH:
        g = df[df["mech"] == m]
        if not len(g):
            continue
        say(f'{m:<24} {len(g):>7} {100*len(g)/len(df):>6.1f}% '
            f'{life.reindex(g.index).median():>19.0f} '
            f'{100*g["ret_total"].median() if g["ret_total"].notna().any() else float("nan"):>16.1f}')
    say(f'{"ВСЕГО":<24} {len(df):>7}')

    say()
    say("=" * 96)
    say("ОСЬ I — ФИНАНСОВЫЙ ИСХОД")
    say("=" * 96)
    say(f'{"класс":<24} {"систем":>7} {"доля":>7} {"медиана итога, %":>17} '
        f'{"медиана maxDD, %":>17}')
    for f in FIN:
        g = df[df["fin"] == f]
        if not len(g):
            continue
        say(f'{f:<24} {len(g):>7} {100*len(g)/len(df):>6.1f}% '
            f'{100*g["ret_total"].median() if g["ret_total"].notna().any() else float("nan"):>16.1f} '
            f'{100*g["maxdd"].median() if g["maxdd"].notna().any() else float("nan"):>16.1f}')

    say()
    say("=" * 96)
    say("МАТРИЦА: механизм × исход (систем)")
    say("=" * 96)
    ct = pd.crosstab(df["mech"], df["fin"]).reindex(index=MECH, columns=FIN).fillna(0)
    hdr = f'{"механизм \\ исход":<24}' + "".join(f'{c:>13}' for c in FIN) + f'{"ИТОГО":>9}'
    say(hdr)
    say("-" * len(hdr))
    for m in MECH:
        if m not in ct.index or ct.loc[m].sum() == 0:
            continue
        say(f'{m:<24}' + "".join(f'{int(ct.loc[m, c]):>13,}' for c in FIN)
            + f'{int(ct.loc[m].sum()):>9,}')
    say("-" * len(hdr))
    say(f'{"ИТОГО":<24}' + "".join(f'{int(ct[c].sum()):>13,}' for c in FIN)
        + f'{len(df):>9,}')

    say()
    say("=" * 96)
    say("СОДЕРЖАТЕЛЬНЫЕ ЯЧЕЙКИ — ради чего строилась схема")
    say("=" * 96)
    for m, f, note in (("смерть у пика", "прибыль", "автор просто ушёл, машина цела"),
                       ("деградация", "прибыль", "механизм сломался, капитал уцелел"),
                       ("деградация", "убыток", "сломалась и потеряла"),
                       ("деградация", "разорение", "сломалась и разорилась"),
                       ("обычная просадка", "убыток", "проиграла в пределах своего риска")):
        g = df[(df["mech"] == m) & (df["fin"] == f)]
        say(f'{m} + {f}: {len(g):>5} систем — {note}')

    ok = df[df["mech"].isin(["обычная просадка", "деградация", "недоказано"])]
    say()
    say("=" * 96)
    say(f"ТЕСТЫ 1.3 НА ВЫБОРКЕ С ДОСТАТОЧНОЙ ИСТОРИЕЙ ({len(ok)} систем)")
    say("=" * 96)
    say(f'{"класс":<20} {"систем":>7} {"медиана фин.просадки":>21} '
        f'{"медиана порога 5 % бутстрапа":>28}')
    for m in ("обычная просадка", "недоказано", "деградация"):
        g = ok[ok["mech"] == m]
        if not len(g):
            continue
        say(f'{m:<20} {len(g):>7} {100*g["dd_final"].median():>20.1f}% '
            f'{100*g["boot_thr5"].median():>27.1f}%')
    say()
    say(f'бутстрап отверг «свой риск» (p<{ALPHA}): '
        f'{int((ok["p_boot"] < ALPHA).sum())} из {len(ok)} '
        f'({100*(ok["p_boot"] < ALPHA).mean():.1f} %)')
    cpd_ok = (ok["p_cpd"] < ALPHA) & (ok["regime_len"] >= MIN_REGIME) & \
             (ok["mean_after"] < ok["mean_before"])
    say(f'разлом подтверждён (p<{ALPHA}, режим ≥{MIN_REGIME} дн, дрейф вниз): '
        f'{int(cpd_ok.sum())} из {len(ok)} ({100*cpd_ok.mean():.1f} %)')
    say(f'СОГЛАСИЕ обоих тестов = деградация: {int((ok["mech"] == "деградация").sum())}')
    say(f'медиана длины нового режима у деградировавших: '
        f'{ok.loc[ok["mech"] == "деградация", "regime_len"].median():.0f} торговых дней')

    say()
    say("=" * 96)
    say("ЧУВСТВИТЕЛЬНОСТЬ К ПОРОГУ ИСТОРИИ (сколько систем вообще допущены к вердикту)")
    say("=" * 96)
    nh = df["n_hist"].dropna()
    say(f'систем с историей до пика ≥ {MIN_HIST} торговых дней: '
        f'{int((nh >= MIN_HIST).sum())} ({100*(nh >= MIN_HIST).mean():.1f} % архива)')
    for th in (50, 100, 250, 500):
        say(f'  порог {th:>4} дн: допущено {int((nh >= th).sum()):>6} систем')
    say(f'медиана истории до пика по архиву: {nh.median():.0f} торговых дней')

    say()
    say(f"файл: {OUT} ({OUT.stat().st_size/1e6:.1f} МБ)")
    LOG.write_text("\n".join(_lines) + "\n")
    say(f"лог: {LOG}")


if __name__ == "__main__":
    main()
