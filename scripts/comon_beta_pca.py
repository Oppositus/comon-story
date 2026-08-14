"""comon_beta_pca.py — БЛОК 6: корреляции. Одна ли это ставка на российский рынок?

План исследования, Блок 6. Вопрос блока — сколько в «доходности
автоследования» собственного сигнала, а сколько рыночной беты. Если популяцией правит
одна общая компонента, то сравнивать системы по CAGR бессмысленно: надо сравнивать
альфы, а бету можно купить за 0 % годовых фьючерсом на индекс.

Разделы:
  6.1 данные и метод (индекс, выравнивание, фильтры);
  6.2 распределение β, α и R² по популяции (+ бета в росте и в падении);
  6.3 сколько систем реально нейтральны — и лучше ли им живётся;
  6.4 PCA дневных: одна ли общая компонента и она ли — индекс;
  6.5 скрытый buy & hold с плечом.

🔴 ОГРАНИЧЕНИЯ.
  • IMOEX — ЦЕНОВОЙ индекс, без дивидендов (~7–9 % годовых). Система, держащая акции,
    получает дивиденды, которых в индексе нет → её α против IMOEX завышена на эту
    величину. Контроль — та же регрессия против MCFTR (индекс полной доходности).
  • Безрисковая ставка в основной регрессии НЕ вычитается (β от этого не зависит).
    Для α это важно: α_raw = α_excess + rf·(1 − β). Поправка считается отдельно на
    подвыборке 2019+ (RUSFAR есть только с 2019-01).
  • t(α) по обычному OLS: автокорреляция дневных доходностей систем автоследования
    (усреднение исполнения, лаг сделок) занижает se → t завышены. Читать как верхнюю
    границу значимости, отсюда порог |t| > 2 применяется с запасом.

Текстовый вывод. Запуск: python comon_beta_pca.py
"""
import gzip
import json
import subprocess
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"

from comon_data import require_raw                  # noqa: E402
require_raw(DIR, "profit")       # первичных данных нет в репозитории — см. DATA.md
OUT = DIR / "beta.csv.gz"
NPROC = 8

MIN_TRADE_DAYS = 100      # фильтр измеримости — общий для Блоков 2–6
MIN_ACTIVITY = 0.10
MIN_COMMON = 250          # общих с индексом дней, иначе β — шум
MIN_NONZERO = 30          # дней с ненулевой доходностью системы
NEUTRAL_BETA = 0.20       # |β| < этого — «рыночно нейтральна» (порог из плана)
BH_BETA = 0.80            # β ≥ этого + corr > BH_CORR — кандидат в buy & hold
BH_CORR = 0.90
ASOF = date(2026, 8, 5)
SERVICE_OWNERS = {2215}   # служебный аккаунт площадки, не автор (см. Блок 5)
PCA_WINDOWS = [("2014-08-01", "2017-08-01"), ("2017-08-01", "2020-08-01"),
               ("2020-08-01", "2023-08-01"), ("2023-08-01", "2026-08-01")]
PCA_COVER = 0.80          # доля дней окна, которые система должна покрывать

_lines = []
_IDX = None               # {дата -> доходность} рабочего индекса, для воркеров

CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv
# Полный прогон читает и пересчитывает β по всем рядам; величины для картинок
# сохраняются рядом, чтобы оформление правилось без пересчёта.
CACHE = ROOT / "results" / "comon_beta_pca_charts.npz"


def chart_pc1(rows):
    """График 20: доля первого фактора против эталона независимости."""
    import comon_charts as ch

    f, ax = ch.fig()
    x = list(range(len(rows)))
    obs = [r["pc1"] for r in rows]
    ind = [r["ind"] for r in rows]
    ax.plot(x, obs, color=ch.BLUE, lw=2.8, marker="o", ms=10, zorder=5,
            label="доля первого общего фактора в колебаниях популяции")
    ax.plot(x, ind, color=ch.ORANGE, lw=2.4, ls="--", marker="s", ms=9, zorder=4,
            label="сколько он забирал бы, будь стратегии независимы")
    ch.value_labels(ax, x, obs, [f"{ch.n_(v, 1)} %" for v in obs], dist=20,
                    color=ch.BLUE)
    ch.value_labels(ax, x, ind, [f"{ch.n_(v, 2)} %" for v in ind], dist=20,
                    color=ch.ORANGE)
    for xi, r in zip(x, rows):
        ax.annotate(f"в {ch.n_(r['pc1']/r['ind'])} раз больше",
                    xy=(xi, np.sqrt(r["pc1"] * r["ind"])), xytext=(0, 0),
                    textcoords="offset points", fontsize=11, color=ch.GREY,
                    ha="center", va="center")
    ax.set_yscale("log")
    ticks = (0.1, 0.3, 1, 3, 10, 30, 100)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels([f"{ch.n_(v, 1)} %" for v in ticks])
    # потолок с запасом: легенда уходит наверх, иначе внизу справа она накрывает
    # подпись последней точки нижней линии
    ax.set_ylim(0.07, 260)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['lab']}\n{ch.n_(r['n'])} стратегий" for r in rows])
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.set_xlabel("трёхлетнее окно")
    ax.set_ylabel("доля общих колебаний")
    ax.set_title("Десять разных стратегий — это примерно одна ставка на рынок")
    ax.legend(loc="upper left", handlelength=2.8, labelspacing=0.6)
    ch.note(ax,
            "В каждом окне берутся стратегии, покрывающие не меньше 80 % его "
            "торговых дней; по их дневным доходностям считаются главные "
            "компоненты. Шкала логарифмическая — иначе нижняя линия сливается "
            "с осью.",
            f"Первый фактор коррелирует с индексом Мосбиржи на "
            f"{ch.n_(min(r['corr'] for r in rows), 2)}–"
            f"{ch.n_(max(r['corr'] for r in rows), 2)}: общая компонента — это "
            f"рынок. Подписчик, разложивший деньги на десять стратегий, получает "
            f"не десять независимых ставок, а примерно одну рыночную с шумом.")
    ch.save(f, 20, "pc1-vs-independence")
    print("    " + "; ".join(f"{r['lab']}: {r['pc1']:.1f} % против {r['ind']:.2f} % "
                             f"(в {r['pc1']/r['ind']:.0f} раз)" for r in rows),
          flush=True)


def chart_beta_alpha(beta, alpha, neutral, sig, live):
    """График 21: карта популяции — бета по горизонтали, альфа по вертикали."""
    import comon_charts as ch

    f, ax = ch.fig(h_px=1060, bottom=0.28)
    b = np.asarray(beta, float)
    a = 100 * np.asarray(alpha, float)
    nt, sg = np.asarray(neutral, bool), np.asarray(sig, bool)
    good = nt & sg & (a > 0)

    # 🔴 Цветом кодируется ТОЛЬКО значимость альфы, а зависимость от рынка —
    # положением по горизонтали: классы «умеренная / сильная / против рынка» это
    # ровно диапазоны беты, и красить их отдельно значило бы дублировать ось
    # цветом. Шесть одновременных оттенков на одном поле проверку палитры не
    # проходят (зелёный и оранжевый неразличимы при протанопии, красный и
    # оранжевый — даже при обычном зрении), три — проходят с запасом: синий и
    # оранжевый различимы на ΔE 18+ при любой форме дальтонизма. Серая масса —
    # намеренно бесцветная: это фон, а не третья серия, и валидатор ругается на
    # её цветность по делу, но здесь это осознанный выбор.
    # 🔴 Классы — ровно те, что в таблице главы: «умеющие» это пересечение
    # нейтральности И значимости (241), а не все значимые подряд (435). Прежняя
    # версия красила по одной значимости, и на картинке синих точек было вдвое
    # больше, чем число, стоящее в подписи под ней.
    good = nt & sg & (a > 0)
    bad_ = nt & sg & (a < 0)
    rest = ~(good | bad_)
    classes = (
        (rest, ch.GREY, "o", 5, 0.20,
         "все остальные, включая зависимые от рынка"),
        (bad_, ch.ORANGE, "s", 26, 0.85, "нейтральные с доказанным минусом"),
        (good, ch.BLUE, "o", 30, 0.85, "нейтральные с доказанной альфой"))
    # Полоса нейтральности ЗАЛИТА, а не отмечена двумя пунктирами: полосу видно
    # как полосу, и подпись к ней встаёт сверху по центру, а не сбоку.
    ax.axvspan(-0.2, 0.2, color=ch.SEQ[0], alpha=0.35, lw=0, zorder=1)
    ax.axvline(0.8, color=ch.PALE, lw=1.6, ls=":", zorder=2)
    handles = []
    for m, color, marker, size, al, lab in classes:
        h = ax.scatter(b[m], a[m], s=size, marker=marker, color=color, alpha=al,
                       lw=0, zorder=3, label=f"{lab} — {ch.n_(m.sum())}")
        handles.append(h)
    ax.axhline(0, color=ch.INK, lw=1.2, zorder=2)
    # три коротких строки, а не две: полоса шириной 0,4 по оси занимает около
    # 170 px, и двухстрочная подпись оказывается шире неё — свисает вправо
    ax.annotate("полоса\nнейтральности\n|бета| < 0,2", xy=(0.0, 148),
                fontsize=11, color=ch.GREY, ha="center", va="center",
                multialignment="center")
    # подпись правой зоны — ПОД легендой: наверху справа стоит легенда
    ax.annotate("бета от 0,8 — рынок\nпод другим именем", xy=(1.5, 78),
                fontsize=11, color=ch.GREY, ha="center", va="center")

    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-120, 175)
    ch.pct_raw(ax, "y")
    ax.set_xlabel("чувствительность к индексу (бета)")
    ax.set_ylabel("альфа — доходность сверх\nрыночной, % годовых")
    ax.set_title("Карта популяции: где на витрине те, ради кого она существует")
    # герой графика — первой строкой легенды, серая масса — последней
    ax.legend(handles=handles[::-1], loc="upper right", handlelength=1.6,
              labelspacing=0.5, markerscale=2.0, fontsize=10)
    ch.note(ax,
            f"База — {ch.n_(len(b))} стратегий, у которых хватает общих с индексом "
            f"дней для оценки беты. Значения за краями поля не показаны.",
            f"Цветом выделены только два класса из таблицы главы — те, что внутри "
            f"полосы нейтральности и с доказанной альфой: {ch.n_(int(good.sum()))} "
            f"с плюсом ({ch.n_(100*good.mean(), 1)} % выборки, примерно одна из "
            f"шестнадцати) и {ch.n_(int(bad_.sum()))} с минусом. У первых медианный "
            f"Sharpe 1,39 при альфе +26,5 % годовых.",
            f"В серой массе есть ещё {ch.n_(int((sg & (a > 0) & ~nt).sum()))} "
            f"стратегий со значимой положительной альфой, но вне полосы: они "
            f"зависят от рынка, и отделить их собственный вклад от рыночного "
            f"нельзя. Справа от пунктира — бета от 0,8: это индекс под другим "
            f"именем, его результат покупается индексным фондом за десятые доли "
            f"процента в год.")
    ch.save(f, 21, "beta-alpha-map")
    print(f"    {len(b)} точек; с доказанной альфой {int(good.sum())} "
          f"({100*good.mean():.1f} %)", flush=True)


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


# ── индексы ───────────────────────────────────────────────────────────────────
def fetch_index(secid, y0=2005, y1=2026):
    """Дневные закрытия индекса MOEX с ISS. Кэш на диске (WebFetch к ISS не работает)."""
    f = DIR / f"idx_{secid}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for y in range(y0, y1 + 1):
        url = ("https://iss.moex.com/iss/history/engines/stock/markets/index/securities/"
               f"{secid}.json?iss.meta=off&from={y}-01-01&till={y}-12-31&limit=100")
        start = 0
        while True:
            j = json.loads(subprocess.check_output(
                ["curl", "-sL", f"{url}&start={start}"], text=True))
            cols, data = j["history"]["columns"], j["history"]["data"]
            if not data:
                break
            i_d, i_c = cols.index("TRADEDATE"), cols.index("CLOSE")
            for row in data:
                if row[i_c]:
                    out[row[i_d]] = float(row[i_c])
            start += len(data)
    f.write_text(json.dumps(out))
    return out


def index_returns(closes):
    """{дата -> простая дневная доходность}; простые, а не логарифмические — rValue тоже прост."""
    d = sorted(closes)
    v = np.array([closes[x] for x in d], dtype=np.float64)
    r = np.zeros(len(v)); r[1:] = v[1:] / v[:-1] - 1.0
    return {dd: float(rr) for dd, rr in zip(d[1:], r[1:])}


# ── ряды систем ───────────────────────────────────────────────────────────────
def load_one(sid):
    f = DIR / "profit" / f"{sid}.json.gz"
    if not f.exists():
        return None
    s = (json.loads(gzip.open(f, "rt").read()).get("data") or {}).get("strategy") or []
    if len(s) < 60:
        return None
    s = s[::-1]
    d = np.array([x["date"] for x in s], dtype=object)
    r = np.array([float(x["rValue"] or 0.0) for x in s], dtype=np.float32)
    return sid, d, r


def ols(y, x):
    """β, α (на день), t(α), R². Обычный МНК с константой."""
    n = len(y)
    if n < 10 or x.std() == 0:
        return (np.nan,) * 4
    X = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    ss = float(e @ e)
    tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss / tot if tot > 0 else np.nan
    s2 = ss / (n - 2)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_a = float(np.sqrt(s2 * xtx_inv[0, 0]))
    t_a = float(b[0] / se_a) if se_a > 0 else np.nan
    return float(b[1]), float(b[0]), t_a, float(r2)


def betas(dates, r, idx):
    """Все β-метрики одной системы против одного индекса."""
    m = np.array([idx.get(d, np.nan) for d in dates], dtype=np.float64)
    ok = np.isfinite(m)
    y, x = r[ok].astype(np.float64), m[ok]
    n = len(y)
    nz = int((y != 0).sum())
    if n < MIN_COMMON or nz < MIN_NONZERO:
        return None
    b, a, t, r2 = ols(y, x)
    corr = float(np.corrcoef(y, x)[0, 1]) if y.std() > 0 else np.nan
    act = y != 0
    b_act = ols(y[act], x[act])[0] if act.sum() >= 100 else np.nan
    up, dn = x > 0, x < 0
    b_up = ols(y[up], x[up])[0] if up.sum() >= 100 else np.nan
    b_dn = ols(y[dn], x[dn])[0] if dn.sum() >= 100 else np.nan
    return dict(n_com=n, n_nz=nz, beta=b, alpha=252 * a, t_alpha=t, r2=r2, corr=corr,
                beta_act=b_act, beta_up=b_up, beta_dn=b_dn,
                act_share=float(act.mean()))


def _work(item):
    sid, d, r = item
    out = {"id": sid}
    for tag, idx in _IDX.items():
        res = betas(d, r, idx)
        if res is None:
            if tag == "imoex":
                return None
            continue
        out.update({f"{k}_{tag}" if tag != "imoex" else k: v for k, v in res.items()})
    return out


def _init(idx):
    global _IDX
    _IDX = idx


def q(v, p):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else np.nan


def draw_from_cache():
    d = np.load(CACHE, allow_pickle=False)
    chart_beta_alpha(d["beta"], d["alpha"], d["neutral"], d["sig"], d["live"])
    if "pc1" in d.files:
        chart_pc1([{"lab": str(l), "n": int(n), "pc1": float(p1), "ind": float(i),
                    "corr": float(c)}
                   for l, n, p1, i, c in zip(d["lab"], d["wn"], d["pc1"],
                                             d["ind"], d["wcorr"])])


def main():
    if ONLY:
        draw_from_cache()
        return
    panel = pd.read_csv(DIR / "panel.csv.gz", index_col="id", low_memory=False)
    S = pd.read_csv(DIR / "sharpe.csv.gz", index_col="id")
    M = S.join(panel[["owner_id", "author", "risk_level", "transaction_rate",
                      "first", "last", "last_trade", "archived_at", "cagr",
                      "maxdd", "skew"]], how="inner", rsuffix="_p")
    M = M[~M["owner_id"].isin(SERVICE_OWNERS)]
    G = M[(M["n_trade"] >= MIN_TRADE_DAYS) & (M["activity"] >= MIN_ACTIVITY)].copy()
    G["live"] = G["is_live"] == True                                    # noqa: E712

    # ── 6.1 ────────────────────────────────────────────────────────────────────
    say("=" * 104)
    say("6.1 ДАННЫЕ И МЕТОД")
    say("=" * 104)
    say("Регрессия дневной доходности каждой системы на дневную доходность индекса:")
    say("  r_i(t) = α + β·r_m(t) + ε.  Индекс — IMOEX с ISS (в ответах Comon серия")
    say("бенчмарка пустая). Контроль — MCFTR (полная доходность, с дивидендами).")
    say("Выравнивание по торговым дням ИНДЕКСА: дни, которых у индекса нет (выходные,")
    say("праздники), выбрасываются вместе с доходностью системы за них.")
    say()
    IDX = {}
    for tag, secid in (("imoex", "IMOEX"), ("mcftr", "MCFTR")):
        cl = fetch_index(secid)
        IDX[tag] = index_returns(cl)
        dd = sorted(cl)
        say(f'{secid:<7} дней {len(cl):>6,}  окно {dd[0]} .. {dd[-1]}')
    say()
    say(f'измеримых систем (≥ {MIN_TRADE_DAYS} торговых дней, активность ≥ '
        f'{100*MIN_ACTIVITY:.0f} %): {len(G):,}')
    ids = list(G.index)
    with Pool(NPROC) as pool:
        ser = [x for x in pool.imap_unordered(load_one, ids, chunksize=100) if x]
    say(f'рядов загружено: {len(ser):,}')
    with Pool(NPROC, initializer=_init, initargs=(IDX,)) as pool:
        rows = [x for x in pool.imap_unordered(_work, ser, chunksize=50) if x]
    B = pd.DataFrame(rows).set_index("id")
    B = B.join(G[["sharpe_ar", "cagr", "vol", "years", "live", "owner_id", "author",
                  "risk_level", "transaction_rate", "followers", "min_sum", "maxdd",
                  "first", "last_trade", "last", "archived_at", "activity"]])
    B.to_csv(OUT)
    say(f'β посчитана для {len(B):,} систем (нужно ≥ {MIN_COMMON} общих дней с индексом')
    say(f'и ≥ {MIN_NONZERO} дней с ненулевой доходностью) → {OUT.name}')

    # ── 6.2 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("6.2 РАСПРЕДЕЛЕНИЕ β, α И R²")
    say("=" * 104)
    hdr = (f'{"величина":<34} {"p5":>8} {"p25":>8} {"медиана":>9} {"p75":>8} '
           f'{"p95":>8} {"среднее":>9}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col in (("β к IMOEX (все общие дни)", "beta"),
                     ("β к IMOEX (только активные дни)", "beta_act"),
                     ("β к MCFTR (с дивидендами)", "beta_mcftr"),
                     ("R² регрессии на IMOEX", "r2"),
                     ("корреляция с IMOEX", "corr"),
                     ("α к IMOEX, % годовых", "alpha"),
                     ("α к MCFTR, % годовых", "alpha_mcftr")):
        v = B[col].dropna()
        sc = 100 if col.startswith("alpha") else 1
        say(f'{lab:<34} {sc*q(v,5):>8.2f} {sc*q(v,25):>8.2f} {sc*q(v,50):>9.2f} '
            f'{sc*q(v,75):>8.2f} {sc*q(v,95):>8.2f} '
            f'{sc*float(stats.trim_mean(v.to_numpy(), 0.01)):>9.2f}')
    say()
    say("«все общие дни» — включая дни, когда система стояла вне рынка (r = 0): такие дни")
    say("тянут β к нулю механически. «только активные дни» — β в те дни, когда система")
    say("реально несла риск; разница между строками показывает, сколько «нейтральности»")
    say("даёт простое отсутствие в рынке. У α показано усечённое среднее (1 %) — витрина")
    say("содержит системы с доходностями в тысячи процентов, обычное среднее ими рушится.")
    say()
    hdr = (f'{"группа":<26} {"систем":>8} {"β мед.":>8} {"R² мед.":>9} '
           f'{"corr мед.":>10} {"α мед., %":>10} {"доля β>0.5":>11} {"доля |β|<0.2":>13}')
    say(hdr)
    say("-" * len(hdr))
    for lab, sub in (("вся выборка", B), ("живые", B[B["live"] == True]),          # noqa: E712
                     ("мёртвые", B[B["live"] == False]),                           # noqa: E712
                     ("история ≥ 3 лет", B[B["years"] >= 3]),
                     ("история < 1 года", B[B["years"] < 1])):
        if not len(sub):
            continue
        say(f'{lab:<26} {len(sub):>8,} {sub["beta"].median():>8.2f} '
            f'{sub["r2"].median():>9.3f} {sub["corr"].median():>10.2f} '
            f'{100*sub["alpha"].median():>9.1f}% '
            f'{100*(sub["beta"] > 0.5).mean():>10.1f}% '
            f'{100*(sub["beta"].abs() < NEUTRAL_BETA).mean():>12.1f}%')
    say()
    say("── бета в росте и в падении рынка ──")
    say("β_up считается только по дням роста индекса, β_dn — только по дням падения.")
    say("Если β_dn > β_up, популяция ловит падения сильнее, чем подъёмы: асимметрия")
    say("не в пользу подписчика (это профиль «продавца риска», а не хеджа).")
    say()
    hdr = (f'{"величина":<34} {"p25":>8} {"медиана":>9} {"p75":>8} {"среднее":>9}')
    say(hdr)
    say("-" * len(hdr))
    for lab, col in (("β в дни роста индекса", "beta_up"),
                     ("β в дни падения индекса", "beta_dn")):
        v = B[col].dropna()
        say(f'{lab:<34} {q(v,25):>8.2f} {q(v,50):>9.2f} {q(v,75):>8.2f} '
            f'{float(stats.trim_mean(v.to_numpy(), 0.01)):>9.2f}')
    d = (B["beta_dn"] - B["beta_up"]).dropna()
    w = stats.wilcoxon(d) if len(d) > 20 else None
    say(f'{"разность β_dn − β_up":<34} {q(d,25):>8.2f} {q(d,50):>9.2f} {q(d,75):>8.2f} '
        f'{float(stats.trim_mean(d.to_numpy(), 0.01)):>9.2f}')
    if w is not None:
        say(f'знаковый тест Уилкоксона на разность: p = {w.pvalue:.2e} '
            f'(доля систем с β_dn > β_up: {100*(d > 0).mean():.1f} %)')

    # ── поправка на безрисковую ставку ─────────────────────────────────────────
    rf = json.loads((DIR / "rusfar.json").read_text())
    rf_mean = float(np.mean(list(rf.values())))
    say()
    say(f'Поправка на ставку: α_raw = α_ex + rf·(1 − β). При средней RUSFAR '
        f'{100*rf_mean:.1f} % годовых')
    say(f'и медианной β {B["beta"].median():.2f} это '
        f'{100*rf_mean*(1-B["beta"].median()):.1f} п.п. годовых, на которые показанная')
    say("α выше настоящей избыточной. Ставка есть только с 2019-01 (RUSFAR), поэтому")
    say("в таблицы она не заводится — держим в уме как сдвиг вверх.")

    # ── 6.3 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("6.3 СКОЛЬКО СИСТЕМ РЕАЛЬНО НЕЙТРАЛЬНЫ")
    say("=" * 104)
    say(f'Нейтральная = |β| < {NEUTRAL_BETA}. Значимая α = |t(α)| > 2 по МНК')
    say("(t завышены из-за автокорреляции — порог мягкий, доля «со значимой α» — верхняя")
    say("граница). Разрез по знаку α важен: значимая ОТРИЦАТЕЛЬНАЯ α — тоже «значимая».")
    say()
    B["neutral"] = B["beta"].abs() < NEUTRAL_BETA
    B["sig"] = B["t_alpha"].abs() > 2

    if CHARTS:
        np.savez(CACHE, beta=B["beta"].to_numpy(float),
                 alpha=B["alpha"].to_numpy(float),
                 neutral=B["neutral"].to_numpy(bool), sig=B["sig"].to_numpy(bool),
                 live=(B["live"] == True).to_numpy(bool))       # noqa: E712
        say("── график 21 ────────────────────────────────────────────────────")
        chart_beta_alpha(B["beta"], B["alpha"], B["neutral"], B["sig"],
                         B["live"] == True)                     # noqa: E712
        say()
    hdr = (f'{"класс":<40} {"систем":>8} {"доля":>8} {"S мед.":>8} {"α мед., %":>10} '
           f'{"живых":>8} {"лет мед.":>9}')
    say(hdr)
    say("-" * len(hdr))
    cls = [("нейтральные |β| < 0.2", B["neutral"]),
           ("  из них α значима и > 0", B["neutral"] & B["sig"] & (B["alpha"] > 0)),
           ("  из них α значима и < 0", B["neutral"] & B["sig"] & (B["alpha"] < 0)),
           ("  из них α не значима", B["neutral"] & ~B["sig"]),
           ("умеренная бета 0.2 ≤ β < 0.8", (B["beta"] >= NEUTRAL_BETA) & (B["beta"] < BH_BETA)),
           ("высокая бета β ≥ 0.8", B["beta"] >= BH_BETA),
           ("отрицательная бета β ≤ −0.2", B["beta"] <= -NEUTRAL_BETA)]
    for lab, m in cls:
        sub = B[m]
        if not len(sub):
            continue
        say(f'{lab:<40} {len(sub):>8,} {100*len(sub)/len(B):>7.1f}% '
            f'{sub["sharpe_ar"].median():>8.2f} {100*sub["alpha"].median():>9.1f}% '
            f'{100*(sub["live"] == True).mean():>7.1f}% '                          # noqa: E712
            f'{sub["years"].median():>9.1f}')
    say()
    say("«доля» — от систем с посчитанной β. «S мед.» — медианный годовой Sharpe (Блок 2).")
    say("«живых» — доля не архивированных на 2026-08-05. «лет мед.» — медианная длина")
    say("истории (это НЕ выживаемость, живые ещё торгуют).")
    say()
    say("── выживаемость по Каплану–Мейеру ──")
    from comon_survival import km, km_at, km_median                       # noqa: E402
    hdr = (f'{"класс":<40} {"систем":>8} {"1 год":>8} {"3 года":>8} {"5 лет":>8} '
           f'{"медиана жизни":>15}')
    say(hdr)
    say("-" * len(hdr))
    for lab, m in (("нейтральные |β| < 0.2", B["neutral"]),
                   ("бета-системы β ≥ 0.2", B["beta"] >= NEUTRAL_BETA),
                   ("высокая бета β ≥ 0.8", B["beta"] >= BH_BETA)):
        sub = B[m]
        T, E = [], []
        for _, r in sub.iterrows():
            if not isinstance(r["first"], str):
                continue
            d0 = date.fromisoformat(r["first"])
            if r["live"] == True:                                          # noqa: E712
                T.append((ASOF - d0).days / 365.25); E.append(0)
            else:
                end = r["last_trade"] if isinstance(r["last_trade"], str) else r["last"]
                if not isinstance(end, str):
                    continue
                T.append((date.fromisoformat(end) - d0).days / 365.25); E.append(1)
        if len(T) < 30:
            continue
        t, s, se = km(np.array(T, dtype=float), np.array(E, dtype=int))
        med = km_median(t, s)
        say(f'{lab:<40} {len(T):>8,} {100*km_at(t,s,se,1)[0]:>7.1f}% '
            f'{100*km_at(t,s,se,3)[0]:>7.1f}% {100*km_at(t,s,se,5)[0]:>7.1f}% '
            f'{(f"{med:.1f} г" if np.isfinite(med) else "не достигнута"):>15}')

    # ── 6.4 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("6.4 PCA: ОДНА ЛИ ОБЩАЯ КОМПОНЕНТА ДВИЖЕТ ВСЕМИ")
    say("=" * 104)
    say("В каждом трёхлетнем окне берём системы, покрывающие ≥ 80 % торговых дней окна,")
    say("считаем главные компоненты КОРРЕЛЯЦИОННОЙ матрицы их дневных доходностей.")
    say("Доля дисперсии первой компоненты — мера «одной общей ставки»: у независимых")
    say("систем она была бы ≈ 1/N, у одинаковых — 100 %. Затем смотрим, ЧТО это за")
    say("компонента: корреляцию её счёта с дневным IMOEX.")
    say()
    SER = {sid: (d, r) for sid, d, r in ser}
    idx = IDX["imoex"]
    hdr = (f'{"окно":<24} {"систем":>8} {"дней":>7} {"PC1":>7} {"PC2":>7} {"PC3":>7} '
           f'{"PC1–5":>8} {"corr(PC1, IMOEX)":>18} {"1/N (эталон)":>14}')
    say(hdr)
    say("-" * len(hdr))
    pca_rows, chart_rows = [], []
    for w0, w1 in PCA_WINDOWS:
        days = sorted(d for d in idx if w0 <= d < w1)
        if len(days) < 400:
            continue
        pos = {d: i for i, d in enumerate(days)}
        Xs, keep = [], []
        for sid, (dd, rr) in SER.items():
            v = np.full(len(days), np.nan, dtype=np.float64)
            for d, x in zip(dd, rr):
                j = pos.get(d)
                if j is not None:
                    v[j] = x
            cov = np.isfinite(v).mean()
            if cov < PCA_COVER:
                continue
            v = np.nan_to_num(v)
            if v.std() == 0 or (v != 0).mean() < MIN_ACTIVITY:
                continue
            Xs.append(v); keep.append(sid)
        if len(Xs) < 30:
            continue
        X = np.array(Xs)
        Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        C = (Z @ Z.T) / Z.shape[1]
        ev, evec = np.linalg.eigh(C)
        ev = ev[::-1]; evec = evec[:, ::-1]
        share = ev / ev.sum()
        pc1 = evec[:, 0] @ Z                      # счёт первой компоненты по дням
        m = np.array([idx[d] for d in days])
        if np.corrcoef(pc1, m)[0, 1] < 0:         # знак компоненты произволен
            pc1 = -pc1; evec[:, 0] = -evec[:, 0]
        c1 = float(np.corrcoef(pc1, m)[0, 1])
        say(f'{w0[:7]} .. {w1[:7]:<13} {len(Xs):>8,} {len(days):>7,} '
            f'{100*share[0]:>6.1f}% {100*share[1]:>6.1f}% {100*share[2]:>6.1f}% '
            f'{100*share[:5].sum():>7.1f}% {c1:>18.2f} {100/len(Xs):>13.2f}%')
        chart_rows.append({"lab": f"{w0[:4]}–{w1[:4]}", "n": len(Xs),
                           "pc1": 100 * float(share[0]),
                           "ind": 100.0 / len(Xs), "corr": c1})
        pca_rows.append((w0, w1, keep, evec[:, 0], share[0], c1,
                         float(np.corrcoef(pc1, m)[0, 1])))
    say()
    if CHARTS:
        d = dict(np.load(CACHE)) if CACHE.exists() else {}
        d.update({"lab": np.array([r["lab"] for r in chart_rows]),
                  "wn": np.array([r["n"] for r in chart_rows], float),
                  "pc1": np.array([r["pc1"] for r in chart_rows]),
                  "ind": np.array([r["ind"] for r in chart_rows]),
                  "wcorr": np.array([r["corr"] for r in chart_rows])})
        np.savez(CACHE, **d)
        say("── график 20 ────────────────────────────────────────────────────")
        chart_pc1(chart_rows)
        say()

    say("PC1..PC3 — доля общей дисперсии, объяснённая 1-й, 2-й, 3-й компонентой; PC1–5 —")
    say("первыми пятью. «1/N» — сколько дала бы первая компонента, будь системы")
    say("независимы (эталон нуля). corr(PC1, IMOEX) — насколько общая компонента и есть")
    say("рынок: близко к 1 = вся популяция едет на индексе.")
    say()
    if pca_rows:
        w0, w1, keep, load, sh1, c1, _ = pca_rows[-1]
        say(f'── нагрузки на PC1 в последнем окне ({w0[:7]} .. {w1[:7]}) ──')
        ld = pd.Series(load, index=keep)
        say(f'доля систем с положительной нагрузкой: {100*(ld > 0).mean():.1f} %')
        say(f'квантили нагрузки: p5 {q(ld,5):+.3f} · медиана {q(ld,50):+.3f} · '
            f'p95 {q(ld,95):+.3f}')
        j = B.index.intersection(ld.index)
        rho = stats.spearmanr(ld.loc[j], B.loc[j, "beta"]).statistic
        say(f'корреляция нагрузки на PC1 с β к IMOEX (Спирмен): {rho:+.3f}')
        say("Если нагрузки одного знака у подавляющего большинства и коррелируют с β —")
        say("первая компонента и есть рыночная бета, а не какой-то общий «стиль».")

    # ── 6.5 ────────────────────────────────────────────────────────────────────
    say()
    say("=" * 104)
    say("6.5 СКРЫТЫЙ BUY & HOLD С ПЛЕЧОМ")
    say("=" * 104)
    say(f'Критерий: β ≥ {BH_BETA} И корреляция с IMOEX > {BH_CORR} И α не значима')
    say("(|t(α)| ≤ 2). Такая система не даёт подписчику ничего, чего он не получил бы,")
    say("купив фьючерс на индекс: её результат — рыночный риск, умноженный на плечо.")
    say()
    say("Строгий критерий по корреляции почти никого не ловит: дневная доходность")
    say("на счёте подписчика — это индекс ПЛЮС шум исполнения (сделки в течение дня,")
    say("не тот инструмент, частичное плечо), и corr падает даже у чистой бета-системы.")
    say("Поэтому ступенчато: как меняется число «просто рынка» при ослаблении порога.")
    say()
    hdr = (f'{"критерий":<46} {"систем":>8} {"доля":>7} {"β мед.":>8} '
           f'{"α мед., %":>10} {"S мед.":>8}')
    say(hdr)
    say("-" * len(hdr))
    crit = [(f'β ≥ {BH_BETA} и corr > {BH_CORR}',
             (B["beta"] >= BH_BETA) & (B["corr"] > BH_CORR)),
            (f'β ≥ {BH_BETA} и corr > 0.8',
             (B["beta"] >= BH_BETA) & (B["corr"] > 0.80)),
            (f'β ≥ {BH_BETA} и corr > 0.7',
             (B["beta"] >= BH_BETA) & (B["corr"] > 0.70)),
            (f'β ≥ {BH_BETA} и corr > 0.7, α не значима',
             (B["beta"] >= BH_BETA) & (B["corr"] > 0.70) & ~B["sig"]),
            ('β ≥ 0.5 и corr > 0.5', (B["beta"] >= 0.5) & (B["corr"] > 0.50)),
            ('β ≥ 0.5 и corr > 0.5, α не значима',
             (B["beta"] >= 0.5) & (B["corr"] > 0.50) & ~B["sig"]),
            ('β ≥ 0.5 и corr > 0.5, активны > 80 % дней',
             (B["beta"] >= 0.5) & (B["corr"] > 0.50) & (B["act_share"] > 0.80))]
    for lab, m in crit:
        sub = B[m]
        if not len(sub):
            say(f'{lab:<46} {0:>8}')
            continue
        say(f'{lab:<46} {len(sub):>8,} {100*len(sub)/len(B):>6.1f}% '
            f'{sub["beta"].median():>8.2f} {100*sub["alpha"].median():>9.1f}% '
            f'{sub["sharpe_ar"].median():>8.2f}')
    bh = (B["beta"] >= 0.5) & (B["corr"] > 0.50) & ~B["sig"]
    say()
    say("Далее «скрытый buy & hold» = самая широкая честная версия: β ≥ 0.5, corr > 0.5,")
    say("α не значима. Это системы, чей результат объясняется рынком и не содержит")
    say("различимого собственного вклада.")
    say()
    hdr = (f'{"класс":<30} {"систем":>8} {"β мед.":>8} {"corr":>7} {"S мед.":>8} '
           f'{"CAGR мед.":>10} {"maxDD мед.":>11} {"живых":>8} {"подписчиков":>12}')
    say(hdr)
    say("-" * len(hdr))
    for lab, m in (("скрытый buy & hold", bh),
                   ("остальные", ~bh)):
        sub = B[m]
        say(f'{lab:<30} {len(sub):>8,} {sub["beta"].median():>8.2f} '
            f'{sub["corr"].median():>7.2f} {sub["sharpe_ar"].median():>8.2f} '
            f'{100*sub["cagr"].median():>9.1f}% {100*sub["maxdd"].median():>10.1f}% '
            f'{100*(sub["live"] == True).mean():>7.1f}% '                          # noqa: E712
            f'{int(sub["followers"].fillna(0).sum()):>12,}')
    say()
    say("«подписчиков» — суммарный followerCount (у мёртвых он обнулён площадкой, так что")
    say("это фактически счёт подписок живых систем класса).")
    say()
    say("── за что голосуют деньги: подписки по группам беты ──")
    say("Считаем по ЖИВЫМ (у мёртвых followerCount обнулён). Вопрос: толпа подписана")
    say("на нейтральные системы или на рыночный риск с плечом?")
    say()
    L = B[(B["live"] == True) & B["followers"].notna()].copy()             # noqa: E712
    tot_f = L["followers"].sum()
    hdr = (f'{"группа по β":<28} {"систем":>8} {"подписок":>10} {"доля подписок":>14} '
           f'{"подписок на систему":>21} {"S мед.":>8}')
    say(hdr)
    say("-" * len(hdr))
    grp = [("|β| < 0.2 (нейтральные)", L["beta"].abs() < NEUTRAL_BETA),
           ("0.2 ≤ β < 0.5", (L["beta"] >= NEUTRAL_BETA) & (L["beta"] < 0.5)),
           ("0.5 ≤ β < 1.0", (L["beta"] >= 0.5) & (L["beta"] < 1.0)),
           ("β ≥ 1.0 (с плечом)", L["beta"] >= 1.0),
           ("β ≤ −0.2 (против рынка)", L["beta"] <= -NEUTRAL_BETA)]
    for lab, m in grp:
        sub = L[m]
        if not len(sub):
            continue
        say(f'{lab:<28} {len(sub):>8,} {int(sub["followers"].sum()):>10,} '
            f'{100*sub["followers"].sum()/tot_f:>13.1f}% '
            f'{sub["followers"].mean():>21.1f} {sub["sharpe_ar"].median():>8.2f}')
    s = L[["beta", "followers"]].dropna()
    say(f'\nкорреляция β с числом подписчиков (Спирмен, живые): '
        f'{stats.spearmanr(s["beta"], s["followers"]).statistic:+.3f} (n = {len(s):,})')
    say()
    say("── отражает ли витрина реальную бету ──")
    say("risk_level — уровень риска из карточки (шкала площадки 0–3), transaction_rate —")
    say("заявленная частота сделок (monthly=1 · weekly=2 · daily=3 · seldom=0).")
    say()
    TR = {"seldom": 0, "monthly": 1, "weekly": 2, "daily": 3}
    B["tr_num"] = B["transaction_rate"].map(TR)
    for lab, col in (("β ↔ risk_level из карточки", "risk_level"),
                     ("β ↔ частота сделок из карточки", "tr_num"),
                     ("β ↔ волатильность системы", "vol"),
                     ("β ↔ Sharpe", "sharpe_ar"),
                     ("β ↔ доля активных дней", "act_share"),
                     ("β ↔ длина истории, лет", "years")):
        s = B[["beta", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(s) < 50 or s[col].nunique() < 2:
            continue
        say(f'{lab:<36} {stats.spearmanr(s["beta"], s[col]).statistic:>+8.3f}   '
            f'(n = {len(s):,})')

    (ROOT / "results" / "comon_beta_pca.log").write_text(
        "\n".join(_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
