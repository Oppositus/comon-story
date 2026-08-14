"""comon_money_lower.py — НИЖНЯЯ ОЦЕНКА ДЕНЕГ (₽) ИЗ ПОДПИСОК: подписки x порог входа.

Пункт «Общее 3» ручной вычитки серии: везде, где
серия говорит «деньги», она на самом деле считает ПОДПИСКИ, взвешенные счётчиком. Здесь
тот же счёт переводится в рубли по единственному денежному признаку, который есть в
данных, — минимальному порогу входа стратегии:

    money_i = followers_i x min_sum_i        (только живые: у архивных счётчик обнулён)

🔴 ЧТО ЭТО ЗНАЧИТ И ЧЕГО НЕ ЗНАЧИТ.
  - СУММА по всем стратегиям — честная нижняя граница: реальный счёт подписчика не
    может быть меньше порога, значит суммарные активы не меньше полученного числа.
  - ДОЛИ (какая часть денег лежит там-то) нижней оценкой НЕ являются. Они верны при
    допущении «у каждого подписчика на счёте ровно порог». Если у дорогих стратегий
    счета ближе к порогу, чем у дешёвых, доли дорогих завышены, и наоборот. Проверить
    это нечем: индивидуальных записей нет.
  - Поэтому каждая таблица ниже показывает ОБА веса — подписки и рубли, — и содержателен
    здесь именно РАЗРЫВ между ними, а не рублёвое число само по себе.

Разделы:
  1 контроль суммы (сверка с 3.94 млрд ₽ и средним чеком глав 4-5);
  2 концентрация: Джини и доли топов в подписках против рублей (стратегии и авторы);
  3 квинтили порога входа — место, где два веса обязаны разойтись сильнее всего;
  4 заявленная частота сделок;
  5 тариф: доля денег у стратегий с вариантом платы от результата;
  6 качество купленного: медианы, взвешенные подписками против рублей;
  7 окно >= 3 лет: чистая доходность, взвешенная подписками против рублей.

Запуск: python comon_money_lower.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comon_tariffs import card_tariffs, work, _init              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data"
NPROC = 8

MIN_TRADE_DAYS = 100      # фильтр измеримости — тот же, что в главах 3-5
MIN_ACTIVITY = 0.10
_lines = []


def say(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def gini(x, w=None):
    """Коэффициент Джини по величине x с весами w (без весов — равные)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    if cxw[-1] <= 0:
        return np.nan
    return float(1 - np.sum((cxw[:-1] + cxw[1:]) * np.diff(cw)) / (cxw[-1] * cw[-1]))


def wmedian(v, w):
    """Взвешенная медиана: значение, где накопленный вес переходит половину."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if not len(v):
        return np.nan
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


CHARTS = "--charts" in sys.argv or "--charts-only" in sys.argv
ONLY = "--charts-only" in sys.argv


def lorenz(v):
    """Точки кривой Лоренца: (доля участников снизу вверх, накопленная доля величины)."""
    v = np.sort(np.asarray(v, dtype=np.float64))
    x = np.arange(len(v) + 1) / len(v)
    y = np.concatenate([[0.0], np.cumsum(v) / v.sum()])
    return x, y


def chart_lorenz(live, au):
    """График 11: концентрация подписок — по авторам и по стратегиям."""
    import comon_charts as ch

    f, ax = ch.fig()
    zero = float((au.followers == 0).mean())

    ax.plot([0, 1], [0, 1], color=ch.PALE, lw=1.6, ls=":", zorder=2,
            label="если бы подписки делились поровну")
    xs, ys = lorenz(live.followers)
    ax.plot(xs, ys, color=ch.ORANGE, lw=2.2, ls="--", zorder=3,
            label=f"по стратегиям ({ch.n_(len(live))}), "
                  f"Джини {ch.n_(gini(live.followers), 3)}")
    xa, ya = lorenz(au.followers)
    ax.plot(xa, ya, color=ch.BLUE, lw=2.6, zorder=4,
            label=f"по авторам ({ch.n_(len(au))}), "
                  f"Джини {ch.n_(gini(au.followers), 3)}")
    ax.plot([zero], [0], marker="o", ms=8, color=ch.BLUE, zorder=5)

    ax.annotate(f"{int((au.followers == 0).sum())} авторов из {len(au)}\n"
                f"({ch.n_(100*zero, 1)} %) — ни одного подписчика",
                # 🔴 подпись — в пустой карман между кривыми и диагональю (справа
                # внизу), в данных, а не в смещении: в смещении она ложилась
                # прямо на диагональ равенства
                xy=(zero, 0.005), xytext=(0.33, 0.13), textcoords="data",
                fontsize=11, color=ch.GREY, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=ch.GREY, lw=1.0))

    ax.legend(loc="upper left", handlelength=2.6, labelspacing=0.55)
    ch.pct(ax, "x")
    ch.pct(ax, "y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("участники рынка, от самых мелких к крупным")
    ax.set_ylabel("накопленная доля всех подписок")
    ax.set_title("Рынок хитов: половину подписок держат пятеро авторов")
    ch.note(ax,
            f"База — {ch.n_(len(live))} живых стратегий и "
            f"{ch.n_(live.followers.sum())} подписок между {ch.n_(len(au))} "
            f"авторами. Кривая Лоренца: чем глубже она провисает под диагональю, "
            f"тем сильнее концентрация.",
            "Верхние пять авторов держат 50,3 % подписок, верхняя десятка — 64,0 %, "
            "верхняя полусотня — 91,2 %. Аудитория собирается вокруг людей плотнее, "
            "чем вокруг стратегий.")
    ch.save(f, 11, "lorenz-concentration")
    print(f"    Джини авторы {gini(au.followers):.3f} / стратегии "
          f"{gini(live.followers):.3f}; без подписчиков {100*zero:.1f} %", flush=True)


def two_weights(df, groups, label):
    """Таблица «доля подписок против доли рублей» по группам."""
    hdr = (f'{label:<28} {"Стратегий":>10} {"Подписок":>10} {"Доля подп.":>11}'
           f' {"Рублей, млн":>12} {"Доля денег":>11} {"Чек, тыс ₽":>11}')
    say(hdr)
    say("-" * len(hdr))
    S, M = df["followers"].sum(), df["money"].sum()
    for name, m in groups:
        d = df[m]
        s, mo = d["followers"].sum(), d["money"].sum()
        chk = mo / s / 1e3 if s else np.nan
        say(f'{name:<28} {len(d):>10,} {int(s):>10,} {100*s/S:>10.1f} %'
            f' {mo/1e6:>12,.0f} {100*mo/M:>10.1f} % {chk:>11,.0f}')
    say("-" * len(hdr))
    say(f'{"ВСЕГО":<28} {len(df):>10,} {int(S):>10,} {100.0:>10.1f} %'
        f' {M/1e6:>12,.0f} {100.0:>10.1f} % {M/S/1e3:>11,.0f}')
    say()


def main():
    panel = pd.read_csv(DIR / "panel.csv.gz", low_memory=False).set_index("id")
    live = panel[panel["is_live"] == 1].copy()
    live["money"] = live["followers"] * live["min_sum"]

    say("=" * 108)
    say("НИЖНЯЯ ОЦЕНКА ДЕНЕГ (₽) ИЗ ПОДПИСОК: подписки x минимальный порог входа")
    say("=" * 108)
    say("Доли в рублях — не нижняя оценка, а счёт при допущении «счёт = порог входа».")
    say("Содержателен разрыв между двумя весами, а не рублёвое число само по себе.")
    say()

    # ── 1. контроль ───────────────────────────────────────────────────────────
    say("=" * 108)
    say("1. КОНТРОЛЬ СУММЫ")
    say("=" * 108)
    S, M = live["followers"].sum(), live["money"].sum()
    say(f'живых стратегий:            {len(live):>12,}')
    say(f'из них с подписками:        {int((live.followers > 0).sum()):>12,}')
    say(f'подписок всего:             {int(S):>12,}')
    say(f'нижняя оценка активов:      {M/1e9:>12,.2f} млрд ₽   (в главах 4-5: 3.94)')
    say(f'средний чек (сумма/подписки): {M/S/1e3:>10,.0f} тыс ₽   (в главах 4-5: 293)')
    say(f'медианный порог входа живой стратегии: {live.min_sum.median()/1e3:>6,.0f} тыс ₽')
    say(f'порог входа, взвешенный подписками (медиана): '
        f'{wmedian(live.min_sum, live.followers)/1e3:>6,.0f} тыс ₽')
    say()

    # ── 2. концентрация ───────────────────────────────────────────────────────
    say("=" * 108)
    say("2. КОНЦЕНТРАЦИЯ: ПОДПИСКИ ПРОТИВ РУБЛЕЙ")
    say("=" * 108)
    ls = live.sort_values("followers", ascending=False)
    lm = live.sort_values("money", ascending=False)
    say(f'{"Верхние N стратегий":<28} {"по подпискам":>16} {"по рублям":>16}')
    say("-" * 62)
    for n in (10, 50, 100, 200):
        say(f'{f"топ-{n}":<28} {100*ls.followers.head(n).sum()/S:>15.1f} %'
            f' {100*lm.money.head(n).sum()/M:>15.1f} %')
    say(f'{"Джини по стратегиям":<28} {gini(live.followers):>15.3f}  {gini(live.money):>15.3f}')
    say()
    au = live.groupby("owner_id").agg(followers=("followers", "sum"),
                                      money=("money", "sum"), n=("followers", "size"))
    au_s = au.sort_values("followers", ascending=False)
    au_m = au.sort_values("money", ascending=False)
    say(f'{"Верхние N авторов":<28} {"по подпискам":>16} {"по рублям":>16}')
    say("-" * 62)
    for n in (5, 10, 50):
        say(f'{f"топ-{n}":<28} {100*au_s.followers.head(n).sum()/S:>15.1f} %'
            f' {100*au_m.money.head(n).sum()/M:>15.1f} %')
    say(f'{"Джини по авторам":<28} {gini(au.followers):>15.3f}  {gini(au.money):>15.3f}')
    say(f'авторов с живыми стратегиями: {len(au):,}; из них без подписок: '
        f'{int((au.followers == 0).sum()):,} ({100*(au.followers == 0).mean():.1f} %)')
    say()

    if CHARTS:
        say("── график 11 ────────────────────────────────────────────────────")
        chart_lorenz(live, au)
        if ONLY:                     # лог не переписываем: прогон неполный
            return

    # ── 2б. авторы: подписки против рублей ────────────────────────────────────
    say("=" * 108)
    say("2б. АВТОРЫ: ПОДПИСКИ ПРОТИВ РУБЛЕЙ (у кого порог входа выше)")
    say("=" * 108)
    au2 = live.groupby("owner_id").agg(
        author=("author", lambda x: x.mode().iat[0]), n=("followers", "size"),
        followers=("followers", "sum"), money=("money", "sum"),
        thr=("min_sum", "median")).reset_index()
    au2["rk_f"] = au2["followers"].rank(ascending=False, method="min").astype(int)
    au2["rk_m"] = au2["money"].rank(ascending=False, method="min").astype(int)
    hdr = (f'{"Автор":<22} {"Стратегий":>10} {"Подписок":>9} {"Доля п.":>8}'
           f' {"Денег, млн ₽":>13} {"Доля д.":>8} {"Порог, тыс":>11}'
           f' {"Ранг п.":>8} {"Ранг д.":>8}')
    for lab, key in (("ТОП-10 ПО ПОДПИСКАМ", "followers"), ("ТОП-10 ПО ДЕНЬГАМ", "money")):
        say(lab)
        say(hdr)
        say("-" * len(hdr))
        for _, r in au2.sort_values(key, ascending=False).head(10).iterrows():
            say(f'{r["author"][:22]:<22} {int(r["n"]):>10,} {int(r["followers"]):>9,}'
                f' {100*r["followers"]/S:>7.1f}% {r["money"]/1e6:>13,.0f}'
                f' {100*r["money"]/M:>7.1f}% {r["thr"]/1e3:>11,.0f}'
                f' {int(r["rk_f"]):>8} {int(r["rk_m"]):>8}')
        say()
    t5f = set(au2.nlargest(5, "followers")["owner_id"])
    t5m = set(au2.nlargest(5, "money")["owner_id"])
    say(f'верхняя пятёрка по подпискам держит {100*au2.nlargest(5,"followers")["followers"].sum()/S:.1f} % подписок; '
        f'верхняя пятёрка по деньгам — {100*au2.nlargest(5,"money")["money"].sum()/M:.1f} % денег; '
        f'общих авторов в этих пятёрках: {len(t5f & t5m)}')
    say()

    # ── 3. квинтили порога входа ──────────────────────────────────────────────
    say("=" * 108)
    say("3. КВИНТИЛИ ПОРОГА ВХОДА (база — 491 живая стратегия с подписками)")
    say("=" * 108)
    sub = live[live.followers > 0].copy()
    sub["q"] = pd.qcut(sub["min_sum"], 5, labels=False, duplicates="drop")
    grp = []
    for q in sorted(sub["q"].dropna().unique()):
        d = sub[sub.q == q]
        grp.append((f'{int(q)+1}: порог {d.min_sum.min()/1e3:,.0f}-{d.min_sum.max()/1e3:,.0f} тыс',
                    sub.q == q))
    two_weights(sub, grp, "Квинтиль порога входа")

    # ── 4. частота сделок ─────────────────────────────────────────────────────
    say("=" * 108)
    say("4. ЗАЯВЛЕННАЯ ЧАСТОТА СДЕЛОК (база — живые с подписками)")
    say("=" * 108)
    ru = {"seldom": "Редко", "monthly": "Раз в месяц",
          "weekly": "Раз в неделю", "daily": "Ежедневно"}
    two_weights(sub, [(ru[k], sub.transaction_rate == k) for k in ru], "Частота")

    # ── 5. тариф ──────────────────────────────────────────────────────────────
    say("=" * 108)
    say("5. ТАРИФ: ГДЕ ЛЕЖАТ ДЕНЬГИ ОТНОСИТЕЛЬНО ПЛАТЫ ОТ РЕЗУЛЬТАТА")
    say("=" * 108)
    with Pool(NPROC) as pool:
        rows = [x for x in pool.imap_unordered(card_tariffs, live.index.tolist(),
                                               chunksize=200) if x]
    T = pd.DataFrame([t for lst in rows for t in lst])
    has_succ = set(T.loc[T["succ"] > 0, "id"])
    m = live.index.isin(list(has_succ))
    two_weights(live, [("Есть вариант с платой от результата", m),
                       ("Только плата от активов", ~m)], "Тарифный набор")
    say(f'плата в год при базовой ставке 6 %: {0.06*M/1e6:,.0f} млн ₽ '
        f'(при льготной 3 % — {0.03*M/1e6:,.0f} млн ₽)')
    say()

    # ── 6. качество купленного ────────────────────────────────────────────────
    say("=" * 108)
    say("6. КАЧЕСТВО КУПЛЕННОГО: ВЗВЕШЕННО ПОДПИСКАМИ ПРОТИВ РУБЛЕЙ")
    say("=" * 108)
    ms = live[(live.n_trade >= MIN_TRADE_DAYS)
              & (live.n_trade / live.n_pts >= MIN_ACTIVITY)].copy()
    say(f'база — {len(ms):,} живые измеримые стратегии, в них '
        f'{int(ms.followers.sum()):,} подписок и {ms.money.sum()/1e9:,.2f} млрд ₽')
    say()
    hdr = (f'{"Величина (медиана)":<26} {"по стратегиям":>15} {"по подпискам":>15}'
           f' {"по рублям":>13}')
    say(hdr)
    say("-" * len(hdr))
    for name, col, fmt in (("Sharpe пожизненный", "sharpe", "{:>15.2f}"),
                           ("Доходность, % годовых", "cagr", "{:>15.1f}"),
                           ("Волатильность, %", "vol", "{:>15.1f}"),
                           ("Просадка, %", "maxdd", "{:>15.1f}"),
                           ("Лет истории", "years", "{:>15.1f}")):
        k = 100.0 if col in ("cagr", "vol", "maxdd") else 1.0
        v = ms[col] * k
        say(f'{name:<26}' + fmt.format(v.median())
            + fmt.format(wmedian(v, ms.followers))
            + fmt.format(wmedian(v, ms.money)).replace(">15", ">13"))
    say()
    for thr, lab in ((1.0, "Sharpe >= 1"), (0.0, "Sharpe > 0")):
        m2 = ms.sharpe >= thr if thr else ms.sharpe > 0
        say(f'{lab:<26} доля стратегий {100*m2.mean():>5.1f} %'
            f' | доля подписок {100*ms.loc[m2, "followers"].sum()/ms.followers.sum():>5.1f} %'
            f' | доля рублей {100*ms.loc[m2, "money"].sum()/ms.money.sum():>5.1f} %')
    say()

    # ── 7. окно >= 3 лет ──────────────────────────────────────────────────────
    say("=" * 108)
    say("7. ОКНО >= 3 ЛЕТ: ЧИСТАЯ ДОХОДНОСТЬ, ВЗВЕШЕННАЯ ПОДПИСКАМИ И РУБЛЯМИ")
    say("=" * 108)
    from comon_sharpe_dist import rusfar                          # noqa: E402
    rf = rusfar()
    # 🔴 Тариф выбирается ТЕМ ЖЕ стабильным правилом «самый дешёвый», что в
    # comon_tariffs (сортировка cost_proxy → succ → mgmt → desc, mergesort). Иначе
    # чистая доходность разъезжается с опубликованной: ничьи по стоимости массовые.
    T["cost_proxy"] = T["mgmt"] + T["succ"] * 0.15
    cheap = (T.sort_values(["cost_proxy", "succ", "mgmt", "desc"], kind="mergesort")
              .groupby("id").first())
    tar = {int(i): {"mgmt": r["mgmt"], "succ": r["succ"], "per": r["per"]}
           for i, r in cheap.iterrows()}
    with Pool(NPROC, initializer=_init, initargs=(tar, rf)) as pool:
        W = [x for x in pool.imap_unordered(work, live.index.tolist(), chunksize=100) if x]
    W = pd.DataFrame(W).set_index("id")
    W = W.join(live[["followers", "money", "min_sum"]], how="inner")
    # 🔴 Колонка «по стратегиям» считается на ВСЕЙ оконной базе (как в главе 5 и
    # заключении), взвешенные — на её подмножестве с подписками: у остальных вес нулевой.
    Wp = W[W.followers > 0]
    say(f'база — {len(W):,} живые стратегии окна (в главе 5 и заключении та же); '
        f'из них с подписками {len(Wp):,}; подписок {int(W.followers.sum()):,}, '
        f'рублей {W.money.sum()/1e9:,.2f} млрд; ставка денежного рынка '
        f'{100*W.rf.median():.1f} % годовых')
    say()
    hdr = (f'{"Величина (медиана)":<30} {"по стратегиям":>15} {"по подпискам":>15}'
           f' {"по рублям":>13}')
    say(hdr)
    say("-" * len(hdr))
    for name, col in (("Валовая доходность, %", "cagr_g"),
                      ("Чистая доходность, %", "cagr_n"),
                      ("Sharpe после тарифа", "sharpe_n"),
                      ("Просадка после тарифа, %", "maxdd_n")):
        k = 1.0 if "sharpe" in col else 100.0
        v, vp = W[col] * k, Wp[col] * k
        say(f'{name:<30} {v.median():>15.2f} {wmedian(vp, Wp.followers):>15.2f}'
            f' {wmedian(vp, Wp.money):>13.2f}')
    say()
    for lab, m2 in (("Чистая выше денежного рынка", W.cagr_n > W.rf),
                    ("Чистая ниже нуля", W.cagr_n < 0)):
        say(f'{lab:<30} доля стратегий {100*m2.mean():>5.1f} %'
            f' | доля подписок {100*W.loc[m2, "followers"].sum()/W.followers.sum():>5.1f} %'
            f' | доля рублей {100*W.loc[m2, "money"].sum()/W.money.sum():>5.1f} %')
    say()

    out = Path(__file__).resolve().parents[1] / "results" / "comon_money_lower.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    say(f"[записано] {out}")


if __name__ == "__main__":
    main()
