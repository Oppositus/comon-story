"""Кто такие подписчики автоследования: анализ followerCount по живым стратегиям Comon.

Данные: data/catalog/v1_live.ndjson.gz (1448 живых, полные поля карточки).
История подписок в API отсутствует (проверено: /followers, /subscribers, /statistics → 404),
поэтому это срез на дату выкачки, а не динамика.

Ключевой приём: сравнение равновзвешенных метрик (какие системы ЕСТЬ на витрине)
со взвешенными по числу подписчиков (какие системы люди РЕАЛЬНО ВЫБРАЛИ).
Разница между ними и есть коллективное предпочтение толпы.
"""
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHARTS = "--charts" in sys.argv

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

from comon_data import require_raw                  # noqa: E402
require_raw(ROOT, "catalog")   # выгрузки витрины в репозитории нет — см. DATA.md
LIVE = os.path.join(ROOT, 'catalog', 'v1_live.ndjson.gz')

rows = [json.loads(l) for l in gzip.open(LIVE, 'rt')]
F = [r['followerCount'] for r in rows]
N, TOT = len(rows), sum(F)


def q(v, p):
    s = sorted(v)
    return s[min(int(p * len(s)), len(s) - 1)]


def wmean(vals, w):
    pairs = [(v, x) for v, x in zip(vals, w) if v is not None and x > 0]
    return sum(v * x for v, x in pairs) / sum(x for _, x in pairs) if pairs else float('nan')


def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else float('nan')


def median(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float('nan')


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 10:
        return float('nan')
    ra, rb = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


def gini(v):
    s = sorted(v)
    n = len(s)
    cum = sum((i + 1) * x for i, x in enumerate(s))
    tot = sum(s)
    return (2 * cum) / (n * tot) - (n + 1) / n if tot else float('nan')


print('=' * 78)
print('1. КАК РАСПРЕДЕЛЕНЫ ПОДПИСЧИКИ ПО ЖИВЫМ СТРАТЕГИЯМ')
print('=' * 78)
print(f'  живых стратегий:              {N}')
print(f'  всего подписок:               {TOT}')
print(f'  стратегий без подписчиков:    {sum(1 for x in F if x == 0)} ({100*sum(1 for x in F if x==0)/N:.1f}%)')
print(f'  подписчиков: медиана={median(F):.0f}  p75={q(F,.75)}  p90={q(F,.90)}  p99={q(F,.99)}  макс={max(F)}')
top = sorted(F, reverse=True)
for k in (10, 50, 100):
    print(f'  доля топ-{k:<3} стратегий в подписках: {100*sum(top[:k])/TOT:.1f}%')
print(f'  коэффициент Джини по подпискам: {gini(F):.3f}')

print()
print('=' * 78)
print('2. ЧТО ЕСТЬ НА ВИТРИНЕ vs ЧТО ВЫБРАЛИ ЛЮДИ (равновзвешенно vs по подписчикам)')
print('=' * 78)
metrics = [
    ('годовая доходность, %/год', 'annualAverageProfit'),
    ('доходность за 365 дней, %', 'profit365Days'),
    ('доходность за 30 дней, %', 'profit30Days'),
    ('макс. просадка, %', 'maxDrawDown'),
    ('CVaR («прогноз просадки»), %', 'conditionalValueAtRisk'),
    ('уровень риска (1-3)', 'riskLevel'),
    ('порог входа minSum, руб', 'minSum'),
    ('индекс торговой активности', 'tradeActivityIndex'),
]
print(f'  {"метрика":<32} {"витрина":>12} {"выбор толпы":>13} {"сдвиг":>10}')
for name, key in metrics:
    vals = [r.get(key) for r in rows]
    a, b = mean(vals), wmean(vals, F)
    print(f'  {name:<32} {a:>12.1f} {b:>13.1f} {b-a:>+10.1f}')

age = []
for r in rows:
    y = int(r['createdAt'][:4]) if r.get('createdAt') else None
    age.append(2026 - y if y else None)
print(f'  {"возраст стратегии, лет":<32} {mean(age):>12.1f} {wmean(age,F):>13.1f} {wmean(age,F)-mean(age):>+10.1f}')


def chart_shelf_vs_choice(items):
    """График 38: витрина против выбора — сдвиг по восьми полям карточки.

    🔴 Поля меряются в разных единицах (проценты, рубли, ярлык 1-3, годы), и на
    одну ось их не положишь. Поэтому сдвиг выражен в долях СОБСТВЕННОГО разброса
    поля по витрине: так «огромный сдвиг по доходности» и «нулевой по просадке»
    становятся сравнимыми, а сырые числа стоят рядом с каждой строкой.
    """
    import comon_charts as ch

    f, ax = ch.fig(h_px=1060, bottom=0.26)
    f.subplots_adjust(left=0.28)
    items = sorted(items, key=lambda it: abs(it['z']))
    ys = list(range(len(items)))
    ax.axvline(0, color=ch.INK, lw=1.4, zorder=3)
    for y, it in zip(ys, items):
        color = ch.BLUE if it['z'] >= 0 else ch.ORANGE
        ax.plot([0, it['z']], [y, y], color=color, lw=3.0, zorder=4)
        ax.plot([it['z']], [y], marker='o', ms=11, color=color, zorder=5)
        side = 1 if it['z'] >= 0 else -1
        ax.annotate(f"{it['a']} → {it['b']}", xy=(it['z'], y),
                    xytext=(14 * side, 0), textcoords='offset points',
                    fontsize=11, color=ch.GREY,
                    ha='left' if side > 0 else 'right', va='center')
    ax.set_yticks(ys)
    ax.set_yticklabels([it['lab'] for it in items], fontsize=11)
    ax.set_ylim(-0.7, len(ys) - 0.3)
    lim = max(abs(it['z']) for it in items) * 1.75
    ax.set_xlim(-lim, lim)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('сдвиг выбора относительно витрины, в долях разброса поля')
    ax.set_title('Выбор смещён по доходности и возрасту, но не по глубине провала')
    ch.note(ax,
            f"База — {ch.n_(N)} живых стратегий витрины и {ch.n_(TOT)} подписок. "
            f"«Витрина» — среднее по стратегиям, «выбор» — то же среднее, "
            f"взвешенное числом подписчиков; рядом с каждой строкой стоят оба "
            f"сырых значения.",
            "Сдвиги приведены к разбросу своего поля, иначе рубли, проценты и "
            "ярлык риска несравнимы. Смещение по доходности огромное, по глубине "
            "просадки — нулевое: выбранные стратегии проседали ровно так же "
            "глубоко, как витрина в целом.",
            "Ярлык уровня риска при этом сдвинут заметно: покупали то, что "
            "помечено как менее рискованное. Но ярлык с реальной глубиной "
            "провала связан слабо (глава 1), поэтому сдвиг по ярлыку и нулевой "
            "сдвиг по просадке противоречия не образуют.")
    ch.save(f, 38, 'shelf-vs-choice')
    print('    ' + '; '.join(f"{it['lab']}: {it['z']:+.2f}σ" for it in items[::-1]))


if CHARTS:
    import comon_charts as _ch

    def _std(v):
        vv = [x for x in v if x is not None]
        m = sum(vv) / len(vv)
        return math.sqrt(sum((x - m) ** 2 for x in vv) / (len(vv) - 1))

    items = []
    for name, key in metrics:
        vals = [r.get(key) for r in rows]
        a, b, sd = mean(vals), wmean(vals, F), _std(vals)
        dec = 0 if key == 'minSum' else 1
        items.append({'lab': name.replace(', %/год', '').replace(', %', '')
                      .replace(', руб', '').replace(' («прогноз просадки»)', ''),
                      'z': (b - a) / sd, 'a': _ch.n_(a, dec), 'b': _ch.n_(b, dec)})
    a, b = mean(age), wmean(age, F)
    items.append({'lab': 'возраст стратегии, лет', 'z': (b - a) / _std(age),
                  'a': _ch.n_(a, 1), 'b': _ch.n_(b, 1)})
    print()
    print('── график 38 ────────────────────────────────────────────────────')
    chart_shelf_vs_choice(items)

print()
print('=' * 78)
print('3. С ЧЕМ СВЯЗАНА ПОПУЛЯРНОСТЬ (ранговая корреляция Спирмена с числом подписчиков)')
print('=' * 78)
for name, key in metrics:
    print(f'  {name:<32} rho = {spearman([r.get(key) for r in rows], F):+.3f}')
print(f'  {"возраст стратегии, лет":<32} rho = {spearman(age, F):+.3f}')

print()
print('=' * 78)
print('4. ПРОФИЛЬ ПО ГРУППАМ ПОПУЛЯРНОСТИ (медианы внутри группы)')
print('=' * 78)
groups = [('без подписчиков', lambda x: x == 0),
          ('1-5', lambda x: 1 <= x <= 5),
          ('6-20', lambda x: 6 <= x <= 20),
          ('21-100', lambda x: 21 <= x <= 100),
          ('101+', lambda x: x > 100)]
hdr = f'  {"группа":<16} {"систем":>7} {"подписок":>9} {"год.дох,%":>10} {"365д,%":>8} {"maxDD,%":>9} {"minSum,тыс":>11} {"возраст":>8}'
print(hdr)
for gname, cond in groups:
    g = [r for r in rows if cond(r['followerCount'])]
    if not g:
        continue
    ga = [2026 - int(r['createdAt'][:4]) for r in g if r.get('createdAt')]
    print(f'  {gname:<16} {len(g):>7} {sum(r["followerCount"] for r in g):>9}'
          f' {median([r.get("annualAverageProfit") for r in g]):>10.1f}'
          f' {median([r.get("profit365Days") for r in g]):>8.1f}'
          f' {median([r.get("maxDrawDown") for r in g]):>9.1f}'
          f' {median([r.get("minSum") for r in g])/1000:>11.0f}'
          f' {median(ga):>8.1f}')

print()
print('=' * 78)
print('5. РИСК, КОТОРЫЙ ТОЛПА ФАКТИЧЕСКИ НЕСЁТ')
print('=' * 78)
for lo, hi, name in [(None, -50, 'глубже −50 %'), (-50, -30, 'от −50 до −30 %'),
                     (-30, -15, 'от −30 до −15 %'), (-15, 0, 'мельче −15 %')]:
    sel = [r for r in rows if r.get('maxDrawDown') is not None
           and (lo is None or r['maxDrawDown'] > lo) and r['maxDrawDown'] <= hi]
    subs = sum(r['followerCount'] for r in sel)
    print(f'  просадка {name:<18} систем {len(sel):>5} ({100*len(sel)/N:>4.1f}%)   '
          f'подписок {subs:>6} ({100*subs/TOT:>4.1f}%)')

print()
print('  Деньги под управлением (оценка снизу: подписчиков x порог входа):')
money = sum(r['followerCount'] * (r.get('minSum') or 0) for r in rows)
print(f'    {money/1e9:.2f} млрд руб при {TOT} подписках, средний чек ~{money/TOT/1e3:.0f} тыс руб')
