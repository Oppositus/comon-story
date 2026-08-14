"""comon_charts.py — общий стиль графиков серии.

Не самостоятельный скрипт: модуль, который импортируют скрипты расчёта. График
рождается ТЕМ ЖЕ скриптом, что и таблица под ним, иначе числа на картинке и в
тексте разойдутся.

Правила, зашитые здесь, а не оставленные на память:
  · палитра валидирована (проверки светлотной полосы, цветности, различимости при
    дальтонизме и для обычного зрения пройдены); порядок слотов фиксирован и
    никогда не циклится — девятой серии не бывает, она сворачивается в «прочее»;
  · цвет назначается по роли: категориальный — для различения объектов,
    последовательный (один тон, светлый → тёмный) — для величины, расходящийся —
    для полярности вокруг нуля;
  · форма маркера и штрих линии остаются вторым носителем различия: подпись у
    линии и своя форма точки помогают там, где цвета сближаются;
  · оси без научной нотации: 2 000 000 пишется «2 млн», не «2·10⁶»;
  · шрифт DejaVu Sans — единственный в venv с полной кириллицей;
  · ширина всех картинок серии — ровно W_PX, поля фиксированы: вёрстка ставит их
    в текст подряд, и разный масштаб виден по толщине линий и кеглю;
  · файлы — в images/ (рядом с текстом, под git), имя несёт номер графика в серии;
  · save() печатает текстовую сводку и обмеряет подписи: оформление проверяется по
    этому обмеру, а не на глаз — 14 проверок, каждая добавлена после реального
    дефекта (подпись шире панели, сноска на названии оси, линия сквозь подпись…).
"""
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")                       # без дисплея: пишем в файл
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.text import Text as _Text   # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Картинки лежат РЯДОМ С ТЕКСТОМ и попадают в git: они часть публикации, а не
# побочный вывод прогона.
OUT = ROOT / "images"

DPI = 150
W_PX = 1600                                 # 🔴 ширина ВСЕХ картинок серии
H_PX = 1000

# ── палитра ──────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"                             # основной текст и оси
GREY = "#52514e"                            # вторичный текст, сноски
PALE = "#b4b3ae"                            # вспомогательные линии уровней

# категориальные слоты: назначаются ПО ПОРЯДКУ и никогда не циклятся
SERIES = ["#2a78d6",    # 1 синий
          "#eb6834",    # 2 оранжевый
          "#1baf7a",    # 3 бирюзовый
          "#eda100",    # 4 жёлтый
          "#e87ba4",    # 5 розовый
          "#008300",    # 6 зелёный
          "#4a3aa7",    # 7 фиолетовый
          "#e34948"]    # 8 красный
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET, RED = SERIES

# последовательная шкала (величина): один тон, светлый → тёмный
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
# расходящаяся шкала (полярность вокруг нуля): два тона и нейтральная середина
DIV_LO, DIV_MID, DIV_HI = BLUE, "#f0efec", ORANGE
BAND = "#9ec5f4"                            # заливка доверительной полосы

DASHES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 12,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#9a9994",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "grid.color": "#e3e2dd",
    "grid.linewidth": 0.7,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})


def fig(w_px=W_PX, h_px=H_PX, bottom=0.24, **kw):
    """Стандартное поле: ширина W_PX у всех графиков серии, поля фиксированы.

    Нижнее поле держит место под сноску note(); графику с двумя строками сноски
    можно передать bottom меньше. Фактический размер файла проверяет save().
    """
    f, ax = plt.subplots(figsize=(w_px / DPI, h_px / DPI), **kw)
    f.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=bottom)
    return f, ax


# ── форматтеры осей: никакой научной нотации ─────────────────────────────────
def pct(ax, axis="y", decimals=0):
    """Ось в процентах: значения приходят долями (0.42 → «42 %»)."""
    f = FuncFormatter(lambda v, _: f"{100*v:.{decimals}f} %".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def pct_raw(ax, axis="y", decimals=0):
    """Ось в процентах, значения уже в процентных пунктах (42 → «42 %»)."""
    f = FuncFormatter(lambda v, _: f"{v:.{decimals}f} %".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def n_(v, dec=0, sign=False):
    """Число по-русски: запятая в дробной части, пробел в тысячах.

    🔴 Заменять точки и запятые в ГОТОВОЙ строке нельзя, и это уже стоило
    испорченных подписей: соседние строковые литералы Python склеивает ДО
    вызова метода, поэтому `"текст. " f"{x:.1f}".replace(".", ",")` меняет и
    точку в конце предложения, и запятую в «Lo, 2002». Формат применяется к
    самому числу, а проза остаётся нетронутой.
    """
    s = f"{v:+,.{dec}f}" if sign else f"{v:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",")


def _human(v):
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f} млрд".replace(".0 ", " ").replace(".", ",")
    if a >= 1e6:
        return f"{v/1e6:.1f} млн".replace(".0 ", " ").replace(".", ",")
    if a >= 1e4:
        return f"{v/1e3:.0f} тыс."
    return f"{v:,.0f}".replace(",", " ")


def human(ax, axis="y"):
    """Крупные числа словами: 2 000 000 → «2,0 млн» (правило §5 договорённостей)."""
    f = FuncFormatter(lambda v, _: _human(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def num(ax, axis="y", decimals=0):
    """Обычные числа с неразрывным разделителем тысяч и запятой в дроби."""
    def g(v, _):
        return f"{v:,.{decimals}f}".replace(",", " ").replace(".", ",")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(g))


def label_line(ax, x, y, text, ha="left", va="bottom", size=11, weight="normal",
               color=INK, pad=0.0):
    """Подпись прямо у линии — чтобы график читался без обращения к легенде."""
    return ax.annotate(text, (x, y), textcoords="offset points",
                       xytext=(4 if ha == "left" else -4, 4 if va == "bottom" else -4),
                       ha=ha, va=va, fontsize=size, fontweight=weight, color=color)


def _crossed(ax, bb, pad=3.0):
    """Проходит ли хоть одна линия панели сквозь прямоугольник подписи."""
    r = ax.figure.canvas.get_renderer()
    for ln in ax.lines:
        d = ln.get_xydata()
        if len(d) < 2:
            continue
        p = ln.get_transform().transform(d)
        seg = np.concatenate([p[:-1] + (p[1:] - p[:-1]) * a
                              for a in np.linspace(0, 1, 40)])
        if ((seg[:, 0] > bb.x0 - pad) & (seg[:, 0] < bb.x1 + pad)
                & (seg[:, 1] > bb.y0 - pad) & (seg[:, 1] < bb.y1 + pad)).any():
            return True
    return False


def value_labels(ax, xs, ys, texts, dist=22, size=10, color=GREY):
    """Числа у точек ломаной, отодвинутые по НОРМАЛИ к самой ломаной.

    Подпись, поставленную просто «сверху», линия перечёркивает везде, где
    ломаная крутая: за половину ширины подписи кривая успевает подняться выше
    отступа. Направление нормали к локальному ходу линии (по соседним точкам)
    уводит подпись в сторону, свободную по построению, а сторону подписи —
    левую или правую — задаёт наклон.
    """
    f = ax.figure
    f.canvas.draw()
    box = ax.get_window_extent()
    pts = ax.transData.transform(np.column_stack([np.asarray(xs, float),
                                                  np.asarray(ys, float)]))
    out = []
    for i, s in enumerate(texts):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        d = b - a
        n = np.hypot(*d) or 1.0
        nx, ny = -d[1] / n, d[0] / n            # нормаль к направлению линии
        # Сторона выбирается не «вверх», а от ХОРДЫ между соседями: во впадине
        # ломаной верхняя полуплоскость занята обеими соседними ветвями, и
        # подпись оказывается ровно между ними — перечёркнутой.
        up = True
        if 0 < i < len(pts) - 1:
            up = pts[i][1] >= (pts[i - 1][1] + pts[i + 1][1]) / 2
        if (ny < 0) == up:
            nx, ny = -nx, -ny
        # Кандидаты перебираются по порядку: своя сторона, зеркало по горизонтали,
        # по вертикали, обе. Годной считается та, где подпись целиком в панели и
        # её не перечёркивает ни одна линия — включая опорные горизонтали (ноль).
        best = None
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            ox, oy = nx * sx, ny * sy
            ha = "center" if abs(ox) < 0.35 else ("right" if ox < 0 else "left")
            t = ax.annotate(s, xy=(xs[i], ys[i]), xytext=(dist * ox, dist * oy),
                            textcoords="offset points", fontsize=size,
                            color=color, ha=ha, va="center")
            f.canvas.draw()
            bb = _Text.get_window_extent(t, renderer=f.canvas.get_renderer())
            inside = box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 \
                and bb.y1 <= box.y1
            if best is None:
                best = (t, ox, oy)
            if inside and not _crossed(ax, bb):
                best = (t, ox, oy)
                break
            t.remove()
        t, ox, oy = best
        if t.figure is None:                    # ни один вариант не подошёл
            ha = "center" if abs(ox) < 0.35 else ("right" if ox < 0 else "left")
            t = ax.annotate(s, xy=(xs[i], ys[i]), xytext=(dist * ox, dist * oy),
                            textcoords="offset points", fontsize=size,
                            color=color, ha=ha, va="center")
        out.append(t)
    return out


def _no_orphan(lines, min_chars=7):
    """Не оставлять на последней строке абзаца одинокое короткое слово.

    Перенос по ширине легко выбрасывает на отдельную строку хвост вроде «2.» —
    висячую строку, которая читается как обрывок. Тянем вниз последнее слово
    предыдущей строки, чтобы внизу оказалось хотя бы два слова.
    """
    while len(lines) > 1 and len(lines[-1]) <= min_chars and " " in lines[-2]:
        head, _, tail = lines[-2].rpartition(" ")
        lines[-2], lines[-1] = head, f"{tail} {lines[-1]}"
        if len(lines[-1]) > min_chars:
            break
    return lines


def note(ax, *lines, size=10, gap_pt=14, step_pt=16, width=95):
    """Служебные строки под осью: база расчёта, единицы, оговорка.

    Отступы заданы в ПУНКТАХ и отсчитываются от нижнего края панели в координатах
    фигуры. Прежняя версия мерила их в долях высоты панели — и на графиках из двух
    панелей строки наезжали друг на друга: доля от низкой панели даёт меньший
    шаг, чем от высокой. Пункты одинаковы всюду.

    Длинные строки matplotlib не переносит сам — переносим по ширине панели;
    явные аргументы задают обязательные переносы (например, по двоеточию).
    Контроль результата — в save(): она измеряет текст и ругается, если вылез.
    """
    f = ax.figure
    box = ax.get_position()
    h_px = f.get_window_extent().y1
    w_px = f.get_window_extent().x1
    # 🔴 Ширина переноса ограничена ФАКТИЧЕСКИМ местом справа от начала сноски.
    # Сноска начинается у левого края панели, а он у графика с широкими подписями
    # делений уезжает вправо — и строка в 95 знаков вылезала за край файла.
    avail = (0.985 - box.x0) * w_px
    # Ширина знака берётся не «на глаз»: строка меряется рендерером, и перенос
    # ужимается, пока самая длинная строка не влезет. Оценка по 0,56 em врала на
    # кириллице и сноска уезжала за край файла.
    f.canvas.draw()
    r = f.canvas.get_renderer()
    fit = int(avail / (size * 0.56 * DPI / 72))
    for _ in range(8):
        out = []
        for s in lines:
            out += _no_orphan(textwrap.wrap(s, width=min(width, fit)) or [""])
        probe = f.text(0, -1, max(out, key=len), fontsize=size)
        wide = probe.get_window_extent(r).width
        probe.remove()
        if wide <= avail:
            break
        fit = int(fit * avail / wide) - 1

    # 🔴 Отсчёт идёт от НИЖНЕЙ ГРАНИЦЫ ВСЕГО, что уже стоит под панелью, — самой
    # панели, подписей делений и названия оси. Прежде отступ брался от панели, и
    # на графиках с двухстрочными подписями делений название оси съезжало вниз
    # ровно на строку сноски: два текста печатались по одной базовой линии.
    f.canvas.draw()
    r = f.canvas.get_renderer()
    bottoms = [ax.get_window_extent().y0]
    if ax.xaxis.label.get_text().strip():
        bottoms.append(ax.xaxis.label.get_window_extent(r).y0)
    for t in ax.get_xticklabels():
        if t.get_text().strip():
            bottoms.append(t.get_window_extent(r).y0)
    base = min(bottoms) / h_px

    # Панель поднимается ровно настолько, чтобы сноска влезла целиком. Иначе
    # нижнее поле приходится подбирать руками под каждый график: число строк
    # зависит от длины текста, а высота подписей делений — от их разбивки.
    placed = []
    for _ in range(4):
        for t in placed:
            t.remove()
        placed = []
        f.canvas.draw()
        r = f.canvas.get_renderer()
        bottoms = [ax.get_window_extent().y0]
        if ax.xaxis.label.get_text().strip():
            bottoms.append(ax.xaxis.label.get_window_extent(r).y0)
        for t in ax.get_xticklabels():
            if t.get_text().strip():
                bottoms.append(t.get_window_extent(r).y0)
        base = min(bottoms) / h_px
        for i, s in enumerate(out):
            y = base - (gap_pt + i * step_pt) / 72.0 * DPI / h_px
            placed.append(f.text(box.x0, y, s, fontsize=size, color=GREY,
                                 ha="left", va="top"))
        f.canvas.draw()
        low = min(t.get_window_extent(f.canvas.get_renderer()).y0 for t in placed)
        if low >= 10:
            break
        # 🔴 Панель не ПОДНИМАЕТСЯ, а становится ниже: верхний край остаётся на
        # месте. Прежде сдвигалась вся панель разом — и вместе с ней уезжал вверх
        # заголовок: у графика 30 от названия осталась нижняя половина, а на
        # графике 13 заголовки панелей обогнали общий заголовок фигуры.
        for a in f.axes:
            pos = a.get_position()
            shift = (12 - low) / h_px
            a.set_position([pos.x0, pos.y0 + shift, pos.width,
                            max(pos.height - shift, 0.15)])
        box = ax.get_position()
    return out


def stack_label(ax, xy, big, small, dx=0, up=True, ha="left", big_size=12,
                small_size=10, color=INK):
    """Число и пояснение к нему у точки — двумя плотными строками.

    Между строками ровно один межстрочный интервал: разнесённые подписи глаз
    читает как две разные аннотации, а не как одну. Число носит цвет своей
    линии только если он передан явно; по умолчанию текст остаётся чернильным.
    """
    if up:
        y_big, y_small, va = 15, 2, "bottom"
    else:
        y_big, y_small, va = -6, -19, "top"
    ax.annotate(big, xy=xy, xytext=(dx, y_big), textcoords="offset points",
                fontsize=big_size, fontweight="bold", ha=ha, va=va, color=color)
    ax.annotate(small, xy=xy, xytext=(dx, y_small), textcoords="offset points",
                fontsize=small_size, color=GREY, ha=ha, va=va)


def save(f, number, slug, verbose=True):
    """Сохранить в images/NN-slug.png и напечатать сводку.

    number — номер графика в серии; slug — короткое имя латиницей.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{number:02d}-{slug}.png"
    _fit_left(f)                            # подвинуть панель, если подписи слева срезаны
    over = _overflow(f)                     # до savefig: рисуем на том же холсте
    f.savefig(p)
    plt.close(f)
    w, h = _png_size(p)                     # из файла, а не из расчёта
    if w != W_PX:
        over.append(f"ширина файла {w} px вместо обязательных {W_PX}")
    if verbose:
        kb = p.stat().st_size / 1024
        print(f"[график {number:>2}] {p.relative_to(ROOT)}  "
              f"{w}×{h} px, {kb:.0f} КБ", flush=True)
        for s in over:
            print(f"    ⚠️ {s}", flush=True)
    return p


def _fit_left(f, pad=6.0):
    """Подвинуть панель вправо, если подписи слева не помещаются в файл.

    Левое поле у всех графиков серии одинаковое, а слева от панели стоит разное:
    подписи делений бывают широкими, название оси — в две строки. Когда сумма не
    влезает, matplotlib молча рисует её за краем файла. Здесь панель ужимается
    ровно на недостачу — так же, как note() ужимает её снизу под сноску.
    """
    for _ in range(3):
        f.canvas.draw()
        r = f.canvas.get_renderer()
        w_px = f.get_window_extent().x1
        moved = False
        for ax in f.axes:
            items = [ax.yaxis.label] + list(ax.get_yticklabels())
            xs = [_tbox(t, r).x0 for t in items if t.get_text().strip()]
            if not xs or min(xs) >= pad:
                continue
            shift = (pad - min(xs)) / w_px
            pos = ax.get_position()
            ax.set_position([pos.x0 + shift, pos.y0,
                             max(pos.width - shift, 0.2), pos.height])
            moved = True
        if not moved:
            return


def _tbox(t, r):
    """Габарит САМОГО текста, без стрелки-выноски.

    У аннотации со стрелкой get_window_extent возвращает объединение текста и
    стрелки, и тогда любая выноска, ведущая к линии, засчитывалась как «линия
    графика пересекает подпись»: стрелка обязана касаться линии, на то она и
    выноска. Базовый метод Text меряет только буквы.
    """
    return _Text.get_window_extent(t, renderer=r)


def _png_size(p):
    """Размер PNG из заголовка файла: расчётный размер и реальный расходились."""
    b = p.read_bytes()[16:24]
    return int.from_bytes(b[:4], "big"), int.from_bytes(b[4:], "big")


def _typos(f):
    """Следы сплошной замены точек и запятых в готовой строке.

    Русское число пишется «10,6», и соблазн получить его заменой в уже собранной
    строке велик — но соседние литералы Python склеивает ДО вызова метода, и
    вместе с числом переделывается проза: точка в конце предложения становится
    запятой, а запятая внутри — пробелом. Ловим три следа этого: запятая перед
    заглавной буквой, двойной пробел, запятая в самом конце подписи.
    """
    bad = []
    wrapped = set(map(id, f.texts))     # строки сноски переносятся по ширине —
    texts = list(f.texts)               # у них запятая в конце строки законна,
    # но САМАЯ НИЖНЯЯ строка сноски — конец прозы, и точка там обязательна
    last_note = min(f.texts, key=lambda t: t.get_position()[1], default=None)
    for ax in f.axes:
        texts += list(ax.texts)
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        lg = ax.get_legend()
        if lg is not None:
            texts += lg.get_texts()
    for t in texts:
        s = t.get_text()
        if not s.strip():
            continue
        short = s[:52].replace("\n", " ")
        # В короткой подписи «(1 448), Джини 0,931» запятая перед заглавной
        # законна — правило применяется только к прозе длиннее строки.
        if len(s) > 60 and re.search(r",\s+[A-ZА-ЯЁ]", s):
            bad.append(f"запятая вместо точки в предложении: «{short}»")
        if "  " in s.replace("\n", " "):
            bad.append(f"двойной пробел (съеденная запятая): «{short}»")
        if id(t) in wrapped and len(s.split()) == 1 and len(s) <= 7:
            bad.append(f"висячая строка сноски: одно слово «{s}»")
        if s.rstrip().endswith(",") and (id(t) not in wrapped or t is last_note):
            bad.append(f"подпись кончается запятой: «{short}»")
    return bad


def _overflow(f, pad=3.0):
    """Дефекты оформления, которые видно только глазами, — измеренные машиной.

    Claude картинок не читает, поэтому осмотр заменяет обмер в тех же пикселях,
    в которых текст окажется в файле. Ловим три вещи, каждая уже случалась:
    подпись вылезла за панель · подпись обрезана краем файла · линия графика
    прошла сквозь подпись (перечеркнула её).
    """
    f.canvas.draw()
    r = f.canvas.get_renderer()
    W, H = f.get_window_extent().x1, f.get_window_extent().y1
    bad = _typos(f)
    # сноски живут в координатах фигуры (f.texts): проверяем, что они не срезаны
    # краем файла и не наехали друг на друга
    prev = None
    # всё, что стоит под панелью и на что сноска может наехать
    under = []
    for ax in f.axes:
        if ax.xaxis.label.get_text().strip():
            under.append(("название оси", ax.xaxis.label.get_window_extent(r)))
        for t in ax.get_xticklabels():
            if t.get_text().strip():
                under.append(("подпись деления", t.get_window_extent(r)))
    for t in sorted([t for t in f.texts if t.get_text().strip()],
                    key=lambda t: -_tbox(t, r).y0):
        bb = _tbox(t, r)
        s = t.get_text()[:52]
        if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
            bad.append(f"сноска обрезана краем файла: «{s}»")
        if prev is not None and bb.y1 > prev - 1:
            bad.append(f"строки сноски наезжают друг на друга: «{s}»")
        for what, ob in under:
            if (bb.y0 < ob.y1 - 1 and bb.y1 > ob.y0 + 1
                    and bb.x0 < ob.x1 - 1 and bb.x1 > ob.x0 + 1):
                bad.append(f"сноска наехала на {what}: «{s}»")
                break
        prev = bb.y0
    # подписи делений: наезд соседних друг на друга (на узкой панели из трёх
    # «100 тыс.» и «200 тыс.» сливаются) и заезд подписей одной панели на другую
    boxes = [ax.get_window_extent() for ax in f.axes]
    for i, ax in enumerate(f.axes):
        lb = sorted([(t.get_window_extent(r), t.get_text())
                     for t in ax.get_xticklabels() if t.get_text().strip()],
                    key=lambda p: p[0].x0)
        for (b1, s1), (b2, s2) in zip(lb, lb[1:]):
            if b2.x0 < b1.x1 - 1:
                bad.append(f"подписи делений наезжают друг на друга: "
                           f"«{s1}» и «{s2}»")
                break
        # название оси не должно быть длиннее самой панели: сноска ужимает
        # панель, и вертикальная подпись перестаёт помещаться по высоте
        bx = ax.get_window_extent()
        for lbl, what, room in ((ax.yaxis.label, "название оси Y",
                                 bx.y1 - bx.y0),
                                (ax.xaxis.label, "название оси X",
                                 bx.x1 - bx.x0)):
            if not lbl.get_text().strip():
                continue
            bb = _tbox(lbl, r)
            long = bb.y1 - bb.y0 if "Y" in what else bb.x1 - bb.x0
            if long > room + 1:
                bad.append(f"{what} длиннее панели на {long - room:.0f} px: "
                           f"«{lbl.get_text()[:40]}»")
            if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
                bad.append(f"{what} вылезло за край файла: "
                           f"«{lbl.get_text()[:40]}»")
        # подписи делений тоже обязаны помещаться в файл
        # только ВИДИМЫЕ деления: matplotlib держит подписи и за пределами
        # выставленных границ, они в файл не попадают
        vis = ([(t, ax.get_ylim()) for t in ax.get_yticklabels()]
               + [(t, ax.get_xlim()) for t in ax.get_xticklabels()])
        for t, (v0, v1) in vis:
            pos = t.get_position()[1 if t in ax.get_yticklabels() else 0]
            if not t.get_text().strip() or not (min(v0, v1) <= pos <= max(v0, v1)):
                continue
            bb = _tbox(t, r)
            if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
                bad.append(f"подпись деления вылезла за край файла: "
                           f"«{t.get_text()[:20]}»")
                break
        side = [(ax.yaxis.label, "название оси")]
        side += [(t, "подпись деления") for t in ax.get_yticklabels()
                 if t.get_text().strip()]
        for t, what in side:
            if not t.get_text().strip():
                continue
            bb = _tbox(t, r)
            for j, ob in enumerate(boxes):
                if j != i and bb.x0 < ob.x1 - 1 and bb.x1 > ob.x0 + 1 \
                        and bb.y0 < ob.y1 - 1 and bb.y1 > ob.y0 + 1:
                    bad.append(f"{what} заехало на соседнюю панель: "
                               f"«{t.get_text()[:40]}»")
                    break

    # заголовки: обрезание верхом файла и порядок «общий заголовок → панели»
    tops = []
    for ax in f.axes:
        t = ax.title
        if not t.get_text().strip():
            continue
        bb = _tbox(t, r)
        tops.append(bb.y1)
        if bb.y1 > H:
            bad.append(f"название панели обрезано верхом файла: "
                       f"«{t.get_text()[:52]}»")
        if bb.x0 < 0 or bb.x1 > W:
            bad.append(f"название панели обрезано краем файла: "
                       f"«{t.get_text()[:52]}»")
    sup = getattr(f, "_suptitle", None)
    if not tops and not (sup is not None and sup.get_text().strip()):
        bad.append("у графика нет названия")
    if sup is not None and sup.get_text().strip():
        bb = _tbox(sup, r)
        if bb.y1 > H:
            bad.append("общий заголовок обрезан верхом файла")
        if tops and bb.y0 < max(tops) - 1:
            bad.append("общий заголовок оказался ниже названий панелей")

    for ax in f.axes:
        box = ax.get_window_extent()
        for t in [t for t in ax.texts if t.get_text().strip()]:
            bb = _tbox(t, r)
            s = t.get_text()[:52].replace("\n", " ")
            if bb.x1 > box.x1 + 1:
                bad.append(f"подпись шире панели на {bb.x1 - box.x1:.0f} px: «{s}»")
            if bb.y1 > box.y1 + 1:
                bad.append(f"подпись вышла за верх панели на {bb.y1 - box.y1:.0f} px: "
                           f"«{s}»")
            if bb.y0 < box.y0 - 1:
                bad.append(f"подпись вышла за низ панели на {box.y0 - bb.y0:.0f} px: "
                           f"«{s}»")
            if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
                bad.append(f"подпись обрезана краем файла: «{s}»")
            lg = ax.get_legend()
            if lg is not None:
                lb = lg.get_window_extent(r)
                if (bb.x0 < lb.x1 - 1 and bb.x1 > lb.x0 + 1
                        and bb.y0 < lb.y1 - 1 and bb.y1 > lb.y0 + 1):
                    bad.append(f"подпись заехала под легенду: «{s}»")
            for ln in ax.lines:
                d = ln.get_xydata()
                if len(d) < 2:
                    continue
                p = ln.get_transform().transform(d)
                # отрезки дискретизируем: линия может пересечь подпись между вершинами
                seg = np.concatenate([
                    p[:-1] + (p[1:] - p[:-1]) * a for a in np.linspace(0, 1, 40)])
                hit = ((seg[:, 0] > bb.x0 - pad) & (seg[:, 0] < bb.x1 + pad)
                       & (seg[:, 1] > bb.y0 - pad) & (seg[:, 1] < bb.y1 + pad))
                if hit.any():
                    bad.append(f"линия графика пересекает подпись «{s}» "
                               f"({int(hit.sum())} точек)")
                    break
    return bad


def link(number, slug):
    """Markdown-вставка картинки для главы (путь — от файла главы)."""
    return f"![рис. {number}](images/{number:02d}-{slug}.png)"
