"""build_html.py — сборка html-версии серии из markdown-глав.

Страницы РОЖДАЮТСЯ ЭТИМ СКРИПТОМ, а не правятся руками: текст живёт в md/, и
любая правка главы должна доезжать до html одним прогоном, иначе две версии
разойдутся — та же логика, что у графиков (график рисует тот же скрипт, что
считает таблицу под ним).

    python build_html.py            собрать всё в ../html
    python build_html.py --check    только проверить, ничего не записывая

Что получается: ../html/index.html (обложка с оглавлением), по странице на
каждую часть серии и общий style.css. Страницы самодостаточны — ни одного
обращения в сеть, все шрифты системные. Картинки берутся из ../images по
относительному пути: он верен и при открытии файла с диска, и на GitHub Pages.

🔴 GitHub НЕ рендерит html внутри репозитория — он покажет исходный код
страницы. Серия читается на GitHub Pages: выкладку делает workflow
.github/workflows/pages.yml, который выдаёт репозиторий как есть, без сборки
(режим Deploy from a branch прогонял всё через Jekyll и падал). Локально
достаточно открыть index.html двойным кликом.

Зависимость: python-markdown (pip install markdown).
"""
import html as htmllib
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "md"
OUT = ROOT / "html"
CHECK = "--check" in sys.argv

# Порядок серии и короткие имена для оглавления. Держим здесь, а не выводим из
# имён файлов: порядок частей — смысловой (интерлюдия между 4 и 5), а не
# алфавитный, и подписи в оглавлении должны быть человеческими.
PARTS = [
    ("comon-story-license.md", "Условия использования",
     "Три лицензии: текст и графики — CC BY-NC-SA 4.0, код — BSD 3-Clause, данные — "
     "производные под CC BY-NC-SA 4.0. Что можно делать, по-русски."),
    ("comon-story-0-intro.md", "Введение",
     "Механика автоследования, юридическая рамка, данные, репрезентативность выборки, "
     "методы и словарь терминов."),
    ("comon-story-1-market.md", "Глава 1. Площадка и витрина",
     "Как устроен рынок автоследования и чему на нём можно верить."),
    ("comon-story-2-survival.md", "Глава 2. Кладбище",
     "Сколько живут стратегии, от чего умирают и что остаётся от тысячи запущенных."),
    ("comon-story-3-returns.md", "Глава 3. Из чего сделана доходность",
     "Сигнал, рыночная бета, плечо и удача — что именно продаётся под видом результата."),
    ("comon-story-4-authors.md", "Глава 4. Продавцы",
     "Кто пишет стратегии, сколько попыток делает и что показывает из них покупателю."),
    ("comon-story-4b-interlude.md", "Интерлюдия. Вторая площадка",
     "Comon и Т-Инвестиции одним счётом: совпадает ли картина на другой витрине."),
    ("comon-story-5-buyers.md", "Глава 5. Покупатели",
     "Кто и как выбирает стратегии — и почему выбор устроен не так, как кажется."),
    ("comon-story-6-conclusion.md", "Заключение",
     "Общий вывод серии: что показали данные и что из этого следует читателю."),
]

CSS = """/* Стиль серии. Единственная таблица стилей на все страницы: правится здесь,
   а не в разметке — так восемь длинных глав остаются одинаковыми на вид. */

:root {
  --bg:        #fcfcfb;
  --surface:   #ffffff;
  --ink:       #171a1c;
  --ink-soft:  #4a5054;
  --ink-faint: #71797e;
  --rule:      #e3e4e1;
  --rule-soft: #eeefec;
  --accent:    #1d5fb0;
  --accent-bg: #eef4fc;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:        #14161a;
    --surface:   #1b1e23;
    --ink:       #e8eaec;
    --ink-soft:  #b3b9bf;
    --ink-faint: #8b9299;
    --rule:      #2c3138;
    --rule-soft: #23272d;
    --accent:    #79b0f0;
    --accent-bg: #1d2734;
  }
}

html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 17px/1.68 "Georgia", "Times New Roman", serif;
  overflow-wrap: break-word;
}

.wrap { max-width: 44rem; margin: 0 auto; padding: 0 1.2rem 6rem; }

/* Шапка с навигацией по частям */
.topbar {
  border-bottom: 1px solid var(--rule);
  background: var(--surface);
  margin-bottom: 2.4rem;
}
.topbar .wrap { padding-top: .85rem; padding-bottom: .85rem; }
.topbar a, .topbar span {
  font: 500 13px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.topbar a { color: var(--ink-soft); text-decoration: none; }
.topbar a:hover { color: var(--accent); text-decoration: underline; }
.topbar .here { color: var(--ink); font-weight: 700; }
.topbar .sep { color: var(--ink-faint); padding: 0 .35rem; }

h1, h2, h3, h4 {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.25;
  color: var(--ink);
}
h1 { font-size: 2rem; font-weight: 700; margin: 2.2rem 0 1.1rem; letter-spacing: -.01em; }
h2 {
  font-size: 1.42rem; font-weight: 700; margin: 3rem 0 1rem;
  padding-top: 1.1rem; border-top: 1px solid var(--rule);
}
h3 { font-size: 1.12rem; font-weight: 650; margin: 2rem 0 .7rem; }
h4 { font-size: 1rem; font-weight: 650; margin: 1.5rem 0 .5rem; color: var(--ink-soft); }

p { margin: 0 0 1.15rem; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }

strong { font-weight: 700; }
em { font-style: italic; color: var(--ink-soft); }

ul, ol { margin: 0 0 1.2rem; padding-left: 1.35rem; }
li { margin: .35rem 0; }
li > ul, li > ol { margin: .35rem 0; }

blockquote {
  margin: 1.4rem 0;
  padding: .1rem 0 .1rem 1.1rem;
  border-left: 3px solid var(--rule);
  color: var(--ink-soft);
}

hr { border: 0; border-top: 1px solid var(--rule); margin: 2.6rem 0; }

/* Таблицы: их в серии много и они широкие — каждая прокручивается внутри себя,
   страница по горизонтали не едет никогда. */
.tablewrap {
  overflow-x: auto;
  margin: 1.5rem 0;
  border: 1px solid var(--rule);
  border-radius: 6px;
  background: var(--surface);
}
table {
  border-collapse: collapse;
  width: 100%;
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
th, td {
  padding: .5rem .7rem;
  text-align: left;
  border-bottom: 1px solid var(--rule-soft);
  white-space: nowrap;
}
th {
  font-weight: 650;
  color: var(--ink-soft);
  background: var(--accent-bg);
  border-bottom: 1px solid var(--rule);
  position: sticky; top: 0;
}
tr:last-child td { border-bottom: 0; }
td:first-child, th:first-child { white-space: normal; min-width: 11rem; }

/* Графики серии: картинка — ссылка на саму себя, открывается в новой вкладке */
img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
a.figure {
  display: block;
  margin: 1.6rem 0;
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: .5rem;
  background: var(--surface);
  cursor: zoom-in;
  text-decoration: none;
}
a.figure:hover { border-color: var(--accent); }

code {
  font: 14px/1.5 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  background: var(--rule-soft);
  padding: .1rem .3rem;
  border-radius: 3px;
}

/* Обложка */
.cover-title { font-size: 2.4rem; margin-top: 3rem; }
.lede { font-size: 1.12rem; color: var(--ink-soft); margin-bottom: 2.4rem; }
.toc { list-style: none; padding: 0; margin: 2rem 0 0; }
.toc li {
  margin: 0 0 .9rem;
  padding: .95rem 1.1rem;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--surface);
}
.toc a {
  font: 650 1.06rem/1.35 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  text-decoration: none;
}
.toc a:hover { text-decoration: underline; }
.toc p { margin: .35rem 0 0; font-size: .95rem; color: var(--ink-soft); }

.foot {
  margin-top: 3.5rem; padding-top: 1.2rem;
  border-top: 1px solid var(--rule);
  font: 13px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink-faint);
}

@media (max-width: 640px) {
  body { font-size: 16px; }
  .wrap { padding: 0 .9rem 4rem; }
  h1, .cover-title { font-size: 1.6rem; }
  h2 { font-size: 1.25rem; }
}
"""

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="topbar"><div class="wrap">{nav}</div></nav>
<main class="wrap">
{body}
<div class="foot">{foot}</div>
</main>
</body>
</html>
"""


def nav_html(current):
    """Шапка: все части серии, текущая — жирным без ссылки."""
    out = ['<a href="index.html">Оглавление</a>']
    for src, short, _ in PARTS:
        label = short.replace("Глава ", "").replace("Интерлюдия. ", "").split(".")[0]
        if src == current:
            out.append(f'<span class="here">{htmllib.escape(label)}</span>')
        else:
            out.append(f'<a href="{src[:-3]}.html">{htmllib.escape(label)}</a>')
    return '<span class="sep">·</span>'.join(out)


def convert(md_text):
    """markdown → html с таблицами; ссылки между главами переводятся на .html."""
    body = markdown.markdown(
        md_text,
        extensions=["tables", "sane_lists", "attr_list"],
        output_format="html5",
    )
    # ссылки на соседние части серии
    body = re.sub(r'href="(comon-story-[0-9a-z-]+)\.md"', r'href="\1.html"', body)

    # Строка навигации и «Дальше: …» есть в самом markdown — они нужны тому, кто
    # читает главы файлами на GitHub. В html их роль уже играют шапка и подвал,
    # поэтому здесь абзац-дубль убирается: два одинаковых ряда ссылок подряд
    # выглядят ошибкой вёрстки. Признак строгий — абзац из ссылок на части серии.
    def drop_nav(m):
        p = m.group(0)
        return "" if len(re.findall(r'href="comon-story-', p)) >= 3 else p

    body = re.sub(r"<p>(?:(?!</p>).)*?</p>", drop_nav, body, flags=re.S)
    body = re.sub(r"<p><strong>Дальше:</strong>.*?</p>\s*", "", body, flags=re.S)
    # каждая таблица — в свой прокручиваемый контейнер
    body = body.replace("<table>", '<div class="tablewrap"><table>')
    body = body.replace("</table>", "</table></div>")

    # Графики кликабельны: в тексте картинка ужата до ширины колонки, а деталей
    # на ней много (подписи, сноска, мелкие деления) — по клику открывается сам
    # файл 1600 px в новой вкладке. rel="noopener" обязателен при target="_blank":
    # без него открытая страница получает ссылку на открывшую через window.opener.
    def wrap_img(m):
        return (f'<a class="figure" href="{m.group(1)}" target="_blank" '
                f'rel="noopener">{m.group(0)}</a>')

    body = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', wrap_img, body)
    return body


def build():
    pages, problems = [], []
    for i, (src, short, blurb) in enumerate(PARTS):
        f = SRC / src
        if not f.exists():
            problems.append(f"нет файла {src}")
            continue
        md_text = f.read_text(encoding="utf-8")
        body = convert(md_text)
        title_m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else short

        nxt = ""
        if i + 1 < len(PARTS):
            n_src, n_short, _ = PARTS[i + 1]
            nxt = f'Дальше: <a href="{n_src[:-3]}.html">{htmllib.escape(n_short)}</a> · '
        foot = nxt + '<a href="index.html">оглавление серии</a>'

        page = PAGE.format(title=htmllib.escape(title), nav=nav_html(src),
                           body=body, foot=foot)
        pages.append((src[:-3] + ".html", page, len(body)))

    # обложка
    items = []
    for src, short, blurb in PARTS:
        items.append(f'  <li><a href="{src[:-3]}.html">{htmllib.escape(short)}</a>'
                     f'<p>{htmllib.escape(blurb)}</p></li>')
    cover_body = (
        '<h1 class="cover-title">Автоследование: исследование</h1>\n'
        '<p class="lede">Автоследование — сервис, где инвестор подписывается на торговую '
        'стратегию другого человека, и сделки автора повторяются на его счёте. '
        'Исследование построено на полной популяции одной площадки — Comon (Финам): '
        '19 978 стратегий, включая 18 529 закрытых. Все метрики пересчитаны из дневных '
        'рядов, а не взяты с витрины.</p>\n'
        '<ul class="toc">\n' + "\n".join(items) + "\n</ul>"
    )
    pages.append(("index.html",
                  PAGE.format(title="Автоследование: исследование",
                              nav='<a href="index.html" class="here">Оглавление</a>',
                              body=cover_body,
                              foot="Данные и код расчётов — в этом же репозитории: "
                                   "<code>data/</code> и <code>scripts/</code>."),
                  len(cover_body)))

    if problems:
        for p in problems:
            print(f"🔴 {p}")
    if CHECK:
        print(f"проверка: {len(pages)} страниц собралось бы, ничего не записано")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "style.css").write_text(CSS, encoding="utf-8")
    for name, text, n in pages:
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"  {name:<34} {len(text)/1024:>7.0f} КБ")
    print(f"  {'style.css':<34} {len(CSS)/1024:>7.0f} КБ")
    print(f"\nсобрано страниц: {len(pages)} → {OUT}")


if __name__ == "__main__":
    build()
