// Выкачка сырья Comon (автоследование Финам): каталог + карточки + дневные ряды доходности.
//
// Зачем: исследование P3 «выживаемость публичных стратегий» — нужна вся популяция,
// включая архивные (мёртвые) системы, иначе метрики съедены survivorship bias.
//
// Доступ: comon.ru закрыт антиботом ServicePipe. Куки (spid/spsc) берём из живого Chrome
// через comon_cdp.js, дальше качаем обычным fetch из Node — проверено, проходит.
// При 403 куки перезабираются автоматически (Chrome должен быть запущен:
//   google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-comon https://www.comon.ru/strategies/ ).
//
// Вежливость к серверу: РОВНО 2 воркера, пауза PAUSE_MS после каждого запроса в каждом
// воркере (~1.7 req/s), backoff при 429/5xx, аварийный стоп после MAX_FAIL_STREAK подряд.
//
//   node comon_fetch.js catalog   # 40 страниц v2 (все, вкл. архивные) + 3 страницы v1 (живые)
//   node comon_fetch.js fetch     # карточка + ряд на каждый id из каталога; идемпотентно (resume)
//
// Сырьё складывается КАК ЕСТЬ (без обрезки бенчмарков), gzip: back-test/data/comon/

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { execSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', 'data', 'comon');
const DIRS = {
  catalog: path.join(ROOT, 'catalog'),
  cards: path.join(ROOT, 'cards'),
  profit: path.join(ROOT, 'profit'),
  log: path.join(ROOT, '_log'),
};
const BASE = 'https://www.comon.ru';
const UA =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36';

const PAUSE_MS = +(process.env.PAUSE_MS || 1200); // пауза в КАЖДОМ воркере после запроса
const WORKERS = +(process.env.WORKERS || 2);      // параллелизация (по требованию — 2)
const REQ_TIMEOUT_MS = 30000;
const RETRIES = 3;
const MAX_FAIL_STREAK = 10;
const MAX_REDIRECT_HOPS = 5;   // 307-челленджей ServicePipe на один запрос

for (const d of Object.values(DIRS)) fs.mkdirSync(d, { recursive: true });
const logStream = fs.createWriteStream(path.join(DIRS.log, 'fetch.log'), { flags: 'a' });
const stateStream = fs.createWriteStream(path.join(DIRS.log, 'state.ndjson'), { flags: 'a' });

function log(msg) {
  const line = `${new Date().toISOString()} ${msg}`;
  logStream.write(line + '\n');
  console.log(line);
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── куки из живого Chrome ────────────────────────────────────────────────────
// Держим банку как Map: ServicePipe периодически прокручивает сессионную куку spsc
// (см. redirect307 ниже), и её нужно уметь подменить в одном месте.
const JAR = new Map();
const jarString = () => [...JAR].map(([k, v]) => `${k}=${v}`).join('; ');

function refreshCookies() {
  const raw = execSync(`node ${path.join(__dirname, 'comon_cdp.js')} cookies`, {
    maxBuffer: 1e8,
  }).toString();
  JAR.clear();
  for (const c of JSON.parse(raw)) JAR.set(c.name, c.value);
  log(`[cookies] обновлены из Chrome, ${JAR.size} шт`);
}

// Применить Set-Cookie из ответа к банке. Возвращает список изменённых имён.
function applySetCookie(res) {
  const changed = [];
  const list = typeof res.headers.getSetCookie === 'function' ? res.headers.getSetCookie() : [];
  for (const raw of list) {
    const [pair] = raw.split(';');
    const i = pair.indexOf('=');
    if (i <= 0) continue;
    const name = pair.slice(0, i).trim();
    const value = pair.slice(i + 1).trim();
    if (JAR.get(name) !== value) { JAR.set(name, value); changed.push(name); }
  }
  return changed;
}

// ── один запрос с ретраями и backoff ─────────────────────────────────────────
let pauseUntil = 0;   // глобальная пауза (backoff): оба воркера ждут
let failStreak = 0;
let redirects = 0;    // сколько раз ServicePipe прокрутил куку через 307
let aborted = null;

async function get(url) {
  for (let attempt = 1; attempt <= RETRIES; attempt++) {
    while (Date.now() < pauseUntil) await sleep(500);
    const t0 = Date.now();
    let status = 0, body = null, err = null;
    try {
      // redirect:'manual' обязателен. ServicePipe периодически отвечает 307 на ТОТ ЖЕ URL,
      // выдавая новую сессионную куку spsc; автоматический редирект в fetch тащит старый
      // заголовок cookie → бесконечная петля («redirect count exceeded»). Поэтому берём
      // Set-Cookie из ответа, кладём в банку и повторяем запрос сами.
      for (let hop = 0; hop <= MAX_REDIRECT_HOPS; hop++) {
        const r = await fetch(BASE + url, {
          redirect: 'manual',
          headers: {
            cookie: jarString(),
            'user-agent': UA,
            accept: 'application/json',
            'accept-language': 'ru-RU,ru;q=0.9',
            referer: BASE + '/strategies/',
          },
          signal: AbortSignal.timeout(REQ_TIMEOUT_MS),
        });
        const changed = applySetCookie(r);
        status = r.status;
        if (status >= 300 && status < 400) {
          await r.text();
          if (!changed.length) { err = `редирект ${status} без новой куки`; break; }
          redirects++;
          await sleep(300);
          continue;                       // повтор того же URL с обновлённой кукой
        }
        body = await r.text();
        break;
      }
    } catch (e) {
      err = String(e).slice(0, 120);
    }
    const ms = Date.now() - t0;

    if (status === 200 && body) {
      failStreak = 0;
      return { ok: true, status, body, ms };
    }
    // 403 — антибот сбросил сессию: перезабрать куки и повторить
    if (status === 403) {
      log(`[403] ${url} — перезабираю куки`);
      try { refreshCookies(); } catch (e) { log(`[403] куки не обновились: ${e}`); }
      await sleep(3000);
      continue;
    }
    if (status === 429 || status >= 500 || err) {
      const wait = [5000, 10000, 20000, 60000][Math.min(attempt - 1, 3)];
      log(`[retry ${attempt}/${RETRIES}] ${url} status=${status} err=${err} → пауза ${wait / 1000}с`);
      pauseUntil = Date.now() + wait;
      continue;
    }
    // прочие коды (404 и т.п.) — не ретраим, это ответ сервера
    return { ok: false, status, body, ms, err };
  }
  return { ok: false, status: 0, body: null, ms: 0, err: 'исчерпаны ретраи' };
}

function writeGz(file, text) {
  fs.writeFileSync(file, zlib.gzipSync(Buffer.from(text, 'utf8')));
}

// ── каталог ──────────────────────────────────────────────────────────────────
async function catalog() {
  refreshCookies();
  for (const [name, url, includeArchived] of [
    ['v2_all', '/api/v2/strategies', true],
    ['v1_live', '/api/v1/strategies', false],
  ]) {
    const out = [];
    let page = 1, totalPages = 1;
    do {
      const q = `${url}?page=${page}&pageSize=500${includeArchived ? '&IncludeArchived=true' : ''}`;
      const r = await get(q);
      if (!r.ok) { log(`[catalog] ОШИБКА ${name} стр.${page}: status=${r.status} ${r.err || ''}`); break; }
      const j = JSON.parse(r.body);
      totalPages = j.paging.totalPages;
      for (const row of j.data) out.push(JSON.stringify(row));
      log(`[catalog] ${name} стр.${page}/${totalPages} +${j.data.length} (всего ${out.length}/${j.paging.totalItems})`);
      page++;
      await sleep(PAUSE_MS);
    } while (page <= totalPages);
    writeGz(path.join(DIRS.catalog, `${name}.ndjson.gz`), out.join('\n') + '\n');
    log(`[catalog] ${name}: записано ${out.length} строк`);
  }
}

// ── карточки + ряды ──────────────────────────────────────────────────────────
function readCatalogIds() {
  const f = path.join(DIRS.catalog, 'v2_all.ndjson.gz');
  if (!fs.existsSync(f)) throw new Error('нет каталога — сначала: node comon_fetch.js catalog');
  const ids = zlib.gunzipSync(fs.readFileSync(f)).toString('utf8')
    .split('\n').filter(Boolean).map(l => JSON.parse(l).id);
  return [...new Set(ids)].sort((a, b) => a - b);
}

async function fetchAll() {
  refreshCookies();
  const ids = readCatalogIds();
  const tasks = [];
  for (const id of ids) {
    if (!fs.existsSync(path.join(DIRS.cards, `${id}.json.gz`)))
      tasks.push({ kind: 'card', id, url: `/api/v1/strategies/${id}`, dir: DIRS.cards });
    if (!fs.existsSync(path.join(DIRS.profit, `${id}.json.gz`)))
      tasks.push({ kind: 'profit', id, url: `/api/v2/strategies/${id}/profit`, dir: DIRS.profit });
  }
  const total = tasks.length;
  log(`[fetch] систем ${ids.length}, к выкачке ${total} файлов (уже на диске ${ids.length * 2 - total}), воркеров ${WORKERS}, пауза ${PAUSE_MS}мс`);
  if (!total) { log('[fetch] всё уже выкачано'); return; }

  let next = 0, done = 0, bad = 0, bytes = 0;
  const t0 = Date.now();

  async function worker(w) {
    while (!aborted) {
      const i = next++;
      if (i >= total) return;
      const t = tasks[i];
      const r = await get(t.url);
      if (r.ok) {
        writeGz(path.join(t.dir, `${t.id}.json.gz`), r.body);
        bytes += r.body.length;
      } else {
        bad++;
        failStreak++;
        if (failStreak >= MAX_FAIL_STREAK) {
          aborted = `${MAX_FAIL_STREAK} подряд неудач (последняя ${t.kind} ${t.id}: status=${r.status} ${r.err || ''})`;
          log(`[СТОП] ${aborted}`);
          return;
        }
      }
      stateStream.write(JSON.stringify({
        ts: new Date().toISOString(), kind: t.kind, id: t.id,
        status: r.status, bytes: r.body ? r.body.length : 0, ms: r.ms, err: r.err || null,
      }) + '\n');
      done++;
      if (done % 200 === 0) {
        const el = (Date.now() - t0) / 1000;
        const rate = done / el;
        const eta = (total - done) / rate / 3600;
        log(`[fetch] ${done}/${total} ошибок=${bad} 307-куки=${redirects} ${rate.toFixed(2)} req/s сырья=${(bytes / 1e6).toFixed(0)}МБ ETA=${eta.toFixed(1)}ч`);
      }
      await sleep(PAUSE_MS);
    }
  }

  await Promise.all(Array.from({ length: WORKERS }, (_, w) => worker(w)));
  const el = (Date.now() - t0) / 1000;
  log(`[fetch] ЗАВЕРШЕНО: ${done}/${total}, ошибок ${bad}, ${(el / 3600).toFixed(2)}ч, сырья ${(bytes / 1e6).toFixed(0)}МБ${aborted ? ' — ПРЕРВАНО: ' + aborted : ''}`);
}

const cmd = process.argv[2];
(async () => {
  if (cmd === 'catalog') await catalog();
  else if (cmd === 'fetch') await fetchAll();
  else { console.error('команды: catalog | fetch'); process.exit(2); }
})().catch(e => { log('ФАТАЛЬНО: ' + (e.stack || e)); process.exit(1); });
