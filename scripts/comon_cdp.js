// Минимальный CDP-клиент к живому Chrome (--remote-debugging-port=9222).
// Нужен потому, что comon.ru закрыт антиботом ServicePipe: curl/WebFetch получают
// JS-challenge, а fetch() внутри страницы same-origin проходит с куками браузера.
//
//   node comon_cdp.js nav  <url>
//   node comon_cdp.js eval <js>          // returnByValue + awaitPromise
//   node comon_cdp.js cookies            // куки comon.ru (для пробы прямого fetch)

const PORT = process.env.CDP_PORT || 9222;

async function firstPage(match = 'comon.ru') {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
  const pages = list.filter(t => t.type === 'page' && t.webSocketDebuggerUrl);
  const p = pages.find(t => t.url.includes(match)) || pages[0];
  if (!p) throw new Error('нет page-вкладки с отладкой');
  return p;
}

function connect(url) {
  const ws = new WebSocket(url);
  let id = 0;
  const waiting = new Map();
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.id && waiting.has(m.id)) {
      const { res, rej } = waiting.get(m.id);
      waiting.delete(m.id);
      m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result);
    }
  });
  const ready = new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', rej);
  });
  return {
    ready,
    send: (method, params = {}) =>
      new Promise((res, rej) => {
        const my = ++id;
        waiting.set(my, { res, rej });
        ws.send(JSON.stringify({ id: my, method, params }));
      }),
    close: () => ws.close(),
  };
}

async function main() {
  const [cmd, arg] = process.argv.slice(2);
  const page = await firstPage();
  const c = connect(page.webSocketDebuggerUrl);
  await c.ready;

  if (cmd === 'nav') {
    await c.send('Page.enable');
    await c.send('Page.navigate', { url: arg });
    await new Promise(r => setTimeout(r, 4000));
    const r = await c.send('Runtime.evaluate', { expression: 'location.href', returnByValue: true });
    console.log(r.result.value);
  } else if (cmd === 'eval') {
    const r = await c.send('Runtime.evaluate', {
      expression: arg,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) {
      console.error('JS-исключение: ' + JSON.stringify(r.exceptionDetails.exception));
      process.exitCode = 1;
    } else {
      const v = r.result.value;
      process.stdout.write(typeof v === 'string' ? v : JSON.stringify(v));
    }
  } else if (cmd === 'cookies') {
    await c.send('Network.enable');
    const r = await c.send('Network.getCookies', { urls: ['https://www.comon.ru/'] });
    console.log(JSON.stringify(r.cookies, null, 1));
  } else {
    console.error('команды: nav <url> | eval <js> | cookies');
    process.exitCode = 2;
  }
  c.close();
}

main().catch(e => {
  console.error(String(e));
  process.exit(1);
});
