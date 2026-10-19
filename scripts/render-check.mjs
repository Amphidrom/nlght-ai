// Verify documentation diagrams from their browser output.
//
// ECharts JSON parsing does not prove that the figure renders. Build the docs
// from the current tree and capture each figure at desktop and narrow widths.
//
//   node scripts/render-check.mjs ingestion/retrieval configuration/principals
//   node scripts/render-check.mjs --all
//   node scripts/render-check.mjs --home
//
// `--all` covers technical documentation under docs/documentation. Use
// `--home` for the home page. The flags may be combined.
//
// Always rebuild and start a fresh preview. A process already bound to the
// port can serve stale screenshots if the new server fails to start.
//
// Playwright and the site live in the landing repository. NLGHT_LANDING points
// there (default ../nlght-landing).

import { createRequire } from 'node:module';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const LANDING = process.env.NLGHT_LANDING ?? path.resolve(ROOT, '..', 'nlght-landing');
const OUT = process.env.RENDER_CHECK_OUT ?? path.join(ROOT, '.render-check');
const PORT = Number(process.env.RENDER_CHECK_PORT ?? 4331);
const WIDTHS = [['desktop', 1600], ['narrow', 480]];

const args = process.argv.slice(2);
const all = args.includes('--all');
const home = args.includes('--home');
let pages = args.filter((a) => !a.startsWith('--'));

if (all) {
  const base = path.join(ROOT, 'docs', 'documentation');
  const walk = (dir) => readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) return walk(full);
    if (!entry.endsWith('.mdx')) return [];
    if (!readFileSync(full, 'utf8').includes('<EChart')) return [];
    return [path.relative(base, full).replace(/\.mdx$/, '')];
  });
  pages = walk(base);
}
if (home) pages.unshift('@home');
if (!pages.length) {
  console.error('usage: render-check.mjs [--all] [--home] <page> [page...]');
  process.exit(2);
}

const run = (cmd, cmdArgs, opts = {}) =>
  spawnSync(cmd, cmdArgs, { cwd: LANDING, stdio: 'inherit', ...opts });

// Any survivor from an earlier run would keep the port and serve a stale build.
spawnSync('sh', ['-c', `lsof -ti tcp:${PORT} | xargs -r kill -9`], { stdio: 'ignore' });

{
  console.log('Importing the working tree and building the site');
  if (run('node', ['scripts/fetch-docs.mjs']).status !== 0) process.exit(1);
  const build = run('./node_modules/.bin/astro', ['build'], { stdio: ['ignore', 'pipe', 'inherit'] });
  if (build.status !== 0) {
    console.error('Build failed: EChart.astro rejects the option before the browser sees it');
    process.exit(1);
  }
}

const require = createRequire(path.join(LANDING, 'package.json'));
const { chromium } = require('playwright');

const server = spawn('./node_modules/.bin/astro', ['preview', '--port', String(PORT)], {
  cwd: LANDING, stdio: 'ignore', detached: true,
});
const stop = () => { try { process.kill(-server.pid); } catch {} };
process.on('exit', stop);
process.on('SIGINT', () => { stop(); process.exit(130); });

const base = `http://localhost:${PORT}`;
let up = false;
for (let i = 0; i < 60; i += 1) {
  if (server.exitCode !== null) break;
  try {
    const res = await fetch(base, { signal: AbortSignal.timeout(1000) });
    if (res.ok || res.status === 404) { up = true; break; }
  } catch { await new Promise((r) => setTimeout(r, 500)); }
}
if (!up) {
  console.error(`Preview did not start on ${PORT}. Screenshots could show a stale build`);
  stop();
  process.exit(1);
}

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
let failures = 0;

for (const page of pages) {
  for (const [label, width] of WIDTHS) {
    const tab = await browser.newPage({ viewport: { width, height: 1200 } });
    if (page === '@home') {
      // The one-time intro overlays the chart, so inspect the settled page.
      await tab.addInitScript(() => localStorage.setItem('ei-intro-seen', '1'));
    }
    const errors = [];
    tab.on('pageerror', (e) => errors.push(String(e)));
    tab.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

    const url = page === '@home' ? `${base}/` : `${base}/documentation/${page}/`;
    const response = await tab.goto(url, { waitUntil: 'networkidle' });
    if (!response?.ok()) {
      console.log(`FAIL ${page} [${label}] HTTP ${response?.status()}`);
      failures += 1; await tab.close(); continue;
    }

    const figures = await tab.$$('figure.nlght-echart');
    const slug = page === '@home' ? 'home' : page.replaceAll('/', '__');
    for (const [index, figure] of figures.entries()) {
      await figure.scrollIntoViewIfNeeded();
      // Charts mount on intersection, then lay out asynchronously.
      await tab.waitForTimeout(900);
      const state = await figure.getAttribute('data-echart-state');
      const file = path.join(OUT, `${slug}--${label}-${index}.png`);
      await figure.screenshot({ path: file });
      const texts = await figure.$$eval('svg text', (nodes) => nodes.map((n) => n.textContent));
      const ok = state === 'ready';
      if (!ok) failures += 1;
      console.log(`${ok ? 'OK' : 'FAIL'} ${page} [${label}] #${index} state=${state} texts=${texts.length}`);
      if (!ok) {
        const status = await figure.$eval('[data-echart-status]', (n) => n.textContent).catch(() => '');
        if (status?.trim()) console.log(`    status: ${status.trim()}`);
      }
    }
    if (!figures.length) console.log(`  ${page} [${label}]: no figures`);
    if (errors.length) {
      failures += 1;
      console.log(`FAIL ${page} [${label}] console: ${errors.slice(0, 3).join(' | ')}`);
    }
    await tab.close();
  }
}

await browser.close();
stop();
console.log(failures ? `\n${failures} problem(s). Inspect screenshots in ${OUT}.` : `\nAll figures ready. Inspect screenshots in ${OUT}.`);
process.exit(failures ? 1 : 0);
