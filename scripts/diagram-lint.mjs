// Static checks for the two binding rules that render silently wrong.
//
// Neither is a syntax error, and both survived a grep-level review:
//   1. A legend on a `graph` series binds to `series.categories`, and each node
//      needs `category`. Without that the legend is configured and draws nothing.
//   2. The visible box label is `series.data[].name`. A `rectGraph.nodes[].label`
//      that differs from it is never shown, so a technical id ends up on screen.
//
// Cheap enough to run on every edit; `render-check.mjs` remains the proof.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const BASE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'docs', 'documentation');
const walk = (dir) => readdirSync(dir).flatMap((e) => {
  const full = path.join(dir, e);
  if (statSync(full).isDirectory()) return walk(full);
  return e.endsWith('.mdx') ? [full] : [];
});

const problems = [];

// A duplicate key in an object literal is silently kept as the *last* one, so
// inserting `label: { color: … }` above an existing `label: { show: true, … }`
// drops the insertion without a word. Scan brace frames for repeated keys.
function duplicateKeys(source, where, name) {
  const frames = [new Map()];
  let i = 0;
  while (i < source.length) {
    const c = source[i];
    if (c === "'" || c === '"' || c === '`') {
      const quote = c;
      i += 1;
      while (i < source.length && source[i] !== quote) i += source[i] === '\\' ? 2 : 1;
      i += 1;
      continue;
    }
    if (c === '/' && source[i + 1] === '/') { while (i < source.length && source[i] !== '\n') i += 1; continue; }
    if (c === '{') { frames.push(new Map()); i += 1; continue; }
    if (c === '}') { frames.pop(); i += 1; continue; }
    if (c === '[') { frames.push(new Map()); i += 1; continue; }
    if (c === ']') { frames.pop(); i += 1; continue; }
    const key = /^([A-Za-z_$][\w$]*)\s*:/.exec(source.slice(i));
    if (key && frames.length) {
      const frame = frames[frames.length - 1];
      const line = source.slice(0, i).split('\n').length;
      if (frame.has(key[1])) {
        problems.push(`${where}: ${name} sets "${key[1]}" twice in one object (lines ${frame.get(key[1])} and ${line}) — the first is silently discarded`);
      } else {
        frame.set(key[1], line);
      }
      i += key[0].length;
      continue;
    }
    i += 1;
  }
}

for (const file of walk(BASE)) {
  const source = readFileSync(file, 'utf8');
  const where = path.relative(path.join(BASE, '..', '..'), file);

  // Blocks may reference an earlier block on the same page, so evaluate them in
  // one shared scope rather than in isolation.
  const scope = {};
  const evaluate = (literal, label) => {
    try {
      const keys = Object.keys(scope);
      return new Function(...keys, `return ${literal};`)(...keys.map((k) => scope[k]));
    } catch (error) {
      problems.push(`${where}: ${label} is not valid JavaScript — ${error.message}`);
      return undefined;
    }
  };

  for (const match of source.matchAll(/^export const (\w+) = (\{[\s\S]*?\n\});$/gm)) {
    const [, name, literal] = match;
    const option = evaluate(literal, name);
    if (!option) continue;
    scope[name] = option;

    duplicateKeys(literal, where, name);
    const series = [option.series].flat().filter(Boolean);
    const graphs = series.filter((s) => s?.type === 'graph');

    if (option.legend) {
      // A legend entry renders only if its name binds to something: a series
      // name, a graph category, or a data item name in a pie-like series.
      const bindable = new Set();
      for (const s of series) {
        if (s?.name) bindable.add(s.name);
        for (const c of s?.categories ?? []) bindable.add(c?.name ?? c);
        if (['pie', 'funnel', 'map', 'radar'].includes(s?.type)) {
          for (const d of s?.data ?? []) if (d?.name) bindable.add(d.name);
        }
      }
      const entries = [option.legend.data ?? []].flat().map((d) => d?.name ?? d).filter(Boolean);
      const unbound = entries.filter((e) => !bindable.has(e));
      if (entries.length && unbound.length === entries.length) {
        problems.push(`${where}: ${name} configures a legend that binds to nothing — it will draw no items (a graph series needs series.categories, or an empty named series per entry)`);
      } else if (unbound.length) {
        problems.push(`${where}: ${name} legend entries bind to nothing: ${unbound.join(', ')}`);
      }
    }

    // A graph series refuses duplicate node names at runtime — it throws
    // "Graph nodes have duplicate name or id" and the figure never renders.
    for (const graph of graphs) {
      const seen = new Set();
      for (const item of graph.data ?? []) {
        const id = item?.name ?? item;
        if (typeof id !== 'string') continue;
        if (seen.has(id)) problems.push(`${where}: ${name} has two graph nodes named "${id}" — the series will not render`);
        seen.add(id);
      }
    }

    // A graph series refuses duplicate node names at runtime — it throws
    // "Graph nodes have duplicate name or id" and the figure never renders.
    for (const graph of graphs) {
      const seen = new Set();
      for (const item of graph.data ?? []) {
        const id = item?.name ?? item;
        if (typeof id !== 'string') continue;
        if (seen.has(id)) problems.push(`${where}: ${name} has two graph nodes named "${id}" — the series will not render`);
        seen.add(id);
      }
    }

    const rectGraphName = `${name}RectGraph`;
    const rectMatch = source.match(new RegExp(`^export const ${rectGraphName} = (\\{[\\s\\S]*?\\n\\});$`, 'm'));
    if (!rectMatch) continue;
    const rectGraph = evaluate(rectMatch[1], rectGraphName);
    if (!rectGraph) continue;
    scope[rectGraphName] = rectGraph;

    const names = (graphs[0]?.data ?? []).map((d) => d?.name ?? d);
    const nodes = rectGraph.nodes ?? [];
    if (names.length !== nodes.length) {
      problems.push(`${where}: ${name} has ${names.length} series data entries but ${nodes.length} rectGraph nodes — the compiler pairs them by index`);
      continue;
    }
    nodes.forEach((node, index) => {
      if (node.label !== undefined && node.label !== names[index]) {
        problems.push(`${where}: ${name} node "${node.id}" carries label "${node.label}" but the drawn text is the series name "${names[index]}" — a technical id would be shown`);
      }
    });
  }
}

for (const problem of problems) console.log(`✗ ${problem}`);
console.log(problems.length ? `\n${problems.length} problem(s).` : 'No dead legends, no hidden labels.');
process.exit(problems.length ? 1 : 0);
