#!/usr/bin/env node
// Biweekly progress deck builder (reporting only; never part of the training runtime).
//
//   node tools/slides/build_deck.js reports/biweekly/<date>/deck.yaml [-o out.pptx]
//
// Content comes from one YAML file written from the result records (every number traceable).
// Look comes from THEME / LAYOUTS below: every frame element (section tab, title, key-point box,
// footer band, citation slot, page number) lives on a slide layout with named placeholders, so the
// deck restyles in one place and stays easy to edit in PowerPoint (and with Claude in PowerPoint).
// Schema: .claude/skills/biweekly-review/SKILL.md.
"use strict";
const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const pptxgen = require("pptxgenjs");

// ---------------------------------------------------------------- theme (blue family, few colours)
const T = {
  primary: "2E6DB4",   // section tab, footer band, table header, card heads
  primaryDark: "1F4E79",
  tint: "DDEBF7",      // card fill, highlighted table cells
  message: "F2F2F2",   // key-point box
  text: "222222",
  sub: "595959",
  line: "BFBFBF",
  accent: "C0392B",    // the one emphasis colour (==text==); use 1-2 times per slide at most
  white: "FFFFFF",
  latin: "Calibri",
  ea: "游ゴシック",
};
const W = 13.333, H = 7.5;           // LAYOUT_WIDE
const M = 0.5;                       // side margin
const BODY_Y = 1.45, FOOT_Y = 7.08;  // header: tab at (0,0), line 0.66, key point 0.76-1.3  // body starts under the key-point box; footer band top
const BODY_H = FOOT_Y - 0.12 - BODY_Y;

// ---------------------------------------------------------------- text helpers
// "**bold**" -> bold run, "==x==" -> accent run. Returns pptxgenjs run array.
// inline citations: "[@key]" in any text becomes "[n]" (numbered by first use across the deck)
let CITE = (k) => `[${k}]`;
function runs(s, base = {}) {
  s = String(s).replace(/\[@([\w-]+(?:\s*,\s*@?[\w-]+)*)\]/g, (_, ks) => `[${ks.split(/\s*,\s*@?/).map((k) => CITE(k)).join(", ")}]`);
  if (s.includes("\n")) {  // soft line break inside one paragraph (keeps a bullet's hanging indent)
    const out = [];
    s.split("\n").forEach((part, i) => {
      const r = runs(part, base);
      if (i > 0 && r.length) r[0].options = { ...r[0].options, softBreakBefore: true };
      out.push(...r);
    });
    return out;
  }
  const out = [];
  const re = /(\*\*[^*]+\*\*|==[^=]+==)/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ text: s.slice(last, m.index), options: { ...base } });
    const t = m[0];
    if (t.startsWith("**")) out.push({ text: t.slice(2, -2), options: { ...base, bold: true } });
    else out.push({ text: t.slice(2, -2), options: { ...base, bold: true, color: T.accent } });
    last = m.index + t.length;
  }
  if (last < s.length) out.push({ text: s.slice(last), options: { ...base } });
  return out;
}
// bullet list -> runs with paragraph breaks; nested arrays become level-2 bullets
function bulletRuns(items, size = 18, opts = {}) {
  const out = [];
  const push = (it, level) => {
    if (Array.isArray(it)) { it.forEach((x) => push(x, level + 1)); return; }
    const para = { bullet: opts.noBullet ? false : { indent: level ? 22 : 18 }, indentLevel: level, paraSpaceAfter: opts.space ?? 6 };
    const parts = runs(it, { fontSize: level ? size - 2 : size, color: T.text });
    Object.assign(parts[0].options, para);
    parts[parts.length - 1].options.breakLine = true;
    out.push(...parts);
  };
  items.forEach((it) => push(it, 0));
  if (out.length) delete out[out.length - 1].options.breakLine;
  return out;
}
const tb = (o) => ({ isTextBox: true, margin: 0.05, fontFace: T.latin, valign: "top", ...o });

// ---------------------------------------------------------------- layouts
function defineLayouts(pres) {
  pres.defineSlideMaster({
    title: "TITLE", background: { color: T.white },
    objects: [
      { rect: { x: 0, y: 0, w: W, h: H, fill: { color: "EEF4FB" } } },
      { rect: { x: 0, y: FOOT_Y, w: W, h: H - FOOT_Y, fill: { color: T.primary } } },
      { placeholder: { options: { name: "event", type: "body", x: M, y: 0.7, w: W - 2 * M, h: 0.5, align: "center", fontSize: 20, color: T.primaryDark, bold: true, fontFace: T.latin }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 1.0, y: 1.6, w: W - 2.0, h: 1.9, align: "center", valign: "middle", fontSize: 32, bold: true, color: T.text, fontFace: T.latin }, text: "" } },
      { placeholder: { options: { name: "info", type: "body", x: 1.5, y: 4.1, w: W - 3.0, h: 2.4, align: "center", valign: "top", fontSize: 18, color: T.text, fontFace: T.latin }, text: "" } },
    ],
  });
  const frame = (extra = []) => [
    { rect: { x: 0, y: FOOT_Y, w: W, h: H - FOOT_Y, fill: { color: T.primary } } },
    { line: { x: 2.55, y: 0.62, w: W - 2.55 - M, h: 0, line: { color: T.line, width: 1 } } },
    { placeholder: { options: { name: "cite", type: "body", x: M, y: FOOT_Y + 0.03, w: W - 2 * M - 1.0, h: H - FOOT_Y - 0.06, fontSize: 11, color: T.white, valign: "middle", fontFace: T.latin }, text: "" } },
    ...extra,
  ];
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: T.white },
    slideNumber: { x: W - M - 0.8, y: FOOT_Y + 0.07, w: 0.8, h: 0.28, fontSize: 12, color: T.white, align: "right", fontFace: T.latin },
    objects: frame([
      { rect: { x: 0, y: 0, w: 2.55, h: 0.62, fill: { color: T.primary } } },
      { placeholder: { options: { name: "section", type: "body", x: 0.15, y: 0, w: 2.38, h: 0.62, fontSize: 20, bold: true, color: T.white, valign: "middle", fontFace: T.latin }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 2.75, y: 0, w: W - 2.75 - M, h: 0.62, fontSize: 24, bold: true, color: T.text, valign: "middle", align: "left", fontFace: T.latin }, text: "" } },
      { rect: { x: M, y: 0.76, w: W - 2 * M, h: 0.54, fill: { color: T.message } } },
      { placeholder: { options: { name: "point", type: "body", x: M + 0.15, y: 0.76, w: W - 2 * M - 0.3, h: 0.54, fontSize: 20, bold: true, color: T.primaryDark, valign: "middle", fontFace: T.latin }, text: "" } },
    ]),
  });
  pres.defineSlideMaster({
    title: "REFS", background: { color: T.white },
    slideNumber: { x: W - M - 0.8, y: FOOT_Y + 0.07, w: 0.8, h: 0.28, fontSize: 12, color: T.white, align: "right", fontFace: T.latin },
    objects: frame([
      { rect: { x: 0, y: 0, w: 2.55, h: 0.62, fill: { color: T.primary } } },
      { placeholder: { options: { name: "section", type: "body", x: 0.15, y: 0, w: 2.38, h: 0.62, fontSize: 20, bold: true, color: T.white, valign: "middle", fontFace: T.latin }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 2.75, y: 0, w: W - 2.75 - M, h: 0.62, fontSize: 24, bold: true, color: T.text, valign: "middle", align: "left", fontFace: T.latin }, text: "" } },
    ]),
  });
  pres.defineSlideMaster({
    title: "TOC", background: { color: T.white },
    slideNumber: { x: W - M - 0.8, y: FOOT_Y + 0.07, w: 0.8, h: 0.28, fontSize: 12, color: T.white, align: "right", fontFace: T.latin },
    objects: [
      { rect: { x: 0, y: 0, w: 3.6, h: FOOT_Y, fill: { color: T.primary } } },
      { rect: { x: 0, y: FOOT_Y, w: W, h: H - FOOT_Y, fill: { color: T.primary } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 2.8, w: 2.7, h: 1.0, fontSize: 32, bold: true, color: T.white, valign: "middle", fontFace: T.latin }, text: "" } },
    ],
  });
}

// ---------------------------------------------------------------- slide bodies
function addFrame(slide, s, citeLabels) {
  slide.addText(s.section || "", { placeholder: "section" });
  slide.addText(runs(s.title || "", {}), { placeholder: "title" });
  if (s.point !== undefined) slide.addText(runs(s.point || "", {}), { placeholder: "point" });
  if (citeLabels.length) slide.addText(citeLabels.join("　"), { placeholder: "cite" });
  // speaker notes: the talk script first; memos follow under an explicit label so they are never read out
  const parts = [];
  if (s.script) parts.push(String(s.script).trim());
  if (s.notes) parts.push("【メモ】\n" + String(s.notes).trim());
  if (parts.length) slide.addNotes(parts.join("\n\n"));
}
function estLines(items, w, size) {
  // calibrated on Mac PowerPoint / Yu Gothic: a 3-column card holds about 15 full-width characters at 18pt
  const perLine = Math.max(8, ((w * 72) / size) * 1.05);
  let n = 0;
  const walk = (it, lvl) => {
    if (Array.isArray(it)) return it.forEach((x) => walk(x, lvl + 1));
    // each soft line break starts a new line
    for (const t of String(it).replace(/\*\*|==/g, "").split("\n")) {
      const wide = [...t].reduce((a, c) => a + (c.charCodeAt(0) > 0x2e7f ? 1 : 0.55), 0);
      n += Math.max(1, Math.ceil(wide / perLine));
    }
  };
  items.forEach((it) => walk(it, 0));
  return n;
}
function textHeight(items, w, size) {
  return estLines(items, w, size) * size * 1.45 / 72 + items.length * 0.08;
}
function cardHeight(cards, w, size) {
  return Math.max(...cards.map((c) => 0.75 + textHeight(c.body || [], w - 0.3, size)));
}
// widest hand-broken line (full-width = 1, half-width = 0.55)
function widestLine(items) {
  let m = 0;
  const walk = (it) => Array.isArray(it) ? it.forEach(walk) : String(it).replace(/\*\*|==|\[@[^\]]*\]/g, "").split("\n")
    .forEach((t) => { m = Math.max(m, [...t].reduce((a, c) => a + (c.charCodeAt(0) > 0x2e7f ? 1 : 0.55), 0)); });
  items.forEach(walk);
  return m;
}
// the largest font size (max..min) whose estimated height fits h and whose hand-broken lines do not wrap again;
// fills the slide instead of leaving the bottom empty
function fitSize(heightAt, h, max, min, items, w) {
  const widest = items ? widestLine(items) : 0;
  for (let z = max; z > min; z--) if (heightAt(z) <= h && (!items || widest <= ((w * 72) / z) * 1.05)) return z;
  return min;
}
// extra space between paragraphs so the leftover height is spread out rather than left at the bottom
function spreadSpace(slack, n) {
  return Math.round(6 + Math.max(0, Math.min(24, (slack * 72 * 0.8) / Math.max(n, 1))));
}
function card(slide, x, y, w, h, head, body, size = 18, space = 6) {
  slide.addShape("rect", { x, y, w, h, fill: { color: T.tint }, line: { color: T.tint }, objectName: `card ${head}` });
  slide.addShape("rect", { x, y, w, h: 0.48, fill: { color: T.primary }, line: { color: T.primary } });
  slide.addText(runs(head, { color: T.white, bold: true, fontSize: 18 }), tb({ x: x + 0.12, y, w: w - 0.24, h: 0.48, valign: "middle" }));
  slide.addText(bulletRuns(body, size, { space }), tb({ x: x + 0.15, y: y + 0.6, w: w - 0.3, h: h - 0.7 }));
}
function sideText(slide, s, x, w) {
  const Y = s._top, HH = FOOT_Y - 0.12 - Y;
  if (!s.side) return;
  const size = s.side_size || fitSize((z) => textHeight(s.side, w - 0.3, z), HH - 0.2, 20, 15, s.side, w - 0.3);
  const space = spreadSpace(HH - 0.3 - textHeight(s.side, w - 0.3, size), s.side.length);
  slide.addText(bulletRuns(s.side, size, { space }), tb({ x, y: Y, w, h: HH }));
}
const BODY = {
  bullets(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    slide.addText(bulletRuns(s.body || [], s.size || 20, { space: 10 }), tb({ x: M, y: Y, w: W - 2 * M, h: HH }));
  },
  cards(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const n = s.cards.length, cols = s.cols || n, rows = Math.ceil(n / cols), gap = 0.3;
    const top = s.lead ? Y + 0.55 : Y;
    if (s.lead) slide.addText(runs(s.lead, { fontSize: 18, color: T.sub }), tb({ x: M, y: Y, w: W - 2 * M, h: 0.45 }));
    const cw = (W - 2 * M - gap * (cols - 1)) / cols, maxH = (FOOT_Y - 0.15 - top - gap * (rows - 1)) / rows;
    // cards run to the bottom of the body area; the font grows (up to 20pt) while the text still fits
    const all = s.cards.flatMap((c) => c.body || []);
    const size = s.size || fitSize((z) => cardHeight(s.cards, cw, z), maxH - 0.2, 20, 16, all, cw - 0.3);
    const ch = s.fill === false ? Math.min(maxH, cardHeight(s.cards, cw, size)) : maxH;
    const space = spreadSpace(ch - 0.25 - cardHeight(s.cards, cw, size), Math.max(...s.cards.map((c) => (c.body || []).length)));
    s.cards.forEach((c, i) => card(slide, M + (i % cols) * (cw + gap), top + Math.floor(i / cols) * (ch + gap), cw, ch, c.head, c.body || [], size, space));
  },
  overview(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    // three blocks on top (goal / questions / approach), a timeline strip at the bottom
    const gap = 0.3, cw = (W - 2 * M - gap * 2) / 3, ch = HH - 1.4;
    const size = fitSize((z) => cardHeight(s.blocks, cw, z), ch - 0.2, 19, 15, s.blocks.flatMap((b) => b.body || []), cw - 0.3);
    const space = spreadSpace(ch - 0.25 - cardHeight(s.blocks, cw, size), Math.max(...s.blocks.map((b) => (b.body || []).length)));
    s.blocks.forEach((b, i) => card(slide, M + i * (cw + gap), Y, cw, ch, b.head, b.body || [], size, space));
    const ty = Y + ch + 0.35, n = s.timeline.length, tw = (W - 2 * M - 0.15 * (n - 1)) / n;
    s.timeline.forEach((t, i) => {
      const x = M + i * (tw + 0.15), cur = !!t.current;
      slide.addShape("homePlate", { x, y: ty, w: tw, h: 1.0, fill: { color: cur ? T.primary : T.message }, line: { color: cur ? T.primary : T.line, width: 1 } });
      slide.addText([
        { text: t.label, options: { bold: true, fontSize: 15, color: cur ? T.white : T.text, breakLine: true } },
        { text: t.note || "", options: { fontSize: 12, color: cur ? T.white : T.sub } },
      ], tb({ x: x + 0.1, y: ty + 0.05, w: tw - 0.45, h: 0.9, valign: "middle" }));
    });
  },
  table(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const t = s.table, sideW = s.side ? (s.side_w || 4.2) : 0;
    const tw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    const hl = new Set((t.highlight || []).map(([r, c]) => `${r},${c}`));
    const size = t.size || 17;
    const header = t.header.map((h) => ({ text: h, options: { bold: true, color: T.white, fill: { color: T.primary }, align: "center", fontSize: size } }));
    const rows = t.rows.map((r, ri) => r.map((v, ci) => ({
      text: runs(String(v) === "" ? "\u00a0" : String(v), { fontSize: size, color: T.text }),
      options: { align: ci === 0 ? "left" : "center", fill: { color: hl.has(`${ri},${ci}`) ? T.tint : (ri % 2 ? "F7F9FC" : T.white) } },
    })));
    const colW = t.col_w ? t.col_w.map((f) => f * tw) : undefined;
    const capTop = t.caption && t.caption_pos === "top";
    if (capTop) slide.addText(runs(t.caption, { fontSize: 12, color: T.sub }), tb({ x: M, y: Y - 0.08, w: W - 2 * M, h: 0.34 }));
    const ty = capTop ? Y + 0.3 : Y;
    // rows share the body height (capped) unless row_h is given, so short tables do not leave the bottom empty
    const availH = FOOT_Y - 0.15 - ty - (t.caption && !capTop ? 0.5 : 0);
    const rowH = t.row_h || Math.min(t.row_max || 0.72, availH / (rows.length + 1));
    slide.addTable([header, ...rows], { x: M, y: ty, w: tw, colW, fontFace: T.latin, border: { type: "solid", pt: 1, color: T.line }, valign: "middle", rowH, margin: 0.06 });
    if (t.caption && !capTop) slide.addText(runs(t.caption, { fontSize: 12, color: T.sub }), tb({ x: M, y: FOOT_Y - 0.5, w: tw, h: 0.35 }));
    sideText(slide, s, W - M - sideW, sideW);
  },
  chart(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const c = s.chart, sideW = s.side ? (s.side_w || 4.2) : 0, cw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    const data = c.series.map((se) => ({ name: se.name, labels: c.categories, values: se.values }));
    slide.addChart(c.kind || "bar", data, {
      x: M, y: Y, w: cw, h: HH, barDir: c.horizontal ? "bar" : "col",
      chartColors: c.colors || (c.series.length > 1 ? [T.primary, "9DC3E6", T.sub] : [T.primary]), invertIfNegative: false, showValue: c.show_values ?? (c.kind || "bar") === "bar",
      lineSize: 2.5, lineDataSymbol: "none", catAxisLabelFrequency: c.label_every || 1, catAxisLabelRotate: c.label_rotate ?? 0, dataLabelFontSize: 12, dataLabelColor: T.text,
      dataLabelFormatCode: c.format || "0.0", catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
      catAxisLabelColor: T.text, valAxisLabelColor: T.sub, valGridLine: { color: "E6E6E6", size: 1 },
      showLegend: c.series.length > 1, legendPos: "b", legendFontSize: 12,
      showValAxisTitle: !!c.y_title, valAxisTitle: c.y_title || "", valAxisTitleFontSize: 12,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
      ...(c.min !== undefined ? { valAxisMinVal: c.min } : {}), ...(c.max !== undefined ? { valAxisMaxVal: c.max } : {}),
    });
    sideText(slide, s, W - M - sideW, sideW);
  },
  figure(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const sideW = s.side ? (s.side_w || 4.6) : 0, fw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    const img = path.resolve(s._dir, s.image);
    const fh = HH - (s.caption ? 0.4 : 0);
    slide.addImage({ path: img, x: M, y: Y, w: fw, h: fh, sizing: { type: "contain", w: fw, h: fh } });
    if (s.caption) slide.addText(runs(s.caption, { fontSize: 12, color: T.sub }), tb({ x: M, y: Y + fh + 0.02, w: fw, h: 0.35 }));
    sideText(slide, s, W - M - sideW, sideW);
  },
  plan(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const lw = 5.4, gap = 0.35, rw = W - 2 * M - lw - gap;
    const mb = s.micro || [], msize = fitSize((z) => cardHeight([{ body: mb }], lw, z), HH - 0.2, 19, 15, mb, lw - 0.3);
    card(slide, M, Y, lw, HH, s.micro_head || "次回まで", mb, msize, spreadSpace(HH - 0.25 - cardHeight([{ body: mb }], lw, msize), mb.length));
    const x = M + lw + gap, rows = s.macro || [];
    slide.addShape("rect", { x, y: Y, w: rw, h: 0.48, fill: { color: T.primary }, line: { color: T.primary } });
    slide.addText(runs(s.macro_head || "修論まで", { color: T.white, bold: true, fontSize: 18 }), tb({ x: x + 0.12, y: Y, w: rw - 0.24, h: 0.48, valign: "middle" }));
    const rh = Math.min(1.2, (HH - 0.6) / Math.max(rows.length, 1));
    rows.forEach((r, i) => {
      const y = Y + 0.6 + i * rh, cur = !!r.current;
      slide.addShape("rect", { x, y, w: 1.5, h: rh - 0.08, fill: { color: cur ? T.primary : T.tint }, line: { color: cur ? T.primary : T.tint } });
      slide.addText(r.period, tb({ x, y, w: 1.5, h: rh - 0.08, align: "center", valign: "middle", fontSize: 15, bold: true, color: cur ? T.white : T.primaryDark }));
      slide.addText(runs(r.item, { fontSize: 16, color: T.text }), tb({ x: x + 1.65, y, w: rw - 1.65, h: rh - 0.08, valign: "middle" }));
    });
  },
};

// ---------------------------------------------------------------- build
async function main() {
  const args = process.argv.slice(2);
  if (!args.length) { console.error("usage: build_deck.js deck.yaml [-o out.pptx]"); process.exit(2); }
  const src = path.resolve(args[0]);
  const dir = path.dirname(src);
  const doc = yaml.load(fs.readFileSync(src, "utf8"));
  // output: -o, else meta.output (e.g. 261019_naito.pptx, yymmdd of the talk), else deck.pptx
  const out = args.includes("-o") ? path.resolve(args[args.indexOf("-o") + 1])
    : doc.meta?.output ? path.resolve(dir, doc.meta.output) : src.replace(/\.ya?ml$/, ".pptx");
  let refs = doc.references || {};
  if (doc.references_file) refs = { ...yaml.load(fs.readFileSync(path.resolve(dir, doc.references_file), "utf8")), ...refs };

  // citation numbers in order of first use across the deck
  const order = [];
  const use = (k) => {
    if (!(k in refs)) throw new Error(`unknown reference key: ${k}`);
    if (!order.includes(k)) order.push(k);
  };
  for (const s of doc.slides) {
    // keys cited inline anywhere in the slide's text, in reading order, then the slide-level cite list
    s._inline = [];
    for (const m of JSON.stringify(s).matchAll(/\[@([\w-]+(?:\s*,\s*@?[\w-]+)*)\]/g))
      for (const k of m[1].split(/\s*,\s*@?/)) { use(k); if (!s._inline.includes(k)) s._inline.push(k); }
    for (const k of s.cite || []) use(k);
  }
  const num = (k) => order.indexOf(k) + 1;
  CITE = (k) => String(num(k));
  // "[n] Qin+ 2024": first author's surname and the year, read from the IEEE string
  const short = (k) => {
    const r = refs[k], first = r.replace(/\s+et al\./, ",").split(/,| and /)[0].trim().split(/\s+/).pop(), year = (r.match(/(19|20)\d\d(?!.*(19|20)\d\d)/) || [""])[0];
    const many = /et al\.|,.*(,| and )/.test(r.split('"')[0]);
    return `[${num(k)}] ${first}${many ? "+" : ""} ${year}`;
  };

  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.theme = { headFontFace: T.latin, bodyFontFace: T.latin };
  pres.author = doc.meta?.name || "";
  pres.title = doc.meta?.title || "";
  defineLayouts(pres);

  // title slide
  const meta = doc.meta || {};
  const ts = pres.addSlide({ masterName: "TITLE" });
  ts.addText(meta.event || "定期ゼミ", { placeholder: "event" });
  ts.addText(meta.title || "", { placeholder: "title" });
  ts.addText([meta.date, ...(meta.affiliation || []), meta.name].filter(Boolean).map((l, i, a) => ({ text: l, options: { breakLine: i < a.length - 1, bold: i === a.length - 1 } })), { placeholder: "info" });

  // table of contents from the sections used
  const sections = [];
  for (const s of doc.slides) if (s.section && !sections.includes(s.section)) sections.push(s.section);
  if (order.length && !sections.includes(doc.references_section || "参考文献")) sections.push(doc.references_section || "参考文献");
  // table of contents; type "agenda" repeats it at a section start with that section highlighted
  const addToc = (current) => {
    const toc = pres.addSlide({ masterName: "TOC" });
    toc.addText("目次", { placeholder: "title" });
    // fit every section between the top margin and the footer band
    const n = Math.max(sections.length, 1), top = 0.5, avail = FOOT_Y - 0.35 - top;
    const gap = n > 7 ? 0.08 : 0.12, rowH = Math.min(0.72, (avail - gap * (n - 1)) / n);
    const fs = rowH < 0.55 ? 18 : 22;
    sections.forEach((sec, i) => {
      const y = top + i * (rowH + gap), on = current === undefined || sec === current;
      toc.addShape("rect", { x: 4.4, y, w: 7.6, h: rowH, fill: { color: sec === current ? T.primary : T.tint }, line: { color: sec === current ? T.primary : T.tint } });
      toc.addText(sec, tb({ x: 4.6, y, w: 7.2, h: rowH, valign: "middle", fontSize: fs, bold: true, color: sec === current ? T.white : (on ? T.primaryDark : "8EA9C8") }));
    });
    return toc;
  };
  addToc(undefined);

  for (const s of doc.slides) {
    s._dir = dir;
    const type = s.type || "bullets";
    if (type === "agenda") { const a = addToc(s.section); if (s.notes) a.addNotes(String(s.notes)); continue; }
    if (!BODY[type]) throw new Error(`unknown slide type: ${type}`);
    // full: a whole-slide body without the key-point box (e.g. one large summary table)
    s._top = s.full ? 0.8 : BODY_Y;
    const slide = pres.addSlide({ masterName: s.full ? "REFS" : "CONTENT" });
    const keys = [...s._inline, ...(s.cite || []).filter((k) => !s._inline.includes(k))].sort((a, b) => num(a) - num(b));
    addFrame(slide, s.full ? { ...s, point: undefined } : s, keys.map(short));
    BODY[type](slide, s);
  }

  // references (IEEE strings, numbered by first use): always one page; the font shrinks to fit
  if (order.length) {
    const refSec = doc.references_section || "参考文献";
    const n = order.length, fsz = n <= 10 ? 13 : n <= 13 ? 12 : n <= 16 ? 11 : n <= 20 ? 10 : 9;
    const slide = pres.addSlide({ masterName: "REFS" });
    addFrame(slide, { section: refSec, title: "参考文献" }, []);
    const items = order.map((k, i) => ({ text: `[${num(k)}] ${refs[k]}`, options: { fontSize: fsz, color: T.text, paraSpaceAfter: fsz <= 10 ? 2 : 4, breakLine: i < n - 1 } }));
    slide.addText(items, tb({ x: M, y: 0.75, w: W - 2 * M, h: FOOT_Y - 0.82, fit: "shrink" }));
  }

  await pres.writeFile({ fileName: out });
  await setEastAsianFont(out);
  console.log(out);
}

// Post-process the written package:
// - Japanese text: pptxgenjs writes the Latin face into every <a:ea>; point East Asian text at T.ea
//   (theme, masters, layouts, slides) so Japanese renders in Yu Gothic and Latin in Calibri.
// - pptxgenjs emits a stray mid-paragraph <a:pPr><a:buNone/></a:pPr> after a styled first run, which
//   drops the bullet; remove it (a <a:pPr> is only valid as the first child of <a:p>).
async function setEastAsianFont(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const parts = Object.keys(zip.files).filter((n) => /^ppt\/(theme\/theme|slideMasters\/slideMaster|slideLayouts\/slideLayout|slides\/slide)\d+\.xml$/.test(n));
  for (const name of parts) {
    let xml = await zip.file(name).async("string");
    xml = xml.replace(/<a:ea typeface="[^"]*"[^>]*\/>/g, `<a:ea typeface="${T.ea}"/>`);
    xml = xml.replace(/(<\/a:r>|<a:br\/>)<a:pPr[^>]*>(?:(?!<\/a:pPr>).)*?<\/a:pPr>/g, "$1");
    zip.file(name, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
