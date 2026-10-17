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
// ---------------------------------------------------------------- sizing and line-break check
// Type sizes are fixed by role. Space left under a block is filled with content (a `below` note, table,
// chart or figure), never by growing the text.
const SZ = { card: 17, side: 16, overview: 16, plan: 17, note: 16, caption: 12 };
// Yu Gothic full-width = 1 em; Latin letters/digits 0.58 em and spaces 0.3 em (Calibri ≈ 0.5 / 0.23, rounded up),
// plus 0.2 em at each Japanese/Latin boundary (PowerPoint's autospace). Deliberately conservative.
const isWide = (c) => c.charCodeAt(0) > 0x2e7f;
const lineW = (t) => {
  const cs = [...t];
  return cs.reduce((a, c, i) => a + (isWide(c) ? 1 : c === " " ? 0.3 : 0.58)
    + (i && c !== " " && cs[i - 1] !== " " && isWide(c) !== isWide(cs[i - 1]) ? 0.2 : 0), 0);
};
const plain = (t) => String(t).replace(/\*\*|==/g, "").replace(/\[@([\w-]+(?:\s*,\s*@?[\w-]+)*)\]/g, (_, ks) => `[${ks.split(/\s*,\s*@?/).map(() => "00").join(", ")}]`);
// characters that fit on one line of a text box `boxW` inches wide (bullet indent included)
const capacity = (boxW, size, bullet = 0) => ((boxW - 0.1) * 72 - (bullet === 0 ? 0 : bullet === 1 ? 18 : 50)) / size;
// every hand-broken line that would wrap again is reported, so line breaks stay where the author put them
const LINT = [];
let SLIDE_NO = 0;
function lintText(where, items, boxW, size, bullets = true) {
  const walk = (it, lvl) => {
    if (Array.isArray(it)) return it.forEach((x) => walk(x, lvl + 1));
    const z = lvl ? size - 2 : size, cap = capacity(boxW, z, bullets ? lvl + 1 : 0);
    for (const t of plain(it).split("\n")) if (lineW(t) > cap - (bullets ? 0.3 : 0.1)) LINT.push(`p${SLIDE_NO} ${where}: ${lineW(t).toFixed(1)} > ${cap.toFixed(1)}  「${t}」`);
  };
  (Array.isArray(items) ? items : [items]).forEach((it) => walk(it, 0));
}
function estLines(items, boxW, size) {
  let n = 0;
  const walk = (it, lvl) => {
    if (Array.isArray(it)) return it.forEach((x) => walk(x, lvl + 1));
    for (const t of plain(it).split("\n")) n += Math.max(1, Math.ceil(lineW(t) / capacity(boxW, size, lvl + 1)));
  };
  items.forEach((it) => walk(it, 0));
  return n;
}
function textHeight(items, boxW, size, space = 6) {
  return estLines(items, boxW, size) * size * 1.22 / 72 + items.length * space / 72 + 0.15;
}
function cardHeight(cards, w, size) {
  return Math.max(...cards.map((c) => 0.62 + textHeight(c.body || [], w - 0.24, size)));
}
function card(slide, x, y, w, h, head, body, size = SZ.card) {
  slide.addShape("rect", { x, y, w, h, fill: { color: T.tint }, line: { color: T.tint }, objectName: `card ${head}` });
  slide.addShape("rect", { x, y, w, h: 0.48, fill: { color: T.primary }, line: { color: T.primary } });
  slide.addText(runs(head, { color: T.white, bold: true, fontSize: 18 }), tb({ x: x + 0.12, y, w: w - 0.24, h: 0.48, valign: "middle" }));
  lintText(`card「${head}」`, body, w - 0.24, size);
  slide.addText(bulletRuns(body, size), tb({ x: x + 0.12, y: y + 0.56, w: w - 0.24, h: h - 0.6 }));
}
function sideText(slide, s, x, w) {
  const Y = s._top, HH = FOOT_Y - 0.12 - Y;
  if (!s.side) return;
  const size = s.side_size || SZ.side;
  lintText("side", s.side, w, size);
  slide.addText(bulletRuns(s.side, size, { space: 8 }), tb({ x, y: Y, w, h: HH }));
}
function drawTable(slide, t, x, y, w) {
  const hl = new Set((t.highlight || []).map(([r, c]) => `${r},${c}`));
  // a row is never shorter than its text needs (CJK line ≈ 1.45 em plus the cell margins)
  const size = t.size || 17, margin = t.margin ?? 0.06, rowH = Math.max(t.row_h || 0.5, size * 1.45 / 72 + 2 * margin);
  const colW = (t.col_w || t.header.map(() => 1 / t.header.length)).map((f) => f * w);
  const header = t.header.map((h) => ({ text: h, options: { bold: true, color: T.white, fill: { color: T.primary }, align: "center", fontSize: size } }));
  const rows = t.rows.map((r, ri) => r.map((v, ci) => ({
    text: runs(String(v) === "" ? " " : String(v), { fontSize: size, color: T.text }),
    options: { align: ci === 0 ? "left" : "center", fill: { color: hl.has(`${ri},${ci}`) ? T.tint : (ri % 2 ? "F7F9FC" : T.white) } },
  })));
  // header cells are bold (≈ 8% wider)
  t.header.forEach((v, ci) => lintText("table header", String(v), (colW[ci] - 0.02) / 1.08, size, false));
  t.rows.forEach((r) => r.forEach((v, ci) => lintText("table", String(v), colW[ci] - 0.02, size, false)));
  slide.addTable([header, ...rows], { x, y, w, colW, fontFace: T.latin, border: { type: "solid", pt: 1, color: T.line }, valign: "middle", rowH, margin });
  let bottom = y + rowH * (rows.length + 1);
  if (bottom > FOOT_Y - 0.1) LINT.push(`p${SLIDE_NO} table runs into the footer by ${(bottom - FOOT_Y + 0.1).toFixed(2)} in`);
  if (t.caption && t.caption_pos !== "top") {
    lintText("caption", t.caption, w, SZ.caption, false);
    slide.addText(runs(t.caption, { fontSize: SZ.caption, color: T.sub }), tb({ x, y: bottom + 0.05, w, h: 0.32 }));
    bottom += 0.4;
  }
  return bottom;
}
function drawChart(slide, c, x, y, w, h) {
  const data = c.series.map((se) => ({ name: se.name, labels: c.categories, values: se.values }));
  slide.addChart(c.kind || "bar", data, {
    x, y, w, h, barDir: c.horizontal ? "bar" : "col",
    chartColors: c.colors || (c.series.length > 1 ? [T.primary, "9DC3E6", T.sub] : [T.primary]), invertIfNegative: false, showValue: c.show_values ?? (c.kind || "bar") === "bar",
    lineSize: 2.5, lineDataSymbol: "none", catAxisLabelFrequency: c.label_every || 1, catAxisLabelRotate: c.label_rotate ?? 0, dataLabelFontSize: 12, dataLabelColor: T.text,
    dataLabelFormatCode: c.format || "0.0", catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    catAxisLabelColor: T.text, valAxisLabelColor: T.sub, valGridLine: { color: "E6E6E6", size: 1 },
    showLegend: c.series.length > 1, legendPos: "b", legendFontSize: 12,
    showValAxisTitle: !!c.y_title, valAxisTitle: c.y_title || "", valAxisTitleFontSize: 12,
    showCatAxisTitle: !!c.x_title, catAxisTitle: c.x_title || "", catAxisTitleFontSize: 12,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    ...(c.min !== undefined ? { valAxisMinVal: c.min } : {}), ...(c.max !== undefined ? { valAxisMaxVal: c.max } : {}),
  });
}
// content placed under the main block: a supplement note, a small table, a chart or a figure.
// It starts right under the block and never stretches the block itself.
function below(slide, s, x, y, w) {
  const b = s.below;
  if (!b) return;
  const bottom = FOOT_Y - 0.15, avail = bottom - y;
  if (avail < 0.5) { LINT.push(`p${SLIDE_NO} below: no room (${avail.toFixed(2)} in)`); return; }
  if (b.title) {
    lintText("below title", b.title, w, 15, false);
    slide.addText(runs(b.title, { fontSize: 15, bold: true, color: T.primaryDark }), tb({ x, y, w, h: 0.34 }));
    y += 0.38;
  }
  if (b.note) {
    const items = Array.isArray(b.note) ? b.note : [b.note], size = b.size || SZ.note, lead = b.head ? 1.9 : 0;
    const tw = w - 0.35 - lead, h = Math.min(bottom - y, textHeight(items, tw, size) + 0.12);
    slide.addShape("rect", { x, y, w, h, fill: { color: T.message }, line: { color: T.message } });
    slide.addShape("rect", { x, y, w: 0.08, h, fill: { color: T.primary }, line: { color: T.primary } });
    if (b.head) slide.addText(runs(b.head, { fontSize: size, bold: true, color: T.primaryDark }), tb({ x: x + 0.2, y, w: lead, h, valign: "middle" }));
    lintText("note", items, tw, size, items.length > 1);
    slide.addText(items.length > 1 ? bulletRuns(items, size, { space: 4 }) : runs(items[0], { fontSize: size, color: T.text }), tb({ x: x + 0.2 + lead, y, w: tw, h, valign: "middle" }));
  } else if (b.table) {
    const need = (b.table.row_h || 0.5) * (b.table.rows.length + 1) + (b.table.caption ? 0.4 : 0);
    if (y + need > bottom + 0.02) LINT.push(`p${SLIDE_NO} below table overflows by ${(y + need - bottom).toFixed(2)} in`);
    drawTable(slide, b.table, x, y, w);
  } else if (b.chart) {
    drawChart(slide, b.chart, x, y, w, bottom - y);
  } else if (b.image) {
    const h = bottom - y - (b.caption ? 0.36 : 0);
    slide.addImage({ path: path.resolve(s._dir, b.image), x, y, w, h, sizing: { type: "contain", w, h } });
    if (b.caption) slide.addText(runs(b.caption, { fontSize: SZ.caption, color: T.sub }), tb({ x, y: y + h + 0.02, w, h: 0.32 }));
  }
}
const BODY = {
  bullets(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    lintText("body", s.body || [], W - 2 * M, s.size || 20);
    slide.addText(bulletRuns(s.body || [], s.size || 20, { space: 10 }), tb({ x: M, y: Y, w: W - 2 * M, h: HH }));
  },
  cards(slide, s) {
    const Y = s._top;
    const n = s.cards.length, cols = s.cols || n, rows = Math.ceil(n / cols), gap = 0.3;
    const top = s.lead ? Y + 0.55 : Y;
    if (s.lead) slide.addText(runs(s.lead, { fontSize: 18, color: T.sub }), tb({ x: M, y: Y, w: W - 2 * M, h: 0.45 }));
    const size = s.size || SZ.card;
    const cw = (W - 2 * M - gap * (cols - 1)) / cols, maxH = (FOOT_Y - 0.15 - top - gap * (rows - 1)) / rows;
    // a card is as tall as its text (all cards in a row share the height); `fill: true` runs it to the bottom
    const ch = s.fill ? maxH : Math.min(maxH, cardHeight(s.cards, cw, size));
    s.cards.forEach((c, i) => card(slide, M + (i % cols) * (cw + gap), top + Math.floor(i / cols) * (ch + gap), cw, ch, c.head, c.body || [], size));
    below(slide, s, M, top + rows * ch + (rows - 1) * gap + 0.25, W - 2 * M);
  },
  overview(slide, s) {
    const Y = s._top;
    // three blocks on top (goal / questions / approach), a timeline strip under them
    const gap = 0.3, cw = (W - 2 * M - gap * 2) / 3, size = SZ.overview, ch = Math.min(3.3, cardHeight(s.blocks, cw, size));
    s.blocks.forEach((b, i) => card(slide, M + i * (cw + gap), Y, cw, ch, b.head, b.body || [], size));
    const ty = Y + ch + 0.25, n = s.timeline.length, tw = (W - 2 * M - 0.15 * (n - 1)) / n;
    s.timeline.forEach((t, i) => {
      const x = M + i * (tw + 0.15), cur = !!t.current;
      slide.addShape("homePlate", { x, y: ty, w: tw, h: 0.75, fill: { color: cur ? T.primary : T.message }, line: { color: cur ? T.primary : T.line, width: 1 } });
      slide.addText([
        { text: t.label, options: { bold: true, fontSize: 15, color: cur ? T.white : T.text, breakLine: true } },
        { text: t.note || "", options: { fontSize: 12, color: cur ? T.white : T.sub } },
      ], tb({ x: x + 0.1, y: ty + 0.03, w: tw - 0.45, h: 0.69, valign: "middle" }));
    });
    below(slide, s, M, ty + 0.75 + 0.2, W - 2 * M);
  },
  table(slide, s) {
    const Y = s._top;
    const t = s.table, sideW = s.side ? (s.side_w || 4.2) : 0;
    const tw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    const capTop = t.caption && t.caption_pos === "top";
    if (capTop) {
      lintText("caption", t.caption, W - 2 * M, SZ.caption, false);
      slide.addText(runs(t.caption, { fontSize: SZ.caption, color: T.sub }), tb({ x: M, y: Y - 0.08, w: W - 2 * M, h: 0.34 }));
    }
    const end = drawTable(slide, t, M, capTop ? Y + 0.3 : Y, tw);
    below(slide, s, M, end + 0.2, tw);
    sideText(slide, s, W - M - sideW, sideW);
  },
  chart(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const sideW = s.side ? (s.side_w || 4.2) : 0, cw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    drawChart(slide, s.chart, M, Y, cw, HH);
    sideText(slide, s, W - M - sideW, sideW);
  },
  figure(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const sideW = s.side ? (s.side_w || 4.6) : 0, fw = W - 2 * M - (sideW ? sideW + 0.3 : 0);
    const img = path.resolve(s._dir, s.image);
    const fh = HH - (s.caption ? 0.4 : 0);
    slide.addImage({ path: img, x: M, y: Y, w: fw, h: fh, sizing: { type: "contain", w: fw, h: fh } });
    if (s.caption) {
      lintText("caption", s.caption, fw, SZ.caption, false);
      slide.addText(runs(s.caption, { fontSize: SZ.caption, color: T.sub }), tb({ x: M, y: Y + fh + 0.02, w: fw, h: 0.35 }));
    }
    sideText(slide, s, W - M - sideW, sideW);
  },
  plan(slide, s) {
    const Y = s._top, HH = FOOT_Y - 0.12 - Y;
    const lw = 5.4, gap = 0.35, rw = W - 2 * M - lw - gap;
    const mb = s.micro || [];
    card(slide, M, Y, lw, Math.min(HH, cardHeight([{ body: mb }], lw, SZ.plan) + 0.1), s.micro_head || "次回まで", mb, SZ.plan);
    const x = M + lw + gap, rows = s.macro || [];
    slide.addShape("rect", { x, y: Y, w: rw, h: 0.48, fill: { color: T.primary }, line: { color: T.primary } });
    slide.addText(runs(s.macro_head || "修論まで", { color: T.white, bold: true, fontSize: 18 }), tb({ x: x + 0.12, y: Y, w: rw - 0.24, h: 0.48, valign: "middle" }));
    const rh = Math.min(0.8, (HH - 0.6) / Math.max(rows.length, 1));
    rows.forEach((r, i) => {
      const y = Y + 0.6 + i * rh, cur = !!r.current;
      slide.addShape("rect", { x, y, w: 1.5, h: rh - 0.08, fill: { color: cur ? T.primary : T.tint }, line: { color: cur ? T.primary : T.tint } });
      slide.addText(r.period, tb({ x, y, w: 1.5, h: rh - 0.08, align: "center", valign: "middle", fontSize: 15, bold: true, color: cur ? T.white : T.primaryDark }));
      lintText("macro", r.item, rw - 1.65, 16, false);
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
    SLIDE_NO = pres.slides.length;
    lintText("title", s.title || "", W - 2.75 - M, 24, false);
    if (!s.full && s.point) lintText("point", s.point, W - 2 * M - 0.3, 20, false);
    addFrame(slide, s.full ? { ...s, point: undefined } : s, keys.map(short));
    BODY[type](slide, s);
  }

  // references (IEEE strings, numbered by first use): always one page; the font shrinks to fit
  if (order.length) {
    const refSec = doc.references_section || "参考文献";
    // the largest size (13..9pt) at which every entry fits above the footer
    const n = order.length, avail = FOOT_Y - 0.9;
    const hAt = (z) => order.reduce((a, k) => a + Math.ceil((`[${num(k)}] ${refs[k]}`.length * 0.47) / capacity(W - 2 * M, z))  /* English prose in Calibri ≈ 0.47 em */, 0) * z * 1.2 / 72 + n * (z <= 10 ? 2 : 4) / 72;
    let fsz = 13;
    while (fsz > 9 && hAt(fsz) > avail) fsz--;
    const slide = pres.addSlide({ masterName: "REFS" });
    addFrame(slide, { section: refSec, title: "参考文献" }, []);
    const items = order.map((k, i) => ({ text: `[${num(k)}] ${refs[k]}`, options: { fontSize: fsz, color: T.text, paraSpaceAfter: fsz <= 10 ? 2 : 4, breakLine: i < n - 1 } }));
    slide.addText(items, tb({ x: M, y: 0.75, w: W - 2 * M, h: FOOT_Y - 0.82, fit: "shrink" }));
  }

  // line-break check: every hand-broken line that would wrap again (fix the text, not the size)
  if (LINT.length) console.error(`line-break check: ${LINT.length} line(s) would wrap\n  ` + LINT.join("\n  "));
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
