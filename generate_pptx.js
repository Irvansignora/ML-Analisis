/**
 * SalesML Analytics - PowerPoint Generator
 * Reads JSON payload from stdin, writes .pptx to output_path
 * Design: Ocean Dark theme — matches the dashboard
 */
"use strict";
const pptxgen = require("pptxgenjs");

// ── READ STDIN ────────────────────────────────────────────────────────────────
let raw = "";
process.stdin.on("data", d => (raw += d));
process.stdin.on("end", () => {
  try {
    const data = JSON.parse(raw);
    buildPresentation(data);
  } catch (e) {
    process.stderr.write("JSON parse error: " + e.message + "\n");
    process.exit(1);
  }
});

// ── PALETTE ───────────────────────────────────────────────────────────────────
const P = {
  bg:      "020B18",
  bg2:     "03142E",
  card:    "071828",
  border:  "0D2D45",
  accent:  "06B6D4",
  sky:     "0EA5E9",
  blue:    "38BDF8",
  green:   "10B981",
  amber:   "F59E0B",
  red:     "EF4444",
  purple:  "8B5CF6",
  pink:    "EC4899",
  white:   "FFFFFF",
  text:    "E0F2FE",
  muted:   "64748B",
  chart_blues:  ["065A82","0D7FAD","0EA5E9","38BDF8","7DD3FC","BAE6FD"],
  chart_greens: ["064E3B","065F46","047857","059669","10B981","34D399"],
  chart_reds:   ["7F1D1D","991B1B","B91C1C","DC2626","EF4444","F87171"],
  chart_multi:  ["06B6D4","10B981","F59E0B","8B5CF6","EC4899","EF4444","38BDF8","34D399"],
};

// ── FORMATTERS ────────────────────────────────────────────────────────────────
function fmtCurrency(v) {
  if (!v || isNaN(v)) return "Rp 0";
  if (v >= 1e12) return "Rp " + (v/1e12).toFixed(1) + "T";
  if (v >= 1e9)  return "Rp " + (v/1e9).toFixed(2)  + "M";
  if (v >= 1e6)  return "Rp " + (v/1e6).toFixed(1)  + "Jt";
  if (v >= 1e3)  return "Rp " + (v/1e3).toFixed(0)  + "K";
  return "Rp " + Math.round(v);
}
function fmtNum(v) {
  if (!v || isNaN(v)) return "0";
  if (v >= 1e9) return (v/1e9).toFixed(1) + "M";
  if (v >= 1e6) return (v/1e6).toFixed(1) + "Jt";
  if (v >= 1e3) return (v/1e3).toFixed(0) + "K";
  return String(Math.round(v));
}
function safe(v, fallback) {
  if (fallback === undefined) fallback = 0;
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return fallback;
  return v;
}
function trunc(str, n) {
  if (n === undefined) n = 18;
  if (!str) return "";
  str = String(str);
  return str.length > n ? str.substring(0, n-1) + "..." : str;
}

// ── LAYOUT HELPERS ────────────────────────────────────────────────────────────
const FOOTER_Y = 5.45;

function addBase(slide, pres) {
  slide.background = { color: P.bg };
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:0, w:10, h:0.055,
    fill:{ color: P.accent }, line:{ color: P.accent }
  });
}

function addFooter(slide, pres, pageNum, total) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:FOOTER_Y, w:10, h:0.175,
    fill:{ color: P.bg2 }, line:{ color: P.border }
  });
  slide.addText("SalesML Analytics Pro  |  Confidential", {
    x:0.35, y:FOOTER_Y+0.03, w:6, h:0.14,
    fontSize:8, color:P.muted, fontFace:"Calibri"
  });
  const ts = new Date().toLocaleDateString("id-ID", {year:"numeric",month:"long",day:"numeric"});
  slide.addText(ts, {
    x:6.3, y:FOOTER_Y+0.03, w:2.4, h:0.14,
    fontSize:8, color:P.muted, fontFace:"Calibri", align:"right"
  });
  if (pageNum && total) {
    slide.addText(pageNum + " / " + total, {
      x:8.8, y:FOOTER_Y+0.03, w:0.9, h:0.14,
      fontSize:8, color:P.accent, fontFace:"Calibri", align:"right"
    });
  }
}

function secLabel(slide, pres, text, x, y, w) {
  if (!w) w = 4.5;
  slide.addShape(pres.shapes.RECTANGLE, {
    x:x, y:y, w:0.055, h:0.28,
    fill:{ color: P.accent }, line:{ color: P.accent }
  });
  slide.addText(text, {
    x:x+0.1, y:y, w:w-0.1, h:0.28,
    fontSize:10.5, bold:true, color:P.blue,
    fontFace:"Calibri", margin:0
  });
}

function addCard(slide, pres, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x:x, y:y, w:w, h:h,
    fill:{ color: P.card },
    line:{ color: P.border, width:0.75 },
    shadow:{ type:"outer", color:"000000", blur:10, offset:3, angle:135, opacity:0.25 }
  });
}

function kpiBlock(slide, pres, x, y, w, h, opts) {
  addCard(slide, pres, x, y, w, h);
  slide.addShape(pres.shapes.RECTANGLE, {
    x:x, y:y, w:w, h:0.04,
    fill:{ color: opts.color }, line:{ color: opts.color }
  });
  slide.addText(opts.icon, { x:x+0.15, y:y+0.1, w:0.55, h:0.42, fontSize:20, margin:0 });
  slide.addText(opts.value, {
    x:x+0.15, y:y+0.5, w:w-0.25, h:0.4,
    fontSize:17, bold:true, color:P.white, fontFace:"Calibri"
  });
  slide.addText(opts.label.toUpperCase(), {
    x:x+0.15, y:y+0.88, w:w-0.25, h:0.22,
    fontSize:7.5, color:P.blue, fontFace:"Calibri", charSpacing:0.8
  });
  if (opts.delta) {
    slide.addText(opts.delta, {
      x:x+0.15, y:y+1.08, w:w-0.25, h:0.2,
      fontSize:9, color:opts.deltaUp ? P.green : P.red, bold:true, fontFace:"Calibri"
    });
  }
}

function insightRow(slide, pres, text, x, y, w) {
  addCard(slide, pres, x, y, w, 0.44);
  slide.addShape(pres.shapes.RECTANGLE, {
    x:x, y:y, w:0.055, h:0.44,
    fill:{ color: P.amber }, line:{ color: P.amber }
  });
  slide.addText(text.replace(/^[^\w\s]+\s*/, ""), {
    x:x+0.12, y:y+0.04, w:w-0.2, h:0.36,
    fontSize:9, color:P.text, fontFace:"Calibri", wrap:true
  });
}

function chartOpts(extra) {
  var base = {
    chartArea: { fill:{ color: P.card } },
    catAxisLabelColor: P.muted,
    valAxisLabelColor: P.muted,
    valGridLine: { color: P.border, size:0.5 },
    catGridLine: { style:"none" },
    showLegend: false
  };
  if (extra) Object.assign(base, extra);
  return base;
}

// ── SLIDE 1 — TITLE ───────────────────────────────────────────────────────────
function slideTitle(pres, data, total) {
  var slide = pres.addSlide();
  slide.background = { color: P.bg };

  slide.addShape(pres.shapes.RECTANGLE, { x:6.8, y:0, w:3.2, h:5.625, fill:{ color:P.sky, transparency:88 }, line:{ color:P.bg } });
  slide.addShape(pres.shapes.RECTANGLE, { x:8.0, y:0, w:2.0, h:5.625, fill:{ color:P.accent, transparency:92 }, line:{ color:P.bg } });
  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.07, fill:{ color:P.accent }, line:{ color:P.accent } });

  slide.addText("SalesML Analytics Pro", { x:0.5, y:0.5, w:6, h:0.28, fontSize:9.5, bold:true, color:P.accent, charSpacing:2.5, fontFace:"Calibri" });
  slide.addText("Sales Performance\nReport", { x:0.5, y:0.9, w:6.1, h:1.65, fontSize:40, bold:true, color:P.white, fontFace:"Calibri", lineSpacingMultiple:1.1 });

  var dr = data.date_range || "Periode Analisis Data";
  slide.addText(dr, { x:0.5, y:2.65, w:6, h:0.3, fontSize:13, color:P.blue, fontFace:"Calibri", italic:true });

  slide.addShape(pres.shapes.RECTANGLE, { x:0.5, y:3.07, w:2.2, h:0.035, fill:{ color:P.accent }, line:{ color:P.accent } });

  var k = data.kpis || {};
  var pills = [
    { label:"Revenue",    val: fmtCurrency(k.total_revenue) },
    { label:"Transaksi",  val: fmtNum(k.total_transactions) },
    { label:"Produk",     val: fmtNum(k.unique_products || 0) },
  ];
  pills.forEach(function(p, i) {
    var px = 0.5 + i * 2.1;
    addCard(slide, pres, px, 3.28, 1.9, 0.72);
    slide.addText(p.val,   { x:px+0.1, y:3.34, w:1.7, h:0.3,  fontSize:14, bold:true, color:P.blue, fontFace:"Calibri" });
    slide.addText(p.label.toUpperCase(), { x:px+0.1, y:3.64, w:1.7, h:0.22, fontSize:7.5, color:P.muted, charSpacing:1 });
  });

  var ts = new Date().toLocaleDateString("id-ID", {year:"numeric",month:"long",day:"numeric"});
  slide.addText("Generated: " + ts, { x:0.5, y:5.15, w:6, h:0.22, fontSize:8.5, color:P.muted, fontFace:"Calibri" });
  addFooter(slide, pres, 1, total);
}

// ── SLIDE 2 — AGENDA ──────────────────────────────────────────────────────────
function slideAgenda(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);

  slide.addText("Agenda", { x:0.5, y:0.22, w:9, h:0.55, fontSize:28, bold:true, color:P.white, fontFace:"Calibri" });

  var items = [
    { n:"01", title:"Executive Summary",     sub:"KPI utama, growth & key insights" },
    { n:"02", title:"Tren Revenue",           sub:"Tren bulanan & MoM growth" },
    { n:"03", title:"Sales Performance",      sub:"Top & bottom produk by revenue" },
    { n:"04", title:"Profitability Analysis", sub:"Margin & profit per produk" },
    { n:"05", title:"Customer & RFM",         sub:"Segmentasi & top customer" },
    { n:"06", title:"Regional & Category",    sub:"Wilayah, kategori & Pareto 80/20" },
  ];

  items.forEach(function(it, i) {
    var col = i % 2, row = Math.floor(i / 2);
    var x = 0.38 + col * 4.82, y = 1.05 + row * 1.27;
    addCard(slide, pres, x, y, 4.55, 1.1);
    slide.addShape(pres.shapes.RECTANGLE, { x:x, y:y, w:0.055, h:1.1, fill:{ color:P.accent }, line:{ color:P.accent } });
    slide.addText(it.n,    { x:x+0.13, y:y+0.1,  w:0.55, h:0.42, fontSize:22, bold:true, color:P.accent, fontFace:"Calibri" });
    slide.addText(it.title,{ x:x+0.72, y:y+0.1,  w:3.7,  h:0.38, fontSize:13, bold:true, color:P.white,  fontFace:"Calibri" });
    slide.addText(it.sub,  { x:x+0.72, y:y+0.52, w:3.7,  h:0.38, fontSize:10, color:P.muted, fontFace:"Calibri" });
  });

  addFooter(slide, pres, 2, total);
}

// ── SLIDE 3 — EXECUTIVE KPI ───────────────────────────────────────────────────
function slideKPI(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);

  slide.addText("Executive Summary", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });
  slide.addText("Ringkasan performa bisnis keseluruhan", { x:0.5, y:0.65, w:9, h:0.25, fontSize:11, color:P.muted, fontFace:"Calibri" });

  var k = data.kpis || {};
  var g = data.growth || {};
  var margin = safe(k.avg_margin, 30);
  var profit = safe(k.total_revenue, 0) * margin / 100;
  var mom = g.mom, yoy = g.yoy;
  var aov = safe(k.avg_order_value, safe(k.total_revenue, 0) / Math.max(safe(k.total_transactions, 1), 1));

  var ROW_Y = [1.0, 2.52];
  var cards = [
    { icon:"💰", value:fmtCurrency(k.total_revenue), label:"Total Revenue",    color:P.accent,
      delta: mom != null ? (mom >= 0 ? "+" : "") + mom.toFixed(1) + "% vs bln lalu" : null, deltaUp: mom >= 0 },
    { icon:"📦", value:fmtNum(k.total_quantity || 0), label:"Total Qty Terjual", color:P.green },
    { icon:"🧾", value:fmtNum(k.total_transactions),  label:"Total Transaksi",   color:P.purple },
    { icon:"🛒", value:fmtCurrency(aov),               label:"Avg Order Value",   color:P.amber },
    { icon:"💎", value:fmtCurrency(profit),            label:"Gross Profit",      color:P.green },
    { icon:"📈", value:margin.toFixed(1) + "%",         label:"Profit Margin",     color:P.sky },
    { icon:"📅", value: mom != null ? (mom>=0?"+":"") + mom.toFixed(1) + "%" : "N/A", label:"Growth MoM", color: mom!=null && mom>=0 ? P.green : P.red },
    { icon:"🗓", value: yoy != null ? (yoy>=0?"+":"") + yoy.toFixed(1) + "%" : "N/A", label:"Growth YoY", color: yoy!=null && yoy>=0 ? P.green : P.red },
  ];

  cards.forEach(function(c, i) {
    var row = Math.floor(i / 4), col = i % 4;
    kpiBlock(slide, pres, 0.3 + col * 2.38, ROW_Y[row], 2.22, 1.38, c);
  });

  var insights = (data.insights || []).slice(0, 2);
  if (insights.length > 0) {
    secLabel(slide, pres, "Key Insights", 0.3, 4.03, 9.4);
    insights.forEach(function(ins, i) {
      insightRow(slide, pres, ins, 0.3 + i * 4.87, 4.37, 4.65);
    });
  }

  addFooter(slide, pres, 3, total);
}

// ── SLIDE 4 — TREN REVENUE ────────────────────────────────────────────────────
function slideTrend(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);
  slide.addText("Tren Revenue", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });

  var trend = data.monthly_trend || [];
  if (trend.length === 0) {
    slide.addText("Data tren tidak tersedia", { x:3, y:2.5, w:4, h:0.5, fontSize:14, color:P.muted, align:"center" });
    addFooter(slide, pres, 4, total); return;
  }

  var labels   = trend.map(function(m){ return m.month || ""; });
  var revenues = trend.map(function(m){ return Math.round(safe(m.revenue)); });

  secLabel(slide, pres, "Revenue Bulanan", 0.3, 0.82, 9.4);
  slide.addChart(pres.charts.BAR, [{ name:"Revenue", labels:labels, values:revenues }],
    Object.assign({ x:0.3, y:1.1, w:9.4, h:2.5, barDir:"col", chartColors:P.chart_blues,
      showValue:false, dataLabelColor:P.white, dataLabelFontSize:7 }, chartOpts())
  );

  var momVals   = trend.slice(1).map(function(m){ return parseFloat(safe(m.revenue_mom, 0).toFixed(1)); });
  var momLabels = labels.slice(1);
  if (momVals.length > 1) {
    secLabel(slide, pres, "MoM Growth (%)", 0.3, 3.72, 9.4);
    slide.addChart(pres.charts.LINE, [{ name:"MoM Growth %", labels:momLabels, values:momVals }],
      Object.assign({ x:0.3, y:3.98, w:9.4, h:1.15, chartColors:[P.amber],
        lineSize:2, lineSmooth:true,
        showValue:true, dataLabelColor:P.white, dataLabelFontSize:7.5 }, chartOpts())
    );
  }

  addFooter(slide, pres, 4, total);
}

// ── SLIDE 5 — TOP & BOTTOM PRODUCTS ──────────────────────────────────────────
function slideSales(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);
  slide.addText("Sales Performance", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });
  slide.addText("Top & bottom produk berdasarkan revenue", { x:0.5, y:0.65, w:9, h:0.25, fontSize:11, color:P.muted });

  var topP = (data.top_products || []).slice(0, 8);
  var botP = (data.bottom_products || []).slice(0, 8);

  if (topP.length > 0) {
    secLabel(slide, pres, "Top 8 Produk by Revenue", 0.3, 1.0, 4.6);
    slide.addChart(pres.charts.BAR,
      [{ name:"Revenue", labels:topP.map(function(p){ return trunc(p.product||p.name||"",20); }), values:topP.map(function(p){ return Math.round(safe(p.revenue)); }) }],
      Object.assign({ x:0.3, y:1.28, w:4.6, h:3.75, barDir:"bar", chartColors:P.chart_blues, showValue:true, dataLabelColor:P.white, dataLabelFontSize:7.5 }, chartOpts())
    );
  }

  if (botP.length > 0) {
    secLabel(slide, pres, "Bottom 8 — Perlu Perhatian", 5.1, 1.0, 4.6);
    slide.addChart(pres.charts.BAR,
      [{ name:"Revenue", labels:botP.map(function(p){ return trunc(p.product||p.name||"",20); }), values:botP.map(function(p){ return Math.round(safe(p.revenue)); }) }],
      Object.assign({ x:5.1, y:1.28, w:4.6, h:3.75, barDir:"bar", chartColors:P.chart_reds, showValue:true, dataLabelColor:P.white, dataLabelFontSize:7.5 }, chartOpts())
    );
  }

  addFooter(slide, pres, 5, total);
}

// ── SLIDE 6 — PROFITABILITY ───────────────────────────────────────────────────
function slideProfit(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);
  slide.addText("Profitability Analysis", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });

  var prods = data.profit_by_product || [];
  if (prods.length === 0) {
    slide.addText("Data profit tidak tersedia", { x:3, y:2.5, w:4, h:0.5, fontSize:14, color:P.muted, align:"center" });
    addFooter(slide, pres, 6, total); return;
  }

  var sorted = prods.slice().sort(function(a,b){ return safe(b.margin_pct)-safe(a.margin_pct); });
  var top8   = sorted.slice(0, 8);
  var bot8   = prods.slice().sort(function(a,b){ return safe(a.margin_pct)-safe(b.margin_pct); }).slice(0, 8);

  var avgM       = prods.reduce(function(s,p){ return s+safe(p.margin_pct); }, 0) / prods.length;
  var totalProfit= prods.reduce(function(s,p){ return s+safe(p.profit); }, 0);
  var totalRev   = prods.reduce(function(s,p){ return s+safe(p.revenue); }, 0);

  kpiBlock(slide, pres, 0.3,    0.82, 3.0, 1.05, { icon:"💎", value:fmtCurrency(totalProfit), label:"Total Profit",  color:P.green });
  kpiBlock(slide, pres, 0.3+3.18, 0.82, 3.0, 1.05, { icon:"📈", value:avgM.toFixed(1)+"%",    label:"Avg Margin",   color:P.sky   });
  kpiBlock(slide, pres, 0.3+6.36, 0.82, 3.0, 1.05, { icon:"💰", value:fmtCurrency(totalRev),  label:"Total Revenue", color:P.accent});

  secLabel(slide, pres, "Top 8 Margin %", 0.3, 1.98, 4.6);
  slide.addChart(pres.charts.BAR,
    [{ name:"Margin %", labels:top8.map(function(p){ return trunc(p.product||"",20); }), values:top8.map(function(p){ return parseFloat(safe(p.margin_pct).toFixed(1)); }) }],
    Object.assign({ x:0.3, y:2.26, w:4.6, h:2.85, barDir:"bar", chartColors:P.chart_greens, showValue:true, dataLabelColor:P.white, dataLabelFontSize:7.5 }, chartOpts())
  );

  secLabel(slide, pres, "Margin Terendah (Waspada!)", 5.1, 1.98, 4.6);
  slide.addChart(pres.charts.BAR,
    [{ name:"Margin %", labels:bot8.map(function(p){ return trunc(p.product||"",20); }), values:bot8.map(function(p){ return parseFloat(safe(p.margin_pct).toFixed(1)); }) }],
    Object.assign({ x:5.1, y:2.26, w:4.6, h:2.85, barDir:"bar", chartColors:P.chart_reds, showValue:true, dataLabelColor:P.white, dataLabelFontSize:7.5 }, chartOpts())
  );

  addFooter(slide, pres, 6, total);
}

// ── SLIDE 7 — CUSTOMER & RFM ──────────────────────────────────────────────────
function slideCustomer(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);
  slide.addText("Customer & RFM Analysis", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });

  var segs = data.rfm_segments || {};
  var topC = (data.top_customers || []).slice(0, 8);
  var segKeys = Object.keys(segs);

  if (segKeys.length > 0) {
    secLabel(slide, pres, "Distribusi Segmen RFM", 0.3, 0.82, 4.6);
    slide.addChart(pres.charts.DOUGHNUT,
      [{ name:"Customer", labels:segKeys, values:segKeys.map(function(k){ return segs[k]; }) }],
      { x:0.3, y:1.1, w:4.6, h:3.95, holeSize:48,
        chartColors:[P.accent,P.green,P.amber,P.purple,P.red],
        chartArea:{ fill:{ color:P.card } },
        showPercent:true, dataLabelColor:P.white, dataLabelFontSize:9,
        showLegend:true, legendPos:"b", legendFontColor:P.text, legendFontSize:9 }
    );
  }

  if (topC.length > 0) {
    secLabel(slide, pres, "Top Customer by Revenue", 5.1, 0.82, 4.6);
    var hdr = [
      { text:"#",        options:{ bold:true, color:P.white, fill:{ color:"0A1E35" }, fontSize:9, fontFace:"Calibri" } },
      { text:"Customer", options:{ bold:true, color:P.white, fill:{ color:"0A1E35" }, fontSize:9, fontFace:"Calibri" } },
      { text:"Revenue",  options:{ bold:true, color:P.white, fill:{ color:"0A1E35" }, fontSize:9, fontFace:"Calibri" } },
      { text:"Freq",     options:{ bold:true, color:P.white, fill:{ color:"0A1E35" }, fontSize:9, fontFace:"Calibri" } },
    ];
    var rows = topC.map(function(c, i) {
      return [
        { text:String(i+1),                                       options:{ color:P.blue,  fontSize:9, fontFace:"Calibri" } },
        { text:trunc(String(c.customer || ""), 18),                options:{ color:P.text,  fontSize:9, fontFace:"Calibri" } },
        { text:fmtCurrency(c.monetary || c.revenue || 0),         options:{ color:P.green, fontSize:9, fontFace:"Calibri" } },
        { text:String(c.frequency || ""),                          options:{ color:P.muted, fontSize:9, fontFace:"Calibri" } },
      ];
    });
    slide.addTable([hdr].concat(rows), {
      x:5.1, y:1.1, w:4.6, h:3.95,
      border:{ pt:0.5, color:P.border },
      rowH:0.43,
    });
  }

  addFooter(slide, pres, 7, total);
}

// ── SLIDE 8 — REGIONAL & CATEGORY ────────────────────────────────────────────
function slideRegionalCategory(pres, data, total) {
  var slide = pres.addSlide();
  addBase(slide, pres);
  slide.addText("Regional & Category", { x:0.5, y:0.2, w:9, h:0.48, fontSize:26, bold:true, color:P.white, fontFace:"Calibri" });

  var regional = (data.regional || []).slice(0, 8);
  var cats     = (data.categories || []).slice(0, 8);

  if (regional.length > 0) {
    secLabel(slide, pres, "Revenue per Wilayah", 0.3, 0.82, 4.6);
    slide.addChart(pres.charts.BAR,
      [{ name:"Revenue", labels:regional.map(function(r){ return trunc(r.region||r.name||"",18); }), values:regional.map(function(r){ return Math.round(safe(r.revenue)); }) }],
      Object.assign({ x:0.3, y:1.1, w:4.6, h:2.5, barDir:"bar", chartColors:P.chart_blues, showValue:true, dataLabelColor:P.white, dataLabelFontSize:8 }, chartOpts())
    );
    slide.addChart(pres.charts.PIE,
      [{ name:"Share", labels:regional.map(function(r){ return trunc(r.region||r.name||"",14); }), values:regional.map(function(r){ return Math.round(safe(r.revenue)); }) }],
      { x:0.3, y:3.72, w:4.6, h:1.45, chartColors:P.chart_multi,
        chartArea:{ fill:{ color:P.card } },
        showPercent:true, dataLabelColor:P.white, dataLabelFontSize:8,
        showLegend:true, legendPos:"r", legendFontColor:P.text, legendFontSize:8 }
    );
  }

  if (cats.length > 0) {
    secLabel(slide, pres, "Revenue per Kategori", 5.1, 0.82, 4.6);
    slide.addChart(pres.charts.BAR,
      [{ name:"Revenue", labels:cats.map(function(c){ return trunc(c.category||c.name||"",18); }), values:cats.map(function(c){ return Math.round(safe(c.revenue)); }) }],
      Object.assign({ x:5.1, y:1.1, w:4.6, h:2.5, barDir:"col", chartColors:P.chart_multi, showValue:true, dataLabelColor:P.white, dataLabelFontSize:8 }, chartOpts())
    );

    var par = data.pareto || {};
    if (par.top_product_count) {
      addCard(slide, pres, 5.1, 3.72, 4.6, 1.45);
      slide.addShape(pres.shapes.RECTANGLE, { x:5.1, y:3.72, w:0.055, h:1.45, fill:{ color:P.amber }, line:{ color:P.amber } });
      slide.addText("Pareto 80/20", { x:5.22, y:3.79, w:4.35, h:0.27, fontSize:10, bold:true, color:P.amber, fontFace:"Calibri" });
      slide.addText(par.top_product_count + " produk (" + Math.round(par.pct||0) + "%) hasilkan 80% revenue.", {
        x:5.22, y:4.1, w:4.35, h:0.28, fontSize:11, bold:true, color:P.white, fontFace:"Calibri"
      });
      slide.addText(par.insight || "", { x:5.22, y:4.4, w:4.35, h:0.66, fontSize:9, color:P.text, fontFace:"Calibri", wrap:true });
    }
  }

  addFooter(slide, pres, 8, total);
}

// ── SLIDE 9 — CLOSING ─────────────────────────────────────────────────────────
function slideClosing(pres, data, total) {
  var slide = pres.addSlide();
  slide.background = { color: P.bg };

  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.07, fill:{ color:P.accent }, line:{ color:P.accent } });
  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:5.555, w:10, h:0.07, fill:{ color:P.accent }, line:{ color:P.accent } });
  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:3.8, h:5.625, fill:{ color:P.sky, transparency:90 }, line:{ color:P.bg } });

  slide.addText("⚡", { x:1.2, y:1.3, w:1.4, h:1.4, fontSize:60, align:"center" });
  slide.addText("Terima Kasih", { x:4.0, y:1.25, w:5.6, h:0.85, fontSize:40, bold:true, color:P.white, fontFace:"Calibri" });
  slide.addText("Data-driven decisions,\nbetter business outcomes.", { x:4.0, y:2.18, w:5.6, h:0.85, fontSize:15, color:P.blue, fontFace:"Calibri", italic:true, lineSpacingMultiple:1.35 });
  slide.addShape(pres.shapes.RECTANGLE, { x:4.0, y:3.18, w:2.2, h:0.04, fill:{ color:P.accent }, line:{ color:P.accent } });

  var k = data.kpis || {};
  slide.addText("Total Revenue: " + fmtCurrency(k.total_revenue), { x:4.0, y:3.35, w:5.6, h:0.28, fontSize:12, color:P.text, fontFace:"Calibri" });
  slide.addText("Total Transaksi: " + fmtNum(k.total_transactions) + "  |  Periode: " + (data.date_range || "—"), { x:4.0, y:3.65, w:5.6, h:0.28, fontSize:11, color:P.muted, fontFace:"Calibri" });
  slide.addText("Generated by SalesML Analytics Pro", { x:4.0, y:4.15, w:5.6, h:0.26, fontSize:9.5, color:P.muted, fontFace:"Calibri" });
  addFooter(slide, pres, total, total);
}

// ── MAIN ──────────────────────────────────────────────────────────────────────
function buildPresentation(data) {
  var pres = new pptxgen();
  pres.layout  = "LAYOUT_16x9";
  pres.author  = "SalesML Analytics Pro";
  pres.title   = "Sales Performance Report";
  pres.subject = "Business Analytics";

  var TOTAL = 9;
  slideTitle(pres, data, TOTAL);
  slideAgenda(pres, data, TOTAL);
  slideKPI(pres, data, TOTAL);
  slideTrend(pres, data, TOTAL);
  slideSales(pres, data, TOTAL);
  slideProfit(pres, data, TOTAL);
  slideCustomer(pres, data, TOTAL);
  slideRegionalCategory(pres, data, TOTAL);
  slideClosing(pres, data, TOTAL);

  var outPath = data.output_path || "/tmp/sales_report.pptx";
  pres.writeFile({ fileName: outPath })
    .then(function(){ process.stdout.write("OK:" + outPath + "\n"); })
    .catch(function(e){ process.stderr.write("WRITE_ERROR:" + e.message + "\n"); process.exit(1); });
}
