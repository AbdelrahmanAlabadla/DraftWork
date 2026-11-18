/** Small dependency-free SVG/HTML charts for the evaluation dashboard. */
import { escapeHtml, num, pct } from "./utils.js";

export const COLORS = {
  success: "#4fae75",
  danger: "#dc6262",
  warning: "#d99a32",
  info: "#6f9fd8",
  accent: "#f36e15",
  muted: "#969ca5",
  text: "#f3f4f6",
};

const WIDTH = 640;
const PADDING = { left: 42, right: 10, top: 12, bottom: 22 };

function axisMax(values) {
  const finite = values.filter((value) => value !== null && Number.isFinite(value));
  if (!finite.length) return 1;
  const max = Math.max(...finite);
  if (max <= 0) return 1;
  // Rates are bounded at 1.0; keep the axis at 100% instead of rounding up to 2.
  if (max <= 1) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(max));
  const scaled = max / magnitude;
  const step = scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * magnitude;
}

function formatAxis(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}k`;
  if (value >= 1) return Math.round(value).toString();
  return value.toFixed(2);
}

export function lineChart({
  labels = [],
  series = [],
  height = 190,
  yFormat = formatAxis,
  yMax = null,
  valueFormat = (value) => num(value),
  tooltipLabels = null,
} = {}) {
  const allValues = series.flatMap((item) => item.values || []);
  const max = yMax ?? axisMax(allValues);
  const innerWidth = WIDTH - PADDING.left - PADDING.right;
  const innerHeight = height - PADDING.top - PADDING.bottom;
  const count = labels.length;
  if (!count) return `<div class="bars-empty">No data in this period</div>`;
  const xAt = (index) => {
    if (count === 1) return PADDING.left + innerWidth / 2;
    return PADDING.left + (index / (count - 1)) * innerWidth;
  };
  const yAt = (value) => PADDING.top + innerHeight - (Math.max(0, value) / max) * innerHeight;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * max);
  const labelEvery = Math.max(1, Math.ceil(count / 6));

  const grid = ticks.map((tick) => {
    const y = yAt(tick);
    return `
      <line class="grid-line" x1="${PADDING.left}" x2="${WIDTH - PADDING.right}" y1="${y}" y2="${y}"></line>
      <text class="axis-label" x="${PADDING.left - 6}" y="${y + 3}" text-anchor="end">${yFormat(tick)}</text>`;
  }).join("");

  const xLabels = labels.map((label, index) => {
    if (index % labelEvery !== 0 && index !== count - 1) return "";
    return `<text class="axis-label" x="${xAt(index)}" y="${height - 6}" text-anchor="middle">${escapeHtml(label)}</text>`;
  }).join("");

  const paths = series.map((item) => {
    const color = item.color || COLORS.accent;
    let path = "";
    let drawing = false;
    (item.values || []).forEach((value, index) => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        drawing = false;
        return;
      }
      const command = drawing ? "L" : "M";
      path += `${command}${xAt(index).toFixed(1)},${yAt(Number(value)).toFixed(1)} `;
      drawing = true;
    });
    const dots = (item.values || []).map((value, index) => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
      const sample = item.sampleCounts?.[index];
      const label = tooltipLabels?.[index] || labels[index];
      const sampleText = sample === null || sample === undefined
        ? ""
        : ` · ${Number(sample).toLocaleString()} ${item.sampleLabel || "sample(s)"}`;
      return `<circle class="series-dot" cx="${xAt(index)}" cy="${yAt(Number(value))}" r="2.6" fill="${color}">
        <title>${escapeHtml(item.name)} · ${escapeHtml(label)} · ${escapeHtml(valueFormat(Number(value)))}${escapeHtml(sampleText)}</title>
      </circle>`;
    }).join("");
    return `<path class="series-line" d="${path.trim()}" stroke="${color}"></path>${dots}`;
  }).join("");

  const legend = series.map((item) => `
    <span class="legend-item"><span class="legend-swatch" style="background:${item.color || COLORS.accent}"></span>${escapeHtml(item.name)}</span>
  `).join("");

  return `
    <div class="chart">
      <svg viewBox="0 0 ${WIDTH} ${height}" role="img" aria-label="Line chart">
        ${grid}${xLabels}${paths}
      </svg>
    </div>
    <div class="legend">${legend}</div>`;
}

export function groupedBars({
  groups = [],
  series = [],
  height = 210,
  yFormat = (value) => (value <= 1 ? pct(value) : formatAxis(value)),
  valueFormat = (value) => num(value),
} = {}) {
  if (!groups.length) return `<div class="bars-empty">No data in this period</div>`;
  const allValues = series.flatMap((item) => item.values || []);
  const max = axisMax(allValues.map((value) => (value === null ? null : Number(value))));
  const innerWidth = WIDTH - PADDING.left - PADDING.right;
  const innerHeight = height - PADDING.top - PADDING.bottom;
  const groupWidth = innerWidth / groups.length;
  const barWidth = Math.min(16, (groupWidth * 0.7) / Math.max(1, series.length));
  const yAt = (value) => PADDING.top + innerHeight - (Math.max(0, value) / max) * innerHeight;
  const ticks = [0, 0.5, 1].map((fraction) => fraction * max);
  const grid = ticks.map((tick) => {
    const y = yAt(tick);
    return `
      <line class="grid-line" x1="${PADDING.left}" x2="${WIDTH - PADDING.right}" y1="${y}" y2="${y}"></line>
      <text class="axis-label" x="${PADDING.left - 6}" y="${y + 3}" text-anchor="end">${yFormat(tick)}</text>`;
  }).join("");
  const bars = groups.map((group, groupIndex) => {
    const startX = PADDING.left + groupIndex * groupWidth + groupWidth / 2
      - (series.length * barWidth) / 2;
    const rects = series.map((item, seriesIndex) => {
      const value = item.values?.[groupIndex];
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
      const x = startX + seriesIndex * barWidth;
      const y = yAt(Number(value));
      return `<rect class="bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(2, barWidth - 2)}"
        height="${Math.max(1, PADDING.top + innerHeight - y).toFixed(1)}" fill="${item.color || COLORS.accent}">
        <title>${escapeHtml(item.name)} · ${escapeHtml(group.label)} · ${escapeHtml(valueFormat(Number(value)))}</title>
        </rect>`;
    }).join("");
    return `${rects}
      <text class="axis-label" x="${PADDING.left + groupIndex * groupWidth + groupWidth / 2}"
        y="${height - 6}" text-anchor="middle">${escapeHtml(group.label)}</text>`;
  }).join("");
  const legend = series.map((item) => `
    <span class="legend-item"><span class="legend-swatch" style="background:${item.color || COLORS.accent}"></span>${escapeHtml(item.name)}</span>
  `).join("");
  return `
    <div class="chart">
      <svg viewBox="0 0 ${WIDTH} ${height}" role="img" aria-label="Grouped bar chart">
        ${grid}${bars}
      </svg>
    </div>
    <div class="legend">${legend}</div>`;
}

export function stackedBar(segments = []) {
  const items = segments.filter((segment) => Number(segment.value || 0) > 0);
  const total = items.reduce((sum, segment) => sum + Number(segment.value || 0), 0);
  if (!total) return `<div class="bars-empty">No data in this period</div>`;
  return `
    <div class="progress" style="height:10px;margin-top:0">
      ${items.map((segment) => `
        <span style="display:inline-block;height:100%;width:${(Number(segment.value) / total) * 100}%;
          background:${segment.color || COLORS.muted}"
          title="${escapeHtml(segment.label)}: ${num(segment.value)}"></span>
      `).join("")}
    </div>
    <div class="legend">
      ${items.map((segment) => `
        <span class="legend-item"><span class="legend-swatch" style="background:${segment.color || COLORS.muted}"></span>
        ${escapeHtml(segment.label)} · ${num(segment.value)}</span>
      `).join("")}
    </div>`;
}
