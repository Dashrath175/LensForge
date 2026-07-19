/**
 * M8 release-figure generator (IMPLEMENTATION.md M8, VALIDATION.md §3):
 * composes the auditable comparison figures committed to docs/ from
 *
 *   1. raw browser captures (rgba8 .bin + .json meta, displayed orientation,
 *      produced by the __lensforge dev hooks and saved by the capture server),
 *   2. measurement reports in docs/results/m8/*.json (V-O/V-I harness output),
 *   3. float64 analytic overlays computed here (Bardeen contour via
 *      src/validation/shadowContour — the same code path the V-I1 test uses).
 *
 * Traced→displayed parity: captures are displayed-orientation (present.wgsl
 * mirrors screen x); all overlay geometry computed in traced coordinates is
 * mapped through x_disp = (width − 1) − x_traced before drawing.
 *
 * Usage: tsx tools/generateFigures.ts <capturesDir> [--write-goldens]
 *   --write-goldens additionally writes tests/golden/<slug>.png from the
 *   golden-* captures (baseline updates are a deliberate, reviewed step —
 *   VALIDATION.md §5).
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { analyticShadowContourPixels, type CameraFraming } from '../src/validation/shadowContour';
import { VI1_FRAMING } from '../src/scene/presets';
import { cropRgba, encodePng } from './png';

const capturesDir = process.argv[2];
const writeGoldens = process.argv.includes('--write-goldens');
if (!capturesDir) {
  console.error('usage: tsx tools/generateFigures.ts <capturesDir> [--write-goldens]');
  process.exit(1);
}

const ROOT = join(import.meta.dirname, '..');
const RESULTS_DIR = join(ROOT, 'docs', 'results', 'm8');
const FIGURES_DIR = join(ROOT, 'docs', 'figures');
const PRESET_SLUGS = ['luminet-1979', 'dngr-gargantua', 'face-on', 'photon-ring-grazing'];
const SHADOW_SPINS = ['0', '0.5', '0.9', '0.998'];

interface Capture {
  rgba: Uint8Array;
  width: number;
  height: number;
}

function readCapture(name: string): Capture | null {
  const binPath = join(capturesDir!, `${name}.bin`);
  const metaPath = join(capturesDir!, `${name}.json`);
  if (!existsSync(binPath) || !existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8')) as { width: number; height: number };
  const rgba = new Uint8Array(readFileSync(binPath));
  if (rgba.length !== meta.width * meta.height * 4) {
    throw new Error(`${name}: ${rgba.length} bytes does not match ${meta.width}×${meta.height}`);
  }
  return { rgba, width: meta.width, height: meta.height };
}

function readResult<T>(name: string): T | null {
  const p = join(RESULTS_DIR, `${name}.json`);
  return existsSync(p) ? (JSON.parse(readFileSync(p, 'utf8')) as T) : null;
}

function pngDataUri(rgba: Uint8Array, width: number, height: number): string {
  return `data:image/png;base64,${encodePng(rgba, width, height).toString('base64')}`;
}

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const written: string[] = [];
function writeFigure(relPath: string, content: string | Buffer): void {
  const p = join(ROOT, relPath);
  mkdirSync(join(p, '..'), { recursive: true });
  writeFileSync(p, content);
  written.push(relPath);
}

// ---------------------------------------------------------------------------
// Preset gallery + goldens
// ---------------------------------------------------------------------------

for (const slug of PRESET_SLUGS) {
  const cap = readCapture(`golden-${slug}`);
  if (!cap) {
    console.warn(`skip: no capture golden-${slug}`);
    continue;
  }
  const png = encodePng(cap.rgba, cap.width, cap.height);
  writeFigure(join('docs', 'figures', 'presets', `${slug}.png`), png);
  if (writeGoldens) writeFigure(join('tests', 'golden', `${slug}.png`), png);
}

// ---------------------------------------------------------------------------
// V-I1: shadow silhouette vs Bardeen contour, four spins
// ---------------------------------------------------------------------------

interface ContourReport {
  spin: number;
  width: number;
  height: number;
  boundaryPoints: number;
  maxDeviationPx: number;
  meanDeviationPx: number;
  pass: boolean;
}

{
  const PANEL = 320;
  const GAP = 18;
  const CAPTION_H = 40;
  const panels: string[] = [];
  let missing = false;
  SHADOW_SPINS.forEach((spinStr, idx) => {
    const cap = readCapture(`shadow-a${spinStr}`);
    const report = readResult<ContourReport>(`vi1-contour-a${spinStr}`);
    if (!cap || !report) {
      missing = true;
      return;
    }
    const spin = Number(spinStr);
    const framing: CameraFraming = { ...VI1_FRAMING, width: cap.width, height: cap.height };
    const contour = analyticShadowContourPixels(spin, framing, 2048).map((p) => ({
      x: cap.width - 1 - p.x, // traced → displayed parity
      y: p.y,
    }));
    const xs = contour.map((p) => p.x);
    const ys = contour.map((p) => p.y);
    const pad = 55;
    const side = Math.ceil(Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 2 * pad);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const crop = cropRgba(cap.rgba, cap.width, cap.height, cx - side / 2, cy - side / 2, side, side);
    const scale = PANEL / crop.width;
    const px = (idx % 2) * (PANEL + GAP);
    const py = Math.floor(idx / 2) * (PANEL + GAP + CAPTION_H);
    const dots = contour
      .filter(
        (p) => p.x >= crop.x0 && p.x < crop.x0 + crop.width && p.y >= crop.y0 && p.y < crop.y0 + crop.height,
      )
      .map(
        (p) =>
          `<circle cx="${(px + (p.x - crop.x0) * scale).toFixed(1)}" cy="${(py + (p.y - crop.y0) * scale).toFixed(1)}" r="0.9" fill="#3bd6c6" fill-opacity="0.85"/>`,
      )
      .join('');
    panels.push(
      `<image x="${px}" y="${py}" width="${PANEL}" height="${(crop.height * scale).toFixed(1)}" href="${pngDataUri(crop.rgba, crop.width, crop.height)}" image-rendering="optimizeQuality"/>` +
        dots +
        `<text x="${px}" y="${py + PANEL + 16}" font-size="13" font-weight="600" fill="#222">a = ${spinStr}</text>` +
        `<text x="${px}" y="${py + PANEL + 32}" font-size="12" fill="#444">max dev ${report.maxDeviationPx.toFixed(2)} px · mean ${report.meanDeviationPx.toFixed(2)} px · ${report.boundaryPoints} boundary px · ${report.pass ? 'PASS' : 'FAIL'}</text>`,
    );
  });
  if (!missing && panels.length === 4) {
    const W = 2 * PANEL + GAP;
    const H = 2 * (PANEL + CAPTION_H) + GAP + 46;
    writeFigure(
      join('docs', 'figures', 'vi1-shadow-contours.svg'),
      `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="system-ui, sans-serif">` +
        `<rect width="100%" height="100%" fill="#fff"/>` +
        `<g transform="translate(0, 40)">${panels.join('')}</g>` +
        `<text x="0" y="16" font-size="15" font-weight="700" fill="#111">V-I1 — rendered Kerr shadow silhouette vs Bardeen analytic contour (dots)</text>` +
        `<text x="0" y="32" font-size="12" fill="#444">${esc('ZAMO observer, r = 1000M, equatorial, fovY 0.03 rad, 1280×720; contour mapped through the exact float64 tetrad inversion (no far-field approximation). Bound: 1 px.')}</text>` +
        `</svg>`,
    );
  } else {
    console.warn('skip: vi1-shadow-contours.svg (missing captures/reports)');
  }
}

// ---------------------------------------------------------------------------
// Log-log chart helper (V-O curves)
// ---------------------------------------------------------------------------

interface Series {
  label: string;
  color: string;
  points: Array<{ x: number; y: number }>;
  /** Scatter only — no connecting polyline (non-monotonic data). */
  pointsOnly?: boolean;
}

function logLogChart(opts: {
  title: string;
  subtitle: string[];
  xLabel: string;
  yLabel: string;
  xRange: [number, number];
  yRange: [number, number];
  series: Series[];
  hLines?: Array<{ y: number; label: string; color: string }>;
  vLines?: Array<{ x: number; label: string }>;
  /** Legend anchor inside the plot area. */
  legend?: 'top-left' | 'bottom-left';
}): string {
  const W = 720;
  const H = 480;
  const M = { l: 64, r: 16, t: 44 + 15 * opts.subtitle.length, b: 46 };
  const lx = (v: number): number =>
    M.l + ((Math.log10(v) - Math.log10(opts.xRange[0])) / (Math.log10(opts.xRange[1]) - Math.log10(opts.xRange[0]))) * (W - M.l - M.r);
  const ly = (v: number): number =>
    H - M.b - ((Math.log10(v) - Math.log10(opts.yRange[0])) / (Math.log10(opts.yRange[1]) - Math.log10(opts.yRange[0]))) * (H - M.t - M.b);
  let grid = '';
  for (let e = Math.ceil(Math.log10(opts.xRange[0])); e <= Math.floor(Math.log10(opts.xRange[1])); e++) {
    const x = lx(10 ** e);
    grid += `<line x1="${x}" y1="${M.t}" x2="${x}" y2="${H - M.b}" stroke="#e3e6ea"/>`;
    grid += `<text x="${x}" y="${H - M.b + 18}" font-size="11" text-anchor="middle" fill="#555">1e${e}</text>`;
  }
  for (let e = Math.ceil(Math.log10(opts.yRange[0])); e <= Math.floor(Math.log10(opts.yRange[1])); e++) {
    const y = ly(10 ** e);
    grid += `<line x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}" stroke="#e3e6ea"/>`;
    grid += `<text x="${M.l - 6}" y="${y + 4}" font-size="11" text-anchor="end" fill="#555">1e${e}</text>`;
  }
  let ref = '';
  for (const h of opts.hLines ?? []) {
    const y = ly(h.y);
    ref += `<line x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}" stroke="${h.color}" stroke-dasharray="6 4"/>`;
    ref += `<text x="${W - M.r - 4}" y="${y - 5}" font-size="11" text-anchor="end" fill="${h.color}">${esc(h.label)}</text>`;
  }
  for (const v of opts.vLines ?? []) {
    const x = lx(v.x);
    ref += `<line x1="${x}" y1="${M.t}" x2="${x}" y2="${H - M.b}" stroke="#999" stroke-dasharray="3 4"/>`;
    ref += `<text x="${x + 4}" y="${M.t + 12}" font-size="11" fill="#666">${esc(v.label)}</text>`;
  }
  let body = '';
  let legendY = opts.legend === 'bottom-left' ? H - M.b - 12 - 17 * (opts.series.length - 1) : M.t + 12;
  for (const s of opts.series) {
    const pts = s.points
      .filter((p) => p.x >= opts.xRange[0] && p.x <= opts.xRange[1] && p.y > 0)
      .sort((p, q) => p.x - q.x);
    const clampY = (v: number): number => Math.min(Math.max(v, opts.yRange[0]), opts.yRange[1]);
    if (!s.pointsOnly) {
      body += `<polyline fill="none" stroke="${s.color}" stroke-width="1.4" points="${pts.map((p) => `${lx(p.x).toFixed(1)},${ly(clampY(p.y)).toFixed(1)}`).join(' ')}"/>`;
    }
    body += pts
      .map((p) => `<circle cx="${lx(p.x).toFixed(1)}" cy="${ly(clampY(p.y)).toFixed(1)}" r="2.6" fill="${s.color}"/>`)
      .join('');
    body += `<circle cx="${M.l + 12}" cy="${legendY}" r="3" fill="${s.color}"/><text x="${M.l + 20}" y="${legendY + 4}" font-size="12" fill="#333">${esc(s.label)}</text>`;
    legendY += 17;
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="system-ui, sans-serif">` +
    `<rect width="100%" height="100%" fill="#fff"/>` +
    `<text x="${M.l}" y="20" font-size="15" font-weight="700" fill="#111">${esc(opts.title)}</text>` +
    opts.subtitle
      .map((line, i) => `<text x="${M.l}" y="${37 + 15 * i}" font-size="12" fill="#444">${esc(line)}</text>`)
      .join('') +
    grid +
    ref +
    body +
    `<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="none" stroke="#889"/>` +
    `<text x="${(M.l + W - M.r) / 2}" y="${H - 10}" font-size="12" text-anchor="middle" fill="#333">${esc(opts.xLabel)}</text>` +
    `<text x="14" y="${(M.t + H - M.b) / 2}" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 14 ${(M.t + H - M.b) / 2})">${esc(opts.yLabel)}</text>` +
    `</svg>`
  );
}

// ---------------------------------------------------------------------------
// V-O1: escape-direction deviation curves
// ---------------------------------------------------------------------------

interface VOReportJson {
  rows: Array<{ bOffset: number | null; arcmin: number | null }>;
}

{
  const SERIES_SPEC: Array<{ file: string; label: string; color: string }> = [
    { file: 'vo1-high-a0', label: 'High tier, a = 0', color: '#2563eb' },
    { file: 'vo1-high-a0.9', label: 'High tier, a = 0.9', color: '#0ea5a3' },
    { file: 'vo1-high-a0.998', label: 'High tier, a = 0.998', color: '#7c3aed' },
    { file: 'vo1-medium-a0.9', label: 'Medium tier, a = 0.9', color: '#d97706' },
    { file: 'vo1-low-a0.9', label: 'Low tier, a = 0.9', color: '#dc2626' },
  ];
  const series: Series[] = [];
  for (const spec of SERIES_SPEC) {
    const rep = readResult<VOReportJson>(spec.file);
    if (!rep) continue;
    // The reported curve is max deviation per |b − b_c| offset (prograde and
    // retrograde sweeps share offsets; the envelope is the bounded quantity).
    const byOffset = new Map<number, number>();
    for (const r of rep.rows) {
      if (r.bOffset === null || r.bOffset <= 0 || r.arcmin === null) continue;
      byOffset.set(r.bOffset, Math.max(byOffset.get(r.bOffset) ?? 0, r.arcmin));
    }
    const points = [...byOffset.entries()].map(([x, y]) => ({ x, y: Math.max(y, 1.1e-4) }));
    if (points.length > 0) series.push({ label: spec.label, color: spec.color, points });
  }
  if (series.length > 0) {
    writeFigure(
      join('docs', 'figures', 'vo1-deviation.svg'),
      logLogChart({
        title: 'V-O1 — GPU f32 vs CPU f64 escape-direction deviation',
        subtitle: [
          'Identical initial rays; max angular error of the asymptotic sky direction per impact-parameter',
          'offset from the critical curve (prograde/retrograde envelope). Left of b − b_c = 1e−2: reported, not bounded.',
        ],
        xLabel: '|b − b_c| (M)',
        yLabel: 'angular deviation (arcmin)',
        xRange: [1e-3, 10],
        yRange: [1e-4, 30],
        series,
        legend: 'bottom-left',
        hLines: [
          { y: 1, label: 'High bound 1′', color: '#2563eb' },
          { y: 3, label: 'Medium bound 3′', color: '#d97706' },
          { y: 10, label: 'Low bound 10′', color: '#dc2626' },
        ],
        vLines: [{ x: 1e-2, label: 'bounded region →' }],
      }),
    );
  } else {
    console.warn('skip: vo1-deviation.svg (no reports)');
  }
}

// ---------------------------------------------------------------------------
// V-O2: disk-hit deviation
// ---------------------------------------------------------------------------

interface VO2ReportJson {
  rows: Array<{ rOracle: number | null; dgRel: number | null; drNormalized: number | null }>;
}

{
  const SPEC: Array<{ file: string; label: string; color: string }> = [
    { file: 'vo2-a0', label: 'a = 0', color: '#2563eb' },
    { file: 'vo2-a0.9', label: 'a = 0.9', color: '#0ea5a3' },
    { file: 'vo2-a0.998', label: 'a = 0.998', color: '#7c3aed' },
  ];
  const gSeries: Series[] = [];
  for (const spec of SPEC) {
    const rep = readResult<VO2ReportJson>(spec.file);
    if (!rep) continue;
    const points = rep.rows
      .filter((r) => r.rOracle !== null && r.dgRel !== null)
      .map((r) => ({ x: r.rOracle!, y: Math.max(r.dgRel!, 1.1e-8) }));
    if (points.length > 0) gSeries.push({ label: spec.label, color: spec.color, points, pointsOnly: true });
  }
  if (gSeries.length > 0) {
    writeFigure(
      join('docs', 'figures', 'vo2-disk-deviation.svg'),
      logLogChart({
        title: 'V-O2 — disk-hit redshift-factor deviation Δg/g',
        subtitle: [
          'GPU f32 vs CPU f64 on identical rays through the f32 ray-init contract;',
          'the radiometric bound is Δg/g < 1e−3 (High tier).',
        ],
        xLabel: 'oracle hit radius r (M)',
        yLabel: 'Δg / g',
        xRange: [1, 30],
        yRange: [1e-8, 1e-2],
        series: gSeries,
        hLines: [{ y: 1e-3, label: 'radiometric bound 1e−3', color: '#dc2626' }],
      }),
    );
  } else {
    console.warn('skip: vo2-disk-deviation.svg (no reports)');
  }
}

// ---------------------------------------------------------------------------
// V-I2: Luminet-framing disk-image boundary geometry
// ---------------------------------------------------------------------------

interface DiskEdgeReportJson {
  spin: number;
  width: number;
  height: number;
  transitions: number;
  matched: number;
  fragile: number;
  maxDevPx: number;
  meanDevPx: number;
  maxRelDev: number;
  maxRelDevBounded: number;
  rows: Array<{ angle: number; rhoOracle: number }>;
  pass: boolean;
}

{
  const cap = readCapture('golden-luminet-1979');
  const rep = readResult<DiskEdgeReportJson>('vi2-disk-edges');
  if (cap && rep) {
    const DISP_W = 880;
    const scale = DISP_W / cap.width;
    const dispH = cap.height * scale;
    const cx = rep.width / 2 - 0.5;
    const cy = rep.height / 2 - 0.5;
    const dots = rep.rows
      .map((r) => {
        const xt = cx + r.rhoOracle * Math.cos(r.angle);
        const yt = cy + r.rhoOracle * Math.sin(r.angle);
        const xd = rep.width - 1 - xt; // traced → displayed parity
        return `<circle cx="${(xd * scale).toFixed(1)}" cy="${(yt * scale).toFixed(1)}" r="2.2" fill="none" stroke="#3bd6c6" stroke-width="1.1"/>`;
      })
      .join('');
    writeFigure(
      join('docs', 'figures', 'vi2-luminet-geometry.svg'),
      `<svg xmlns="http://www.w3.org/2000/svg" width="${DISP_W}" height="${(dispH + 107).toFixed(0)}" font-family="system-ui, sans-serif">` +
        `<rect width="100%" height="100%" fill="#fff"/>` +
        `<text x="0" y="16" font-size="15" font-weight="700" fill="#111">V-I2 — Luminet (1979) framing: disk-image boundary geometry vs float64 oracle</text>` +
        `<text x="0" y="33" font-size="12" fill="#444">${esc('Schwarzschild, disk to 20M, viewed 10° above the plane (the [R2] photograph geometry).')}</text>` +
        `<text x="0" y="48" font-size="12" fill="#444">${esc('Circles: oracle (float64) boundary radii bisected along 48 spokes — primary and secondary image edges.')}</text>` +
        `<g transform="translate(0, 59)"><image width="${DISP_W}" height="${dispH.toFixed(1)}" href="${pngDataUri(cap.rgba, cap.width, cap.height)}"/>${dots}</g>` +
        `<text x="0" y="${(dispH + 79).toFixed(0)}" font-size="12" fill="#333">matched ${rep.matched}/${rep.transitions} boundary transitions (${rep.fragile} status-fragile) · max deviation ${rep.maxDevPx.toFixed(2)} px (mean ${rep.meanDevPx.toFixed(2)} px) · shape deviation ${(100 * rep.maxRelDevBounded).toFixed(2)} % where resolvable (ρ ≥ 50 px)</text>` +
        `<text x="0" y="${(dispH + 97).toFixed(0)}" font-size="12" fill="#333">bounds: ≤ 1 px absolute, &lt; 2 % shape — ${rep.pass ? 'PASS' : 'FAIL'}</text>` +
        `</svg>`,
    );
  } else {
    console.warn('skip: vi2-luminet-geometry.svg (missing capture/report)');
  }
}

// ---------------------------------------------------------------------------
// V-I3: beaming asymmetry
// ---------------------------------------------------------------------------

interface BeamingReportJson {
  spin: number;
  ratioGpu: number;
  ratioOracle: number;
  relDiff: number;
  meanPixelRelDiff: number;
  maxPixelRelDiff: number;
  sampledLeft: number;
  sampledRight: number;
  pass: boolean;
}

{
  const cap = readCapture('golden-dngr-gargantua');
  const reports = ['vi3-beaming-a0', 'vi3-beaming-a0.9']
    .map((f) => readResult<BeamingReportJson>(f))
    .filter((r): r is BeamingReportJson => r !== null);
  if (cap && reports.length > 0) {
    const DISP_W = 720;
    const scale = DISP_W / cap.width;
    const dispH = cap.height * scale;
    // The harness ratio is traced-orientation left/right = receding/approaching
    // (the approaching side is screen-right in the raw HDR); the physical
    // approaching/receding ratio is its reciprocal.
    const lines = reports
      .flatMap((r, i) => [
        `<text x="0" y="${(dispH + 96 + 34 * i).toFixed(0)}" font-size="12" fill="#333">a = ${r.spin}: approaching/receding luminance ratio — GPU ${(1 / r.ratioGpu).toFixed(3)} vs oracle ${(1 / r.ratioOracle).toFixed(3)} (rel diff ${r.relDiff.toExponential(1)}; bound 5 %)</text>`,
        `<text x="14" y="${(dispH + 111 + 34 * i).toFixed(0)}" font-size="12" fill="#333">mean per-pixel HDR deviation ${(100 * r.meanPixelRelDiff).toFixed(2)} % over ${r.sampledLeft + r.sampledRight} sampled disk pixels · ${r.pass ? 'PASS' : 'FAIL'}</text>`,
      ])
      .join('');
    writeFigure(
      join('docs', 'figures', 'vi3-beaming.svg'),
      `<svg xmlns="http://www.w3.org/2000/svg" width="${DISP_W}" height="${(dispH + 108 + 34 * reports.length).toFixed(0)}" font-family="system-ui, sans-serif">` +
        `<rect width="100%" height="100%" fill="#fff"/>` +
        `<text x="0" y="16" font-size="15" font-weight="700" fill="#111">V-I3 — Doppler beaming asymmetry vs float64 oracle</text>` +
        `<text x="0" y="33" font-size="12" fill="#444">${esc('Measurement framing: near-edge-on ZAMO at r = 60M (VALIDATION.md §3); every sampled disk pixel is')}</text>` +
        `<text x="0" y="48" font-size="12" fill="#444">${esc('re-shaded by the oracle (intersection → g → NT flux → Planck → XYZ → Y).')}</text>` +
        `<text x="0" y="63" font-size="12" fill="#444">${esc('Shown: the DNGR Gargantua preset (a = 0.998), approaching side left — the [R5] convention.')}</text>` +
        `<g transform="translate(0, 74)"><image width="${DISP_W}" height="${dispH.toFixed(1)}" href="${pngDataUri(cap.rgba, cap.width, cap.height)}"/></g>` +
        lines +
        `</svg>`,
    );
  } else {
    console.warn('skip: vi3-beaming.svg (missing capture/reports)');
  }
}

console.log(`wrote ${written.length} files:\n  ${written.join('\n  ')}`);
