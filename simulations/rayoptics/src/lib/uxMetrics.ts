import type { AnalysisSummary, CockpitMetric, CockpitRisk, KpiStatus, OpticalModel } from "./types";
import {
  applyProjectRequirements,
  applyRequirementRiskStatuses,
  defaultProjectRequirements,
  type ProjectRequirements
} from "./requirements";

export type { CockpitMetric, CockpitRisk } from "./types";

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function statusFor(value: number, warn: number, fail: number, direction: "high-good" | "low-good"): KpiStatus {
  if (direction === "high-good") {
    if (value <= fail) return "fail";
    if (value <= warn) return "warn";
    return "pass";
  }
  if (value >= fail) return "fail";
  if (value >= warn) return "warn";
  return "pass";
}

function statusFromCount(failCount: number, warnCount: number): KpiStatus {
  if (failCount > 0) return "fail";
  if (warnCount > 0) return "warn";
  return "pass";
}

export function deriveMetrics(
  model: OpticalModel | null,
  warnings: string[] = [],
  errors: string[] = [],
  summary: AnalysisSummary | null = null,
  requirements: ProjectRequirements = defaultProjectRequirements
) {
  if (summary) {
    const metrics = applyProjectRequirements(summary.metrics, requirements);
    return {
      metrics,
      risks: applyRequirementRiskStatuses(summary.risks, metrics),
      complexity: summary.counts.complexity,
      maxSemi: summary.counts.maxSemiDiameter,
      fieldMax: summary.counts.maxField,
      poweredSurfaces: summary.counts.poweredSurfaces,
      wavelengthCount: summary.counts.wavelengthCount,
      surfaceCount: summary.counts.surfaceCount
    };
  }

  const surfaces = model?.surfaces ?? [];
  const fields = model?.system.field.fields ?? [];
  const wavelengths = model?.system.wavelengths.values.filter((value) => value !== null) ?? [];
  const poweredSurfaces = surfaces.filter((surface) => surface.radius !== null && Math.abs(surface.curvature) > 1e-6);
  const stop = model?.stopSurface ?? null;
  const maxSemi = Math.max(1, ...surfaces.map((surface) => surface.semiDiameter || 0));
  const fieldMax = Math.max(0, ...fields.map((field) => Math.hypot(field.x ?? 0, field.y ?? 0)));
  const curvatureLoad = poweredSurfaces.reduce((sum, surface) => sum + Math.min(2.4, Math.abs(surface.curvature) * 35), 0);
  const complexity = poweredSurfaces.length + curvatureLoad * 0.3 + Math.max(0, fields.length - 3) * 0.35 + Math.max(0, wavelengths.length - 3) * 0.25;
  const warnCount = warnings.length;
  const failCount = errors.length;

  const mtf = clamp(0.9 - complexity * 0.038 - warnCount * 0.025 - failCount * 0.16, 0.08, 0.96);
  const illumination = clamp(1 - fieldMax / Math.max(maxSemi * 3.6, 1) - Math.max(0, poweredSurfaces.length - 6) * 0.035, 0.22, 0.98);
  const distortion = clamp(fieldMax * 0.16 + curvatureLoad * 0.085 + warnCount * 0.22, 0.05, 5.5);
  const cra = clamp((fieldMax / Math.max(maxSemi, 1)) * 18 + poweredSurfaces.length * 1.1 + (stop ?? 0) * 0.32, 4, 35);
  const tolerancePct = clamp(96 - complexity * 2.4 - distortion * 1.7 - warnCount * 4.5 - failCount * 20, 35, 98.5);
  const snr = clamp(44 - fieldMax * 0.55 - Math.max(0, cra - 18) * 0.42 - warnCount * 1.4 - failCount * 8, 16, 48);
  const traceScore = failCount > 0 ? 0 : warnCount > 0 ? 0.68 : 1;

  const metrics: CockpitMetric[] = [
    {
      key: "trace",
      label: "Trace Health",
      value: failCount > 0 ? `${failCount}` : warnCount > 0 ? `${warnCount}` : "0",
      unit: failCount > 0 ? "errors" : "warnings",
      target: "0 errors",
      status: statusFromCount(failCount, warnCount),
      score: traceScore,
      source: "diagnostic"
    },
    {
      key: "mtf",
      label: "MTF50 Proxy",
      value: mtf.toFixed(2),
      target: ">= 0.30",
      status: statusFor(mtf, 0.42, 0.3, "high-good"),
      score: mtf,
      source: "proxy"
    },
    {
      key: "illumination",
      label: "Rel. Illum.",
      value: illumination.toFixed(2),
      target: ">= 0.50",
      status: statusFor(illumination, 0.62, 0.5, "high-good"),
      score: illumination,
      source: "proxy"
    },
    {
      key: "distortion",
      label: "Distortion",
      value: distortion.toFixed(2),
      unit: "%",
      target: "<= 2.00%",
      status: statusFor(distortion, 2, 3.25, "low-good"),
      score: clamp(1 - distortion / 5.5, 0, 1),
      source: "proxy"
    },
    {
      key: "cra",
      label: "CRA Max",
      value: cra.toFixed(1),
      unit: "deg",
      target: "<= 25 deg",
      status: statusFor(cra, 25, 30, "low-good"),
      score: clamp(1 - cra / 35, 0, 1),
      source: "proxy"
    },
    {
      key: "yield",
      label: "Tol. Proxy",
      value: tolerancePct.toFixed(1),
      unit: "%",
      target: ">= 80%",
      status: statusFor(tolerancePct, 86, 80, "high-good"),
      score: tolerancePct / 100,
      source: "proxy"
    },
    {
      key: "snr",
      label: "Corner SNR",
      value: snr.toFixed(1),
      unit: "dB",
      target: ">= 40 dB",
      status: statusFor(snr, 40, 32, "high-good"),
      score: clamp(snr / 48, 0, 1),
      source: "proxy"
    }
  ];

  const risks: CockpitRisk[] = [
    {
      label: "Worst Field",
      value: fieldMax > 0 ? fieldMax.toFixed(2) : "On-axis",
      status: metrics.find((metric) => metric.key === "illumination")?.status ?? "pass",
      detail: `${fields.length || 1} fields, ${wavelengths.length || 1} wavelengths`,
      source: "model"
    },
    {
      label: "Surface Sensitivity",
      value: poweredSurfaces.length ? `S${poweredSurfaces[Math.min(poweredSurfaces.length - 1, 2)].index}` : "n/a",
      status: metrics.find((metric) => metric.key === "yield")?.status ?? "pass",
      detail: `${poweredSurfaces.length} powered surfaces`,
      source: "proxy"
    },
    {
      label: "Sensor Coupling",
      value: `${cra.toFixed(1)} deg`,
      status: metrics.find((metric) => metric.key === "cra")?.status ?? "pass",
      detail: "CRA / microlens margin",
      source: "proxy"
    },
    {
      label: "Scene Risk",
      value: snr < 32 ? "High" : snr < 40 ? "Medium" : "Low",
      status: metrics.find((metric) => metric.key === "snr")?.status ?? "pass",
      detail: "Assumed-scene SNR proxy",
      source: "proxy"
    }
  ];

  const requirementMetrics = applyProjectRequirements(metrics, requirements);

  return {
    metrics: requirementMetrics,
    risks: applyRequirementRiskStatuses(risks, requirementMetrics),
    complexity,
    maxSemi,
    fieldMax,
    poweredSurfaces: poweredSurfaces.length,
    wavelengthCount: wavelengths.length,
    surfaceCount: surfaces.length
  };
}

export function statusLabel(status: KpiStatus) {
  if (status === "pass") return "PASS";
  if (status === "warn") return "WARN";
  return "FAIL";
}
