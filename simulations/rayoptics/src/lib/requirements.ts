import type { CockpitMetric, CockpitRisk, KpiStatus } from "./types";

export interface ProjectRequirements {
  spotMaxUm: number;
  mtfMinLpMm: number;
  mtfProxyMin: number;
  illuminationMin: number;
  distortionMaxPct: number;
  craMaxDeg: number;
  toleranceMinPct: number;
  snrMinDb: number;
}

export type ProjectRequirementKey = keyof ProjectRequirements;
export type RequirementProfileId = "reference" | "mobile-wide" | "automotive-wide" | "lidar-receiver" | "arvr-eyepiece" | "endoscope";
export type RequirementProfileSelection = RequirementProfileId | "custom";
export type RequirementDirection = "min" | "max";

export interface RequirementDefinition {
  metricKey: string;
  targetKey: ProjectRequirementKey;
  label: string;
  direction: RequirementDirection;
  unit?: string;
  min: number;
  max?: number;
  step: number;
  decimals: number;
}

export interface RequirementProfile {
  id: RequirementProfileId;
  label: string;
  domain: string;
  description: string;
  requirements: ProjectRequirements;
}

export const defaultProjectRequirements: ProjectRequirements = {
  spotMaxUm: 10,
  mtfMinLpMm: 20,
  mtfProxyMin: 0.3,
  illuminationMin: 0.5,
  distortionMaxPct: 2,
  craMaxDeg: 25,
  toleranceMinPct: 86,
  snrMinDb: 30
};

export const defaultRequirementProfile: RequirementProfileId = "reference";

export const requirementProfiles: RequirementProfile[] = [
  {
    id: "reference",
    label: "50 mm Reference",
    domain: "General",
    description: "Balanced sequential-lens baseline for example models.",
    requirements: defaultProjectRequirements
  },
  {
    id: "mobile-wide",
    label: "Mobile Wide",
    domain: "Camera",
    description: "Compact wide camera target with tighter CRA and distortion pressure.",
    requirements: {
      spotMaxUm: 6,
      mtfMinLpMm: 45,
      mtfProxyMin: 0.42,
      illuminationMin: 0.55,
      distortionMaxPct: 3,
      craMaxDeg: 18,
      toleranceMinPct: 82,
      snrMinDb: 28
    }
  },
  {
    id: "automotive-wide",
    label: "Automotive Wide",
    domain: "ADAS",
    description: "Wide-FOV safety camera target emphasizing corner illumination, SNR, CRA, and robustness.",
    requirements: {
      spotMaxUm: 8,
      mtfMinLpMm: 30,
      mtfProxyMin: 0.35,
      illuminationMin: 0.6,
      distortionMaxPct: 5,
      craMaxDeg: 24,
      toleranceMinPct: 90,
      snrMinDb: 32
    }
  },
  {
    id: "lidar-receiver",
    label: "LiDAR Receiver",
    domain: "Receiver",
    description: "Large-pupil receiver target that favors throughput and alignment robustness.",
    requirements: {
      spotMaxUm: 18,
      mtfMinLpMm: 8,
      mtfProxyMin: 0.22,
      illuminationMin: 0.72,
      distortionMaxPct: 1.5,
      craMaxDeg: 16,
      toleranceMinPct: 88,
      snrMinDb: 35
    }
  },
  {
    id: "arvr-eyepiece",
    label: "AR/VR Eyepiece",
    domain: "Display",
    description: "Wide-angle eyepiece target with stricter distortion and field quality checks.",
    requirements: {
      spotMaxUm: 12,
      mtfMinLpMm: 25,
      mtfProxyMin: 0.32,
      illuminationMin: 0.5,
      distortionMaxPct: 1,
      craMaxDeg: 35,
      toleranceMinPct: 80,
      snrMinDb: 24
    }
  },
  {
    id: "endoscope",
    label: "Endoscope",
    domain: "Medical",
    description: "Compact optical-system target favoring high throughput and close-range image quality.",
    requirements: {
      spotMaxUm: 5,
      mtfMinLpMm: 60,
      mtfProxyMin: 0.45,
      illuminationMin: 0.65,
      distortionMaxPct: 8,
      craMaxDeg: 28,
      toleranceMinPct: 84,
      snrMinDb: 30
    }
  }
];

export const requirementDefinitions: RequirementDefinition[] = [
  {
    metricKey: "spot",
    targetKey: "spotMaxUm",
    label: "Spot RMS",
    direction: "max",
    unit: "um",
    min: 0.1,
    max: 200,
    step: 0.1,
    decimals: 1
  },
  {
    metricKey: "mtf",
    targetKey: "mtfMinLpMm",
    label: "MTF50",
    direction: "min",
    unit: "lp/mm",
    min: 0.1,
    max: 300,
    step: 0.5,
    decimals: 1
  },
  {
    metricKey: "illumination",
    targetKey: "illuminationMin",
    label: "Rel. Illum.",
    direction: "min",
    min: 0.05,
    max: 1,
    step: 0.01,
    decimals: 2
  },
  {
    metricKey: "distortion",
    targetKey: "distortionMaxPct",
    label: "Distortion",
    direction: "max",
    unit: "%",
    min: 0.01,
    max: 25,
    step: 0.05,
    decimals: 2
  },
  {
    metricKey: "cra",
    targetKey: "craMaxDeg",
    label: "CRA",
    direction: "max",
    unit: "deg",
    min: 0.1,
    max: 90,
    step: 0.1,
    decimals: 1
  },
  {
    metricKey: "yield",
    targetKey: "toleranceMinPct",
    label: "Tolerance",
    direction: "min",
    unit: "%",
    min: 0,
    max: 100,
    step: 0.5,
    decimals: 1
  },
  {
    metricKey: "snr",
    targetKey: "snrMinDb",
    label: "Corner SNR",
    direction: "min",
    unit: "dB",
    min: 0,
    max: 80,
    step: 0.5,
    decimals: 1
  }
];

const riskMetricMap = new Map<string, string>([
  ["Worst Field", "illumination"],
  ["Surface Sensitivity", "yield"],
  ["Spot Quality", "spot"],
  ["Sensor Coupling", "cra"],
  ["Scene Risk", "snr"]
]);

const profileMap = new Map(requirementProfiles.map((profile) => [profile.id, profile]));

export function requirementsForProfile(profileId: RequirementProfileId): ProjectRequirements {
  return { ...(profileMap.get(profileId) ?? profileMap.get(defaultRequirementProfile)!).requirements };
}

export function profileLabel(profileId: RequirementProfileSelection) {
  if (profileId === "custom") return "Custom";
  return profileMap.get(profileId)?.label ?? "Reference";
}

export function profileDescription(profileId: RequirementProfileSelection, baseProfileId: RequirementProfileId = defaultRequirementProfile) {
  if (profileId === "custom") return `Edited from ${profileLabel(baseProfileId)}`;
  return profileMap.get(profileId)?.description ?? "";
}

export function sanitizeRequirementTarget(key: ProjectRequirementKey, rawValue: number, requirements: ProjectRequirements) {
  const definition = requirementDefinitions.find((item) => item.targetKey === key);
  if (!definition || !Number.isFinite(rawValue)) return requirements[key];
  const bounded = Math.min(definition.max ?? Number.POSITIVE_INFINITY, Math.max(definition.min, rawValue));
  const scale = 1 / definition.step;
  return Math.round(bounded * scale) / scale;
}

export function applyProjectRequirements(metrics: CockpitMetric[], requirements: ProjectRequirements): CockpitMetric[] {
  return metrics.map((metric) => {
    const definition = definitionForMetric(metric);
    if (!definition) return metric;
    const target = requirements[definition.targetKey];
    const targetLabel = targetString(definition, target);
    const value = Number(metric.value);
    if (!Number.isFinite(value)) {
      return {
        ...metric,
        target: targetLabel,
        status: metric.status === "pass" ? "warn" : metric.status,
        note: appendRequirementNote(metric.note, targetLabel, "The metric is not numerically available, so this requirement is not proven.")
      };
    }
    return {
      ...metric,
      target: targetLabel,
      status: statusFromRequirement(value, target, definition.direction),
      score: scoreFromRequirement(value, target, definition.direction),
      note: appendRequirementNote(metric.note, targetLabel)
    };
  });
}

export function applyRequirementRiskStatuses(risks: CockpitRisk[], metrics: CockpitMetric[]): CockpitRisk[] {
  const metricByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  return risks.map((risk) => {
    const metricKey = riskMetricMap.get(risk.label);
    const metric = metricKey ? metricByKey.get(metricKey) : null;
    return metric ? { ...risk, status: metric.status } : risk;
  });
}

export function definitionForMetric(metric: CockpitMetric): RequirementDefinition | null {
  if (metric.key === "mtf" && metric.unit !== "lp/mm") {
    return {
      metricKey: "mtf",
      targetKey: "mtfProxyMin",
      label: "MTF50 Proxy",
      direction: "min",
      min: 0.01,
      max: 1,
      step: 0.01,
      decimals: 2
    };
  }
  return requirementDefinitions.find((item) => item.metricKey === metric.key) ?? null;
}

export function targetString(definition: RequirementDefinition, target: number) {
  const comparator = definition.direction === "min" ? ">=" : "<=";
  const value = formatTargetValue(target, definition.decimals);
  return `${comparator} ${value}${definition.unit ? ` ${definition.unit}` : ""}`;
}

function statusFromRequirement(value: number, target: number, direction: RequirementDirection): KpiStatus {
  if (direction === "min") {
    if (value >= target) return "pass";
    if (value >= target * 0.8) return "warn";
    return "fail";
  }
  if (value <= target) return "pass";
  if (value <= target * 1.25) return "warn";
  return "fail";
}

function scoreFromRequirement(value: number, target: number, direction: RequirementDirection) {
  if (target <= 0) return value <= target ? 1 : 0;
  if (direction === "min") return clamp(value / target, 0, 1);
  if (value <= target) return 1;
  return clamp(target / value, 0, 1);
}

function formatTargetValue(value: number, decimals: number) {
  return value.toFixed(decimals).replace(/0+$/, "").replace(/\.$/, "");
}

function appendRequirementNote(note: string | null | undefined, target: string, extra?: string) {
  const suffix = `Project requirement target: ${target}.${extra ? ` ${extra}` : ""}`;
  return note ? `${note} ${suffix}` : suffix;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
