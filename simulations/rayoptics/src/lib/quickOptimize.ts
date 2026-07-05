import type { ProjectRequirementKey, ProjectRequirements } from "./requirements";
import type { QuickOptimizeObjective, QuickOptimizeOperandKey, QuickOptimizeTargets, QuickOptimizeWeights } from "./types";

export interface QuickOptimizeOperandDefinition {
  key: QuickOptimizeOperandKey;
  metricKey: string | null;
  label: string;
  targetKey: ProjectRequirementKey | null;
  targetLabel: string;
}

export const quickOptimizeObjectiveOptions: Array<{ value: QuickOptimizeObjective; label: string }> = [
  { value: "balanced", label: "Balanced" },
  { value: "spot", label: "Spot RMS" },
  { value: "distortion", label: "Distortion" },
  { value: "throughput", label: "Throughput" },
  { value: "cra", label: "CRA" },
  { value: "custom", label: "Custom Weights" }
];

export const quickOptimizeOperandDefinitions: QuickOptimizeOperandDefinition[] = [
  { key: "spot", metricKey: "spot", label: "Spot RMS", targetKey: "spotMaxUm", targetLabel: "max spot" },
  { key: "distortion", metricKey: "distortion", label: "Distortion", targetKey: "distortionMaxPct", targetLabel: "max distortion" },
  { key: "throughput", metricKey: "illumination", label: "Throughput", targetKey: "illuminationMin", targetLabel: "min throughput" },
  { key: "cra", metricKey: "cra", label: "CRA", targetKey: "craMaxDeg", targetLabel: "max CRA" },
  { key: "first_order", metricKey: null, label: "First Order", targetKey: null, targetLabel: "valid EFL/F/#" }
];

export const quickOptimizeWeightPresets: Record<QuickOptimizeObjective, QuickOptimizeWeights> = {
  balanced: { spot: 0.42, distortion: 0.2, throughput: 0.16, cra: 0.12, first_order: 0.1 },
  spot: { spot: 0.62, distortion: 0.12, throughput: 0.08, cra: 0.08, first_order: 0.1 },
  distortion: { spot: 0.2, distortion: 0.48, throughput: 0.1, cra: 0.1, first_order: 0.12 },
  throughput: { spot: 0.18, distortion: 0.1, throughput: 0.52, cra: 0.1, first_order: 0.1 },
  cra: { spot: 0.18, distortion: 0.1, throughput: 0.1, cra: 0.5, first_order: 0.12 },
  custom: { spot: 0.42, distortion: 0.2, throughput: 0.16, cra: 0.12, first_order: 0.1 }
};

export function quickOptimizeObjectiveLabel(objective: QuickOptimizeObjective) {
  return quickOptimizeObjectiveOptions.find((item) => item.value === objective)?.label ?? "Balanced";
}

export function weightsForQuickOptimizeObjective(objective: QuickOptimizeObjective): QuickOptimizeWeights {
  return { ...quickOptimizeWeightPresets[objective] };
}

export function sanitizeQuickOptimizeWeights(value: unknown, fallback: QuickOptimizeWeights = quickOptimizeWeightPresets.balanced): QuickOptimizeWeights {
  const source = value && typeof value === "object" ? (value as Partial<Record<QuickOptimizeOperandKey, unknown>>) : {};
  return quickOptimizeOperandDefinitions.reduce((weights, operand) => {
    const numeric = Number(source[operand.key]);
    weights[operand.key] = Number.isFinite(numeric) ? Math.min(1, Math.max(0, numeric)) : fallback[operand.key];
    return weights;
  }, {} as QuickOptimizeWeights);
}

export function quickOptimizeWeightTotal(weights: QuickOptimizeWeights) {
  return quickOptimizeOperandDefinitions.reduce((sum, operand) => sum + Math.max(0, weights[operand.key] || 0), 0);
}

export function normalizedQuickOptimizePercent(weights: QuickOptimizeWeights, key: QuickOptimizeOperandKey) {
  const total = quickOptimizeWeightTotal(weights);
  return total > 0 ? (Math.max(0, weights[key] || 0) / total) * 100 : 0;
}

export function quickOptimizeTargetsFromRequirements(requirements: ProjectRequirements): QuickOptimizeTargets {
  return {
    spot: requirements.spotMaxUm,
    distortion: requirements.distortionMaxPct,
    throughput: requirements.illuminationMin,
    cra: requirements.craMaxDeg
  };
}
