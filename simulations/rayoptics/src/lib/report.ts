import type {
  AnalysisSummary,
  CockpitMetric,
  CockpitRisk,
  ExampleCheckResponse,
  OpticalModel,
  QuickOptimizeObjective,
  QuickOptimizeResult,
  QuickOptimizeWeights,
  ToleranceSweepResult,
  ToleranceSweepScope,
  VariantSnapshot
} from "./types";
import type { ProjectRequirements, RequirementProfileId, RequirementProfileSelection } from "./requirements";

export interface ReportSectionSelection {
  model: boolean;
  requirements: boolean;
  firstOrder: boolean;
  sensor: boolean;
  counts: boolean;
  metrics: boolean;
  risks: boolean;
  diagnostics: boolean;
  lensData: boolean;
  compatibility: boolean;
  quickOptimize: boolean;
  optimizationSettings: boolean;
  toleranceSweep: boolean;
  variants: boolean;
}

export interface ReportWorkflowState {
  quickOptimizeSettings?: {
    objective: QuickOptimizeObjective;
    iterations: number;
    maxEvaluations: number;
    stepScale: number;
    operandWeights: QuickOptimizeWeights;
  };
  toleranceSweep?: {
    settings: {
      scope: ToleranceSweepScope;
      perturbationPct: number;
      maxSurfaces: number;
    };
    result: ToleranceSweepResult | null;
  };
}

export const defaultReportSections: ReportSectionSelection = {
  model: true,
  requirements: true,
  firstOrder: true,
  sensor: true,
  counts: true,
  metrics: true,
  risks: true,
  diagnostics: true,
  lensData: true,
  compatibility: true,
  quickOptimize: true,
  optimizationSettings: true,
  toleranceSweep: true,
  variants: true
};

export function buildReportSnapshot(
  model: OpticalModel | null,
  summary: AnalysisSummary | null,
  metrics: CockpitMetric[],
  risks: CockpitRisk[],
  warnings: string[],
  errors: string[],
  requirements: ProjectRequirements,
  requirementProfile: RequirementProfileSelection,
  requirementProfileBase: RequirementProfileId,
  autosave: { path: string | null; savedAt: string | null; draftCount: number | null },
  compatibility: ExampleCheckResponse | null = null,
  quickOptimize: QuickOptimizeResult | null = null,
  sections: ReportSectionSelection = defaultReportSections,
  variants: VariantSnapshot[] = [],
  workflowState: ReportWorkflowState = {}
) {
  return {
    generatedAt: new Date().toISOString(),
    includedSections: sections,
    model: model && sections.model
      ? {
          id: model.id,
          name: model.name || "Optical Design Cockpit",
          filename: model.filename,
          surfaceCount: model.surfaces.length,
          stopSurface: model.stopSurface,
          updateMode: model.updateMode,
          autosave
      }
      : null,
    requirements: sections.requirements
      ? {
          profile: requirementProfile,
          baseProfile: requirementProfileBase,
          targets: requirements,
          ...requirements
        }
      : null,
    firstOrder: sections.firstOrder ? summary?.firstOrder ?? null : null,
    sensor: sections.sensor ? summary?.sensor ?? null : null,
    counts: sections.counts ? summary?.counts ?? null : null,
    metrics: sections.metrics ? metrics : [],
    risks: sections.risks ? risks : [],
    diagnostics: sections.diagnostics
      ? {
          warnings,
          errors
        }
      : null,
    compatibility: sections.compatibility ? compatibility : null,
    quickOptimize: sections.quickOptimize ? quickOptimize : null,
    optimizationSettings: sections.optimizationSettings ? workflowState.quickOptimizeSettings ?? null : null,
    toleranceSweep: sections.toleranceSweep ? workflowState.toleranceSweep ?? null : null,
    variants: sections.variants ? variants : [],
    lensData: model && sections.lensData
      ? {
          surfaces: model.surfaces,
          system: model.system
        }
      : null
  };
}

export function exportReportSnapshot(snapshot: ReturnType<typeof buildReportSnapshot>) {
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${safeFilename(snapshot.model?.name ?? "rayoptics")}-analysis-snapshot.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function safeFilename(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "rayoptics";
}
