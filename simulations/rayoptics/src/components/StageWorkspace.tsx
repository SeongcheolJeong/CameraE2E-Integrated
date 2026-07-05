import {
  AlertTriangle,
  Aperture,
  BarChart3,
  Camera,
  Check,
  ChevronRight,
  CircleDot,
  Download,
  Eye,
  FileText,
  Gauge,
  GitCompare,
  Layers3,
  LineChart,
  Microscope,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  Trash2,
  Waves
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { buildReportSnapshot, defaultReportSections, exportReportSnapshot, type ReportSectionSelection } from "../lib/report";
import {
  definitionForMetric,
  profileDescription,
  profileLabel,
  requirementDefinitions,
  requirementProfiles,
  targetString,
  type ProjectRequirementKey,
  type ProjectRequirements,
  type RequirementProfileId,
  type RequirementProfileSelection
} from "../lib/requirements";
import { useWorkbench } from "../lib/store";
import type { CockpitMetric, CockpitRisk } from "../lib/uxMetrics";
import {
  normalizedQuickOptimizePercent,
  quickOptimizeObjectiveOptions,
  quickOptimizeOperandDefinitions,
  quickOptimizeObjectiveLabel as quickOptimizeLabel,
  quickOptimizeWeightTotal
} from "../lib/quickOptimize";
import type {
  AnalysisSummary,
  ExampleCheckResponse,
  ExampleCheckResult,
  KpiStatus,
  MetricSource,
  NewModelSpec,
  OpticalModel,
  QuickOptimizeObjective,
  QuickOptimizeOperandKey,
  QuickOptimizeResult,
  QuickOptimizeWeights,
  SensorAssumptions,
  SensorPatch,
  ToleranceSweepResult,
  ToleranceSweepScope,
  VariantSnapshot,
  WorkflowStage
} from "../lib/types";
import { deriveMetrics, statusLabel } from "../lib/uxMetrics";
import { AnalysisPanel } from "./AnalysisPanel";
import { LensDataEditor } from "./LensDataEditor";
import { OpticalLayout } from "./OpticalLayout";
import { workflowStages } from "./WorkflowRail";

export function StageWorkspace({
  activeStage,
  onStageChange
}: {
  activeStage: WorkflowStage;
  onStageChange: (stage: WorkflowStage) => void;
}) {
  const model = useWorkbench((state) => state.model);
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);
  const isBusy = useWorkbench((state) => state.isBusy);
  const refresh = useWorkbench((state) => state.refresh);
  const summary = useWorkbench((state) => state.summary);
  const requirements = useWorkbench((state) => state.requirements);
  const requirementProfile = useWorkbench((state) => state.requirementProfile);
  const requirementProfileBase = useWorkbench((state) => state.requirementProfileBase);
  const setRequirementTarget = useWorkbench((state) => state.setRequirementTarget);
  const setRequirementProfile = useWorkbench((state) => state.setRequirementProfile);
  const resetRequirements = useWorkbench((state) => state.resetRequirements);
  const compatibility = useWorkbench((state) => state.compatibility);
  const quickOptimizeResult = useWorkbench((state) => state.quickOptimizeResult);
  const quickOptimizeObjective = useWorkbench((state) => state.quickOptimizeObjective);
  const quickOptimizeIterations = useWorkbench((state) => state.quickOptimizeIterations);
  const quickOptimizeMaxEvaluations = useWorkbench((state) => state.quickOptimizeMaxEvaluations);
  const quickOptimizeStepScale = useWorkbench((state) => state.quickOptimizeStepScale);
  const quickOptimizeWeights = useWorkbench((state) => state.quickOptimizeWeights);
  const toleranceSweepResult = useWorkbench((state) => state.toleranceSweepResult);
  const toleranceSweepScope = useWorkbench((state) => state.toleranceSweepScope);
  const toleranceSweepPerturbationPct = useWorkbench((state) => state.toleranceSweepPerturbationPct);
  const toleranceSweepMaxSurfaces = useWorkbench((state) => state.toleranceSweepMaxSurfaces);
  const variants = useWorkbench((state) => state.variants);
  const isCheckingExamples = useWorkbench((state) => state.isCheckingExamples);
  const runExampleChecks = useWorkbench((state) => state.runExampleChecks);
  const autosavePath = useWorkbench((state) => state.autosavePath);
  const autosaveSavedAt = useWorkbench((state) => state.autosaveSavedAt);
  const draftCount = useWorkbench((state) => state.draftCount);
  const metrics = useMemo(() => deriveMetrics(model, warnings, errors, summary, requirements), [errors, model, requirements, summary, warnings]);
  const stage = workflowStages.find((item) => item.id === activeStage) ?? workflowStages[1];

  return (
    <section className="stageWorkspace">
      <header className="stageHeader">
        <div className="stageTitle">
          <span className="eyebrow">Optical Design Cockpit</span>
          <h1>{stage.label}</h1>
        </div>
        <div className="stageTabs">
          {workflowStages.map((item) => (
            <button key={item.id} className={item.id === activeStage ? "active" : ""} onClick={() => onStageChange(item.id)}>
              {item.short}
            </button>
          ))}
        </div>
        <button className="runButton" onClick={() => void refresh()} disabled={isBusy}>
          {isBusy ? <RefreshCw size={16} /> : <Play size={16} />}
          {isBusy ? "Running" : "Run Trace"}
        </button>
      </header>
      <div className="stageBody">
        {activeStage === "project" ? (
          <ProjectStage
            metrics={metrics.metrics}
            risks={metrics.risks}
            requirements={requirements}
            requirementProfile={requirementProfile}
            requirementProfileBase={requirementProfileBase}
            onRequirementChange={setRequirementTarget}
            onRequirementProfileChange={setRequirementProfile}
            onRequirementReset={resetRequirements}
            compatibility={compatibility}
            isCheckingExamples={isCheckingExamples}
            onRunExampleChecks={runExampleChecks}
          />
        ) : null}
        {activeStage === "optical-model" ? <OpticalModelStage /> : null}
        {activeStage === "analysis" ? <AnalysisDiagnosticsStage model={model} metrics={metrics.metrics} risks={metrics.risks} /> : null}
        {activeStage === "optimization" ? <OptimizationToleranceStage mode="optimization" metrics={metrics.metrics} risks={metrics.risks} /> : null}
        {activeStage === "tolerance" ? <OptimizationToleranceStage mode="tolerance" metrics={metrics.metrics} risks={metrics.risks} /> : null}
        {activeStage === "sensor" ? <SensorStage metrics={metrics.metrics} risks={metrics.risks} sensor={summary?.sensor} /> : null}
        {activeStage === "scene" ? <SceneStage metrics={metrics.metrics} risks={metrics.risks} /> : null}
        {activeStage === "compare" ? <CompareStage metrics={metrics.metrics} risks={metrics.risks} /> : null}
        {activeStage === "report" ? (
          <ReportStage
            model={model}
            summary={summary}
            metrics={metrics.metrics}
            risks={metrics.risks}
            warnings={warnings}
            errors={errors}
            requirements={requirements}
            requirementProfile={requirementProfile}
            requirementProfileBase={requirementProfileBase}
            autosavePath={autosavePath}
            autosaveSavedAt={autosaveSavedAt}
            draftCount={draftCount}
            compatibility={compatibility}
            quickOptimizeResult={quickOptimizeResult}
            quickOptimizeObjective={quickOptimizeObjective}
            quickOptimizeIterations={quickOptimizeIterations}
            quickOptimizeMaxEvaluations={quickOptimizeMaxEvaluations}
            quickOptimizeStepScale={quickOptimizeStepScale}
            quickOptimizeWeights={quickOptimizeWeights}
            toleranceSweepResult={toleranceSweepResult}
            toleranceSweepScope={toleranceSweepScope}
            toleranceSweepPerturbationPct={toleranceSweepPerturbationPct}
            toleranceSweepMaxSurfaces={toleranceSweepMaxSurfaces}
            variants={variants}
          />
        ) : null}
      </div>
    </section>
  );
}

function ProjectStage({
  metrics,
  risks,
  requirements,
  requirementProfile,
  requirementProfileBase,
  onRequirementChange,
  onRequirementProfileChange,
  onRequirementReset,
  compatibility,
  isCheckingExamples,
  onRunExampleChecks
}: {
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
  requirements: ProjectRequirements;
  requirementProfile: RequirementProfileSelection;
  requirementProfileBase: RequirementProfileId;
  onRequirementChange: (key: ProjectRequirementKey, value: number) => void;
  onRequirementProfileChange: (profile: RequirementProfileId) => void;
  onRequirementReset: () => void;
  compatibility: ExampleCheckResponse | null;
  isCheckingExamples: boolean;
  onRunExampleChecks: () => Promise<void>;
}) {
  const newModel = useWorkbench((state) => state.newModel);
  const isBusy = useWorkbench((state) => state.isBusy);
  const model = useWorkbench((state) => state.model);
  const starterSpecs: Array<{ name: string; profile: RequirementProfileId; spec: NewModelSpec; note: string }> = [
    { name: "50 mm Reference", profile: "reference", spec: { efl: 50, epd: 12.5, fov: 20 }, note: "baseline singlet" },
    { name: "Mobile Wide Starter", profile: "mobile-wide", spec: { efl: 4.2, epd: 1.8, fov: 72 }, note: "short EFL / wide field" },
    { name: "Automotive Wide Starter", profile: "automotive-wide", spec: { efl: 6, epd: 4, fov: 95 }, note: "wide FOV starter" },
    { name: "LiDAR Receiver Starter", profile: "lidar-receiver", spec: { efl: 25, epd: 18, fov: 8 }, note: "large pupil starter" },
    { name: "AR/VR Eyepiece Starter", profile: "arvr-eyepiece", spec: { efl: 22, epd: 7, fov: 60 }, note: "wide-angle starter" },
    { name: "Endoscope Starter", profile: "endoscope", spec: { efl: 2.8, epd: 1.2, fov: 80 }, note: "compact starter" }
  ];

  return (
    <div className="projectStage">
      <section className="workPanel requirementsPanel">
        <PanelHeader icon={Target} title="Requirements" action="Guided" />
        <RequirementTargetEditor
          metrics={metrics}
          requirements={requirements}
          requirementProfile={requirementProfile}
          requirementProfileBase={requirementProfileBase}
          onRequirementChange={onRequirementChange}
          onRequirementProfileChange={onRequirementProfileChange}
          onRequirementReset={onRequirementReset}
        />
        <div className="requirementMatrix">
          {metrics.slice(1).map((metric) => (
            <MetricRequirement key={metric.key} metric={metric} />
          ))}
        </div>
      </section>
      <section className="workPanel templatePanel">
        <PanelHeader icon={Layers3} title="Starter Specs" action="Local" />
        <div className="templateGrid">
          {starterSpecs.map((template) => (
            <button
              key={template.name}
              onClick={() => {
                onRequirementProfileChange(template.profile);
                void newModel(template.spec);
              }}
              disabled={isBusy}
              title={`${template.spec.efl} mm EFL / ${template.spec.epd} mm EPD / ${template.spec.fov} deg field · ${profileLabel(template.profile)} requirements`}
            >
              <Aperture size={18} />
              <strong>{template.name}</strong>
              <span>
                EFL {formatCompact(template.spec.efl)} · EPD {formatCompact(template.spec.epd)} · FOV {formatCompact(template.spec.fov)}
              </span>
              <em>
                {template.note} · {profileLabel(template.profile)}
              </em>
            </button>
          ))}
        </div>
      </section>
      <section className="workPanel compatibilityPanel">
        <PanelHeader icon={ShieldCheck} title="Example QA" action={compatibility ? `${compatibility.passed}/${compatibility.total} pass` : "Not Run"} />
        <CompatibilityPanel compatibility={compatibility} isChecking={isCheckingExamples} onRun={onRunExampleChecks} />
      </section>
      <section className="workPanel modelSnapshot">
        <PanelHeader icon={Settings2} title="Current Configuration" action={model?.filename ? "Loaded" : "Unsaved"} />
        <div className="snapshotGrid">
          <Snapshot label="Model" value={model?.name || "New optical model"} />
          <Snapshot label="Surfaces" value={String(model?.surfaces.length ?? 0)} />
          <Snapshot label="Stop" value={model?.stopSurface === null || model?.stopSurface === undefined ? "n/a" : `S${model.stopSurface}`} />
          <Snapshot label="Fields" value={String(model?.system.field.fields.length ?? 0)} />
          <Snapshot label="Waves" value={String(model?.system.wavelengths.values.length ?? 0)} />
          <Snapshot label="Units" value="mm / nm" />
        </div>
      </section>
      <section className="workPanel riskPanel">
        <PanelHeader icon={AlertTriangle} title="Worst Cases" action="Live" />
        <div className="riskStack">
          {risks.map((risk) => (
            <RiskRow key={risk.label} risk={risk} />
          ))}
        </div>
      </section>
    </div>
  );
}

function OpticalModelStage() {
  return (
    <div className="modelStage">
      <section className="workPanel ldePanel">
        <PanelHeader icon={Aperture} title="Lens Data Editor" action="Sequential" />
        <LensDataEditor />
      </section>
      <section className="workPanel layoutPanel">
        <PanelHeader icon={Eye} title="Layout / Analysis" action="RayOptics" />
        <AnalysisPanel />
      </section>
    </div>
  );
}

function AnalysisDiagnosticsStage({
  model,
  metrics,
  risks
}: {
  model: OpticalModel | null;
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
}) {
  const selectedSurface = useWorkbench((state) => state.selectedSurface);
  const worstFieldRisk = risks.find((risk) => risk.label === "Worst Field");
  const surfaceRisk = risks.find((risk) => risk.label === "Surface Sensitivity");
  const sensorRisk = risks.find((risk) => risk.label === "Sensor Coupling") ?? risks.find((risk) => risk.label === "Scene Risk");
  const diagnosticMetrics = pickMetrics(metrics, ["spot", "mtf", "illumination", "distortion", "cra", "yield"]);
  const causeNodes: Array<[string, string, KpiStatus]> = [
    ["Field", worstFieldRisk?.value ?? fieldScopeLabel(model), metricStatus(metrics, "illumination")],
    ["Wavelength", referenceWavelengthLabel(model), metricStatus(metrics, "mtf")],
    ["Surface", selectedSurfaceLabel(model, selectedSurface, surfaceRisk), metricStatus(metrics, "yield")],
    ["Sensor Impact", sensorRisk?.value ?? metricValueLabel(metrics, "snr"), metricStatus(metrics, "snr")]
  ];
  const rootCauseStatus = worstStatus(causeNodes.map(([, , status]) => status));

  return (
    <div className="analysisDiagnosticsStage">
      <section className="workPanel worstCases">
        <PanelHeader icon={AlertTriangle} title="Worst Cases" action="Ranked" />
        <div className="riskStack large">
          {risks.map((risk, index) => (
            <RiskRow key={risk.label} risk={risk} rank={index + 1} />
          ))}
        </div>
      </section>
      <section className="workPanel liveAnalysis">
        <PanelHeader icon={LineChart} title="RayOptics Analysis" action="Live" />
        <AnalysisPanel />
      </section>
      <section className="workPanel diagnosticMaps">
        <PanelHeader icon={BarChart3} title="Evidence" action="Source-coded" />
        <div className="evidenceGrid">
          {diagnosticMetrics.map((metric) => (
            <MetricRequirement key={metric.key} metric={metric} />
          ))}
        </div>
      </section>
      <section className="workPanel rootCause">
        <PanelHeader icon={CircleDot} title="Root Cause" action={statusLabel(rootCauseStatus)} />
        <div className="causeFlow">
          {causeNodes.map(([label, value, status], index) => (
            <div key={`${label}-${value}`} className={`causeNode ${status}`}>
              <span>{label}</span>
              <strong>{value}</strong>
              {index < 3 ? <ChevronRight size={16} /> : null}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function OptimizationToleranceStage({
  mode,
  metrics,
  risks
}: {
  mode: "optimization" | "tolerance";
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
}) {
  const isBusy = useWorkbench((state) => state.isBusy);
  const quickOptimize = useWorkbench((state) => state.quickOptimize);
  const quickOptimizeResult = useWorkbench((state) => state.quickOptimizeResult);
  const quickOptimizeObjective = useWorkbench((state) => state.quickOptimizeObjective);
  const quickOptimizeIterations = useWorkbench((state) => state.quickOptimizeIterations);
  const quickOptimizeMaxEvaluations = useWorkbench((state) => state.quickOptimizeMaxEvaluations);
  const quickOptimizeStepScale = useWorkbench((state) => state.quickOptimizeStepScale);
  const quickOptimizeWeights = useWorkbench((state) => state.quickOptimizeWeights);
  const setQuickOptimizeObjective = useWorkbench((state) => state.setQuickOptimizeObjective);
  const setQuickOptimizeIterations = useWorkbench((state) => state.setQuickOptimizeIterations);
  const setQuickOptimizeMaxEvaluations = useWorkbench((state) => state.setQuickOptimizeMaxEvaluations);
  const setQuickOptimizeStepScale = useWorkbench((state) => state.setQuickOptimizeStepScale);
  const setQuickOptimizeWeight = useWorkbench((state) => state.setQuickOptimizeWeight);
  const resetQuickOptimizeWeights = useWorkbench((state) => state.resetQuickOptimizeWeights);
  const toleranceSweepResult = useWorkbench((state) => state.toleranceSweepResult);
  const toleranceSweepScope = useWorkbench((state) => state.toleranceSweepScope);
  const toleranceSweepPerturbationPct = useWorkbench((state) => state.toleranceSweepPerturbationPct);
  const toleranceSweepMaxSurfaces = useWorkbench((state) => state.toleranceSweepMaxSurfaces);
  const setToleranceSweepScope = useWorkbench((state) => state.setToleranceSweepScope);
  const setToleranceSweepPerturbationPct = useWorkbench((state) => state.setToleranceSweepPerturbationPct);
  const setToleranceSweepMaxSurfaces = useWorkbench((state) => state.setToleranceSweepMaxSurfaces);
  const runToleranceSweep = useWorkbench((state) => state.runToleranceSweep);
  const requirements = useWorkbench((state) => state.requirements);
  const setRequirementTarget = useWorkbench((state) => state.setRequirementTarget);
  const model = useWorkbench((state) => state.model);
  const toleranceMetric = metrics.find((metric) => metric.key === "yield");
  const toleranceValue = Number(toleranceMetric?.value ?? 86.7);
  const toleranceGaugeValue = Number.isFinite(toleranceValue) ? toleranceValue : 0;
  const trackedMetrics = pickMetrics(metrics, ["spot", "mtf", "distortion", "cra", "yield"]);
  const surfaceRisk = risks.find((risk) => risk.label === "Surface Sensitivity") ?? risks[0];
  const stressAction = toleranceMetric?.source === "computed" ? "Computed" : toleranceMetric?.source === "proxy" ? "Proxy" : "Unavailable";
  const sweepAction = toleranceSweepResult ? statusLabel(toleranceSweepResult.status) : stressAction;
  const variableRows = model?.surfaces.filter((surface) => surface.variable.trim()) ?? [];
  const variableCount = variableRows.reduce((count, surface) => count + surface.variable.split(",").filter(Boolean).length, 0);
  const variableSummary = variableRows.length
    ? `${variableCount} flags on ${variableRows.map((surface) => `S${surface.index}:${surface.variable}`).join(" ")}`
    : "No active variables";
  const meritWeightTotal = quickOptimizeWeightTotal(quickOptimizeWeights);

  return (
    <div className="optimizationStage">
      <section className="workPanel goalWeights">
        <PanelHeader icon={SlidersHorizontal} title={mode === "optimization" ? "Optimization Setup" : "Stress Sweep"} action={mode === "optimization" ? (variableCount ? "Variables Marked" : "Not Configured") : stressAction} />
        {mode === "optimization" ? (
          <>
            <CapabilityRow label="Merit Function" value={variableCount ? `${objectiveLabel(quickOptimizeObjective)} metric search` : "Not configured"} status={variableCount ? "warn" : "fail"} source={variableCount ? "model" : "unsupported"} />
            <CapabilityRow label="Variables" value={variableSummary} status={variableCount ? "warn" : "fail"} source={variableCount ? "model" : "unsupported"} />
            <CapabilityRow label="Solver" value="Bounded coordinate search; no DLS constraints" status="warn" source="unsupported" />
            <CapabilityRow label="Operand Weights" value={meritWeightTotal > 0 ? `${Math.round(meritWeightTotal * 100)} raw points` : "All zero"} status={meritWeightTotal > 0 ? "warn" : "fail"} source="model" />
            <div className="quickOptimizeControls">
              <label>
                <span>Objective</span>
                <select value={quickOptimizeObjective} onChange={(event) => setQuickOptimizeObjective(event.target.value as QuickOptimizeObjective)} disabled={isBusy}>
                  {quickOptimizeObjectiveOptions.map((objective) => (
                    <option key={objective.value} value={objective.value}>
                      {objective.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Iterations</span>
                <input type="number" min={1} max={5} value={quickOptimizeIterations} onChange={(event) => setQuickOptimizeIterations(Number(event.target.value))} disabled={isBusy} />
              </label>
              <label>
                <span>Evaluations</span>
                <input type="number" min={4} max={160} value={quickOptimizeMaxEvaluations} onChange={(event) => setQuickOptimizeMaxEvaluations(Number(event.target.value))} disabled={isBusy} />
              </label>
              <label>
                <span>Step</span>
                <input type="number" min={0.1} max={5} step={0.1} value={quickOptimizeStepScale} onChange={(event) => setQuickOptimizeStepScale(Number(event.target.value))} disabled={isBusy} />
              </label>
            </div>
            <MeritOperandEditor
              metrics={metrics}
              requirements={requirements}
              weights={quickOptimizeWeights}
              isBusy={isBusy}
              onWeightChange={setQuickOptimizeWeight}
              onTargetChange={setRequirementTarget}
              onReset={resetQuickOptimizeWeights}
            />
            <button className="primaryWide" onClick={() => void quickOptimize()} disabled={!variableCount || isBusy}>
              {isBusy ? <RefreshCw size={16} /> : <Play size={16} />}
              {isBusy ? "Optimizing" : variableCount ? "Run Quick Optimize" : "Mark Variables First"}
            </button>
          </>
        ) : (
          <>
            {toleranceMetric ? <MetricRequirement metric={toleranceMetric} /> : null}
            <CapabilityRow label="Sweep Type" value="Deterministic perturbation, not Monte Carlo" status="warn" source="computed" />
            <CapabilityRow label="Worst Case" value={toleranceSweepResult?.worstCase ?? surfaceRisk?.detail ?? "n/a"} status={toleranceSweepResult?.status ?? surfaceRisk?.status ?? "warn"} source="computed" />
            <div className="quickOptimizeControls">
              <label>
                <span>Scope</span>
                <select value={toleranceSweepScope} onChange={(event) => setToleranceSweepScope(event.target.value as ToleranceSweepScope)} disabled={isBusy}>
                  <option value="powered">Powered Surfaces</option>
                  <option value="variables">Marked Variables</option>
                </select>
              </label>
              <label>
                <span>Perturb %</span>
                <input type="number" min={0.05} max={5} step={0.05} value={toleranceSweepPerturbationPct} onChange={(event) => setToleranceSweepPerturbationPct(Number(event.target.value))} disabled={isBusy} />
              </label>
              <label>
                <span>Max Items</span>
                <input type="number" min={1} max={20} value={toleranceSweepMaxSurfaces} onChange={(event) => setToleranceSweepMaxSurfaces(Number(event.target.value))} disabled={isBusy} />
              </label>
              <label>
                <span>Cases</span>
                <input type="text" value={`${toleranceSweepMaxSurfaces * 2} max`} readOnly />
              </label>
            </div>
            <button className="primaryWide" onClick={() => void runToleranceSweep()} disabled={isBusy || !model}>
              {isBusy ? <RefreshCw size={16} /> : <Play size={16} />}
              {isBusy ? "Sweeping" : "Run Quick Tolerance Sweep"}
            </button>
          </>
        )}
      </section>
      <section className="workPanel paretoPanel">
        <PanelHeader icon={GitCompare} title={mode === "optimization" ? "Variant Space" : "Stress Case Outcome"} action={mode === "optimization" ? "Current Only" : sweepAction} />
        {mode === "optimization" ? (
          quickOptimizeResult ? (
            <QuickOptimizeOutcome result={quickOptimizeResult} />
          ) : (
            <UnavailablePanel
              title="No Quick Optimize Run"
              rows={[
                ["Current Model", "Available"],
                ["Search Type", "Coordinate search"],
                ["Solver Output", variableCount ? "Ready" : "Mark variables first"]
              ]}
            />
          )
        ) : toleranceSweepResult ? (
          <ToleranceSweepOutcome result={toleranceSweepResult} />
        ) : (
          <StressOutcome metric={toleranceMetric} risk={surfaceRisk} />
        )}
      </section>
      <section className="workPanel toleranceYield">
        <PanelHeader icon={ShieldCheck} title="Tolerance Robustness" action={toleranceMetric ? statusLabel(toleranceMetric.status) : "PASS"} />
        <div className={`yieldGauge ${toleranceMetric?.status ?? "pass"}`}>
          <svg viewBox="0 0 120 70" aria-label="Tolerance robustness gauge">
            <path d="M15 60 A45 45 0 0 1 105 60" />
            <path d="M15 60 A45 45 0 0 1 105 60" style={{ strokeDasharray: `${Math.max(8, toleranceGaugeValue)} 100` }} />
          </svg>
          <strong>{Number.isFinite(toleranceValue) ? `${toleranceValue.toFixed(1)}%` : "n/a"}</strong>
          <span>{toleranceMetric?.target ?? "Target n/a"}</span>
        </div>
      </section>
      <section className="workPanel sensitivityLayout">
        <PanelHeader icon={Aperture} title="Surface Sensitivity" action="By Surface" />
        <div className="smallLayout">
          <OpticalLayout />
        </div>
      </section>
      <section className="workPanel variantDiff">
        <PanelHeader icon={BarChart3} title={mode === "optimization" ? "Optimization Metrics" : "Stress Metrics"} action="Current" />
        {mode === "tolerance" && toleranceSweepResult ? (
          <ToleranceSweepTable result={toleranceSweepResult} />
        ) : (
          <table className="diffTable metricDiffTable">
            <thead>
              <tr>
                <th>Metric</th>
                <td>Current</td>
                <td>Target</td>
                <td>Source</td>
              </tr>
            </thead>
            <tbody>
              {trackedMetrics.map((metric) => (
                <tr key={metric.key}>
                  <th>{metric.label}</th>
                  <td className={metric.status === "pass" ? "positive" : "negative"}>
                    {metric.value}
                    {metric.unit ? ` ${metric.unit}` : ""}
                  </td>
                  <td>{metric.target}</td>
                  <td>{metric.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="workPanel yieldKillers">
        <PanelHeader icon={AlertTriangle} title="Risk Drivers" action="Ranked" />
        <div className="riskStack compareRiskStack">
          {risks.slice(0, 5).map((risk, index) => (
            <RiskRow key={risk.label} risk={risk} rank={index + 1} />
          ))}
        </div>
      </section>
    </div>
  );
}

function MeritOperandEditor({
  metrics,
  requirements,
  weights,
  isBusy,
  onWeightChange,
  onTargetChange,
  onReset
}: {
  metrics: CockpitMetric[];
  requirements: ProjectRequirements;
  weights: Record<QuickOptimizeOperandKey, number>;
  isBusy: boolean;
  onWeightChange: (key: QuickOptimizeOperandKey, weight: number) => void;
  onTargetChange: (key: ProjectRequirementKey, value: number) => void;
  onReset: () => void;
}) {
  const metricsByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  const total = quickOptimizeWeightTotal(weights);

  return (
    <div className="meritOperandEditor">
      <header>
        <strong>Merit Operands</strong>
        <button onClick={onReset} disabled={isBusy}>
          Reset
        </button>
      </header>
      <div className="meritOperandRows">
        {quickOptimizeOperandDefinitions.map((operand) => {
          const metric = operand.metricKey ? metricsByKey.get(operand.metricKey) : null;
          const requirementDefinition = operand.targetKey ? requirementDefinitions.find((definition) => definition.targetKey === operand.targetKey) : null;
          const target = operand.targetKey ? requirements[operand.targetKey] : null;
          const weight = weights[operand.key] ?? 0;
          const normalized = normalizedQuickOptimizePercent(weights, operand.key);
          return (
            <div key={operand.key} className="meritOperandRow">
              <div className="meritOperandName">
                <strong>{operand.label}</strong>
                <span>{metric ? metricDisplay(metric) : "validity gate"}</span>
              </div>
              <label className="meritTarget">
                <span>{operand.targetLabel}</span>
                {requirementDefinition && operand.targetKey && target !== null ? (
                  <input
                    type="number"
                    min={requirementDefinition.min}
                    max={requirementDefinition.max}
                    step={requirementDefinition.step}
                    value={target}
                    onChange={(event) => onTargetChange(operand.targetKey as ProjectRequirementKey, Number(event.target.value))}
                    disabled={isBusy}
                    title={targetString(requirementDefinition, target)}
                  />
                ) : (
                  <b>required</b>
                )}
              </label>
              <label className="meritWeight">
                <span>Weight</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={weight}
                  onChange={(event) => onWeightChange(operand.key, Number(event.target.value))}
                  disabled={isBusy}
                />
              </label>
              <div className="meritContribution">
                <strong>{Math.round(weight * 100)}</strong>
                <span>{total > 0 ? `${normalized.toFixed(0)}% norm` : "0% norm"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SensorStage({ metrics, risks, sensor }: { metrics: CockpitMetric[]; risks: CockpitRisk[]; sensor?: SensorAssumptions }) {
  const isBusy = useWorkbench((state) => state.isBusy);
  const patchSensor = useWorkbench((state) => state.patchSensor);
  const snrStatus = metricStatus(metrics, "snr");
  const craStatus = metricStatus(metrics, "cra");
  const snrMetric = metrics.find((metric) => metric.key === "snr");
  const sensorMetrics = pickMetrics(metrics, ["snr", "cra", "illumination", "mtf", "spot", "distortion"]);
  const sensorRisks = risks.filter((risk) => ["Sensor Coupling", "Scene Risk"].includes(risk.label));
  const commitSensorValue = (field: keyof SensorPatch, value: number) => {
    void patchSensor({ [field]: value });
  };

  return (
    <div className="sensorStage">
      <section className="workPanel sensorConfig">
        <PanelHeader icon={Camera} title="Sensor Model" action={sensor ? "Editable" : "Reference"} />
        <div className="sensorEditor">
          <SensorNumberField label="Pixel Pitch" value={sensor?.pixelPitchUm} unit="um" min={0.2} max={30} step={0.05} disabled={isBusy} onCommit={(value) => commitSensorValue("pixelPitchUm", value)} />
          <SensorNumberField label="QE" value={sensor?.quantumEfficiency} unit="ratio" min={0.01} max={1} step={0.01} disabled={isBusy} onCommit={(value) => commitSensorValue("quantumEfficiency", value)} />
          <SensorNumberField label="Full Well" value={sensor?.fullWellE} unit="e-" min={100} max={10000000} step={100} disabled={isBusy} onCommit={(value) => commitSensorValue("fullWellE", value)} />
          <SensorNumberField label="Read Noise" value={sensor?.readNoiseE} unit="e- RMS" min={0} max={100} step={0.1} disabled={isBusy} onCommit={(value) => commitSensorValue("readNoiseE", value)} />
          <SensorNumberField label="Dark Noise" value={sensor?.darkNoiseE} unit="e-" min={0} max={1000} step={0.1} disabled={isBusy} onCommit={(value) => commitSensorValue("darkNoiseE", value)} />
          <SensorNumberField label="CRA Limit" value={sensor?.microlensCraLimitDeg} unit="deg" min={0.1} max={90} step={0.1} disabled={isBusy} status={craStatus} onCommit={(value) => commitSensorValue("microlensCraLimitDeg", value)} />
          <SensorNumberField label="Transmission" value={sensor?.opticalTransmission} unit="ratio" min={0.01} max={1} step={0.01} disabled={isBusy} onCommit={(value) => commitSensorValue("opticalTransmission", value)} />
          <SensorNumberField label="Exposure" value={sensor?.exposureMs} unit="ms" min={0.001} max={60000} step={0.1} disabled={isBusy} onCommit={(value) => commitSensorValue("exposureMs", value)} />
          <SensorNumberField label="Scene L" value={sensor?.sceneLuminanceCdM2} unit="cd/m2" min={0.001} max={1000000} step={1} disabled={isBusy} onCommit={(value) => commitSensorValue("sceneLuminanceCdM2", value)} />
          <SensorNumberField label="Ref. Wavelength" value={sensor?.referenceWavelengthNm} unit="nm" min={100} max={30000} step={1} disabled={isBusy} onCommit={(value) => commitSensorValue("referenceWavelengthNm", value)} />
          <ConfigRow label="CFA" value="RGGB" swatch />
        </div>
      </section>
      <section className="workPanel pixelStack">
        <PanelHeader icon={Layers3} title="Pixel Stack & Chief Ray Geometry" action="Assumption" />
        <div className="pixelCrossSection">
          <span>Microlens</span>
          <span>Color Filter</span>
          <span>OCL</span>
          <span>Photodiode</span>
          <i />
        </div>
      </section>
      <section className="workPanel cameraOutput">
        <PanelHeader
          icon={Eye}
          title="Sensor Coupling"
          action={snrMetric ? snrMetric.source : "Unavailable"}
        />
        <div className={`sensorCouplingSummary ${snrStatus}`}>
          <span>{snrMetric?.label ?? "Assumed Corner SNR"}</span>
          <strong>{snrMetric ? `${snrMetric.value}${snrMetric.unit ? ` ${snrMetric.unit}` : ""}` : "n/a"}</strong>
          <em>{snrMetric?.target ?? "Target n/a"}</em>
          <p>{snrMetric?.note ?? "No sensor coupling estimate is available for the current model."}</p>
        </div>
      </section>
      <section className="workPanel analysisMaps">
        <PanelHeader icon={BarChart3} title="Sensor-Coupled Metrics" action="Current" />
        <div className="sensorMetricGrid">
          {sensorMetrics.map((metric) => (
            <MetricRequirement key={metric.key} metric={metric} />
          ))}
        </div>
      </section>
      <section className="workPanel perceptionRisk">
        <PanelHeader icon={Microscope} title="Perception Interface" action="No Model" />
        <div className="riskStack compareRiskStack">
          {sensorRisks.map((risk, index) => (
            <RiskRow key={risk.label} risk={risk} rank={index + 1} />
          ))}
          <CapabilityRow label="Perception Model" value="Not connected" status="fail" source="unsupported" />
          <CapabilityRow label="Scene Dataset" value="Not loaded" status="fail" source="unsupported" />
        </div>
      </section>
    </div>
  );
}

function SceneStage({ metrics, risks }: { metrics: CockpitMetric[]; risks: CockpitRisk[] }) {
  const sceneMetrics = pickMetrics(metrics, ["spot", "mtf", "illumination", "distortion", "cra", "snr"]);

  return (
    <div className="sceneStage">
      <section className="workPanel scenarioMatrix">
        <PanelHeader icon={Microscope} title="Scene Validation Matrix" action="Not Configured" />
        <UnavailablePanel
          title="No Scene Dataset Loaded"
          rows={[
            ["Scenario assets", "Unavailable"],
            ["Object detector", "Not connected"],
            ["ISP pipeline", "Not connected"],
            ["Validation result", "Optical metrics only"]
          ]}
        />
      </section>
      <section className="workPanel sceneViewer">
        <PanelHeader icon={Eye} title="Validation Scene" action="No Asset" />
        <div className="sceneMetricGrid">
          {sceneMetrics.map((metric) => (
            <MetricRequirement key={metric.key} metric={metric} />
          ))}
        </div>
      </section>
      <section className="workPanel perceptionChain">
        <PanelHeader icon={Target} title="Optics to Perception Chain" action="Optical Risk" />
        <div className="chainList">
          {risks.map((risk) => (
            <RiskRow key={risk.label} risk={risk} />
          ))}
          {metrics.slice(1, 4).map((metric) => (
            <MetricRequirement key={metric.key} metric={metric} />
          ))}
        </div>
      </section>
    </div>
  );
}

function CompareStage({ metrics, risks }: { metrics: CockpitMetric[]; risks: CockpitRisk[] }) {
  const model = useWorkbench((state) => state.model);
  const variants = useWorkbench((state) => state.variants);
  const isBusy = useWorkbench((state) => state.isBusy);
  const captureVariant = useWorkbench((state) => state.captureVariant);
  const deleteVariant = useWorkbench((state) => state.deleteVariant);
  const clearVariants = useWorkbench((state) => state.clearVariants);
  const trackedMetrics = pickMetrics(metrics, ["spot", "mtf", "illumination", "distortion", "cra", "yield", "snr"]);
  const passCount = trackedMetrics.filter((metric) => metric.status === "pass").length;
  const candidate = variants[0] ?? null;
  const baseline = variants[1] ?? null;
  const canCompare = Boolean(candidate && baseline);

  return (
    <div className="compareStage">
      <section className="workPanel tradePlot">
        <PanelHeader icon={GitCompare} title="Variant Set" action={`${variants.length} snapshots`} />
        <div className="compareActions">
          <button className="primaryWide" onClick={() => captureVariant()} disabled={!model || isBusy}>
            <Camera size={15} />
            Capture Current
          </button>
          <button onClick={() => clearVariants()} disabled={!variants.length || isBusy}>
            <Trash2 size={15} />
            Clear
          </button>
        </div>
        {variants.length ? (
          <div className="variantSnapshotList">
            {variants.map((variant, index) => (
              <div key={variant.id} className={`variantSnapshot ${index < 2 ? "active" : ""}`}>
                <div className="variantSnapshotMain">
                  <strong>{variant.name}</strong>
                  <span>{variant.modelName}</span>
                </div>
                <div className="variantSnapshotMeta">
                  <em>{sourceLabel(variant.source)}</em>
                  <span>{formatSnapshotTime(variant.createdAt)}</span>
                  <span>{variant.surfaceCount} surfaces</span>
                  <span>{variant.variableSummary}</span>
                </div>
                <div className="variantSnapshotScore">
                  <b>{variant.metrics.filter((metric) => metric.status === "pass").length}/{variant.metrics.length || 1}</b>
                  <button title={`Delete ${variant.name}`} onClick={() => deleteVariant(variant.id)} disabled={isBusy}>
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="compareNotice">
            <CapabilityRow label="Baseline Model" value="Not captured" status="warn" source="model" />
            <CapabilityRow label="Variant Set" value="Capture current model or run Quick Optimize" status="warn" source="assumption" />
          </div>
        )}
      </section>
      <section className="workPanel variantTablePanel">
        <PanelHeader icon={BarChart3} title={canCompare ? "Latest Delta" : "Current Metric Scores"} action={canCompare ? "Latest pair" : `${passCount}/${trackedMetrics.length} pass`} />
        {canCompare && baseline && candidate ? (
          <VariantDeltaTable baseline={baseline} candidate={candidate} trackedKeys={trackedMetrics.map((metric) => metric.key)} />
        ) : (
          <MetricScorePlot metrics={trackedMetrics} />
        )}
      </section>
      <section className="workPanel surfaceDiff">
        <PanelHeader icon={Aperture} title={canCompare ? "Prescription Movement" : "Risk Drivers"} action={canCompare ? "Latest pair" : "Current"} />
        {canCompare && baseline && candidate ? (
          <PrescriptionDeltaTable baseline={baseline} candidate={candidate} />
        ) : (
          <div className="riskStack compareRiskStack">
            {risks.slice(0, 5).map((risk, index) => (
              <RiskRow key={risk.label} risk={risk} rank={index + 1} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function VariantDeltaTable({ baseline, candidate, trackedKeys }: { baseline: VariantSnapshot; candidate: VariantSnapshot; trackedKeys: string[] }) {
  const rows = compareVariantMetrics(baseline, candidate, trackedKeys);
  return (
    <table className="diffTable large variantDeltaTable">
      <thead>
        <tr>
          <th>Metric</th>
          <td>{baseline.name}</td>
          <td>{candidate.name}</td>
          <td>Score Δ</td>
          <td>Status</td>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.key}>
            <th>{row.label}</th>
            <td>{row.before}</td>
            <td>{row.after}</td>
            <td className={row.scoreDelta >= 0 ? "positive" : "negative"}>{row.scoreDelta >= 0 ? `+${row.scoreDelta.toFixed(3)}` : row.scoreDelta.toFixed(3)}</td>
            <td className={row.status === "pass" ? "positive" : "negative"}>{statusLabel(row.status)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PrescriptionDeltaTable({ baseline, candidate }: { baseline: VariantSnapshot; candidate: VariantSnapshot }) {
  const rows = comparePrescription(baseline, candidate);
  if (!rows.length) {
    return (
      <div className="compareNotice prescriptionEmpty">
        <CapabilityRow label="Prescription Delta" value="No radius, thickness, glass, semi-diameter, or conic movement" status="pass" source="model" />
      </div>
    );
  }
  return (
    <table className="diffTable large prescriptionDeltaTable">
      <thead>
        <tr>
          <th>Surf</th>
          <td>Field</td>
          <td>Before</td>
          <td>After</td>
          <td>Δ</td>
        </tr>
      </thead>
      <tbody>
        {rows.slice(0, 14).map((row) => (
          <tr key={`${row.surface}-${row.field}`}>
            <th>{row.surface}</th>
            <td>{row.field}</td>
            <td>{row.before}</td>
            <td>{row.after}</td>
            <td className={row.deltaClass}>{row.delta}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function compareVariantMetrics(baseline: VariantSnapshot, candidate: VariantSnapshot, trackedKeys: string[]) {
  const baselineByKey = new Map(baseline.metrics.map((metric) => [metric.key, metric]));
  const candidateByKey = new Map(candidate.metrics.map((metric) => [metric.key, metric]));
  const keys = trackedKeys.length ? trackedKeys : Array.from(new Set([...baselineByKey.keys(), ...candidateByKey.keys()]));
  return keys.flatMap((key) => {
    const before = baselineByKey.get(key);
    const after = candidateByKey.get(key);
    if (!before || !after) return [];
    return [
      {
        key,
        label: after.label,
        before: metricDisplay(before),
        after: metricDisplay(after),
        scoreDelta: after.score - before.score,
        status: after.status
      }
    ];
  });
}

function comparePrescription(baseline: VariantSnapshot, candidate: VariantSnapshot) {
  const baselineByIndex = new Map(baseline.surfaces.map((surface) => [surface.index, surface]));
  const candidateByIndex = new Map(candidate.surfaces.map((surface) => [surface.index, surface]));
  const changedRows = candidate.surfaces.flatMap((surface) => {
    const before = baselineByIndex.get(surface.index);
    if (!before) {
      return [
        {
          surface: surfaceLabel(candidate, surface.index),
          field: "Surface",
          before: "missing",
          after: surface.label || surface.type,
          delta: "added",
          deltaClass: "positive"
        }
      ];
    }
    const rows = [
      numericSurfaceDelta(candidate, surface.index, "Radius", before.radius, surface.radius),
      numericSurfaceDelta(candidate, surface.index, "Thickness", before.thickness, surface.thickness),
      textSurfaceDelta(candidate, surface.index, "Glass", before.glass, surface.glass),
      numericSurfaceDelta(candidate, surface.index, "Semi-Dia", before.semiDiameter, surface.semiDiameter),
      numericSurfaceDelta(candidate, surface.index, "Conic", before.conic, surface.conic)
    ];
    return rows.filter((row): row is NonNullable<typeof row> => Boolean(row));
  });
  const removedRows = baseline.surfaces.flatMap((surface) => {
    if (candidateByIndex.has(surface.index)) return [];
    return [
      {
        surface: surfaceLabel(baseline, surface.index),
        field: "Surface",
        before: surface.label || surface.type,
        after: "missing",
        delta: "removed",
        deltaClass: "negative"
      }
    ];
  });
  return [...changedRows, ...removedRows];
}

function numericSurfaceDelta(variant: VariantSnapshot, index: number, field: string, before: number | null, after: number | null) {
  if (before === null && after === null) return null;
  if (before === null || after === null) {
    return {
      surface: surfaceLabel(variant, index),
      field,
      before: before === null ? "n/a" : formatCompact(before),
      after: after === null ? "n/a" : formatCompact(after),
      delta: before === null ? "set" : "cleared",
      deltaClass: before === null ? "positive" : "negative"
    };
  }
  const delta = after - before;
  if (Math.abs(delta) < 1e-8) return null;
  return {
    surface: surfaceLabel(variant, index),
    field,
    before: formatCompact(before),
    after: formatCompact(after),
    delta: delta >= 0 ? `+${formatCompact(delta)}` : formatCompact(delta),
    deltaClass: delta >= 0 ? "positive" : "negative"
  };
}

function textSurfaceDelta(variant: VariantSnapshot, index: number, field: string, before: string, after: string) {
  if (before === after) return null;
  return {
    surface: surfaceLabel(variant, index),
    field,
    before: before || "air",
    after: after || "air",
    delta: "changed",
    deltaClass: "positive"
  };
}

function ReportStage({
  model,
  summary,
  metrics,
  risks,
  warnings,
  errors,
  requirements,
  requirementProfile,
  requirementProfileBase,
  autosavePath,
  autosaveSavedAt,
  draftCount,
  compatibility,
  quickOptimizeResult,
  quickOptimizeObjective,
  quickOptimizeIterations,
  quickOptimizeMaxEvaluations,
  quickOptimizeStepScale,
  quickOptimizeWeights,
  toleranceSweepResult,
  toleranceSweepScope,
  toleranceSweepPerturbationPct,
  toleranceSweepMaxSurfaces,
  variants
}: {
  model: OpticalModel | null;
  summary: AnalysisSummary | null;
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
  warnings: string[];
  errors: string[];
  requirements: ProjectRequirements;
  requirementProfile: RequirementProfileSelection;
  requirementProfileBase: RequirementProfileId;
  autosavePath: string | null;
  autosaveSavedAt: string | null;
  draftCount: number | null;
  compatibility: ExampleCheckResponse | null;
  quickOptimizeResult: QuickOptimizeResult | null;
  quickOptimizeObjective: QuickOptimizeObjective;
  quickOptimizeIterations: number;
  quickOptimizeMaxEvaluations: number;
  quickOptimizeStepScale: number;
  quickOptimizeWeights: QuickOptimizeWeights;
  toleranceSweepResult: ToleranceSweepResult | null;
  toleranceSweepScope: ToleranceSweepScope;
  toleranceSweepPerturbationPct: number;
  toleranceSweepMaxSurfaces: number;
  variants: VariantSnapshot[];
}) {
  type ReportSectionId = keyof ReportSectionSelection;
  const [sections, setSections] = useState<ReportSectionSelection>(() => ({ ...defaultReportSections }));
  const snapshot = useMemo(
    () =>
      buildReportSnapshot(model, summary, metrics, risks, warnings, errors, requirements, requirementProfile, requirementProfileBase, {
        path: autosavePath,
        savedAt: autosaveSavedAt,
        draftCount
      }, compatibility, quickOptimizeResult, sections, variants, {
        quickOptimizeSettings: {
          objective: quickOptimizeObjective,
          iterations: quickOptimizeIterations,
          maxEvaluations: quickOptimizeMaxEvaluations,
          stepScale: quickOptimizeStepScale,
          operandWeights: quickOptimizeWeights
        },
        toleranceSweep: {
          settings: {
            scope: toleranceSweepScope,
            perturbationPct: toleranceSweepPerturbationPct,
            maxSurfaces: toleranceSweepMaxSurfaces
          },
          result: toleranceSweepResult
        }
      }),
    [
      autosavePath,
      autosaveSavedAt,
      compatibility,
      draftCount,
      errors,
      metrics,
      model,
      quickOptimizeIterations,
      quickOptimizeMaxEvaluations,
      quickOptimizeObjective,
      quickOptimizeResult,
      quickOptimizeStepScale,
      quickOptimizeWeights,
      requirementProfile,
      requirementProfileBase,
      requirements,
      risks,
      sections,
      summary,
      toleranceSweepMaxSurfaces,
      toleranceSweepPerturbationPct,
      toleranceSweepResult,
      toleranceSweepScope,
      variants,
      warnings
    ]
  );
  const reportChecks: Array<{ id: ReportSectionId; label: string }> = [
    { id: "model", label: "Model identity" },
    { id: "requirements", label: "Project requirements" },
    { id: "lensData", label: "Sequential lens data" },
    { id: "metrics", label: `${metrics.length} analysis metrics` },
    { id: "risks", label: `${risks.length} ranked risks` },
    { id: "firstOrder", label: summary?.firstOrder ? "First-order values" : "First-order fallback pending" },
    { id: "sensor", label: summary?.sensor ? "Sensor assumptions" : "Sensor assumptions pending" },
    { id: "counts", label: "Model counts" },
    { id: "compatibility", label: compatibility ? `${compatibility.total} example QA checks` : "Example QA not run" },
    { id: "quickOptimize", label: quickOptimizeResult ? `Quick Optimize ${quickOptimizeResult.status}` : "Quick Optimize not run" },
    { id: "optimizationSettings", label: `Optimizer setup ${objectiveLabel(quickOptimizeObjective)}` },
    { id: "toleranceSweep", label: toleranceSweepResult ? `Tolerance sweep ${toleranceSweepResult.status}` : "Tolerance sweep not run" },
    { id: "variants", label: `${variants.length} compare snapshots` },
    { id: "diagnostics", label: `${warnings.length} warnings / ${errors.length} errors` }
  ];
  const selectedCount = reportChecks.filter((item) => sections[item.id]).length;

  function toggleSection(id: ReportSectionId, checked: boolean) {
    setSections((current) => ({ ...current, [id]: checked }));
  }

  return (
    <div className="reportStage">
      <section className="workPanel reportBuilder">
        <PanelHeader icon={FileText} title="Report Builder" action={model ? `${selectedCount}/${reportChecks.length}` : "No model"} />
        <div className="reportChecks">
          {reportChecks.map((item) => (
            <label key={item.id}>
              <input type="checkbox" checked={sections[item.id]} onChange={(event) => toggleSection(item.id, event.target.checked)} />
              <span>{item.label}</span>
            </label>
          ))}
        </div>
        <button className="primaryWide" disabled={!model || selectedCount === 0} onClick={() => exportReportSnapshot(snapshot)}>
          <Download size={16} />
          Export JSON Snapshot
        </button>
      </section>
      <section className="workPanel reportPreview">
        <PanelHeader icon={FileText} title="Preview" action="JSON" />
        <div className="reportPage">
          <h2>{model?.name || "Optical Design Cockpit"}</h2>
          <p>{model?.filename ?? "Unsaved local model"}</p>
          {sections.metrics ? (
            <div className="reportMetricGrid">
              {metrics.slice(0, 8).map((metric) => (
                <MetricRequirement key={metric.key} metric={metric} />
              ))}
            </div>
          ) : null}
          <pre className="reportJsonPreview">{JSON.stringify(snapshot, null, 2)}</pre>
        </div>
      </section>
    </div>
  );
}

function MetricScorePlot({ metrics }: { metrics: CockpitMetric[] }) {
  return (
    <div className="metricScorePlot">
      {metrics.map((metric) => (
        <div key={metric.key} className={`metricScoreRow ${metric.status}`}>
          <div>
            <span>{metric.label}</span>
            <strong>
              {metric.value}
              {metric.unit ? ` ${metric.unit}` : ""}
            </strong>
          </div>
          <i>
            <b style={{ width: `${Math.round(metric.score * 100)}%` }} />
          </i>
          <em>
            {metric.target} · {metric.source}
          </em>
        </div>
      ))}
    </div>
  );
}

function RequirementTargetEditor({
  metrics,
  requirements,
  requirementProfile,
  requirementProfileBase,
  onRequirementChange,
  onRequirementProfileChange,
  onRequirementReset
}: {
  metrics: CockpitMetric[];
  requirements: ProjectRequirements;
  requirementProfile: RequirementProfileSelection;
  requirementProfileBase: RequirementProfileId;
  onRequirementChange: (key: ProjectRequirementKey, value: number) => void;
  onRequirementProfileChange: (profile: RequirementProfileId) => void;
  onRequirementReset: () => void;
}) {
  const metricByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  const editableMetrics = requirementDefinitions
    .map((definition) => metricByKey.get(definition.metricKey))
    .filter((metric): metric is CockpitMetric => Boolean(metric));

  return (
    <div className="requirementEditor">
      <div className="requirementEditorHeader">
        <span title={profileDescription(requirementProfile, requirementProfileBase)}>
          {profileLabel(requirementProfile)} targets
        </span>
        <button type="button" onClick={onRequirementReset}>
          Reset
        </button>
      </div>
      <div className="requirementProfileStrip" role="group" aria-label="Requirement profile">
        {requirementProfiles.map((profile) => (
          <button
            key={profile.id}
            type="button"
            className={requirementProfile !== "custom" && profile.id === requirementProfileBase ? "active" : ""}
            title={profile.description}
            onClick={() => onRequirementProfileChange(profile.id)}
          >
            <strong>{profile.label}</strong>
            <span>{profile.domain}</span>
          </button>
        ))}
      </div>
      <div className="requirementTargetGrid">
        {editableMetrics.map((metric) => {
          const definition = definitionForMetric(metric);
          if (!definition) return null;
          const value = requirements[definition.targetKey];
          return (
            <label key={`${metric.key}-${definition.targetKey}`} className={`requirementTarget ${metric.status}`}>
              <span>{definition.label}</span>
              <div>
                <b>{definition.direction === "min" ? ">=" : "<="}</b>
                <input
                  aria-label={`${definition.label} requirement target`}
                  type="number"
                  min={definition.min}
                  max={definition.max}
                  step={definition.step}
                  value={value}
                  onChange={(event) => onRequirementChange(definition.targetKey, Number(event.target.value))}
                />
                {definition.unit ? <em>{definition.unit}</em> : null}
              </div>
              <strong>{targetString(definition, value)}</strong>
            </label>
          );
        })}
      </div>
    </div>
  );
}

function CompatibilityPanel({
  compatibility,
  isChecking,
  onRun
}: {
  compatibility: ExampleCheckResponse | null;
  isChecking: boolean;
  onRun: () => Promise<void>;
}) {
  const overallStatus: KpiStatus | null = compatibility
    ? compatibility.failed > 0
      ? "fail"
      : compatibility.warned > 0
        ? "warn"
        : "pass"
    : null;

  return (
    <div className="compatibilityQa">
      <button className="primaryWide qaRunButton" disabled={isChecking} onClick={() => void onRun()}>
        {isChecking ? <RefreshCw size={16} /> : <Play size={16} />}
        {isChecking ? "Running Example QA" : "Run Example QA"}
      </button>
      {compatibility ? (
        <>
          <div className={`qaSummary ${overallStatus ?? ""}`}>
            <span>{formatCheckedAt(compatibility.checkedAt)}</span>
            <strong>
              {compatibility.passed} pass / {compatibility.warned} warn / {compatibility.failed} fail
            </strong>
          </div>
          <div className="qaCheckList">
            {compatibility.checks.map((check) => (
              <CompatibilityCheckRow key={`${check.path}-${check.durationMs}`} check={check} />
            ))}
          </div>
        </>
      ) : (
        <div className="emptyState compact">No compatibility run yet.</div>
      )}
    </div>
  );
}

function CompatibilityCheckRow({ check }: { check: ExampleCheckResult }) {
  const failedStage = check.stages.find((stage) => stage.status === "fail");
  const warningStage = check.stages.find((stage) => stage.status === "warn");
  const detail = failedStage?.detail ?? warningStage?.detail ?? `${check.stages.length} stages completed.`;

  return (
    <details className={`qaCheck ${check.status}`}>
      <summary title={check.path}>
        <b>{statusLabel(check.status)}</b>
        <span>
          <strong>{check.label}</strong>
          <em>
            {check.kind} · {check.surfaceCount ?? "n/a"} surf · {check.durationMs} ms
          </em>
        </span>
      </summary>
      <div className="qaStageList">
        {check.stages.map((stage) => (
          <span key={`${check.path}-${stage.name}`} className={stage.status}>
            {stage.status === "pass" ? <Check size={12} /> : <AlertTriangle size={12} />}
            <strong>{stage.name}</strong>
            <em>{stage.detail}</em>
          </span>
        ))}
      </div>
      {check.warnings.length || check.errors.length ? (
        <div className="qaDiagnostics">
          {[...check.errors, ...check.warnings].slice(0, 4).map((message) => (
            <span key={message}>{message}</span>
          ))}
        </div>
      ) : (
        <p>{detail}</p>
      )}
    </details>
  );
}

function pickMetrics(metrics: CockpitMetric[], keys: string[]) {
  const byKey = new Map(metrics.map((metric) => [metric.key, metric]));
  return keys.map((key) => byKey.get(key)).filter((metric): metric is CockpitMetric => Boolean(metric));
}

function PanelHeader({ icon: Icon, title, action }: { icon: typeof Aperture; title: string; action: string }) {
  return (
    <header className="panelHeader">
      <div>
        <Icon size={17} />
        <strong>{title}</strong>
      </div>
      <span>{action}</span>
    </header>
  );
}

function MetricRequirement({ metric }: { metric: CockpitMetric }) {
  return (
    <div className={`metricRequirement ${metric.status}`}>
      <span>{metric.label}</span>
      <strong>
        {metric.value}
        {metric.unit ? ` ${metric.unit}` : ""}
      </strong>
      <em>
        {metric.target}
        <b className={`sourceBadge ${metric.source}`} title={metric.note ?? metric.source}>
          {metric.source}
        </b>
      </em>
    </div>
  );
}

function RiskRow({ risk, rank }: { risk: CockpitRisk; rank?: number }) {
  return (
    <div className={`riskRow ${risk.status}`}>
      {rank ? <b>{rank}</b> : null}
      <div>
        <strong>{risk.label}</strong>
        <span>
          {risk.detail} · {risk.source}
        </span>
      </div>
      <em>{risk.value}</em>
    </div>
  );
}

function Snapshot({ label, value }: { label: string; value: string }) {
  return (
    <div className="snapshot">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ConfigRow({ label, value, status, swatch }: { label: string; value: string; status?: KpiStatus; swatch?: boolean }) {
  return (
    <div className={`configRow ${status ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {swatch ? <i className="cfaSwatch" /> : null}
    </div>
  );
}

function SensorNumberField({
  label,
  value,
  unit,
  min,
  max,
  step,
  disabled,
  status,
  onCommit
}: {
  label: string;
  value: number | undefined;
  unit: string;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  status?: KpiStatus;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(value === undefined ? "" : String(value));

  useEffect(() => {
    setDraft(value === undefined ? "" : String(value));
  }, [value]);

  function commit() {
    const numeric = Number(draft);
    if (!Number.isFinite(numeric) || value === undefined) {
      setDraft(value === undefined ? "" : String(value));
      return;
    }
    const clamped = Math.min(max, Math.max(min, numeric));
    setDraft(String(clamped));
    if (Math.abs(clamped - value) > 1.0e-9) {
      onCommit(clamped);
    }
  }

  return (
    <label className={`sensorField ${status ?? ""}`}>
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft}
        disabled={disabled || value === undefined}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
      <em>{unit}</em>
    </label>
  );
}

function CapabilityRow({
  label,
  value,
  status,
  source
}: {
  label: string;
  value: string;
  status: KpiStatus;
  source: MetricSource;
}) {
  return (
    <div className={`capabilityRow ${status}`}>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <b className={`sourceBadge ${source}`}>{source}</b>
    </div>
  );
}

function UnavailablePanel({ title, rows }: { title: string; rows: Array<[string, string]> }) {
  return (
    <div className="unavailablePanel">
      <strong>{title}</strong>
      <div>
        {rows.map(([label, value]) => (
          <span key={label}>
            <em>{label}</em>
            <b>{value}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

function StressOutcome({ metric, risk }: { metric?: CockpitMetric; risk?: CockpitRisk }) {
  return (
    <div className="stressOutcome">
      <div className={`stressScore ${metric?.status ?? "warn"}`}>
        <span>{metric?.label ?? "Tol. Robustness"}</span>
        <strong>
          {metric ? `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}` : "n/a"}
        </strong>
        <em>{metric?.target ?? "Target n/a"}</em>
      </div>
      <div className="stressNote">
        <span>{risk?.label ?? "Surface Sensitivity"}</span>
        <strong>{risk?.value ?? "n/a"}</strong>
        <p>{metric?.note ?? risk?.detail ?? "No stress case result available."}</p>
      </div>
    </div>
  );
}

function ToleranceSweepOutcome({ result }: { result: ToleranceSweepResult }) {
  return (
    <div className="quickOptimizeOutcome toleranceSweepOutcome">
      <div className={`stressScore ${result.status}`}>
        <span>Quick Tolerance Sweep</span>
        <strong>{result.worstScore !== null ? `${Math.round(result.worstScore * 100)}%` : "n/a"}</strong>
        <em>
          {result.attemptedCases} cases · {result.perturbationPct}% perturb
        </em>
      </div>
      <div className="stressNote">
        <span>{result.scope === "variables" ? "Marked Variables" : "Powered Surfaces"}</span>
        <strong>{result.worstCase ?? "No sweep case"}</strong>
        <p>
          {result.passedCases} pass · {result.warnedCases} warn · {result.failedCases} fail. Deterministic perturbation only, not statistical yield.
        </p>
      </div>
      <div className="moveList">
        {result.cases.slice(0, 8).map((testCase) => (
          <span key={`${testCase.label}-${testCase.after}`}>
            <b>{testCase.label}</b>
            <em>
              {statusLabel(testCase.status)} · score {formatCompact(testCase.score)}
            </em>
          </span>
        ))}
        {result.warnings.map((warning) => (
          <span key={warning}>
            <b>Note</b>
            <em>{warning}</em>
          </span>
        ))}
      </div>
    </div>
  );
}

function ToleranceSweepTable({ result }: { result: ToleranceSweepResult }) {
  if (!result.cases.length) {
    return (
      <div className="compareNotice prescriptionEmpty">
        <CapabilityRow label="Tolerance Sweep" value={result.warnings[0] ?? "No cases were generated"} status="warn" source="computed" />
      </div>
    );
  }
  return (
    <table className="diffTable large toleranceSweepTable">
      <thead>
        <tr>
          <th>Case</th>
          <td>Score</td>
          <td>Spot</td>
          <td>Thr.</td>
          <td>Dist.</td>
          <td>CRA</td>
        </tr>
      </thead>
      <tbody>
        {result.cases.slice(0, 16).map((testCase) => (
          <tr key={`${testCase.label}-${testCase.after}`}>
            <th title={testCase.traceFailures.join("; ") || testCase.label}>
              {testCase.label}
              {testCase.traceFailures.length ? <b className="caseFailureCount"> {testCase.traceFailures.length}f</b> : null}
            </th>
            <td className={testCase.status === "pass" ? "positive" : "negative"}>{Math.round(testCase.score * 100)}%</td>
            <td>{formatNullable(testCase.spotRmsUm, " um")}</td>
            <td>{formatNullable(testCase.throughput, "")}</td>
            <td>{formatNullable(testCase.distortionPct, "%")}</td>
            <td>{formatNullable(testCase.craDeg, " deg")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function QuickOptimizeOutcome({ result }: { result: QuickOptimizeResult }) {
  const status: KpiStatus = result.status === "improved" ? "pass" : result.status === "no-change" ? "warn" : "fail";

  return (
    <div className="quickOptimizeOutcome">
      <div className={`stressScore ${status}`}>
        <span>Quick Optimize · {objectiveLabel(result.objective)}</span>
        <strong>{result.improvement >= 0 ? `+${result.improvement.toFixed(4)}` : result.improvement.toFixed(4)}</strong>
        <em>
          {result.evaluations} evaluations · {result.iterations} iterations
        </em>
      </div>
      <div className="stressNote">
        <span>{result.status}</span>
        <strong>
          {result.baselineScore.toFixed(4)} → {result.finalScore.toFixed(4)}
        </strong>
        <p>{result.message}</p>
      </div>
      <div className="moveList">
        {result.moves.length ? (
          result.moves.slice(0, 8).map((move) => (
            <span key={`${move.label}-${move.score}-${move.after}`}>
              <b>{move.label}</b>
              <em>
                {formatCompact(move.before)} → {formatCompact(move.after)}
              </em>
            </span>
          ))
        ) : (
          <span>
            <b>Variables</b>
            <em>{result.variables.length ? result.variables.join(", ") : "None"}</em>
          </span>
        )}
      </div>
    </div>
  );
}

function metricDisplay(metric: CockpitMetric) {
  return `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`;
}

function sourceLabel(source: VariantSnapshot["source"]) {
  if (source === "quick-optimize-before") return "Quick Optimize Before";
  if (source === "quick-optimize-after") return "Quick Optimize After";
  if (source === "open") return "Opened";
  if (source === "new") return "New";
  return "Manual";
}

function formatSnapshotTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "time n/a";
  return date.toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function surfaceLabel(variant: VariantSnapshot, index: number) {
  if (index === 0) return "OBJ";
  if (index === variant.surfaces.length - 1) return "IMG";
  return `S${index}`;
}

function formatCompact(value: number) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return value.toPrecision(4);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function formatNullable(value: number | null, unit: string) {
  return value === null || value === undefined ? "n/a" : `${formatCompact(value)}${unit}`;
}

function metricStatus(metrics: CockpitMetric[], key: string) {
  return metrics.find((metric) => metric.key === key)?.status ?? "pass";
}

function worstStatus(statuses: KpiStatus[]) {
  if (statuses.includes("fail")) return "fail";
  if (statuses.includes("warn")) return "warn";
  return "pass";
}

function metricValueLabel(metrics: CockpitMetric[], key: string) {
  const metric = metrics.find((item) => item.key === key);
  if (!metric) return "n/a";
  return `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`;
}

function objectiveLabel(objective: QuickOptimizeObjective) {
  return quickOptimizeLabel(objective);
}

function fieldScopeLabel(model: OpticalModel | null) {
  const fields = model?.system.field.fields ?? [];
  if (!fields.length) return "No fields";
  const maxField = fields.reduce((max, field) => Math.max(max, Math.hypot(field.x ?? 0, field.y ?? 0)), 0);
  return maxField > 0 ? `${formatCompact(maxField)} field` : "On-axis";
}

function referenceWavelengthLabel(model: OpticalModel | null) {
  const wavelengths = model?.system.wavelengths.values ?? [];
  if (!wavelengths.length) return "No wavelengths";
  const reference = model?.system.wavelengths.reference ?? 0;
  const selected = wavelengths[Math.max(0, Math.min(reference, wavelengths.length - 1))] ?? wavelengths.find((value) => value !== null);
  return selected === null || selected === undefined ? "n/a" : `${formatCompact(selected)} nm`;
}

function selectedSurfaceLabel(model: OpticalModel | null, selectedSurface: number | null, surfaceRisk?: CockpitRisk) {
  if (surfaceRisk?.value && surfaceRisk.value !== "n/a") return surfaceRisk.value;
  if (!model?.surfaces.length) return "No surface";
  const surface = model.surfaces.find((item) => item.index === selectedSurface) ?? model.surfaces.find((item) => item.isStop) ?? model.surfaces[1] ?? model.surfaces[0];
  if (!surface) return "No surface";
  if (surface.index === 0) return "OBJ";
  if (surface.index === model.surfaces.length - 1) return "IMG";
  return `S${surface.index}`;
}

function formatCheckedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Last run time unavailable";
  return `Last run ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}
