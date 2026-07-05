import { AlertTriangle, Aperture, CircleDot, Database, FlaskConical, History, MapPin, RadioTower, Waves } from "lucide-react";
import { useMemo } from "react";
import { useWorkbench } from "../lib/store";
import type { KpiStatus, RecentModelFile, WorkbenchFileStatus, WorkflowStage } from "../lib/types";
import { profileLabel } from "../lib/requirements";
import { deriveMetrics, statusLabel } from "../lib/uxMetrics";
import { workflowStages } from "./WorkflowRail";

export function ProjectNavigator({
  activeStage,
  onStageChange
}: {
  activeStage: WorkflowStage;
  onStageChange: (stage: WorkflowStage) => void;
}) {
  const model = useWorkbench((state) => state.model);
  const examples = useWorkbench((state) => state.examples);
  const recentFiles = useWorkbench((state) => state.recentFiles);
  const workbenchFileStatus = useWorkbench((state) => state.workbenchFileStatus);
  const openModel = useWorkbench((state) => state.openModel);
  const isBusy = useWorkbench((state) => state.isBusy);
  const selectedSurface = useWorkbench((state) => state.selectedSurface);
  const setSelectedSurface = useWorkbench((state) => state.setSelectedSurface);
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);
  const summary = useWorkbench((state) => state.summary);
  const requirements = useWorkbench((state) => state.requirements);
  const requirementProfile = useWorkbench((state) => state.requirementProfile);
  const metrics = useMemo(() => deriveMetrics(model, warnings, errors, summary, requirements), [errors, model, requirements, summary, warnings]);
  const currentStage = workflowStages.find((stage) => stage.id === activeStage);
  const traceMetric = metrics.metrics.find((metric) => metric.key === "trace");
  const computedMetricCount = metrics.metrics.filter((metric) => metric.source === "computed").length;
  const proxyMetricCount = metrics.metrics.filter((metric) => metric.source === "proxy").length;
  const assumptionMetricCount = metrics.metrics.filter((metric) => metric.source === "assumption").length;
  const capabilityStack: Array<{ label: string; value: string; status: KpiStatus }> = [
    {
      label: "Trace",
      value: traceMetric ? `${traceMetric.value} ${traceMetric.unit ?? ""}`.trim() : "not run",
      status: traceMetric?.status ?? "warn"
    },
    {
      label: "Computed",
      value: `${computedMetricCount} metrics`,
      status: computedMetricCount > 0 ? "pass" : "warn"
    },
    {
      label: "Proxy/Assumption",
      value: `${proxyMetricCount + assumptionMetricCount} marked`,
      status: proxyMetricCount + assumptionMetricCount > 0 ? "warn" : "pass"
    },
    {
      label: "Requirements",
      value: profileLabel(requirementProfile),
      status: requirementProfile === "custom" ? "warn" : "pass"
    },
    {
      label: "Scene Data",
      value: "not loaded",
      status: "fail"
    }
  ];
  const representativeExamples = examples
    .filter((example) => /\.(roa|seq|zmx)$/i.test(example.path))
    .slice(0, 7);

  return (
    <aside className="projectNavigator">
      <section className="navigatorBlock currentContext">
        <span className="eyebrow">Workflow</span>
        <strong>{currentStage?.short ?? "Workbench"}</strong>
        <div className="stagePills">
          {workflowStages.slice(0, 6).map((stage) => (
            <button key={stage.id} className={stage.id === activeStage ? "active" : ""} onClick={() => onStageChange(stage.id)}>
              {stage.short}
            </button>
          ))}
        </div>
      </section>

      <section className="navigatorBlock">
        <div className="blockTitle">
          <Database size={16} />
          Workbench File
        </div>
        <WorkbenchFileState status={workbenchFileStatus} />
      </section>

      <section className="navigatorBlock">
        <div className="blockTitle">
          <Aperture size={16} />
          Model Tree
        </div>
        <div className="treeList">
          {model?.surfaces.slice(0, 14).map((surface) => (
            <button
              key={surface.index}
              className={surface.index === selectedSurface ? "selected" : ""}
              onClick={() => setSelectedSurface(surface.index)}
              title={`${surface.index}: ${surface.label || surface.type}`}
            >
              <span>{surface.isStop ? "STOP" : surface.index === 0 ? "OBJ" : surface.index === model.surfaces.length - 1 ? "IMG" : `S${surface.index}`}</span>
              <strong>{surface.label || surface.glass || surface.type}</strong>
            </button>
          ))}
        </div>
      </section>

      <section className="navigatorBlock">
        <div className="blockTitle">
          <CircleDot size={16} />
          Requirement Status
        </div>
        <div className="requirementList">
          {metrics.metrics.slice(1, 6).map((metric) => (
            <div key={metric.key} className={`requirementRow ${metric.status}`}>
              <span>{metric.label}</span>
              <strong>
                {metric.value}
                {metric.unit ? ` ${metric.unit}` : ""}
              </strong>
              <em>{statusLabel(metric.status)}</em>
              <b className={`sourceBadge ${metric.source}`} title={metric.note ?? metric.source}>
                {metric.source}
              </b>
            </div>
          ))}
        </div>
      </section>

      <section className="navigatorBlock">
        <div className="blockTitle">
          <MapPin size={16} />
          Fields / Wavelengths
        </div>
        <div className="fieldGrid">
          {(model?.system.field.fields.length ? model.system.field.fields : [{ x: 0, y: 0 }]).map((field, index) => (
            <span key={`${field.x}-${field.y}-${index}`}>
              F{index + 1}
              <strong>{field.y ?? 0}</strong>
            </span>
          ))}
        </div>
        <div className="waveChips">
          {(model?.system.wavelengths.values.length ? model.system.wavelengths.values : [null]).map((value, index) => (
            <span key={`${value}-${index}`}>
              <Waves size={12} />
              {value === null ? "n/a" : `${value.toFixed(0)} nm`}
            </span>
          ))}
        </div>
      </section>

      <section className="navigatorBlock">
        <div className="blockTitle">
          <RadioTower size={16} />
          Capability Stack
        </div>
        <div className="scenarioList">
          {capabilityStack.map((item) => (
            <span key={item.label} className={item.status}>
              {item.label}
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
      </section>

      {recentFiles.length ? (
        <section className="navigatorBlock">
          <div className="blockTitle">
            <History size={16} />
            Recent Files
          </div>
          <RecentFileList files={recentFiles.slice(0, 5)} isBusy={isBusy} onOpen={openModel} />
        </section>
      ) : null}

      <section className="navigatorBlock">
        <div className="blockTitle">
          <FlaskConical size={16} />
          Examples
        </div>
        <div className="exampleList">
          {representativeExamples.map((example) => (
            <button key={example.path} onClick={() => void openModel(example.path)} title={example.path}>
              {example.label}
            </button>
          ))}
        </div>
      </section>

      {(warnings.length || errors.length) ? (
        <section className="navigatorBlock compactWarning">
          <div className="blockTitle">
            <AlertTriangle size={16} />
            Active Diagnostics
          </div>
          <strong>{errors.length} errors</strong>
          <span>{warnings.length} warnings</span>
        </section>
      ) : null}
    </aside>
  );
}

function WorkbenchFileState({ status }: { status: WorkbenchFileStatus | null }) {
  const className = status?.hasWorkbench ? "pass" : status?.state === "new" ? "warn" : "warn";
  return (
    <div className={`workbenchFileState ${className}`}>
      <span>{status ? status.state : "new"}</span>
      <strong>{status?.label ?? "Unsaved model"}</strong>
      <em>{status?.detail ?? "No workbench sidecar has been written yet."}</em>
      <small>{status ? formatRelativeTime(status.at) : "current session"}</small>
    </div>
  );
}

function RecentFileList({
  files,
  isBusy,
  onOpen
}: {
  files: RecentModelFile[];
  isBusy: boolean;
  onOpen: (path: string) => Promise<boolean>;
}) {
  return (
    <div className="recentFileList">
      {files.map((file) => (
        <button key={file.path} onClick={() => void onOpen(file.path)} disabled={isBusy} title={file.path}>
          <span>{file.hasWorkbench ? "UX" : file.kind}</span>
          <strong>{file.label}</strong>
          <em>{formatRelativeTime(file.touchedAt)}</em>
        </button>
      ))}
    </div>
  );
}

function formatRelativeTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "recent";
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return "now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.round(hours / 24);
  return `${days}d`;
}
