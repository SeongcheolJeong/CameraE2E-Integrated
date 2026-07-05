import { Activity, AlertTriangle, Gauge, Save, ShieldCheck } from "lucide-react";
import { useMemo } from "react";
import { useWorkbench } from "../lib/store";
import { deriveMetrics, statusLabel } from "../lib/uxMetrics";
import { Diagnostics } from "./Diagnostics";

export function KpiDashboard() {
  const model = useWorkbench((state) => state.model);
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);
  const dirty = useWorkbench((state) => state.dirty);
  const analysisStale = useWorkbench((state) => state.analysisStale);
  const summary = useWorkbench((state) => state.summary);
  const requirements = useWorkbench((state) => state.requirements);
  const autosaveSavedAt = useWorkbench((state) => state.autosaveSavedAt);
  const autosavePath = useWorkbench((state) => state.autosavePath);
  const metrics = useMemo(() => deriveMetrics(model, warnings, errors, summary, requirements), [errors, model, requirements, summary, warnings]);
  const efl = summary?.firstOrder.efl;
  const fno = summary?.firstOrder.fno;
  const dashboardMetrics = metrics.metrics.filter((metric) => ["trace", "spot", "illumination", "distortion", "cra", "yield", "snr"].includes(metric.key));
  const surfaceCount = model?.surfaces.length ?? metrics.surfaceCount;

  return (
    <footer className="kpiDashboard">
      <div className="kpiStatus">
        <div className="statusStack">
          <span className={`readyDot ${analysisStale ? "stale" : dirty ? "dirty" : ""}`} />
          <strong>{analysisStale ? "Stale" : dirty ? "Dirty" : "Ready"}</strong>
          <span>{analysisStale ? "Refresh required" : `${surfaceCount} surfaces`}</span>
        </div>
        <div className="statusStack">
          <Gauge size={15} />
          <strong>{formatFirstOrder(efl)}</strong>
          <span>EFL mm</span>
        </div>
        <div className="statusStack">
          <ShieldCheck size={15} />
          <strong>{formatFirstOrder(fno)}</strong>
          <span>F/#</span>
        </div>
        <div className="statusStack" title={autosavePath ?? "No draft file has been written yet."}>
          <Save size={15} />
          <strong>{formatAutosaveTime(autosaveSavedAt)}</strong>
          <span>Draft</span>
        </div>
      </div>
      <div className="kpiStrip">
        {dashboardMetrics.map((metric) => (
          <div key={metric.key} className={`kpiCard ${metric.status}`}>
            <div>
              <span>{metric.label}</span>
              <em>{statusLabel(metric.status)}</em>
            </div>
            <strong>
              {metric.value}
              {metric.unit ? <small>{metric.unit}</small> : null}
            </strong>
            <p>{metric.target}</p>
            <b className={`sourceBadge ${metric.source}`} title={metric.note ?? metric.source}>
              {metric.source}
            </b>
            <span className="scoreTrack">
              <span style={{ width: `${Math.round(metric.score * 100)}%` }} />
            </span>
          </div>
        ))}
      </div>
      <div className="diagnosticDock">
        <div className="dockTitle">
          {errors.length ? <AlertTriangle size={15} /> : <Activity size={15} />}
          Diagnostics
        </div>
        <Diagnostics />
      </div>
    </footer>
  );
}

function formatFirstOrder(value: number | null | undefined) {
  if (value === null || value === undefined) return "n/a";
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}

function formatAutosaveTime(value: string | null) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "n/a";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
