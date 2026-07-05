import {
  Activity,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  Database,
  FileJson,
  Gauge,
  Image,
  Layers,
  Play,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Zap
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  exportDataset,
  exportReport,
  fetchAssetStatus,
  fetchPresets,
  runOptimization,
  runSimulation
} from "./api";
import type {
  AssetStatus,
  Candidate,
  DatasetResult,
  OptimizationResult,
  PresetsResponse,
  ReportResult,
  Settings,
  SimulationResult
} from "./types";

const workflow = [
  { id: "goal", label: "Goal", icon: Gauge },
  { id: "configure", label: "Configure", icon: SlidersHorizontal },
  { id: "simulate", label: "Simulate", icon: Play },
  { id: "optimize", label: "Optimize", icon: BarChart3 },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "report", label: "Report", icon: FileJson }
];

const pipeline = [
  { id: "scene_photons", label: "Scene" },
  { id: "oi_photons", label: "Optics" },
  { id: "sensor_raw", label: "Sensor" },
  { id: "ip_srgb", label: "ISP" },
  { id: "metrics", label: "Metrics" }
];

function App() {
  const [presets, setPresets] = useState<PresetsResponse | null>(null);
  const [assets, setAssets] = useState<AssetStatus | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [optimization, setOptimization] = useState<OptimizationResult | null>(null);
  const [dataset, setDataset] = useState<DatasetResult | null>(null);
  const [report, setReport] = useState<ReportResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [jsonUnlocked, setJsonUnlocked] = useState(false);

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchPresets(), fetchAssetStatus()])
      .then(([presetPayload, assetPayload]) => {
        if (!mounted) return;
        setPresets(presetPayload);
        setSettings(presetPayload.default_settings);
        setAssets(assetPayload);
      })
      .catch((exc: Error) => setError(exc.message));
    return () => {
      mounted = false;
    };
  }, []);

  const fdtdAvailable = Boolean(assets?.assets.fdtd_lut?.available);
  const tcadAvailable = Boolean(assets?.assets.tcad?.available);

  const updateSetting = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((current) => (current ? { ...current, [key]: value } : current));
  };

  const applyCameraPreset = (presetId: string) => {
    const preset = presets?.camera_presets.find((item) => item.id === presetId);
    if (!preset) {
      updateSetting("cameraPreset", presetId);
      return;
    }
    setSettings((current) =>
      current
        ? {
            ...current,
            cameraPreset: presetId,
            fovDeg: Number(preset.hfov_deg.toFixed(2)),
            pixelSizeUm: Number(preset.pixel_pitch_um.toFixed(2)),
            fNumber: Number(preset.f_number.toFixed(2))
          }
        : current
    );
  };

  const perform = async (action: string, callback: () => Promise<void>) => {
    setBusy(action);
    setError(null);
    try {
      await callback();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(null);
    }
  };

  if (!settings || !presets) {
    return (
      <main className="boot">
        <Camera size={28} />
        <span>Loading CameraE2E Workbench</span>
      </main>
    );
  }

  const metricById = Object.fromEntries((simulation?.metrics ?? []).map((item) => [item.id, item]));
  const activeBadges = simulation?.fidelity_badges ?? optimization?.fidelity_badges ?? assets?.fidelity_badges ?? [];
  const outputDir = dataset?.dataset_root ?? settings.outputDir;

  return (
    <div className="app-shell">
      <aside className="workflow-rail" aria-label="Workflow">
        <div className="rail-mark">
          <Camera size={22} />
        </div>
        {workflow.map((item) => {
          const Icon = item.icon;
          const active =
            (item.id === "simulate" && simulation) ||
            (item.id === "optimize" && optimization) ||
            (item.id === "dataset" && dataset) ||
            item.id === "configure";
          return (
            <button className={`rail-item ${active ? "active" : ""}`} key={item.id} title={item.label}>
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </aside>

      <main className="workbench">
        <header className="topbar">
          <div className="title-stack">
            <h1>CameraE2E Workbench</h1>
            <span>{busy ? `Running ${busy}` : "Local research-grade simulation and RAW export"}</span>
          </div>
          <label className="top-select">
            <span>Project Preset</span>
            <select value={settings.cameraPreset} onChange={(event) => applyCameraPreset(event.target.value)}>
              {presets.camera_presets.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <div className="run-meta">
            <span>Seed {settings.seed}</span>
            <span>{outputDir}</span>
          </div>
          <div className="top-actions">
            <button
              className="primary"
              disabled={Boolean(busy)}
              onClick={() => perform("Simulation", async () => setSimulation(await runSimulation(settings)))}
            >
              <Play size={16} />
              Run Simulation
            </button>
            <button
              className="secondary"
              disabled={Boolean(busy)}
              onClick={() => perform("Report", async () => setReport(await exportReport(settings)))}
            >
              <FileJson size={16} />
              Export Report
            </button>
          </div>
        </header>

        {error && <div className="error-strip">{error}</div>}

        <section className="workspace-grid">
          <section className="center-stack">
            <div className="pipeline-band">
              {pipeline.map((stage, index) => {
                const available =
                  stage.id === "metrics"
                    ? Boolean(simulation)
                    : Boolean(simulation?.stage_summaries.find((item) => item.name === stage.id)?.available);
                return (
                  <div className={`pipeline-node ${available ? "done" : ""}`} key={stage.id}>
                    <span>{stage.label}</span>
                    {available ? <CheckCircle2 size={14} /> : <ChevronRight size={14} />}
                    {index < pipeline.length - 1 && <div className="pipeline-line" />}
                  </div>
                );
              })}
            </div>

            <section className="core-layout">
              <div className="configure-panel panel">
                <PanelHeader icon={<Settings2 size={18} />} title="Configure" action="8 core knobs" />
                <div className="control-grid">
                  <SelectControl
                    label="Camera preset"
                    value={settings.cameraPreset}
                    onChange={applyCameraPreset}
                    options={presets.camera_presets.map((item) => ({ value: item.id, label: item.name }))}
                  />
                  <NumberControl
                    label="FOV"
                    value={settings.fovDeg}
                    min={35}
                    max={120}
                    step={0.5}
                    suffix="deg"
                    onChange={(value) => updateSetting("fovDeg", value)}
                  />
                  <NumberControl
                    label="Pixel size"
                    value={settings.pixelSizeUm}
                    min={1}
                    max={8}
                    step={0.05}
                    suffix="um"
                    onChange={(value) => updateSetting("pixelSizeUm", value)}
                  />
                  <SelectControl
                    label="CFA"
                    value={settings.cfaPreset}
                    onChange={(value) => updateSetting("cfaPreset", value)}
                    options={[
                      { value: "bayer_rgb", label: "Bayer RGB" },
                      { value: "quad_bayer_rgb", label: "Quad Bayer RGB" },
                      { value: "quad_bayer_bggr", label: "Quad Bayer BGGR" }
                    ]}
                  />
                  <SelectControl
                    label="OCL model"
                    value={settings.oclMode}
                    onChange={(value) => updateSetting("oclMode", value)}
                    options={[
                      { value: "off", label: "Off" },
                      { value: "centered", label: "Centered" },
                      { value: "optimal", label: "Optimal" }
                    ]}
                  />
                  <NumberControl
                    label="Exposure"
                    value={settings.exposureMs}
                    min={0.5}
                    max={20}
                    step={0.1}
                    suffix="ms"
                    onChange={(value) => updateSetting("exposureMs", value)}
                  />
                  <NumberControl
                    label="Lens PSF radius"
                    value={settings.lensPsfRadiusUm}
                    min={0.5}
                    max={8}
                    step={0.1}
                    suffix="um"
                    onChange={(value) => updateSetting("lensPsfRadiusUm", value)}
                  />
                  <ToggleControl
                    label="FDTD LUT"
                    checked={settings.fdtdEnabled}
                    disabled={!fdtdAvailable}
                    onChange={(value) => updateSetting("fdtdEnabled", value)}
                    detail={fdtdAvailable ? "LUT-backed" : "Missing LUT"}
                  />
                </div>
                <button className="text-button" onClick={() => setAdvancedOpen((value) => !value)}>
                  <SlidersHorizontal size={15} />
                  Advanced
                </button>
                {advancedOpen && (
                  <div className="advanced-panel">
                    <SelectControl
                      label="Scene"
                      value={settings.sceneType}
                      onChange={(value) => updateSetting("sceneType", value)}
                      options={[
                        { value: "macbeth", label: "Macbeth" },
                        { value: "slanted bar", label: "Slanted bar" },
                        { value: "uniform", label: "Uniform" }
                      ]}
                    />
                    <SelectControl
                      label="OCL group"
                      value={settings.oclGroupShape}
                      onChange={(value) => updateSetting("oclGroupShape", value)}
                      options={[
                        { value: "1x1", label: "1x1" },
                        { value: "2x2", label: "2x2" }
                      ]}
                    />
                    <NumberControl
                      label="OCL equalization"
                      value={settings.oclEqualization}
                      min={0}
                      max={1}
                      step={0.1}
                      onChange={(value) => updateSetting("oclEqualization", value)}
                    />
                    <SelectControl
                      label="Binning"
                      value={String(settings.binningFactor)}
                      onChange={(value) => updateSetting("binningFactor", Number(value))}
                      options={[
                        { value: "1", label: "Off" },
                        { value: "2", label: "2x proxy" }
                      ]}
                    />
                    <SelectControl
                      label="FDTD mode"
                      value={settings.fdtdMode}
                      onChange={(value) => updateSetting("fdtdMode", value)}
                      options={[
                        { value: "qe", label: "QE" },
                        { value: "qe+field", label: "QE + field" },
                        { value: "qe+field+crosstalk", label: "QE + field + crosstalk" }
                      ]}
                    />
                    <ToggleControl
                      label="TCAD collection"
                      checked={settings.tcadEnabled}
                      disabled={!tcadAvailable}
                      onChange={(value) => updateSetting("tcadEnabled", value)}
                      detail={tcadAvailable ? "calibration-required" : "Unavailable"}
                    />
                    <ToggleControl
                      label="HW ISP delay"
                      checked={settings.hwIspEnabled}
                      onChange={(value) => updateSetting("hwIspEnabled", value)}
                      detail="slower run"
                    />
                    <NumberControl
                      label="Search budget"
                      value={settings.maxCandidates}
                      min={4}
                      max={96}
                      step={1}
                      onChange={(value) => updateSetting("maxCandidates", Math.round(value))}
                    />
                  </div>
                )}
              </div>

              <div className="preview-panel panel">
                <PanelHeader icon={<Image size={18} />} title="Live Preview" action={simulation ? `${simulation.elapsed_ms} ms` : "not run"} />
                <div className="preview-frame">
                  {simulation?.preview_png ? (
                    <img src={simulation.preview_png} alt="CameraE2E simulation preview" />
                  ) : (
                    <div className="empty-state">
                      <Activity size={22} />
                      <span>Run Simulation to render actual output</span>
                    </div>
                  )}
                </div>
                <div className="metric-strip">
                  {["rgb_mean", "raw_std", "rgb_clip_fraction", "frame_count"].map((id) => (
                    <MetricTile key={id} metric={metricById[id]} />
                  ))}
                </div>
              </div>
            </section>

            <section className="optimization-panel panel">
              <PanelHeader icon={<Sparkles size={18} />} title="Quick Optimization" action={`${settings.maxCandidates} candidates`} />
              <div className="optimization-actions">
                <div className="objective-chips">
                  <span>maximize RGB mean</span>
                  <span>minimize clipping</span>
                  <span>target RAW std</span>
                </div>
                <button
                  className="primary compact"
                  disabled={Boolean(busy)}
                  onClick={() => perform("Optimization", async () => setOptimization(await runOptimization(settings)))}
                >
                  <Zap size={15} />
                  Quick Optimization
                </button>
              </div>
              <div className="optimization-grid">
                <ParetoPlot points={optimization?.pareto_points ?? []} />
                <CandidateTable candidates={optimization?.top_cases ?? []} best={optimization?.best_case ?? null} />
              </div>
            </section>
          </section>

          <aside className="right-panel">
            <section className="panel fidelity-panel">
              <PanelHeader icon={<Layers size={18} />} title="Fidelity" action={assets?.ok ? "registry ok" : "check"} />
              <div className="badge-stack">
                {activeBadges.map((badge) => (
                  <div className={`fidelity-badge tone-${badge.tone}`} key={badge.label}>
                    <span>{badge.label}</span>
                    <strong>{badge.tier}</strong>
                    <em>{badge.available ? (badge.active ? "active" : "available") : "unavailable"}</em>
                  </div>
                ))}
              </div>
              {assets?.validation.warnings?.[0] && (
                <div className="warning-box">
                  <strong>{assets.validation.warnings[0].entry}</strong>
                  <span>{assets.validation.warnings[0].message}</span>
                </div>
              )}
            </section>

            <section className="panel candidate-panel">
              <PanelHeader icon={<BarChart3 size={18} />} title="Best Candidate" action={optimization ? `${optimization.elapsed_ms} ms` : "not run"} />
              {optimization?.best_case ? (
                <CandidateSummary candidate={optimization.best_case} />
              ) : (
                <div className="empty-small">Optimization results will appear here.</div>
              )}
            </section>

            <section className="panel json-panel">
              <PanelHeader icon={<FileJson size={18} />} title="Configure JSON" action={jsonUnlocked ? "editable" : "locked"} />
              <label className="unlock-row">
                <input type="checkbox" checked={jsonUnlocked} onChange={(event) => setJsonUnlocked(event.target.checked)} />
                Unlock editor
              </label>
              <textarea
                value={JSON.stringify(settings, null, 2)}
                readOnly={!jsonUnlocked}
                onChange={(event) => {
                  if (!jsonUnlocked) return;
                  try {
                    setSettings(JSON.parse(event.target.value) as Settings);
                  } catch {
                    return;
                  }
                }}
              />
            </section>
          </aside>
        </section>

        <section className="artifact-drawer">
          <div className="drawer-header">
            <div>
              <strong>RAW Dataset Factory</strong>
              <span>RAW NPZ + RGB preview + labels JSON + metadata JSONL</span>
            </div>
            <div className="drawer-actions">
              <SelectControl
                label="Selection"
                value={settings.datasetSelection}
                onChange={(value) => updateSetting("datasetSelection", value)}
                options={[
                  { value: "best", label: "Best" },
                  { value: "top", label: "Top" },
                  { value: "pareto", label: "Pareto" }
                ]}
              />
              <NumberControl
                label="Cases"
                value={settings.datasetCaseCount}
                min={1}
                max={8}
                step={1}
                onChange={(value) => updateSetting("datasetCaseCount", Math.round(value))}
              />
              <button
                className="secondary"
                disabled={Boolean(busy)}
                onClick={() => perform("Dataset Export", async () => setDataset(await exportDataset(settings)))}
              >
                <Database size={16} />
                Export Dataset
              </button>
            </div>
          </div>
          <div className="artifact-grid">
            <ArtifactItem label="RAW NPZ" value={dataset?.records?.[0]?.raw ?? "--"} />
            <ArtifactItem label="RGB Preview" value={dataset?.records?.[0]?.rgb ?? "--"} />
            <ArtifactItem label="Labels JSON" value={dataset?.records?.[0]?.labels ?? "--"} />
            <ArtifactItem label="Dataset Manifest" value={dataset?.manifest_path ?? "--"} />
            <ArtifactItem label="Report JSON" value={report?.json_path ?? "--"} />
            <ArtifactItem label="Report HTML" value={report?.html_path ?? "--"} />
          </div>
        </section>
      </main>
    </div>
  );
}

function PanelHeader({ icon, title, action }: { icon: React.ReactNode; title: string; action: string }) {
  return (
    <div className="panel-header">
      <div>
        {icon}
        <h2>{title}</h2>
      </div>
      <span>{action}</span>
    </div>
  );
}

function SelectControl({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((item) => (
          <option value={item.value} key={item.value}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="control number-control">
      <span>{label}</span>
      <div>
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {suffix && <em>{suffix}</em>}
      </div>
    </label>
  );
}

function ToggleControl({
  label,
  checked,
  disabled,
  detail,
  onChange
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  detail?: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className={`toggle-control ${disabled ? "disabled" : ""}`}>
      <span>
        {label}
        {detail && <em>{detail}</em>}
      </span>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <i />
    </label>
  );
}

function MetricTile({ metric }: { metric?: { label: string; value: number | string | null; unit: string; area: string } }) {
  return (
    <div className="metric-tile">
      <span>{metric?.label ?? "--"}</span>
      <strong>{metric?.value ?? "--"}</strong>
      <em>{metric?.unit ?? ""}</em>
    </div>
  );
}

function ParetoPlot({
  points
}: {
  points: Array<{ case_index: number; score: number | null; x: number | null; y: number | null; clip: number | null }>;
}) {
  const valid = points.filter((item) => item.x !== null && item.y !== null) as Array<{
    case_index: number;
    score: number | null;
    x: number;
    y: number;
    clip: number | null;
  }>;
  const bounds = useMemo(() => {
    if (!valid.length) return null;
    const xs = valid.map((item) => item.x);
    const ys = valid.map((item) => item.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    };
  }, [valid]);

  if (!bounds) {
    return (
      <div className="plot-empty">
        <BarChart3 size={22} />
        <span>Run Quick Optimization to populate Pareto points</span>
      </div>
    );
  }

  const scaleX = (x: number) => 32 + ((x - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-9)) * 236;
  const scaleY = (y: number) => 168 - ((y - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-9)) * 126;

  return (
    <div className="pareto-card">
      <svg viewBox="0 0 300 210" role="img" aria-label="Pareto scatter">
        <line x1="32" y1="168" x2="274" y2="168" />
        <line x1="32" y1="168" x2="32" y2="34" />
        {valid.map((item) => (
          <circle
            key={item.case_index}
            cx={scaleX(item.x)}
            cy={scaleY(item.y)}
            r={item.clip && item.clip > 0 ? 6 : 5}
          />
        ))}
        <text x="34" y="196">
          RAW std
        </text>
        <text x="36" y="26">
          RGB mean
        </text>
      </svg>
    </div>
  );
}

function CandidateTable({ candidates, best }: { candidates: Candidate[]; best: Candidate | null }) {
  if (!candidates.length) {
    return <div className="candidate-table empty-small">No evaluated candidates yet.</div>;
  }
  return (
    <div className="candidate-table">
      <table>
        <thead>
          <tr>
            <th>Case</th>
            <th>Score</th>
            <th>Pixel um</th>
            <th>CFA</th>
            <th>PSF um</th>
          </tr>
        </thead>
        <tbody>
          {candidates.slice(0, 6).map((item) => (
            <tr className={item.case_index === best?.case_index ? "best" : ""} key={item.case_index}>
              <td>{item.case_index}</td>
              <td>{item.score ?? "--"}</td>
              <td>{String(item.parameters["sensor.pixel_size"] ?? "--")}</td>
              <td>{String(item.parameters["sensor.cfa_preset"] ?? "--")}</td>
              <td>{String(item.parameters["optics.si_psf_radius_um"] ?? "--")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CandidateSummary({ candidate }: { candidate: Candidate }) {
  const entries = Object.entries(candidate.parameters).slice(0, 7);
  return (
    <div className="candidate-summary">
      <div className="score-circle">
        <span>Score</span>
        <strong>{candidate.score ?? "--"}</strong>
      </div>
      <div className="param-list">
        {entries.map(([key, value]) => (
          <div key={key}>
            <span>{key.replace("sensor.", "S.").replace("optics.", "O.")}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function ArtifactItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="artifact-item">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}

export default App;
