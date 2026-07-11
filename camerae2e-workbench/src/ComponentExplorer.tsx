import {
  Aperture,
  ArrowLeft,
  ArrowRight,
  Check,
  CircleAlert,
  Cpu,
  GitCompareArrows,
  LoaderCircle,
  Microscope,
  Play,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  X
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  applyComponentModule,
  artifactUrl,
  compareComponentModules,
  evaluateComponentModules,
  fetchLensComponent,
  fetchSensorComponent,
  searchLensComponents,
  searchSensorComponents
} from "./api";
import type {
  ComponentSearchResponse,
  ComponentSelection,
  JobRecord,
  LensComponent,
  ModuleApplicationResponse,
  ModuleCompatibilityResponse,
  ProjectPayload,
  SensorComponent,
  StudyRecord,
  StudySpec
} from "./types";

type ExplorerProps = {
  project: ProjectPayload;
  study: StudyRecord;
  draft: StudySpec;
  comparisonJob: JobRecord | null;
  activeJob: boolean;
  onClose: () => void;
  onApplied: (payload: ModuleApplicationResponse) => void;
  onJobSubmitted: (job: JobRecord) => void;
  onError: (message: string) => void;
};

type Tab = "lenses" | "sensors" | "modules";

export default function ComponentExplorer({
  project,
  study,
  draft,
  comparisonJob,
  activeJob,
  onClose,
  onApplied,
  onJobSubmitted,
  onError
}: ExplorerProps) {
  const [tab, setTab] = useState<Tab>("lenses");
  const [lensQuery, setLensQuery] = useState("");
  const [sensorQuery, setSensorQuery] = useState("");
  const [lensCompany, setLensCompany] = useState("");
  const [sensorMaker, setSensorMaker] = useState("");
  const [requirePsf, setRequirePsf] = useState(false);
  const [requireDti, setRequireDti] = useState(false);
  const [lensSimulationReady, setLensSimulationReady] = useState(true);
  const [simulationReady, setSimulationReady] = useState(true);
  const [lensPage, setLensPage] = useState(1);
  const [sensorPage, setSensorPage] = useState(1);
  const [lenses, setLenses] = useState<ComponentSearchResponse<LensComponent> | null>(null);
  const [sensors, setSensors] = useState<ComponentSearchResponse<SensorComponent> | null>(null);
  const [selectedLensId, setSelectedLensId] = useState(draft.baseline.lens.model_id ?? "");
  const [selectedSensorId, setSelectedSensorId] = useState(draft.baseline.sensor.model_id ?? "");
  const [lensDetail, setLensDetail] = useState<Record<string, any> | null>(null);
  const [sensorDetail, setSensorDetail] = useState<Record<string, any> | null>(null);
  const [tray, setTray] = useState<ComponentSelection[]>([]);
  const [compatibility, setCompatibility] = useState<ModuleCompatibilityResponse | null>(null);
  const [pairCompatibility, setPairCompatibility] = useState<ModuleCompatibilityResponse["candidates"][number] | null>(null);
  const [loading, setLoading] = useState<string | null>("catalog");

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void searchLensComponents({
        query: lensQuery,
        company: lensCompany,
        require_psf: requirePsf || undefined,
        simulation_ready: lensSimulationReady || undefined,
        page: lensPage,
        page_size: 40,
        sort: "company"
      })
        .then(setLenses)
        .catch((error: Error) => onError(error.message))
        .finally(() => setLoading((current) => current === "catalog" ? null : current));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [lensQuery, lensCompany, requirePsf, lensSimulationReady, lensPage, onError]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void searchSensorComponents({
        query: sensorQuery,
        manufacturer: sensorMaker,
        has_dti: requireDti || undefined,
        simulation_ready: simulationReady || undefined,
        page: sensorPage,
        page_size: 40,
        sort: "pixel_pitch"
      })
        .then(setSensors)
        .catch((error: Error) => onError(error.message));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [sensorQuery, sensorMaker, requireDti, simulationReady, sensorPage, onError]);

  useEffect(() => {
    if (!selectedLensId) {
      setLensDetail(null);
      return;
    }
    void fetchLensComponent(selectedLensId).then(setLensDetail).catch(() => setLensDetail(null));
  }, [selectedLensId]);

  useEffect(() => {
    if (!selectedSensorId) {
      setSensorDetail(null);
      return;
    }
    void fetchSensorComponent(selectedSensorId).then(setSensorDetail).catch(() => setSensorDetail(null));
  }, [selectedSensorId]);

  useEffect(() => {
    if (!tray.length) {
      setCompatibility(null);
      return;
    }
    setLoading("compatibility");
    void evaluateComponentModules(draft.requirements, tray)
      .then(setCompatibility)
      .catch((error: Error) => onError(error.message))
      .finally(() => setLoading(null));
  }, [tray, draft.requirements, onError]);

  const selectedLens = useMemo(
    () => lenses?.items.find((item) => item.id === selectedLensId) ?? lensDetail as LensComponent | null,
    [lenses, lensDetail, selectedLensId]
  );
  const selectedSensor = useMemo(
    () => sensors?.items.find((item) => item.id === selectedSensorId) ?? sensorDetail as SensorComponent | null,
    [sensors, sensorDetail, selectedSensorId]
  );
  const selectedPair = useMemo<ComponentSelection | null>(
    () => selectedLens && selectedSensor
      ? { lens_id: selectedLens.id, sensor_id: selectedSensor.id, use_geometric_psf: Boolean(selectedLens.psf_available) }
      : null,
    [selectedLens, selectedSensor]
  );

  useEffect(() => {
    let cancelled = false;
    if (!selectedPair) {
      setPairCompatibility(null);
      return () => { cancelled = true; };
    }
    void evaluateComponentModules(draft.requirements, [selectedPair])
      .then((payload) => {
        if (!cancelled) setPairCompatibility(payload.candidates[0] ?? null);
      })
      .catch(() => {
        if (!cancelled) setPairCompatibility(null);
      });
    return () => { cancelled = true; };
  }, [selectedPair, draft.requirements]);

  const addPair = () => {
    if (!selectedPair) return;
    if (tray.some((item) => item.lens_id === selectedPair.lens_id && item.sensor_id === selectedPair.sensor_id)) return;
    if (tray.length >= 4) return;
    setTray((current) => [...current, selectedPair]);
    setTab("modules");
  };

  const applyPair = async (selection: ComponentSelection) => {
    setLoading("apply");
    try {
      const payload = await applyComponentModule(project.info.id, study.id, selection);
      onApplied(payload);
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(null);
    }
  };

  const runComparison = async () => {
    if (tray.length < 2) return;
    setLoading("compare");
    try {
      const payload = await compareComponentModules(
        project.info.id,
        study.id,
        tray,
        draft.scenes[0]?.id
      );
      onJobSubmitted(payload.job);
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(null);
    }
  };

  const currentCompatibility = pairCompatibility;

  return (
    <div className="component-explorer" role="dialog" aria-modal="true" aria-label="Camera component explorer">
      <header className="explorer-header">
        <div><Microscope size={19} /><strong>Component Explorer</strong><span>ADAS module selection</span></div>
        <div className="explorer-tabs" role="tablist">
          <button className={tab === "lenses" ? "active" : ""} onClick={() => setTab("lenses")}><Aperture size={15} /> Lenses <span>{lenses?.total ?? 0}</span></button>
          <button className={tab === "sensors" ? "active" : ""} onClick={() => setTab("sensors")}><Cpu size={15} /> Sensors <span>{sensors?.total ?? 0}</span></button>
          <button className={tab === "modules" ? "active" : ""} onClick={() => setTab("modules")}><GitCompareArrows size={15} /> Modules <span>{tray.length}</span></button>
        </div>
        <button className="icon-button" title="Close explorer" onClick={onClose}><X size={18} /></button>
      </header>

      <div className="module-selection-bar">
        <SelectionSlot icon={Aperture} label="Lens" value={selectedLens ? `${selectedLens.company} ${selectedLens.id}` : "Not selected"} onClick={() => setTab("lenses")} />
        <SelectionSlot icon={Cpu} label="Sensor" value={selectedSensor ? `${selectedSensor.manufacturer} ${selectedSensor.device_name}` : "Not selected"} onClick={() => setTab("sensors")} />
        <span className={`module-state ${currentCompatibility?.status ?? "pending"}`}>{currentCompatibility?.status ?? "not evaluated"}</span>
        <button className="secondary-button" disabled={!selectedPair || tray.length >= 4} onClick={addPair}><GitCompareArrows size={15} /> Add to Compare</button>
        <button className="primary-button" disabled={!selectedPair || !currentCompatibility || currentCompatibility.status === "incompatible" || loading === "apply"} onClick={() => selectedPair && void applyPair(selectedPair)}><Check size={15} /> Use as Baseline</button>
      </div>

      {tab === "lenses" && (
        <div className="explorer-body">
          <aside className="catalog-filters">
            <FilterSearch value={lensQuery} onChange={(value) => { setLensQuery(value); setLensPage(1); }} placeholder="Lens, patent, company" />
            <FilterSelect label="Company" value={lensCompany} options={(lenses?.facets.companies ?? []) as string[]} onChange={(value) => { setLensCompany(value); setLensPage(1); }} />
            <label className="check-filter"><input type="checkbox" checked={lensSimulationReady} onChange={(event) => { setLensSimulationReady(event.target.checked); setLensPage(1); }} /> Simulation-ready only</label>
            <label className="check-filter"><input type="checkbox" checked={requirePsf} onChange={(event) => { setRequirePsf(event.target.checked); setLensPage(1); }} /> Geometric PSF available</label>
            <EvidenceLegend />
          </aside>
          <CatalogTable kind="lens" items={lenses?.items ?? []} selectedId={selectedLensId} onSelect={setSelectedLensId} />
          <ComponentInspector kind="lens" item={lensDetail ?? selectedLens} />
          <Pager page={lensPage} total={lenses?.total ?? 0} pageSize={40} onChange={setLensPage} />
        </div>
      )}

      {tab === "sensors" && (
        <div className="explorer-body">
          <aside className="catalog-filters">
            <FilterSearch value={sensorQuery} onChange={(value) => { setSensorQuery(value); setSensorPage(1); }} placeholder="Sensor, code, manufacturer" />
            <FilterSelect label="Manufacturer" value={sensorMaker} options={(sensors?.facets.manufacturers ?? []) as string[]} onChange={(value) => { setSensorMaker(value); setSensorPage(1); }} />
            <label className="check-filter"><input type="checkbox" checked={simulationReady} onChange={(event) => { setSimulationReady(event.target.checked); setSensorPage(1); }} /> Simulation-ready only</label>
            <label className="check-filter"><input type="checkbox" checked={requireDti} onChange={(event) => { setRequireDti(event.target.checked); setSensorPage(1); }} /> DTI structure</label>
            <EvidenceLegend />
          </aside>
          <CatalogTable kind="sensor" items={sensors?.items ?? []} selectedId={selectedSensorId} onSelect={setSelectedSensorId} />
          <ComponentInspector kind="sensor" item={sensorDetail ?? selectedSensor} />
          <Pager page={sensorPage} total={sensors?.total ?? 0} pageSize={40} onChange={setSensorPage} />
        </div>
      )}

      {tab === "modules" && (
        <div className="module-workspace">
          <section className="module-matrix-pane">
            <div className="module-pane-heading"><SlidersHorizontal size={16} /><strong>Compatibility Matrix</strong><span>{compatibility ? `${compatibility.status_counts.compatible ?? 0} compatible` : "No candidates"}</span></div>
            {compatibility ? <CompatibilityMatrix payload={compatibility} onRemove={(index) => setTray((items) => items.filter((_, itemIndex) => itemIndex !== index))} onApply={(index) => void applyPair(tray[index])} /> : <ExplorerEmpty label="Add a lens and sensor combination" />}
          </section>
          <section className="module-results-pane">
            <div className="module-pane-heading"><Play size={16} /><strong>Same-scene Evaluation</strong><span>{comparisonJob?.status ?? "not run"}</span></div>
            <ComparisonResults projectId={project.info.id} job={comparisonJob} />
          </section>
          <footer className="compare-action-bar">
            <span>{draft.scenes[0]?.name} · seed {draft.seed} · native geometry / downsampled simulation</span>
            <button className="primary-button" disabled={tray.length < 2 || activeJob || compatibility?.candidates.some((item) => item.status === "incompatible") || loading === "compare"} onClick={() => void runComparison()}>{activeJob ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />} Compare on Scene</button>
          </footer>
        </div>
      )}

      {loading && <div className="explorer-loading"><LoaderCircle className="spin" size={16} /> {loading}</div>}
    </div>
  );
}

function SelectionSlot({ icon: Icon, label, value, onClick }: { icon: typeof Aperture; label: string; value: string; onClick: () => void }) {
  return <button className="selection-slot" onClick={onClick}><Icon size={16} /><span><small>{label}</small><strong>{value}</strong></span></button>;
}

function FilterSearch({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return <label className="catalog-search"><Search size={15} /><input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} /></label>;
}

function FilterSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="filter-control"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="">All</option>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>;
}

function EvidenceLegend() {
  return <div className="evidence-legend"><span><i className="source" /> Source-derived</span><span><i className="proxy" /> Proxy</span><span><i className="unknown" /> Unknown</span></div>;
}

function CatalogTable({ kind, items, selectedId, onSelect }: { kind: "lens" | "sensor"; items: Array<LensComponent | SensorComponent>; selectedId: string; onSelect: (id: string) => void }) {
  return (
    <div className="catalog-table-wrap">
      <table className="catalog-table">
        <thead>{kind === "lens" ? <tr><th>Company / Patent</th><th>Focal</th><th>F/#</th><th>FOV</th><th>Image H</th><th>PSF</th><th>Evidence</th></tr> : <tr><th>Manufacturer / Sensor</th><th>Pixel</th><th>Resolution</th><th>CFA</th><th>Shutter</th><th>DTI</th><th>Evidence</th></tr>}</thead>
        <tbody>{items.map((item) => kind === "lens" ? <LensRow key={item.id} item={item as LensComponent} selected={item.id === selectedId} onSelect={onSelect} /> : <SensorRow key={item.id} item={item as SensorComponent} selected={item.id === selectedId} onSelect={onSelect} />)}</tbody>
      </table>
    </div>
  );
}

function LensRow({ item, selected, onSelect }: { item: LensComponent; selected: boolean; onSelect: (id: string) => void }) {
  return <tr className={selected ? "selected" : ""} onClick={() => onSelect(item.id)}><td><strong>{item.company}</strong><small>{item.publication_number} · {item.configuration}</small></td><td>{number(item.focal_length_mm)} mm</td><td>{number(item.f_number)}</td><td>{number(item.field_of_view_deg)}°</td><td>{number(item.image_height_mm)} mm</td><td>{item.psf_available ? <Check size={14} /> : "--"}</td><td><EvidenceMeter value={item.evidence_completeness} /></td></tr>;
}

function SensorRow({ item, selected, onSelect }: { item: SensorComponent; selected: boolean; onSelect: (id: string) => void }) {
  return <tr className={selected ? "selected" : ""} onClick={() => onSelect(item.id)}><td><strong>{item.manufacturer}</strong><small>{item.device_name} · {item.code}</small></td><td>{number(item.pixel_pitch_um)} um</td><td>{number(item.resolution_mp)} MP</td><td>{item.cfa_pattern ?? "unknown"}</td><td>{item.shutter ?? "unknown"}</td><td>{flag(item.has_dti)}</td><td><EvidenceMeter value={item.evidence_completeness} /></td></tr>;
}

function EvidenceMeter({ value }: { value: number }) {
  return <span className="evidence-meter"><i style={{ width: `${Math.round(value * 100)}%` }} /> <small>{Math.round(value * 100)}%</small></span>;
}

function ComponentInspector({ kind, item }: { kind: "lens" | "sensor"; item: Record<string, any> | null }) {
  if (!item) return <aside className="component-inspector"><ExplorerEmpty label={`Select a ${kind}`} /></aside>;
  const lens = kind === "lens";
  return (
    <aside className="component-inspector">
      <div className="inspector-title">{lens ? <Aperture size={17} /> : <Cpu size={17} />}<span><strong>{lens ? `${item.company} ${item.id}` : `${item.manufacturer} ${item.device_name}`}</strong><small>{lens ? item.publication_number : item.code}</small></span></div>
      <InspectorGrid values={lens ? [
        ["Focal length", numberUnit(item.focal_length_mm, "mm")], ["F-number", number(item.f_number)], ["Max FOV", numberUnit(item.field_of_view_deg, "deg")], ["Image height", numberUnit(item.image_height_mm, "mm")], ["Surfaces", String(item.surface_count ?? "unknown")], ["Aspheres", String(item.asphere_count ?? "unknown")], ["Airy 550nm", numberUnit(item.airy_disk_diameter_um_550, "um")], ["Geometric PSF", item.psf_available ? "available" : "missing"]
      ] : [
        ["Pixel pitch", numberUnit(item.pixel_pitch_um, "um")], ["Resolution", numberUnit(item.resolution_mp, "MP")], ["Native size", item.native_size_rc?.join(" x ") ?? "unknown"], ["Geometry", item.geometry_source ?? "unknown"], ["Modality", item.sensor_modality ?? "unknown"], ["CFA", item.cfa_pattern ?? "unknown"], ["Frame RAW", item.module_configurable ? "supported" : "blocked"], ["Shutter", item.shutter ?? "unknown"], ["Illumination", item.illumination ?? "unknown"], ["Microlens", item.microlens_type ?? "unknown"]
      ]} />
      {!lens && <div className="feature-flags">{[["DTI", item.has_dti], ["PDAF", item.has_pdaf], ["HDR", item.has_hdr], ["LOFIC", item.has_lofic], ["Stacked", item.is_stacked], ["NIR", item.is_nir]].map(([label, value]) => <span className={value === true ? "yes" : value === false ? "no" : "unknown"} key={String(label)}>{String(label)} {flag(value as boolean | null)}</span>)}</div>}
      {lens && item.surfaces && <div className="surface-summary"><span>Prescription</span><strong>{item.surfaces.length} normalized surfaces</strong><small>{item.simulation_model} · patent-derived</small></div>}
      <div className="inspector-boundary"><ShieldAlert size={15} /><span>{item.truth_boundary ?? (lens ? "Patent-derived proxy" : "Source-derived proxy")}</span></div>
    </aside>
  );
}

function InspectorGrid({ values }: { values: string[][] }) {
  return <div className="inspector-grid">{values.map(([label, value]) => <span key={label}><small>{label}</small><strong>{value}</strong></span>)}</div>;
}

function CompatibilityMatrix({ payload, onRemove, onApply }: { payload: ModuleCompatibilityResponse; onRemove: (index: number) => void; onApply: (index: number) => void }) {
  const rows: Array<[string, string, string]> = [
    ["HFOV", "hfov_deg", "deg"], ["Object @ range", "object_pixels_at_range", "px"], ["Sensor diagonal", "sensor_diagonal_mm", "mm"], ["Airy sampling", "airy_diameter_pixels", "px"], ["Pixel rate", "pixel_rate_mpix_s", "Mpix/s"], ["Low-light proxy", "relative_photon_area_proxy", "rel"], ["Complexity", "complexity_index", "index"], ["Evidence", "evidence_completeness", "ratio"]
  ];
  return (
    <div className="compatibility-scroll">
      <table className="compatibility-matrix"><thead><tr><th>Metric</th>{payload.candidates.map((item, index) => <th key={item.id}><span>{item.lens.company} + {item.sensor.manufacturer}</span><small className={item.status}>{item.status}{item.pareto ? " · Pareto" : ""}</small><button title="Remove" onClick={() => onRemove(index)}><X size={13} /></button></th>)}</tr></thead><tbody>{rows.map(([label, key, unit]) => <tr key={key}><th>{label}</th>{payload.candidates.map((item) => <td key={item.id}>{numberUnit(item.derived[key] as number | null, unit)}</td>)}</tr>)}<tr><th>Failed gates</th>{payload.candidates.map((item) => <td className="gate-cell" key={item.id}>{item.failed_gate_ids.length ? item.failed_gate_ids.join(", ") : item.not_evaluable_ids.length ? `unknown: ${item.not_evaluable_ids.join(", ")}` : <Check size={14} />}</td>)}</tr></tbody></table>
      <div className="matrix-actions">{payload.candidates.map((item, index) => <button className="secondary-button" disabled={item.status === "incompatible"} key={item.id} onClick={() => onApply(index)}>Use {index + 1} as Baseline</button>)}</div>
      <div className="matrix-boundary"><CircleAlert size={14} />{payload.truth_boundary}</div>
    </div>
  );
}

function ComparisonResults({ projectId, job }: { projectId: string; job: JobRecord | null }) {
  if (!job) return <ExplorerEmpty label="Run a same-scene comparison" />;
  if (job.status === "queued" || job.status === "running") return <ExplorerEmpty label={`Comparison ${Math.round(job.progress * 100)}%`} loading />;
  if (job.status !== "succeeded") return <ExplorerEmpty label={job.error ?? `Comparison ${job.status}`} />;
  const evaluations = (job.result?.evaluations ?? []) as Array<Record<string, any>>;
  return <div className="comparison-results">{evaluations.map((item, index) => {
    const result = item.evaluation ?? {};
    const preview = result.preview_artifact_hash ? artifactUrl(projectId, result.preview_artifact_hash) : null;
    return <article className="comparison-result" key={`${item.selection?.lens_id}-${item.selection?.sensor_id}`}><header><strong>Module {index + 1}</strong><span>{item.compatibility?.status}</span></header>{preview ? <img src={preview} alt={`Module ${index + 1} camera output`} /> : <div className="result-placeholder" />}<dl><div><dt>Target</dt><dd>{number(result.evaluation?.target_score)}</dd></div><div><dt>Clip</dt><dd>{number(result.metrics?.artifact?.rgb_high_clip_fraction)}</dd></div><div><dt>RAW std</dt><dd>{number(result.metrics?.artifact?.raw_std)}</dd></div><div><dt>Fidelity</dt><dd>{String(result.fidelity?.effective ?? "--")}</dd></div></dl></article>;
  })}</div>;
}

function Pager({ page, total, pageSize, onChange }: { page: number; total: number; pageSize: number; onChange: (page: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return <div className="catalog-pager"><button title="Previous page" disabled={page <= 1} onClick={() => onChange(page - 1)}><ArrowLeft size={15} /></button><span>{page} / {pages}</span><button title="Next page" disabled={page >= pages} onClick={() => onChange(page + 1)}><ArrowRight size={15} /></button></div>;
}

function ExplorerEmpty({ label, loading = false }: { label: string; loading?: boolean }) {
  return <div className="explorer-empty">{loading ? <LoaderCircle className="spin" size={20} /> : <GitCompareArrows size={20} />}<span>{label}</span></div>;
}

function number(value: unknown): string {
  if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value))) return "--";
  const numeric = Number(value);
  return Math.abs(numeric) >= 100 ? numeric.toFixed(0) : Math.abs(numeric) >= 10 ? numeric.toFixed(1) : numeric.toFixed(2);
}

function numberUnit(value: unknown, unit: string): string {
  const formatted = number(value);
  return formatted === "--" ? formatted : `${formatted} ${unit}`;
}

function flag(value: boolean | null | undefined): string {
  return value === true ? "yes" : value === false ? "no" : "unknown";
}
