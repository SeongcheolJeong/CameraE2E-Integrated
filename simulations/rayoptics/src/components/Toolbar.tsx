import { FileDown, FilePlus2, FolderOpen, HelpCircle, Menu, Play, RefreshCw, RotateCcw, RotateCw, Save, Settings } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { buildReportSnapshot, exportReportSnapshot } from "../lib/report";
import { useWorkbench } from "../lib/store";
import type { AnalysisScale, ExampleFile, OpticalModel, PatentCompany, PatentDbStatus, PatentSearchResult, RecentModelFile, UpdateMode } from "../lib/types";
import { deriveMetrics } from "../lib/uxMetrics";

const updateModes: Array<{ value: UpdateMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "manual", label: "Manual" },
  { value: "layout-only", label: "Layout Only" },
  { value: "editors-only", label: "Editors Only" }
];

type FileDialogMode = "open" | "save";
type ControlDialog = "menu" | "settings";
type RadiusDisplay = "radius" | "curvature";

const analysisScales: Array<{ value: AnalysisScale; label: string }> = [
  { value: "same", label: "Same Axes" },
  { value: "all", label: "Fit Each" }
];

export function Toolbar() {
  const examples = useWorkbench((state) => state.examples);
  const recentFiles = useWorkbench((state) => state.recentFiles);
  const updateMode = useWorkbench((state) => state.updateMode);
  const model = useWorkbench((state) => state.model);
  const summary = useWorkbench((state) => state.summary);
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);
  const requirements = useWorkbench((state) => state.requirements);
  const requirementProfile = useWorkbench((state) => state.requirementProfile);
  const requirementProfileBase = useWorkbench((state) => state.requirementProfileBase);
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
  const autosavePath = useWorkbench((state) => state.autosavePath);
  const autosaveSavedAt = useWorkbench((state) => state.autosaveSavedAt);
  const draftCount = useWorkbench((state) => state.draftCount);
  const dirty = useWorkbench((state) => state.dirty);
  const canUndo = useWorkbench((state) => state.canUndo);
  const canRedo = useWorkbench((state) => state.canRedo);
  const newModel = useWorkbench((state) => state.newModel);
  const openModel = useWorkbench((state) => state.openModel);
  const openPatent = useWorkbench((state) => state.openPatent);
  const saveModel = useWorkbench((state) => state.saveModel);
  const refresh = useWorkbench((state) => state.refresh);
  const undo = useWorkbench((state) => state.undo);
  const redo = useWorkbench((state) => state.redo);
  const setUpdateMode = useWorkbench((state) => state.setUpdateMode);
  const isBusy = useWorkbench((state) => state.isBusy);
  const [fileDialog, setFileDialog] = useState<FileDialogMode | null>(null);
  const [controlDialog, setControlDialog] = useState<ControlDialog | null>(null);

  const defaultSavePath = suggestSavePath(model);
  const metrics = useMemo(() => deriveMetrics(model, warnings, errors, summary, requirements), [errors, model, requirements, summary, warnings]);
  const reportSnapshot = useMemo(
    () =>
      buildReportSnapshot(model, summary, metrics.metrics, metrics.risks, warnings, errors, requirements, requirementProfile, requirementProfileBase, {
        path: autosavePath,
        savedAt: autosaveSavedAt,
        draftCount
      }, compatibility, quickOptimizeResult, undefined, variants, {
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
      metrics.metrics,
      metrics.risks,
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
      summary,
      toleranceSweepMaxSurfaces,
      toleranceSweepPerturbationPct,
      toleranceSweepResult,
      toleranceSweepScope,
      variants,
      warnings
    ]
  );

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <button className="iconButton" title="Main menu" onClick={() => setControlDialog("menu")}>
            <Menu size={17} />
          </button>
          <span className="brandMark" />
          <div>
            <strong>Optical Design Cockpit</strong>
            <span>{model?.filename ?? model?.name ?? "unsaved model"}</span>
          </div>
        </div>
        <div className="toolGroup">
          <button title="New model" onClick={() => void newModel()} disabled={isBusy}>
            <FilePlus2 size={17} />
            New
          </button>
          <button title="Open local path" onClick={() => setFileDialog("open")} disabled={isBusy}>
            <FolderOpen size={17} />
            Open
          </button>
          <button title="Save as .roa" onClick={() => setFileDialog("save")} disabled={!model || isBusy}>
            <Save size={17} />
            Save
          </button>
          <button title="Export JSON analysis snapshot" onClick={() => exportReportSnapshot(reportSnapshot)} disabled={!model || isBusy}>
            <FileDown size={17} />
            Export
          </button>
        </div>
        <div className="toolGroup compact">
          <button title={canUndo ? "Undo last lens edit" : "No edit to undo"} disabled={!canUndo || isBusy} onClick={() => void undo()}>
            <RotateCcw size={17} />
          </button>
          <button title={canRedo ? "Redo lens edit" : "No edit to redo"} disabled={!canRedo || isBusy} onClick={() => void redo()}>
            <RotateCw size={17} />
          </button>
          <button title="Refresh model and active analysis" onClick={() => void refresh()} disabled={!model || isBusy}>
            <RefreshCw size={17} />
            Refresh
          </button>
          <button className="runTraceTop" title="Run active analysis" onClick={() => void refresh()} disabled={!model || isBusy}>
            <Play size={17} />
            Run
          </button>
        </div>
        <div className="modeGroup" role="group" aria-label="Update mode">
          {updateModes.map((mode) => (
            <button
              key={mode.value}
              className={mode.value === updateMode ? "active" : ""}
              onClick={() => setUpdateMode(mode.value)}
              title={`Update mode: ${mode.label}`}
              disabled={isBusy}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <div className="windowTools">
          <button title="RayOptics documentation" onClick={openRayOpticsDocs}>
            <HelpCircle size={16} />
          </button>
          <button title="Workbench settings" onClick={() => setControlDialog("settings")}>
            <Settings size={16} />
          </button>
        </div>
        {dirty ? <span className="dirtyDot" title="Unsaved changes" /> : null}
      </header>
      {controlDialog === "menu" ? (
        <CommandDialog
          model={model}
          isBusy={isBusy}
          onCancel={() => setControlDialog(null)}
          onNew={async () => {
            await newModel();
            setControlDialog(null);
          }}
          onOpen={() => {
            setControlDialog(null);
            setFileDialog("open");
          }}
          onSave={() => {
            setControlDialog(null);
            setFileDialog("save");
          }}
          onExport={() => exportReportSnapshot(reportSnapshot)}
          onSettings={() => setControlDialog("settings")}
          onDocs={openRayOpticsDocs}
        />
      ) : null}
      {controlDialog === "settings" ? <SettingsDialog onCancel={() => setControlDialog(null)} /> : null}
      {fileDialog ? (
        <FileDialog
          mode={fileDialog}
          examples={examples}
          recentFiles={recentFiles}
          model={model}
          defaultPath={fileDialog === "save" ? defaultSavePath : model?.filename ?? ""}
          isBusy={isBusy}
          onCancel={() => setFileDialog(null)}
          onOpen={async (path) => {
            const ok = await openModel(path);
            if (ok) {
              setFileDialog(null);
              return null;
            }
            return useWorkbench.getState().errors[0] ?? "Open failed.";
          }}
          onOpenPatent={async (simulationId) => {
            const ok = await openPatent(simulationId);
            if (ok) {
              setFileDialog(null);
              return null;
            }
            return useWorkbench.getState().errors[0] ?? "Patent DB open failed.";
          }}
          onSave={async (path, overwrite) => {
            const ok = await saveModel(path, overwrite);
            if (ok) {
              setFileDialog(null);
              return null;
            }
            return useWorkbench.getState().errors[0] ?? "Save failed.";
          }}
        />
      ) : null}
    </>
  );
}

function openRayOpticsDocs() {
  window.open("https://ray-optics.readthedocs.io/en/latest/", "_blank", "noopener,noreferrer");
}

function suggestSavePath(model: OpticalModel | null) {
  const fallback = "/Users/seongcheoljeong/RayOptics/design.roa";
  if (!model?.filename) return fallback;
  const filename = model.filename;
  const basename = filename.split("/").pop() || "design.roa";
  const roaName = basename.replace(/\.[^.]+$/, "") + ".roa";
  if (filename.includes("/site-packages/rayoptics/") || filename.includes("/rayoptics-env/lib/")) {
    return `/Users/seongcheoljeong/RayOptics/${roaName}`;
  }
  return filename.endsWith(".roa") ? filename : filename.replace(/\.[^.]+$/, ".roa");
}

function FileDialog({
  mode,
  examples,
  recentFiles,
  model,
  defaultPath,
  isBusy,
  onCancel,
  onOpen,
  onOpenPatent,
  onSave
}: {
  mode: FileDialogMode;
  examples: ExampleFile[];
  recentFiles: RecentModelFile[];
  model: OpticalModel | null;
  defaultPath: string;
  isBusy: boolean;
  onCancel: () => void;
  onOpen: (path: string) => Promise<string | null>;
  onOpenPatent: (simulationId: string) => Promise<string | null>;
  onSave: (path: string, overwrite?: boolean) => Promise<string | null>;
}) {
  const [path, setPath] = useState(defaultPath);
  const [query, setQuery] = useState("");
  const [patentQuery, setPatentQuery] = useState("");
  const [patentCompany, setPatentCompany] = useState("all");
  const [patentStatus, setPatentStatus] = useState<PatentDbStatus | null>(null);
  const [patentCompanies, setPatentCompanies] = useState<PatentCompany[]>([]);
  const [patentResults, setPatentResults] = useState<PatentSearchResult[]>([]);
  const [patentLoading, setPatentLoading] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [confirmOverwrite, setConfirmOverwrite] = useState(false);
  const trimmedPath = path.trim();
  const isUnsupportedSmx = /\.smx$/i.test(trimmedPath);
  const filteredExamples = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return examples
      .filter((example) => /\.(roa|seq|zmx)$/i.test(example.path))
      .filter((example) => !normalized || `${example.label} ${example.path}`.toLowerCase().includes(normalized))
      .slice(0, 12);
  }, [examples, query]);
  const filteredRecent = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return recentFiles
      .filter((file) => !normalized || `${file.label} ${file.path}`.toLowerCase().includes(normalized))
      .slice(0, 6);
  }, [query, recentFiles]);

  useEffect(() => {
    if (mode !== "open") return;
    let cancelled = false;
    async function loadPatentDb() {
      try {
        const [statusPayload, companiesPayload] = await Promise.all([api.patentStatus(), api.patentCompanies()]);
        if (cancelled) return;
        setPatentStatus(statusPayload);
        setPatentCompanies(companiesPayload.companies.filter((company) => company.camerae2eReady > 0));
      } catch (error) {
        if (!cancelled) {
          setPatentStatus({ path: "", exists: false, summary: null });
          setDialogError(error instanceof Error ? error.message : String(error));
        }
      }
    }
    void loadPatentDb();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  useEffect(() => {
    if (mode !== "open" || patentStatus?.exists === false) return;
    let cancelled = false;
    async function loadPatentResults() {
      setPatentLoading(true);
      try {
        const payload = await api.patentSearch({
          company: patentCompany === "all" ? undefined : patentCompany,
          query: patentQuery.trim() || undefined,
          status: "camerae2e_ready",
          limit: 60
        });
        if (!cancelled) {
          setPatentResults(payload.results);
        }
      } catch (error) {
        if (!cancelled) {
          setPatentResults([]);
          setDialogError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (!cancelled) {
          setPatentLoading(false);
        }
      }
    }
    void loadPatentResults();
    return () => {
      cancelled = true;
    };
  }, [mode, patentCompany, patentQuery, patentStatus?.exists]);

  async function submit() {
    if (!trimmedPath || isUnsupportedSmx || isBusy) return;
    setDialogError(null);
    if (mode === "open") {
      setDialogError(await onOpen(trimmedPath));
      return;
    }
    const error = await onSave(trimmedPath, confirmOverwrite);
    setDialogError(error);
    setConfirmOverwrite(Boolean(error?.startsWith("File already exists:")));
  }

  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="fileDialog" role="dialog" aria-modal="true" aria-label={mode === "open" ? "Open model" : "Save model"}>
        <div className="fileDialogHeader">
          <div>
            <span className="eyebrow">{mode === "open" ? "Open" : "Save"}</span>
            <h2>{mode === "open" ? "Model File" : model?.name ?? "Unsaved Model"}</h2>
          </div>
          <button onClick={onCancel}>Cancel</button>
        </div>
        <div className="fileDialogBody">
          <label className="pathField">
            <span>{mode === "open" ? "Path" : "Save Path"}</span>
            <input
              value={path}
              onChange={(event) => {
                setPath(event.target.value);
                setDialogError(null);
                setConfirmOverwrite(false);
              }}
              autoFocus
            />
          </label>
          {isUnsupportedSmx ? <div className="dialogWarning">SMX is marked experimental; use ROA, SEQ, or ZMX.</div> : null}
          {dialogError ? <div className="dialogWarning" aria-live="polite">{dialogError}</div> : null}
          {mode === "open" ? (
            <div className="examplePicker">
              {filteredRecent.length ? (
                <>
                  <span className="pickerLabel">Recent</span>
                  <div className="exampleResults recentResults">
                    {filteredRecent.map((file) => (
                      <button
                        key={file.path}
                        onClick={() => {
                          setDialogError(null);
                          void onOpen(file.path).then(setDialogError);
                        }}
                        disabled={isBusy}
                        title={file.path}
                      >
                        <span>{file.hasWorkbench ? "Workbench" : file.kind}</span>
                        <strong>{file.label}</strong>
                      </button>
                    ))}
                  </div>
                </>
              ) : null}
              <div className="patentPicker">
                <div className="pickerHeader">
                  <span className="pickerLabel">Patent DB</span>
                  <span>
                    {patentStatus?.summary
                      ? `${patentStatus.summary.camerae2eReady} ready / ${patentStatus.summary.simulationResults} rows`
                      : patentStatus?.exists === false
                        ? "not found"
                        : "loading"}
                  </span>
                </div>
                {patentStatus?.exists === false ? (
                  <div className="dialogWarning">Lens Patent DB was not found. Generate the CameraE2E patent dataset first.</div>
                ) : (
                  <>
                    <div className="patentControls">
                      <label className="pathField">
                        <span>Company</span>
                        <select value={patentCompany} onChange={(event) => setPatentCompany(event.target.value)}>
                          <option value="all">All ready companies</option>
                          {patentCompanies.map((company) => (
                            <option key={company.companySlug} value={company.companySlug}>
                              {company.company} ({company.camerae2eReady})
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="pathField">
                        <span>Search</span>
                        <input
                          value={patentQuery}
                          onChange={(event) => setPatentQuery(event.target.value)}
                          placeholder="Canon, p0014, US10310255B2"
                        />
                      </label>
                    </div>
                    <div className="exampleResults patentResults" aria-busy={patentLoading}>
                      {patentResults.slice(0, 12).map((patent) => (
                        <button
                          key={patent.simulationId}
                          onClick={() => {
                            setDialogError(null);
                            void onOpenPatent(patent.simulationId).then(setDialogError);
                          }}
                          disabled={isBusy || patentLoading}
                          title={`${patent.publicationNumber} ${patent.simulationId}`}
                        >
                          <span>
                            {patent.company} · {patent.configuration}
                          </span>
                          <strong>{patent.publicationNumber || patent.simulationId}</strong>
                          <em>
                            {formatPatentNumber(patent.focalLengthMm, "mm")} · F/{formatPatentNumber(patent.fNumber)} · {patent.surfaceCount}S / {patent.asphereCount}A
                          </em>
                        </button>
                      ))}
                      {!patentLoading && !patentResults.length ? <div className="emptyPickerState">No ready patent rows match the filter.</div> : null}
                    </div>
                  </>
                )}
              </div>
              <label className="pathField">
                <span>Examples</span>
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Sasian, telephoto, zmx" />
              </label>
              <div className="exampleResults">
                {filteredExamples.map((example) => (
                  <button
                    key={example.path}
                    onClick={() => {
                      setDialogError(null);
                      void onOpen(example.path).then(setDialogError);
                    }}
                    disabled={isBusy}
                    title={example.path}
                  >
                    <span>{example.kind}</span>
                    <strong>{example.label.replace(`${example.kind}: `, "")}</strong>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
        <div className="dialogActions">
          <span>{mode === "save" ? "ROA" : "ROA / SEQ / ZMX"}</span>
          <button className={confirmOverwrite ? "primaryWide destructiveAction" : "primaryWide"} onClick={() => void submit()} disabled={!trimmedPath || isUnsupportedSmx || isBusy}>
            {mode === "open" ? "Open" : confirmOverwrite ? "Overwrite" : "Save"}
          </button>
        </div>
      </section>
    </div>
  );
}

function formatPatentNumber(value: number | null, unit = "") {
  if (value === null || !Number.isFinite(value)) return "n/a";
  const text = Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return unit ? `${text} ${unit}` : text;
}

function CommandDialog({
  model,
  isBusy,
  onCancel,
  onNew,
  onOpen,
  onSave,
  onExport,
  onSettings,
  onDocs
}: {
  model: OpticalModel | null;
  isBusy: boolean;
  onCancel: () => void;
  onNew: () => Promise<void>;
  onOpen: () => void;
  onSave: () => void;
  onExport: () => void;
  onSettings: () => void;
  onDocs: () => void;
}) {
  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="fileDialog commandDialog" role="dialog" aria-modal="true" aria-label="Main menu">
        <div className="fileDialogHeader">
          <div>
            <span className="eyebrow">Menu</span>
            <h2>{model?.name ?? "Optical Design Cockpit"}</h2>
          </div>
          <button onClick={onCancel}>Close</button>
        </div>
        <div className="commandList">
          <button onClick={() => void onNew()} disabled={isBusy}>
            <FilePlus2 size={17} />
            <span>New Model</span>
            <em>Sequential starter</em>
          </button>
          <button onClick={onOpen} disabled={isBusy}>
            <FolderOpen size={17} />
            <span>Open Model</span>
            <em>ROA / SEQ / ZMX</em>
          </button>
          <button onClick={onSave} disabled={!model || isBusy}>
            <Save size={17} />
            <span>Save As</span>
            <em>ROA file</em>
          </button>
          <button onClick={onExport} disabled={!model || isBusy}>
            <FileDown size={17} />
            <span>Export Snapshot</span>
            <em>JSON analysis</em>
          </button>
          <button onClick={onSettings}>
            <Settings size={17} />
            <span>Workbench Settings</span>
            <em>Display and analysis</em>
          </button>
          <button onClick={onDocs}>
            <HelpCircle size={17} />
            <span>RayOptics Docs</span>
            <em>ReadTheDocs</em>
          </button>
        </div>
      </section>
    </div>
  );
}

function SettingsDialog({ onCancel }: { onCancel: () => void }) {
  const model = useWorkbench((state) => state.model);
  const updateMode = useWorkbench((state) => state.updateMode);
  const radiusDisplay = useWorkbench((state) => state.radiusDisplay);
  const sampling = useWorkbench((state) => state.sampling);
  const scale = useWorkbench((state) => state.scale);
  const analysisFieldIndex = useWorkbench((state) => state.analysisFieldIndex);
  const analysisWavelengthIndex = useWorkbench((state) => state.analysisWavelengthIndex);
  const setUpdateMode = useWorkbench((state) => state.setUpdateMode);
  const setRadiusDisplay = useWorkbench((state) => state.setRadiusDisplay);
  const setSampling = useWorkbench((state) => state.setSampling);
  const setScale = useWorkbench((state) => state.setScale);
  const setAnalysisFieldIndex = useWorkbench((state) => state.setAnalysisFieldIndex);
  const setAnalysisWavelengthIndex = useWorkbench((state) => state.setAnalysisWavelengthIndex);

  function resetAnalysisScope() {
    setAnalysisFieldIndex(null);
    setAnalysisWavelengthIndex(null);
    setScale("same");
    setSampling(21);
  }

  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="fileDialog settingsDialog" role="dialog" aria-modal="true" aria-label="Workbench settings">
        <div className="fileDialogHeader">
          <div>
            <span className="eyebrow">Settings</span>
            <h2>Workbench Controls</h2>
          </div>
          <button onClick={onCancel}>Close</button>
        </div>
        <div className="settingsGrid">
          <label className="settingField">
            <span>Update Mode</span>
            <select value={updateMode} onChange={(event) => setUpdateMode(event.target.value as UpdateMode)}>
              {updateModes.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
          </label>
          <div className="settingField">
            <span>Lens Editor Units</span>
            <div className="settingsSegment">
              {(["radius", "curvature"] satisfies RadiusDisplay[]).map((mode) => (
                <button key={mode} className={radiusDisplay === mode ? "active" : ""} onClick={() => setRadiusDisplay(mode)}>
                  {mode === "radius" ? "Radius" : "Curvature"}
                </button>
              ))}
            </div>
          </div>
          <label className="settingField">
            <span>Analysis Field</span>
            <select
              value={analysisFieldIndex === null ? "all" : String(analysisFieldIndex)}
              onChange={(event) => setAnalysisFieldIndex(event.target.value === "all" ? null : Number(event.target.value))}
            >
              <option value="all">All Fields</option>
              {model?.system.field.fields.map((field, index) => (
                <option key={`${field.x}-${field.y}-${index}`} value={index}>
                  F{index + 1} ({formatNullable(field.x)}, {formatNullable(field.y)})
                </option>
              ))}
            </select>
          </label>
          <label className="settingField">
            <span>Analysis Wavelength</span>
            <select
              value={analysisWavelengthIndex === null ? "all" : String(analysisWavelengthIndex)}
              onChange={(event) => setAnalysisWavelengthIndex(event.target.value === "all" ? null : Number(event.target.value))}
            >
              <option value="all">All Wavelengths</option>
              {model?.system.wavelengths.values.map((wavelength, index) => (
                <option key={`${wavelength}-${index}`} value={index}>
                  W{index + 1} {wavelength === null ? "n/a" : `${wavelength.toPrecision(6)} nm`}
                </option>
              ))}
            </select>
          </label>
          <label className="settingField">
            <span>Plot Scale</span>
            <select value={scale} onChange={(event) => setScale(event.target.value as AnalysisScale)}>
              {analysisScales.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
          </label>
          <label className="settingField">
            <span>Sampling</span>
            <input type="number" min={5} max={65} step={2} value={sampling} onChange={(event) => setSampling(Number(event.target.value))} />
          </label>
        </div>
        <div className="dialogActions">
          <button onClick={resetAnalysisScope}>Reset Analysis Scope</button>
          <button className="primaryWide" onClick={onCancel}>
            Done
          </button>
        </div>
      </section>
    </div>
  );
}

function formatNullable(value: number | null | undefined) {
  if (value === null || value === undefined) return "n/a";
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}
