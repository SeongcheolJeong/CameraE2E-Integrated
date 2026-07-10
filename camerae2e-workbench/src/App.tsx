import {
  Activity,
  Archive,
  BarChart3,
  Boxes,
  Camera,
  Check,
  ChevronDown,
  CircleAlert,
  Database,
  FileChartColumn,
  FlaskConical,
  Gauge,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Play,
  RefreshCw,
  Save,
  ScanSearch,
  ShieldCheck,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  artifactUrl,
  bootstrapProject,
  fetchAssetStatus,
  fetchBenchmarkStatus,
  fetchJob,
  fetchJobs,
  fetchProject,
  scenePreviewUrl,
  submitOperation,
  updateStudy
} from "./api";
import type {
  ArtifactRecord,
  AssetStatus,
  BenchmarkStatus,
  DesignVariable,
  JobRecord,
  ProjectPayload,
  StudyRecord,
  StudySpec
} from "./types";

const workflow = [
  { id: "requirements", label: "Requirements", icon: Gauge },
  { id: "design-space", label: "Design Space", icon: SlidersHorizontal },
  { id: "study", label: "Study", icon: Activity },
  { id: "candidates", label: "Candidates", icon: BarChart3 },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "calibration", label: "Calibration", icon: FlaskConical },
  { id: "report", label: "Report", icon: FileChartColumn }
];

const terminalStatuses = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

function App() {
  const [project, setProject] = useState<ProjectPayload | null>(null);
  const [study, setStudy] = useState<StudyRecord | null>(null);
  const [draft, setDraft] = useState<StudySpec | null>(null);
  const [assets, setAssets] = useState<AssetStatus | null>(null);
  const [benchmarkStatus, setBenchmarkStatus] = useState<BenchmarkStatus | null>(null);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState<string | null>("Loading project");
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState("requirements");
  const [selectedCandidate, setSelectedCandidate] = useState<number>(0);
  const [executeSolvers, setExecuteSolvers] = useState(false);
  const [datasetCases, setDatasetCases] = useState(200);
  const [previewTab, setPreviewTab] = useState<"source" | "ideal" | "output" | "overlay">("source");
  const [calibration, setCalibration] = useState({
    kind: "generic",
    measured_path: "",
    simulated_path: ""
  });

  const refreshJobs = useCallback(async (projectId: string, studyId: string) => {
    const listed = await fetchJobs(projectId, studyId);
    const detailed = await Promise.all(
      listed.slice(0, 20).map((item) => fetchJob(projectId, item.id).catch(() => item))
    );
    setJobs([...detailed, ...listed.slice(20)]);
  }, []);

  useEffect(() => {
    let mounted = true;
    bootstrapProject()
      .then(async (payload) => {
        if (!mounted) return;
        const rememberedStudy = window.localStorage.getItem(
          `camerae2e:last-study:${payload.info.id}`
        );
        const firstStudy = payload.studies.find((item) => item.id === rememberedStudy)
          ?? [...payload.studies].sort((left, right) => left.created_at.localeCompare(right.created_at))[0];
        if (!firstStudy) throw new Error("Project has no study");
        setProject(payload);
        setStudy(firstStudy);
        setDraft(firstStudy.spec);
        const [assetPayload, benchmarkPayload] = await Promise.all([
          fetchAssetStatus(payload.info.id),
          fetchBenchmarkStatus(payload.info.id, firstStudy.id),
          refreshJobs(payload.info.id, firstStudy.id)
        ]);
        if (mounted) {
          setAssets(assetPayload);
          setBenchmarkStatus(benchmarkPayload);
        }
      })
      .catch((exc: Error) => setError(exc.message))
      .finally(() => mounted && setBusy(null));
    return () => {
      mounted = false;
    };
  }, [refreshJobs]);

  useEffect(() => {
    if (!activeJobId || !project || !study) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const current = await fetchJob(project.info.id, activeJobId);
        if (cancelled) return;
        setJobs((items) => [current, ...items.filter((item) => item.id !== current.id)]);
        if (terminalStatuses.has(current.status)) {
          setActiveJobId(null);
          setBusy(null);
          if (current.status === "failed") setError(current.error ?? "Job failed");
          await refreshJobs(project.info.id, study.id);
          if (current.kind === "train_detector" && current.status === "succeeded") {
            const refreshed = await fetchProject(project.info.id);
            const refreshedStudy = refreshed.studies.find((item) => item.id === study.id);
            setProject(refreshed);
            if (refreshedStudy) {
              setStudy(refreshedStudy);
              setDraft(refreshedStudy.spec);
              setDirty(false);
            }
          }
          setBenchmarkStatus(await fetchBenchmarkStatus(project.info.id, study.id));
          return;
        }
        window.setTimeout(poll, 700);
      } catch (exc) {
        if (!cancelled) {
          setActiveJobId(null);
          setBusy(null);
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [activeJobId, project, study, refreshJobs]);

  const latest = useCallback(
    (kind: string) => jobs.find((item) => item.kind === kind && item.status === "succeeded") ?? null,
    [jobs]
  );

  const evaluationJob = latest("evaluate");
  const preflightJob = latest("benchmark_preflight");
  const requirementsJob = latest("requirements_evaluate");
  const sensitivityJob = latest("sensitivity");
  const optimizationJob = latest("optimize");
  const validationJob = latest("validate_candidate");
  const datasetJob = latest("dataset_export");
  const reportJob = latest("report");
  const evaluation = evaluationJob?.result ?? null;
  const sensitivity = sensitivityJob?.result ?? null;
  const optimizationResult = optimizationJob?.result ?? null;
  const optimization = (
    draft?.target_profile === "adas_yolo_perception" &&
    optimizationResult &&
    !optimizationResult.successive_halving
      ? null
      : optimizationResult
  );
  const candidates = (optimization?.top_cases ?? []) as Array<Record<string, any>>;
  const selected = candidates[Math.min(selectedCandidate, Math.max(candidates.length - 1, 0))] ?? null;

  const artifacts = useMemo(() => {
    const seen = new Set<string>();
    const output: Array<{ role: string; artifact: ArtifactRecord }> = [];
    for (const job of jobs) {
      for (const item of job.artifacts ?? []) {
        if (!seen.has(item.artifact.hash)) {
          seen.add(item.artifact.hash);
          output.push(item);
        }
      }
    }
    return output;
  }, [jobs]);

  const activeJob = activeJobId ? jobs.find((item) => item.id === activeJobId) : null;
  const perceptionAssetsReady = Boolean(
    draft?.target_profile !== "adas_yolo_perception" ||
      (draft.perception_model_path && draft.scenes[0]?.label_path)
  );
  const preflight = (preflightJob?.result ?? optimization?.preflight ?? benchmarkStatus?.preflight ?? null) as Record<string, any> | null;
  const preflightCameraOutput = (preflight?.camera_output ?? null) as Record<string, any> | null;
  const failedPreflightIds = ((preflight?.checks ?? []) as Array<Record<string, any>>)
    .filter((item) => !item.pass)
    .map((item) => String(item.id));
  const fidelityGateFailed = failedPreflightIds.some((id) => id.startsWith("fidelity_"));
  const detectorGateFailed = failedPreflightIds.some((id) =>
    ["detector_model", "detector_load", "source_map50", "source_map50_95", "source_recall"].includes(id)
  );
  const optimizationReady = Boolean(
    draft?.target_profile !== "adas_yolo_perception" || preflight?.ready
  );

  const mutate = (callback: (current: StudySpec) => StudySpec) => {
    setDraft((current) => (current ? callback(current) : current));
    setDirty(true);
  };

  const persist = async (): Promise<StudyRecord> => {
    if (!project || !study || !draft) throw new Error("Study is not loaded");
    if (!dirty) return study;
    setBusy("Saving study");
    const updated = await updateStudy(project.info.id, study.id, draft);
    setStudy(updated);
    setDraft(updated.spec);
    setProject((current) =>
      current
        ? { ...current, studies: current.studies.map((item) => (item.id === updated.id ? updated : item)) }
        : current
    );
    setDirty(false);
    setBusy(null);
    return updated;
  };

  const switchStudy = async (studyId: string) => {
    if (!project || studyId === study?.id || dirty || activeJobId) return;
    const next = project.studies.find((item) => item.id === studyId);
    if (!next) return;
    setBusy("Opening study");
    setError(null);
    setJobs([]);
    setBenchmarkStatus(null);
    setStudy(next);
    setDraft(next.spec);
    setSelectedCandidate(0);
    setPreviewTab("source");
    try {
      const benchmark = await fetchBenchmarkStatus(project.info.id, next.id);
      await refreshJobs(project.info.id, next.id);
      setBenchmarkStatus(benchmark);
      window.localStorage.setItem(`camerae2e:last-study:${project.info.id}`, next.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(null);
    }
  };

  const run = async (kind: string, request: Record<string, unknown> = {}) => {
    if (!project || !study) return;
    setError(null);
    try {
      const currentStudy = await persist();
      if (kind === "validate_candidate" && executeSolvers) {
        const accepted = window.confirm(
          "Run the selected local physics solvers? This can consume substantial CPU time."
        );
        if (!accepted) return;
      }
      if (kind === "train_detector") {
        const accepted = window.confirm(
          "Train KITTI YOLO for up to 30 epochs and run the 50-scene preflight? This can take substantial local compute time."
        );
        if (!accepted) return;
      }
      setBusy(operationLabel(kind));
      const response = await submitOperation(project.info.id, currentStudy.id, kind, request);
      setJobs((items) => [response.job, ...items]);
      setActiveJobId(response.job.id);
    } catch (exc) {
      setBusy(null);
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const scrollTo = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  if (!project || !study || !draft) {
    return (
      <main className="boot-screen">
        <Camera size={28} />
        <LoaderCircle className="spin" size={20} />
        <span>{error ?? "Opening CameraE2E project"}</span>
      </main>
    );
  }

  const scene = draft.scenes[0];
  const sceneUrl = scene.image_path
    ? scenePreviewUrl(project.info.id, study.id, scene.id)
    : null;
  const outputHash = evaluation?.preview_artifact_hash as string | null;
  const outputUrl = outputHash ? artifactUrl(project.info.id, outputHash) : null;
  const previewHashes = (evaluation?.preview_artifact_hashes ?? {}) as Record<string, string>;
  const previewUrls = {
    source: sceneUrl,
    ideal: previewHashes.ideal_recapture ? artifactUrl(project.info.id, previewHashes.ideal_recapture) : null,
    output: outputUrl,
    overlay: previewHashes.detector_overlay ? artifactUrl(project.info.id, previewHashes.detector_overlay) : null
  };
  const fidelity = (evaluation?.fidelity ?? {}) as Record<string, any>;

  return (
    <div className="app-shell">
      <aside className="workflow-rail">
        <button className="brand-mark" title="CameraE2E Workbench" onClick={() => scrollTo("requirements")}>
          <Camera size={22} />
        </button>
        <nav aria-label="Study workflow">
          {workflow.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`rail-item ${activeSection === item.id ? "active" : ""}`}
                onClick={() => scrollTo(item.id)}
                title={item.label}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="workbench">
        <header className="topbar">
          <div>
            <div className="product-line">CameraE2E Workbench <span>v2</span></div>
            <h1>{project.info.name}</h1>
            <p>{draft.name}</p>
          </div>
          <div className="topbar-controls">
            <label className="compact-field study-picker">
              <span>Study</span>
              <select data-testid="study-selector" value={study.id} disabled={dirty || Boolean(activeJobId)} onChange={(event) => void switchStudy(event.target.value)}>
                {project.studies.map((item) => <option key={item.id} value={item.id}>{item.spec.name}</option>)}
              </select>
            </label>
            <label className="compact-field">
              <span>Search fidelity</span>
              <select
                value={draft.fidelity_policy.search_level}
                onChange={(event) =>
                  mutate((current) => ({
                    ...current,
                    fidelity_policy: {
                      ...current.fidelity_policy,
                      search_level: event.target.value as "L0_analytic" | "L1_lut"
                    }
                  }))
                }
              >
                <option value="L0_analytic">L0 Analytic</option>
                <option value="L1_lut">L1 LUT-backed</option>
              </select>
            </label>
            <div className="seed-box"><span>Seed</span><strong>{draft.seed}</strong></div>
            <button className="secondary-button" disabled={!dirty || Boolean(activeJobId)} onClick={() => void persist()}>
              <Save size={16} /> Save
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <CircleAlert size={18} />
            <span>{error}</span>
            <button title="Dismiss" onClick={() => setError(null)}><X size={16} /></button>
          </div>
        )}

        {activeJob && (
          <div className="job-banner">
            <LoaderCircle className="spin" size={17} />
            <strong>{operationLabel(activeJob.kind)}</strong>
            <div className="progress-track"><span style={{ width: `${Math.max(activeJob.progress * 100, 4)}%` }} /></div>
            <code>{activeJob.id.slice(-8)}</code>
          </div>
        )}

        <section id="requirements" className="workspace-section">
          <SectionHeading icon={Gauge} title="Requirements" meta={`${draft.requirements.mission.toUpperCase()} mission`} />
          <div className="requirements-grid">
            <NumberControl
              label="Horizontal FOV"
              value={draft.requirements.hfov.value}
              unit="deg"
              onChange={(value) => mutate((current) => ({
                ...current,
                requirements: { ...current.requirements, hfov: { ...current.requirements.hfov, value } },
                baseline: { ...current.baseline, lens: { ...current.baseline.lens, hfov_deg: value } }
              }))}
            />
            <NumberControl
              label="Minimum object"
              value={draft.requirements.min_object_height.value}
              unit="px"
              onChange={(value) => mutate((current) => ({
                ...current,
                requirements: { ...current.requirements, min_object_height: { ...current.requirements.min_object_height, value } }
              }))}
            />
            <NumberControl
              label="Detection range"
              value={draft.requirements.detection_range.value}
              unit="m"
              onChange={(value) => mutate((current) => ({
                ...current,
                requirements: { ...current.requirements, detection_range: { ...current.requirements.detection_range, value } }
              }))}
            />
            <NumberControl
              label="Maximum latency"
              value={draft.requirements.max_latency.value}
              unit="ms"
              onChange={(value) => mutate((current) => ({
                ...current,
                requirements: { ...current.requirements, max_latency: { ...current.requirements.max_latency, value } }
              }))}
            />
            <div className="requirement-wide">
              <span className="control-label">Target classes</span>
              <div className="chips">
                {draft.requirements.target_classes.map((label) => <span className="chip" key={label}>{label}</span>)}
              </div>
            </div>
            <div className="requirement-wide source-row">
              <div><span className="control-label">Scene evidence</span><strong>{scene.name}</strong></div>
              <FidelityBadge level={scene.source_kind} />
              <code title={scene.image_path ?? "Synthetic scene"}>{scene.image_path ?? scene.scene_type}</code>
            </div>
          </div>
        </section>

        <section id="design-space" className="workspace-section">
          <SectionHeading icon={SlidersHorizontal} title="Design Space" meta={`${draft.design_variables.filter((item) => item.enabled).length} active variables`} />
          <div className="config-grid">
            <NumberControl label="Pixel size" value={draft.baseline.sensor.pixel_size_um} unit="um" onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, pixel_size_um: value } } }))} />
            <NumberControl label="F-number" value={draft.baseline.lens.f_number} unit="f/#" onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, lens: { ...current.baseline.lens, f_number: value } } }))} />
            <NumberControl label="Exposure" value={draft.baseline.sensor.exposure_ms} unit="ms" onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, exposure_ms: value } } }))} />
            <NumberControl label="PSF radius" value={draft.baseline.lens.psf_radius_um} unit="um" onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, lens: { ...current.baseline.lens, psf_radius_um: value } } }))} />
            <SelectControl label="CFA" value={draft.baseline.sensor.cfa_preset} options={["bayer_rgb", "quad_bayer_rgb", "quad_bayer_bggr"]} onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, cfa_preset: value } } }))} />
            <SelectControl label="QE profile" value={draft.baseline.sensor.qe_profile} options={["isetcam_default_rgb", "onsemi_ar0132at", "sony_imx363"]} onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, qe_profile: value } } }))} />
            <NumberControl label="Binning" value={draft.baseline.sensor.binning_factor} unit="x" onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, binning_factor: Math.max(1, Math.round(value)) } } }))} />
            <SelectControl label="OCL" value={draft.baseline.sensor.ocl_mode} options={["off", "centered", "optimal"]} onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, sensor: { ...current.baseline.sensor, ocl_mode: value } } }))} />
            <SelectControl label="Demosaic" value={draft.baseline.isp.demosaic_method} options={["bilinear", "adaptive laplacian"]} onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, isp: { ...current.baseline.isp, demosaic_method: value } } }))} />
            <SelectControl label="CCM" value={draft.baseline.isp.ccm_method} options={["mcc_optimized", "esser_optimized"]} onChange={(value) => mutate((current) => ({ ...current, baseline: { ...current.baseline, isp: { ...current.baseline.isp, ccm_method: value } } }))} />
          </div>
          <details className="axis-drawer">
            <summary><Settings2 size={16} /> Optimization axes <ChevronDown size={15} /></summary>
            <div className="axis-table">
              {draft.design_variables.map((variable, index) => (
                <AxisRow
                  key={variable.path}
                  variable={variable}
                  onChange={(next) => mutate((current) => ({
                    ...current,
                    design_variables: current.design_variables.map((item, itemIndex) => itemIndex === index ? next : item)
                  }))}
                />
              ))}
            </div>
          </details>
        </section>

        <section id="study" className="workspace-section">
          <SectionHeading icon={Activity} title="Study" meta={evaluationJob ? `baseline ${evaluationJob.status}` : "not evaluated"} />
          <div className="study-toolbar">
            <SelectControl
              label="Target"
              value={draft.target_profile}
              options={["adas_yolo_perception", "raw_quality"]}
              onChange={(value) => mutate((current) => ({ ...current, target_profile: value as StudySpec["target_profile"] }))}
            />
            <SelectControl
              label="Search"
              value={draft.search_method}
              options={["latin_hypercube", "surrogate", "gaussian_process", "evolutionary", "grid"]}
              onChange={(value) => mutate((current) => ({ ...current, search_method: value as StudySpec["search_method"] }))}
            />
            <NumberControl label="Budget" value={draft.search_budget} unit="cases" onChange={(value) => mutate((current) => ({ ...current, search_budget: Math.max(1, Math.round(value)) }))} />
            <div className={`target-readiness ${optimizationReady ? "ready" : "blocked"}`}>
              {optimizationReady ? <Check size={16} /> : <TriangleAlert size={16} />}
              <span>{optimizationReady ? "Target valid" : fidelityGateFailed ? "Fidelity invalid" : perceptionAssetsReady ? "Preflight required" : "Model or labels missing"}</span>
            </div>
          </div>
          <div className="benchmark-gate">
            <div>
              <ShieldCheck size={18} />
              <span>
                <strong>KITTI benchmark gate</strong>
                <small>{preflight ? `${preflight.scene_count} scenes · ${preflight.status}` : `${draft.benchmark.quick_scene_count} quick / ${draft.benchmark.final_scene_count} final`}</small>
              </span>
            </div>
            <div className="benchmark-actions">
              <button className="secondary-button" disabled={Boolean(activeJobId) || !perceptionAssetsReady || draft.target_profile !== "adas_yolo_perception"} onClick={() => void run("benchmark_preflight", { scene_count: draft.benchmark.quick_scene_count })}><ShieldCheck size={17} /> Run Preflight</button>
              <button className="secondary-button" disabled={Boolean(activeJobId) || !optimizationReady} onClick={() => void run("benchmark_run", { scene_count: draft.benchmark.quick_scene_count, include_robustness: true })}><ScanSearch size={17} /> Run Benchmark</button>
              <button className="secondary-button" title="Evaluate engineering requirement gates" disabled={Boolean(activeJobId)} onClick={() => void run("requirements_evaluate")}><Gauge size={17} /> Requirements</button>
            </div>
          </div>
          {preflight && !preflight.ready && (
            <div className="invalid-target">
              <TriangleAlert size={17} />
              <div><strong>{fidelityGateFailed ? "Invalid camera fidelity" : "Invalid perception target"}</strong><span>{failedPreflightIds.join(" · ")}</span></div>
              {fidelityGateFailed && draft.fidelity_policy.search_level !== "L0_analytic" ? (
                <button className="secondary-button" disabled={Boolean(activeJobId)} onClick={() => mutate((current) => ({ ...current, fidelity_policy: { ...current.fidelity_policy, search_level: "L0_analytic" } }))}>Use L0 Analytic</button>
              ) : detectorGateFailed ? (
                <button className="secondary-button" disabled={Boolean(activeJobId)} onClick={() => void run("train_detector", { execute: true, epochs: 30, activate: true })}>Train KITTI YOLO</button>
              ) : null}
            </div>
          )}
          {preflightCameraOutput && preflightCameraOutput.scene_count > 0 && (
            <div className="fidelity-sanity">
              <span><small>Executed fidelity</small><strong>{String(preflightCameraOutput.fidelity?.effective ?? draft.fidelity_policy.search_level).replaceAll("_", " ")}</strong></span>
              <span><small>Recall retention</small><strong>{formatNumber(preflightCameraOutput.recall_retention)}</strong></span>
              <span><small>Channel gain mismatch</small><strong>{formatNumber(preflightCameraOutput.relative_channel_gain_imbalance)}</strong></span>
              <span><small>Ideal/output SSIM</small><strong>{formatNumber(preflightCameraOutput.ideal_output_ssim)}</strong></span>
            </div>
          )}
          <div className="action-bar">
            <button className="primary-button" disabled={Boolean(activeJobId) || !perceptionAssetsReady} onClick={() => void run("evaluate")}><Play size={17} /> Evaluate Baseline</button>
            <button className="secondary-button" disabled={Boolean(activeJobId) || !optimizationReady} onClick={() => void run("sensitivity")}><ScanSearch size={17} /> Explore Design Space</button>
            <button className="secondary-button" disabled={Boolean(activeJobId) || !optimizationReady} onClick={() => void run("optimize")}><Sparkles size={17} /> Optimize</button>
          </div>

          <div className="preview-workspace">
            <div className="preview-tabs" role="tablist" aria-label="Simulation preview stage">
              {(["source", "ideal", "output", "overlay"] as const).map((tab) => (
                <button key={tab} role="tab" aria-selected={previewTab === tab} className={previewTab === tab ? "active" : ""} disabled={tab !== "source" && !previewUrls[tab]} onClick={() => setPreviewTab(tab)}>{({ source: "Scene Input", ideal: "Ideal Re-capture", output: "Camera Output", overlay: "Detector Overlay" } as const)[tab]}</button>
              ))}
            </div>
            <figure className="preview-frame single-preview">
              <figcaption>
                <span>{({ source: "Scene Input", ideal: "Ideal Re-capture", output: "Camera Output", overlay: "Detector Overlay" } as const)[previewTab]}</span>
                <FidelityBadge level={previewTab === "source" ? scene.source_kind : previewTab === "ideal" ? "geometry_only" : String(fidelity.effective ?? draft.fidelity_policy.search_level)} />
              </figcaption>
              {previewUrls[previewTab] ? <img src={String(previewUrls[previewTab])} alt={previewTab === "source" ? "Source scene without detector boxes" : `${previewTab} preview`} /> : <EmptyVisual icon={ImageIcon} label="Evaluate baseline" />}
            </figure>
          </div>

          <div className="metric-strip">
            <Metric label="Target Score" value={evaluation?.evaluation?.target_score} unit="" />
            <Metric label="RGB Mean" value={evaluation?.metrics?.color?.rgb_mean} unit="linear" />
            <Metric label="RAW Std" value={evaluation?.metrics?.artifact?.raw_std} unit="V/equiv" />
            <Metric label="Highlight Clip" value={evaluation?.metrics?.artifact?.rgb_high_clip_fraction ?? evaluation?.metrics?.artifact?.rgb_clip_fraction} unit="ratio" />
          </div>
          {(requirementsJob?.result?.result || evaluation?.requirement_gates) && <RequirementSummary payload={(requirementsJob?.result?.result ?? evaluation?.requirement_gates) as Record<string, any>} />}
          {evaluation?.truth_boundary && <TruthBoundary text={String(evaluation.truth_boundary)} />}
        </section>

        <section id="candidates" className="workspace-section">
          <SectionHeading icon={BarChart3} title="Candidates" meta={optimization ? `${optimization.final_feasible_count ?? optimization.feasible_count}/${optimization.finalist_count ?? optimization.case_count} finalists feasible · ${optimization.case_count} screened` : "not optimized"} />
          <div className="analysis-grid">
            <div className="analysis-pane">
              <h3>Sensitivity</h3>
              {sensitivity?.axes?.length ? (
                <div className="sensitivity-list">
                  {(sensitivity.axes as Array<Record<string, any>>).slice(0, 8).map((axis) => (
                    <div className="sensitivity-row" key={axis.path}>
                      <span>{shortPath(String(axis.path))}</span>
                      <div><i style={{ width: `${influenceWidth(Number(axis.relative_influence))}%` }} /></div>
                      <strong>{formatNumber(axis.score_span)}</strong>
                    </div>
                  ))}
                </div>
              ) : <EmptyVisual icon={ScanSearch} label="Run design-space exploration" />}
            </div>
            <div className="analysis-pane">
              <h3>Pareto: score vs clipping</h3>
              <ParetoPlot cases={(optimization?.pareto_front ?? optimization?.top_cases ?? []) as Array<Record<string, any>>} />
            </div>
          </div>

          <div className="candidate-table-wrap">
            <table className="candidate-table">
              <thead><tr><th>Rank</th><th>Score</th><th>mAP</th><th>Recall</th><th>Small Obj</th><th>Clip</th><th>Pixel</th><th>CFA</th><th>PSF</th><th>Evidence</th></tr></thead>
              <tbody>
                {candidates.map((candidate, index) => (
                  <tr key={candidate.case_index} className={`${selectedCandidate === index ? "selected " : ""}${candidate.finalist ? "finalist" : "screened"}`} onClick={() => setSelectedCandidate(index)}>
                    <td>{index + 1}</td>
                    <td>{formatNumber(candidate.target_score)}</td>
                    <td>{formatNumber(candidate.perception_metrics?.map50_95)}</td>
                    <td>{formatNumber(candidate.perception_metrics?.recall50)}</td>
                    <td>{formatNumber(candidate.perception_metrics?.small_object_recall50)}</td>
                    <td>{formatNumber(candidate.metrics?.artifact?.rgb_high_clip_fraction ?? candidate.metrics?.artifact?.rgb_clip_fraction)}</td>
                    <td>{formatMicrons(candidate.parameters?.["sensor.pixel_size"])}</td>
                    <td>{candidate.parameters?.["sensor.cfa_preset"] ?? "-"}</td>
                    <td>{formatNumber(candidate.parameters?.["optics.si_psf_radius_um"])}</td>
                    <td><span className={`candidate-state ${candidate.feasible ? "pass" : "fail"}`} title={candidate.finalist ? "Evaluated at the final scene budget" : `Eliminated after ${candidate.scene_count ?? 0} scenes`}>{candidate.finalist ? "finalist" : `screened ${candidate.scene_count ?? 0}`}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!candidates.length && <div className="table-empty"><BarChart3 size={20} /> Run optimization to rank candidates</div>}
          </div>

          <div className="candidate-actions">
            <label className="switch-control">
              <input type="checkbox" checked={executeSolvers} onChange={(event) => setExecuteSolvers(event.target.checked)} />
              <span>Execute local solvers</span>
            </label>
            <button
              className="secondary-button"
              disabled={!selected || Boolean(activeJobId)}
              onClick={() => void run("validate_candidate", {
                candidate: selected,
                execute: executeSolvers,
                acknowledge_expensive: executeSolvers
              })}
            >
              <Boxes size={17} /> Validate Candidate
            </button>
            {validationJob && <span className="inline-status"><Check size={15} /> {validationJob.result?.promotion?.state ?? `${validationJob.result?.solver_results?.length ?? 0} evidence records`}</span>}
          </div>
        </section>

        <section id="dataset" className="workspace-section compact-section">
          <SectionHeading icon={Database} title="RAW Dataset Factory" meta={datasetJob ? `${datasetJob.result?.case_count ?? 0} cases` : "no export"} />
          <div className="horizontal-form">
            <NumberControl label="Scenes" value={datasetCases} unit="frames" onChange={(value) => setDatasetCases(Math.max(1, Math.round(value)))} />
            <div className="format-tokens"><span>RAW NPZ</span><span>RGB PNG</span><span>Labels JSON</span><span>Metadata</span></div>
            <button className="primary-button" disabled={Boolean(activeJobId) || !optimization?.best_case} onClick={() => void run("dataset_export", { selection: "best", case_count: 1, scene_count: datasetCases })}><Archive size={17} /> Export Best Candidate</button>
          </div>
          {datasetJob?.result?.dataset_root && <code className="output-path">{String(datasetJob.result.dataset_root)}</code>}
        </section>

        <section id="calibration" className="workspace-section compact-section">
          <SectionHeading icon={FlaskConical} title="Calibration" meta="measured vs simulated" />
          <div className="calibration-form">
            <SelectControl label="Evidence" value={calibration.kind} options={["generic", "qe", "angular_response", "ptc", "mtf", "color", "latency"]} onChange={(value) => setCalibration((current) => ({ ...current, kind: value }))} />
            <TextControl label="Measured file" value={calibration.measured_path} onChange={(value) => setCalibration((current) => ({ ...current, measured_path: value }))} />
            <TextControl label="Simulated file" value={calibration.simulated_path} onChange={(value) => setCalibration((current) => ({ ...current, simulated_path: value }))} />
            <button className="secondary-button" disabled={!calibration.measured_path || !calibration.simulated_path || Boolean(activeJobId)} onClick={() => void run("calibrate", calibration)}><FlaskConical size={17} /> Fit Evidence</button>
          </div>
        </section>

        <section id="report" className="workspace-section compact-section">
          <SectionHeading icon={FileChartColumn} title="Decision Report" meta={reportJob ? "generated" : "not generated"} />
          <div className="report-row">
            <div>
              <strong>{draft.name}</strong>
              <span>{project.artifact_summary.count + artifacts.length} evidence artifacts</span>
            </div>
            <button className="primary-button" disabled={Boolean(activeJobId)} onClick={() => void run("report")}><FileChartColumn size={17} /> Build Report</button>
            {reportJob?.result?.html_artifact?.hash && (
              <a className="secondary-button" href={artifactUrl(project.info.id, String(reportJob.result.html_artifact.hash))} target="_blank" rel="noreferrer"><ImageIcon size={17} /> Open HTML</a>
            )}
          </div>
        </section>
      </main>

      <aside className="evidence-panel">
        <div className="inspector-header"><Layers3 size={18} /><strong>Evidence</strong></div>
        <div className="fidelity-ladder">
          {[
            ["L0", "Analytic", "validated"],
            ["L1", "LUT-backed", assets?.validation?.stale_dependency_count ? "stale" : "available"],
            ["L2", "Solver", validationJob ? "evidence" : "not run"],
            ["L3", "Calibrated", latest("calibrate") ? "scoped" : "missing"]
          ].map(([level, label, state]) => (
            <div className={`fidelity-step ${String(draft.fidelity_policy.search_level).startsWith(level) ? "active" : ""}`} key={level}>
              <span>{level}</span><div><strong>{label}</strong><small>{state}</small></div>
            </div>
          ))}
        </div>
        <div className="inspector-block">
          <span className="inspector-label">Target</span>
          <strong>{draft.target_profile.replaceAll("_", " ")}</strong>
          {draft.perception_model_path && <code title={draft.perception_model_path}>{fileName(draft.perception_model_path)}</code>}
        </div>
        <div className="inspector-block">
          <span className="inspector-label">Registry</span>
          <div className="status-line"><span>Artifacts</span><strong>{assets?.project_artifacts?.artifact_count ?? project.artifact_summary.count}</strong></div>
          <div className="status-line"><span>Camera assets</span><strong>{project.camera_asset_summary.count}</strong></div>
          <div className="status-line"><span>Lineage</span><strong className={assets?.validation?.stale_dependency_count ? "warn" : "ok"}>{assets?.validation?.stale_dependency_count ? `${assets.validation.stale_dependency_count} stale` : "current"}</strong></div>
          <div className="status-line"><span>DB gate</span><strong className={assets?.validation?.ok ? "ok" : "warn"}>{assets?.validation?.ok ? "pass" : "warning"}</strong></div>
          <div className="status-line"><span>KITTI scenes</span><strong>{benchmarkStatus?.inventory.available_scene_count ?? assets?.benchmark?.available_scene_count ?? 0}</strong></div>
        </div>
        <div className="inspector-block">
          <span className="inspector-label">Selected candidate</span>
          {selected ? (
            <>
              <strong>Case {selected.case_index}</strong>
              <span>Score {formatNumber(selected.target_score)}</span>
              <FidelityBadge level={String(selected.evidence_state ?? selected.fidelity?.effective ?? draft.fidelity_policy.search_level)} />
              <div className="score-breakdown">
                {Object.entries(selected.score_components ?? {}).map(([name, component]) => (
                  <div key={name}><span>{name.replaceAll("_", " ")}</span><strong>{formatNumber((component as Record<string, any>).contribution)}</strong></div>
                ))}
              </div>
              {selected.requirement_summary?.failed_gate_ids?.length ? <small className="failed-gates">Fails: {selected.requirement_summary.failed_gate_ids.join(", ")}</small> : null}
            </>
          ) : <span>No candidate selected</span>}
        </div>
        <div className="truth-mini">
          <TriangleAlert size={16} />
          <span>No result is promoted to product sign-off without complete measured evidence.</span>
        </div>
      </aside>

      <footer className="artifact-drawer">
        <div className="drawer-heading"><Database size={17} /><strong>Artifacts</strong><span>{artifacts.length}</span></div>
        <div className="artifact-list">
          {artifacts.slice(0, 12).map((item) => (
            <a key={`${item.role}-${item.artifact.hash}`} href={artifactUrl(project.info.id, item.artifact.hash)} target="_blank" rel="noreferrer" title={item.artifact.hash}>
              <span>{item.role}</span><code>{item.artifact.hash.slice(0, 8)}</code><small>{formatBytes(item.artifact.size_bytes)}</small>
            </a>
          ))}
          {!artifacts.length && <span className="artifact-empty">Run a study operation to create immutable evidence.</span>}
        </div>
      </footer>
    </div>
  );
}

function SectionHeading({ icon: Icon, title, meta }: { icon: typeof Gauge; title: string; meta: string }) {
  return <div className="section-heading"><div><Icon size={18} /><h2>{title}</h2></div><span>{meta}</span></div>;
}

function NumberControl({ label, value, unit, onChange }: { label: string; value: number; unit: string; onChange: (value: number) => void }) {
  return <label className="control"><span className="control-label">{label}</span><div className="input-with-unit"><input type="number" value={Number.isFinite(value) ? value : 0} step="any" onChange={(event) => onChange(Number(event.target.value))} /><span>{unit}</span></div></label>;
}

function SelectControl({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="control"><span className="control-label">{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option} value={option}>{option.replaceAll("_", " ")}</option>)}</select></label>;
}

function TextControl({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="control text-control"><span className="control-label">{label}</span><input type="text" value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function AxisRow({ variable, onChange }: { variable: DesignVariable; onChange: (value: DesignVariable) => void }) {
  return <div className="axis-row">
    <input type="checkbox" checked={variable.enabled} onChange={(event) => onChange({ ...variable, enabled: event.target.checked })} aria-label={`Enable ${variable.path}`} />
    <code>{variable.path}</code>
    <span>{variable.unit}</span>
    <span>{variable.values?.map(String).join(" / ") ?? `${variable.minimum} - ${variable.maximum}`}</span>
    <FidelityBadge level={variable.readiness_tier} />
  </div>;
}

function Metric({ label, value, unit }: { label: string; value: unknown; unit: string }) {
  return <div className="metric"><span>{label}</span><strong>{formatNumber(value)}</strong><small>{unit}</small></div>;
}

function FidelityBadge({ level }: { level: string }) {
  const normalized = level.toLowerCase();
  const tone = normalized.includes("calibrated") ? "calibrated" : normalized.includes("solver") || normalized.includes("calibration") ? "solver" : normalized.includes("lut") || normalized.includes("proxy") ? "proxy" : "analytic";
  return <span className={`fidelity-badge ${tone}`}>{level.replaceAll("_", " ")}</span>;
}

function TruthBoundary({ text }: { text: string }) {
  return <div className="truth-boundary"><TriangleAlert size={16} /><span>{text}</span></div>;
}

function RequirementSummary({ payload }: { payload: Record<string, any> }) {
  const gates = (payload.gates ?? []) as Array<Record<string, any>>;
  const failed = gates.filter((item) => item.hard && item.pass === false);
  const unavailable = gates.filter((item) => item.status === "not_evaluable");
  return <div className={`requirement-summary ${failed.length ? "failed" : "passed"}`}>
    {failed.length ? <TriangleAlert size={16} /> : <Check size={16} />}
    <strong>{failed.length ? `${failed.length} requirement gates failed` : "Engineering gates passed"}</strong>
    <span>{failed.map((item) => item.id).join(" · ") || `${unavailable.length} calibration-dependent checks pending`}</span>
  </div>;
}

function EmptyVisual({ icon: Icon, label }: { icon: typeof Camera; label: string }) {
  return <div className="empty-visual"><Icon size={24} /><span>{label}</span></div>;
}

function ParetoPlot({ cases }: { cases: Array<Record<string, any>> }) {
  if (!cases.length) return <EmptyVisual icon={BarChart3} label="Run optimization" />;
  const scores = cases.map((item) => Number(item.target_score ?? 0));
  const clips = cases.map((item) => Number(item.metrics?.artifact?.rgb_high_clip_fraction ?? item.metrics?.artifact?.rgb_clip_fraction ?? 0));
  const maxScore = Math.max(...scores, 1e-9);
  const maxClip = Math.max(...clips, 1e-9);
  return <div className="pareto-plot" aria-label="Target score versus clipping risk">
    <span className="y-label">Target score</span>
    <div className="plot-area">
      {cases.map((item, index) => (
        <i key={`${item.case_index}-${index}`} style={{ left: `${8 + 84 * (clips[index] / maxClip)}%`, bottom: `${8 + 84 * (scores[index] / maxScore)}%` }} title={`Case ${item.case_index}: ${formatNumber(scores[index])}`} />
      ))}
    </div>
    <span className="x-label">Clipping risk</span>
  </div>;
}

function operationLabel(kind: string): string {
  return ({ evaluate: "Evaluate Baseline", benchmark_preflight: "Run Benchmark Preflight", benchmark_run: "Run Benchmark", train_detector: "Train KITTI YOLO", requirements_evaluate: "Evaluate Requirements", sensitivity: "Explore Design Space", optimize: "Optimize", validate_candidate: "Validate Candidate", dataset_export: "Export Dataset", calibrate: "Fit Calibration", report: "Build Report" } as Record<string, string>)[kind] ?? kind;
}

function formatNumber(value: unknown): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) return number.toExponential(3);
  return number.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
}

function formatMicrons(value: unknown): string {
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number * 1e6)} um` : "-";
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function shortPath(path: string): string {
  return path.split(".").slice(-2).join(".");
}

function influenceWidth(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  return Math.min(100, Math.max(4, 15 * Math.log10(1 + value * 10)));
}

function fileName(path: string): string {
  return path.split("/").pop() ?? path;
}

export default App;
