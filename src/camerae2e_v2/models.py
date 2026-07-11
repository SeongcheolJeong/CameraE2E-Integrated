"""Typed domain contracts for CameraE2E v2."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SceneType = Literal["macbeth", "slanted_bar", "uniform", "rgb_file", "multispectral_file"]
SearchMethod = Literal[
    "grid",
    "random",
    "latin_hypercube",
    "evolutionary",
    "surrogate",
    "gaussian_process",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FidelityLevel(StrEnum):
    ANALYTIC = "L0_analytic"
    LUT = "L1_lut"
    SOLVER = "L2_solver"
    CALIBRATED = "L3_calibrated"


class ReadinessTier(StrEnum):
    MISSING = "missing"
    AVAILABLE = "available"
    PROXY = "proxy"
    VALIDATED = "validated"
    CALIBRATION_REQUIRED = "calibration_required"
    CALIBRATED = "calibrated"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class CandidateEvidenceState(StrEnum):
    ANALYTIC_SCREENED = "analytic_screened"
    LUT_VALIDATED = "lut_validated"
    SOLVER_EVIDENCED = "solver_evidenced"
    CALIBRATION_REQUIRED = "calibration_required"
    CALIBRATED = "calibrated"


class AssetKind(StrEnum):
    LENS_DESIGN = "lens_design"
    SENSOR_PRODUCT = "sensor_product"
    PIXEL_STACK = "pixel_stack"
    OPTICAL_LUT = "optical_lut"
    MATERIAL_NK = "material_nk"
    ISP_PROFILE = "isp_profile"
    CALIBRATION = "calibration"
    PERCEPTION_MODEL = "perception_model"
    CAMERA_PRESET = "camera_preset"


class Quantity(StrictModel):
    value: float
    unit: str = Field(min_length=1)


class RequirementSet(StrictModel):
    mission: Literal["adas", "mobile", "general"] = "adas"
    hfov: Quantity = Field(default_factory=lambda: Quantity(value=81.4, unit="deg"))
    target_classes: list[str] = Field(default_factory=lambda: ["Car", "Pedestrian", "Cyclist"])
    min_object_height: Quantity = Field(default_factory=lambda: Quantity(value=12, unit="px"))
    detection_range: Quantity = Field(default_factory=lambda: Quantity(value=80, unit="m"))
    illuminance_range: tuple[float, float] = (1.0, 100_000.0)
    max_latency: Quantity = Field(default_factory=lambda: Quantity(value=100, unit="ms"))
    max_sensor_diagonal: Quantity | None = Field(
        default_factory=lambda: Quantity(value=8.0, unit="mm")
    )
    reference_object_height: Quantity = Field(default_factory=lambda: Quantity(value=1.5, unit="m"))
    min_frame_rate: Quantity = Field(default_factory=lambda: Quantity(value=30.0, unit="fps"))
    max_pixel_rate: Quantity = Field(default_factory=lambda: Quantity(value=300.0, unit="Mpixel/s"))
    max_rolling_shutter: Quantity = Field(default_factory=lambda: Quantity(value=20.0, unit="ms"))
    max_clip_fraction: float = Field(default=0.01, ge=0.0, le=1.0)
    min_snr_db: float = Field(default=20.0, ge=0.0)
    reference_wavelength_nm: float = Field(default=550.0, gt=0.0)
    notes: str = ""

    @field_validator("illuminance_range")
    @classmethod
    def validate_illuminance(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] < 0 or value[1] <= value[0]:
            raise ValueError("illuminance_range must be nonnegative and increasing")
        return value


class LensConfig(StrictModel):
    model_id: str | None = None
    hfov_deg: float = Field(default=81.4, gt=1.0, lt=179.0)
    f_number: float = Field(default=1.8, gt=0.5)
    focal_length_mm: float | None = Field(default=None, gt=0.0)
    psf_radius_um: float = Field(default=2.0, ge=0.0)
    distortion_model: str = "none"
    relative_illumination_model: str = "analytic_cos4"


class SensorConfig(StrictModel):
    model_id: str | None = None
    rows: int = Field(default=375, gt=0)
    cols: int = Field(default=1242, gt=0)
    native_rows: int | None = Field(default=None, gt=0)
    native_cols: int | None = Field(default=None, gt=0)
    geometry_source: str = "configured_readout"
    pixel_size_um: float = Field(default=3.75, gt=0.0)
    pixel_fill_factor: float = Field(default=0.75, gt=0.0, le=1.0)
    simulation_pixel_size_um: float | None = Field(default=None, gt=0.0)
    simulation_fill_factor: float | None = Field(default=None, gt=0.0, le=1.0)
    cfa_preset: str = "bayer_rgb"
    qe_profile: str = "isetcam_default_rgb"
    exposure_ms: float = Field(default=4.0, gt=0.0)
    analog_gain: float = Field(default=1.0, gt=0.0)
    read_noise_e: float | None = Field(default=None, ge=0.0)
    full_well_e: float | None = Field(default=None, gt=0.0)
    binning_factor: int = Field(default=1, ge=1)
    ocl_mode: str = "centered"
    ocl_group_shape: str = "2x2"
    ocl_equalization: float = Field(default=0.5, ge=0.0, le=1.0)
    bit_depth: int = Field(default=12, ge=8, le=24)
    frame_rate_fps: float = Field(default=30.0, gt=0.0)
    row_time_us: float = Field(default=20.0, gt=0.0)


class ISPConfig(StrictModel):
    demosaic_method: str = "bilinear"
    ccm_method: str = "mcc_optimized"
    ccm_matrix: list[list[float]] | None = None
    tone_method: str = "default"
    gamma: float = Field(default=2.2, gt=0.0)
    ccm_regularization: float = Field(default=0.02, ge=0.0)
    ccm_row_sum_tolerance: float = Field(default=0.15, ge=0.0, le=1.0)
    ccm_max_abs: float = Field(default=4.0, gt=0.0)

    @field_validator("ccm_matrix")
    @classmethod
    def validate_ccm(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("ccm_matrix must be a 3x3 matrix")
        return value


class CameraModule(StrictModel):
    id: str = Field(default_factory=lambda: new_id("camera"))
    name: str = "KITTI-style ADAS baseline"
    lens: LensConfig = Field(default_factory=LensConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    isp: ISPConfig = Field(default_factory=ISPConfig)
    hw_isp_enabled: bool = False
    hw_isp_frames: int = Field(default=2, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneCase(StrictModel):
    id: str = Field(default_factory=lambda: new_id("scene"))
    name: str = "Baseline scene"
    source_kind: Literal["physical", "measured_proxy", "display_rgb_proxy", "synthetic"] = (
        "synthetic"
    )
    scene_type: SceneType = "macbeth"
    image_path: str | None = None
    label_path: str | None = None
    mean_luminance_cd_m2: float = Field(default=80.0, gt=0.0)
    illuminance_lux: float | None = Field(default=None, ge=0.0)
    weather: str = "clear"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self) -> SceneCase:
        if self.scene_type in {"rgb_file", "multispectral_file"} and not self.image_path:
            raise ValueError(f"{self.scene_type} requires image_path")
        return self


class GeometryTransform(StrictModel):
    source_size_rc: tuple[int, int]
    requested_sensor_size_rc: tuple[int, int]
    active_sensor_size_rc: tuple[int, int]
    readout_size_rc: tuple[int, int]
    output_size_rc: tuple[int, int]
    cfa_block_rc: tuple[int, int] = (2, 2)
    binning_factor: int = Field(default=1, ge=1)
    scale_xy: tuple[float, float]
    offset_xy: tuple[float, float] = (0.0, 0.0)
    hfov_deg: float = Field(gt=0.0, lt=180.0)
    alignment_warning: str | None = None

    def transform_bbox(self, bbox_xyxy: list[float] | tuple[float, ...]) -> list[float]:
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coordinates")
        sx, sy = self.scale_xy
        ox, oy = self.offset_xy
        rows, cols = self.output_size_rc
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        return [
            float(min(max((x1 * sx) + ox, 0.0), cols)),
            float(min(max((y1 * sy) + oy, 0.0), rows)),
            float(min(max((x2 * sx) + ox, 0.0), cols)),
            float(min(max((y2 * sy) + oy, 0.0), rows)),
        ]


class RobustnessCase(StrictModel):
    name: str
    kind: Literal["exposure_ev", "psf_scale", "ocl_equalization", "low_light"]
    amount: float


def default_robustness_cases() -> list[RobustnessCase]:
    return [
        RobustnessCase(name="Exposure -1 EV", kind="exposure_ev", amount=-1.0),
        RobustnessCase(name="Exposure +1 EV", kind="exposure_ev", amount=1.0),
        RobustnessCase(name="Defocus / PSF", kind="psf_scale", amount=1.5),
        RobustnessCase(name="CRA mismatch", kind="ocl_equalization", amount=0.2),
        RobustnessCase(name="Low illumination", kind="low_light", amount=0.25),
    ]


class BenchmarkSuite(StrictModel):
    name: str = "KITTI ADAS benchmark"
    metric_version: str = "adas_yolo_perception_v1"
    source_root: str | None = None
    quick_scene_count: int = Field(default=50, ge=1, le=2000)
    final_scene_count: int = Field(default=200, ge=1, le=10000)
    halving_scene_counts: tuple[int, ...] = (8, 20, 50)
    final_candidate_count: int = Field(default=3, ge=1, le=100)
    stratification: Literal["adas_class_and_object_size", "ordered"] = "adas_class_and_object_size"
    robustness_cases: list[RobustnessCase] = Field(default_factory=default_robustness_cases)
    detector_map50_min: float = Field(default=0.50, ge=0.0, le=1.0)
    detector_map50_95_min: float = Field(default=0.30, ge=0.0, le=1.0)
    detector_recall_min: float = Field(default=0.50, ge=0.0, le=1.0)
    ideal_recall_retention_min: float = Field(default=0.90, ge=0.0, le=1.0)
    ideal_ssim_min: float = Field(default=0.85, ge=0.0, le=1.0)
    fidelity_scene_count: int = Field(default=3, ge=1, le=100)
    fidelity_recall_retention_min: float = Field(default=0.20, ge=0.0, le=1.0)
    fidelity_color_imbalance_max: float = Field(default=0.25, ge=0.0)
    fidelity_ssim_min: float = Field(default=0.45, ge=0.0, le=1.0)
    uncertainty_bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    minimum_meaningful_score_delta: float = Field(default=0.005, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> BenchmarkSuite:
        if self.final_scene_count < self.quick_scene_count:
            raise ValueError("final_scene_count must be >= quick_scene_count")
        if any(value <= 0 for value in self.halving_scene_counts):
            raise ValueError("halving_scene_counts must be positive")
        if tuple(sorted(self.halving_scene_counts)) != self.halving_scene_counts:
            raise ValueError("halving_scene_counts must be increasing")
        return self


class BenchmarkPreflightResult(StrictModel):
    ready: bool
    status: str
    scene_count: int
    model_path: str | None = None
    checks: list[dict[str, Any]] = Field(default_factory=list)
    source_baseline: dict[str, Any] = Field(default_factory=dict)
    ideal_recapture: dict[str, Any] = Field(default_factory=dict)
    camera_output: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    benchmark_manifest: dict[str, Any] = Field(default_factory=dict)


class RequirementGateResult(StrictModel):
    feasible: bool
    gates: list[dict[str, Any]] = Field(default_factory=list)
    derived: dict[str, Any] = Field(default_factory=dict)


class SceneAggregateMetrics(StrictModel):
    scene_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    per_scene: list[dict[str, Any]] = Field(default_factory=list)


class CandidatePromotionResult(StrictModel):
    case_id: str
    state: CandidateEvidenceState
    previous_state: CandidateEvidenceState = CandidateEvidenceState.ANALYTIC_SCREENED
    evidence_artifacts: list[str] = Field(default_factory=list)
    benchmark_rerun: dict[str, Any] | None = None
    calibration_blocked: bool = True
    message: str = ""


class DesignVariable(StrictModel):
    path: str = Field(min_length=3)
    unit: str = "dimensionless"
    values: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    steps: int = Field(default=3, ge=2, le=100)
    readiness_tier: ReadinessTier = ReadinessTier.VALIDATED
    enabled: bool = True

    @model_validator(mode="after")
    def validate_domain(self) -> DesignVariable:
        if self.values is None:
            if self.minimum is None or self.maximum is None:
                raise ValueError("design variable requires values or minimum/maximum")
            if self.maximum <= self.minimum:
                raise ValueError("design variable maximum must exceed minimum")
        elif not self.values:
            raise ValueError("design variable values must not be empty")
        return self


class Objective(StrictModel):
    id: str
    metric: str
    direction: Literal["maximize", "minimize", "target"] = "maximize"
    weight: float = Field(default=1.0, gt=0.0)
    target: float | None = None
    display_name: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Objective:
        if self.direction == "target" and self.target is None:
            raise ValueError("target objective requires target")
        return self


class Constraint(StrictModel):
    id: str
    metric: str
    operator: Literal["<=", ">=", "<", ">", "=="]
    value: float
    hard: bool = True
    display_name: str | None = None


class FidelityPolicy(StrictModel):
    search_level: FidelityLevel = FidelityLevel.ANALYTIC
    validation_level: FidelityLevel = FidelityLevel.SOLVER
    allow_proxy: bool = True
    require_fresh_lineage: bool = True
    require_calibration_for_signoff: bool = True
    solver_timeout_s: int = Field(default=3600, gt=0)


def default_design_variables() -> list[DesignVariable]:
    return [
        DesignVariable(path="sensor.integration_time", unit="s", values=[0.002, 0.004, 0.006]),
        DesignVariable(path="sensor.pixel_size", unit="m", values=[2.8e-6, 3.75e-6, 4.5e-6]),
        DesignVariable(
            path="sensor.cfa_preset", unit="enum", values=["bayer_rgb", "quad_bayer_rgb"]
        ),
        DesignVariable(path="optics.fnumber", unit="f/#", values=[1.4, 1.8, 2.4]),
        DesignVariable(
            path="optics.si_psf_radius_um",
            unit="um",
            values=[1.0, 2.0, 3.5],
            readiness_tier=ReadinessTier.PROXY,
        ),
        DesignVariable(
            path="ip.sensor_conversion_method",
            unit="enum",
            values=["mcc_optimized", "esser_optimized"],
        ),
    ]


def default_objectives() -> list[Objective]:
    return [
        Objective(
            id="signal_support",
            metric="metrics.color.rgb_mean",
            direction="maximize",
            weight=1.0,
            display_name="Signal support",
        ),
        Objective(
            id="clip_risk",
            metric="metrics.artifact.rgb_high_clip_fraction",
            direction="minimize",
            weight=0.8,
            display_name="Clipping risk",
        ),
        Objective(
            id="noise_target",
            metric="metrics.artifact.raw_std",
            direction="target",
            target=0.02,
            weight=0.35,
            display_name="RAW noise target",
        ),
    ]


class StudyCreate(StrictModel):
    name: str = "ADAS camera trade study"
    requirements: RequirementSet = Field(default_factory=RequirementSet)
    baseline: CameraModule = Field(default_factory=CameraModule)
    scenes: list[SceneCase] = Field(default_factory=lambda: [SceneCase()])
    design_variables: list[DesignVariable] = Field(default_factory=default_design_variables)
    objectives: list[Objective] = Field(default_factory=default_objectives)
    constraints: list[Constraint] = Field(default_factory=list)
    fidelity_policy: FidelityPolicy = Field(default_factory=FidelityPolicy)
    benchmark: BenchmarkSuite = Field(default_factory=BenchmarkSuite)
    target_profile: Literal["raw_quality", "adas_yolo_perception"] = "raw_quality"
    perception_model_path: str | None = None
    seed: int = 42
    search_budget: int = Field(default=24, ge=1, le=1000)
    search_method: SearchMethod = "latin_hypercube"

    @field_validator("scenes")
    @classmethod
    def validate_scenes(cls, value: list[SceneCase]) -> list[SceneCase]:
        if not value:
            raise ValueError("study requires at least one scene")
        return value


class StudyRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("study"))
    project_id: str
    revision: int = 1
    status: str = "draft"
    spec: StudyCreate
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProjectInfo(StrictModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    name: str
    path: str
    schema_version: str = "camerae2e_project_v2"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ArtifactRecord(StrictModel):
    hash: str
    artifact_type: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    media_type: str
    fidelity_level: FidelityLevel
    readiness_tier: ReadinessTier
    source: str
    dependencies: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CameraAssetRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("asset"))
    kind: AssetKind
    name: str = Field(min_length=1)
    version: str = "1"
    artifact_hashes: list[str]
    fidelity_level: FidelityLevel
    readiness_tier: ReadinessTier
    parameters: dict[str, Any] = Field(default_factory=dict)
    valid_domain: dict[str, Any] = Field(default_factory=dict)
    source: str
    validation: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("artifact_hashes")
    @classmethod
    def validate_artifact_hashes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("camera asset requires at least one artifact hash")
        return value


class ComponentSelection(StrictModel):
    lens_id: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    name: str | None = None
    use_geometric_psf: bool = True


class ModuleEvaluationRequest(StrictModel):
    requirements: RequirementSet = Field(default_factory=RequirementSet)
    candidates: list[ComponentSelection] = Field(min_length=1, max_length=4)


class ModuleBaselineRequest(StrictModel):
    selection: ComponentSelection
    allow_incompatible: bool = False


class ModuleCompareRequest(StrictModel):
    candidates: list[ComponentSelection] = Field(min_length=2, max_length=4)
    scene_id: str | None = None
    allow_incompatible: bool = False


class JobRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    study_id: str | None = None
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    log: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobSubmitResponse(StrictModel):
    job: JobRecord
    poll_url: str


class DatasetExportRequest(StrictModel):
    selection: Literal["baseline", "best", "top", "pareto"] = "best"
    case_count: int = Field(default=2, ge=1, le=1000)
    scene_count: int | None = Field(default=None, ge=1, le=10000)
    include_tiff: bool = False
    include_stage_outputs: bool = False
    include_uncertainty: bool = False


class ReportRequest(StrictModel):
    title: str | None = None
    include_candidates: int = Field(default=8, ge=1, le=100)


class CalibrationRequest(StrictModel):
    kind: Literal["qe", "angular_response", "ptc", "mtf", "color", "latency", "generic"] = "generic"
    measured_path: str
    simulated_path: str
    measured_key: str | None = None
    simulated_key: str | None = None
    valid_min: float | None = None
    valid_max: float | None = None
    min_sample_count: int = Field(default=6, ge=3)
    min_r2: float = Field(default=0.95, ge=-1.0, le=1.0)
    max_normalized_rmse: float = Field(default=0.05, gt=0.0)
    notes: str = ""


def path_exists(value: str | None) -> bool:
    return bool(value and Path(value).expanduser().exists())
