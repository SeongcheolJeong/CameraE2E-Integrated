from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


UpdateMode = Literal["auto", "manual", "layout-only", "editors-only"]
MetricStatus = Literal["pass", "warn", "fail"]
MetricSource = Literal["computed", "model", "proxy", "diagnostic", "unsupported", "assumption"]


class NewModelRequest(BaseModel):
    efl: float = 50.0
    epd: float = 12.5
    fov: float = 20.0


class ModelOpenRequest(BaseModel):
    path: str


class PatentOpenRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    simulation_id: str = Field(alias="simulationId")


class ModelSaveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    model_id: str = Field(alias="modelId")
    path: str | None = None
    overwrite: bool = False
    workbench: dict[str, Any] | None = None


class DraftAutosaveRequest(BaseModel):
    workbench: dict[str, Any] | None = None


class DraftRestoreRequest(BaseModel):
    path: str


class DraftAutosaveResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    model_id: str = Field(alias="modelId")
    path: str
    saved_at: str = Field(alias="savedAt")
    draft_count: int = Field(alias="draftCount")
    pruned_count: int = Field(alias="prunedCount")


class ModelSettingsPatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    update_mode: UpdateMode | None = Field(default=None, alias="updateMode")


class SurfacePatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    label: str | None = None
    radius: float | None = None
    curvature: float | None = None
    thickness: float | None = None
    glass: str | None = None
    catalog: str | None = None
    semi_diameter: float | None = Field(default=None, alias="semiDiameter")
    conic: float | None = None
    mode: Literal["transmit", "reflect", "dummy", "phantom"] | None = None
    stop: bool | None = None
    variable: str | None = None


class SurfaceCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    after: int | None = None
    radius: float = 0.0
    thickness: float = 1.0
    glass: str = "air"
    catalog: str = ""
    semi_diameter: float = Field(default=5.0, alias="semiDiameter")


class SystemPatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    aperture_value: float | None = Field(default=None, alias="apertureValue")
    field_value: float | None = Field(default=None, alias="fieldValue")
    field_x_values: list[float] | None = Field(default=None, alias="fieldXValues")
    field_y_values: list[float] | None = Field(default=None, alias="fieldYValues")
    field_weights: list[float] | None = Field(default=None, alias="fieldWeights")
    focus_shift: float | None = Field(default=None, alias="focusShift")
    defocus_range: float | None = Field(default=None, alias="defocusRange")
    wavelength_values: list[float] | None = Field(default=None, alias="wavelengthValues")
    wavelength_weights: list[float] | None = Field(default=None, alias="wavelengthWeights")
    wavelength_reference_index: int | None = Field(default=None, alias="wavelengthReferenceIndex")


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    scale: Literal["all", "same"] = "same"
    sampling: int = Field(default=21, ge=5, le=65)
    field_index: int | None = Field(default=None, alias="fieldIndex", ge=0)
    wavelength_index: int | None = Field(default=None, alias="wavelengthIndex", ge=0)


class ExampleCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    paths: list[str] | None = None
    limit: int = Field(default=6, ge=1, le=50)
    include_analyses: bool = Field(default=True, alias="includeAnalyses")


class ExampleCheckStageDTO(BaseModel):
    name: str
    status: MetricStatus
    detail: str


class ExampleCheckResultDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    label: str
    path: str
    kind: str
    status: MetricStatus
    duration_ms: int = Field(alias="durationMs")
    model_name: str | None = Field(alias="modelName")
    surface_count: int | None = Field(alias="surfaceCount")
    field_count: int | None = Field(alias="fieldCount")
    wavelength_count: int | None = Field(alias="wavelengthCount")
    stages: list[ExampleCheckStageDTO]
    warnings: list[str]
    errors: list[str]


class ExampleCheckResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    checked_at: str = Field(alias="checkedAt")
    total: int
    passed: int
    warned: int
    failed: int
    checks: list[ExampleCheckResultDTO]


class QuickOptimizeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    objective: Literal["balanced", "spot", "distortion", "throughput", "cra", "custom"] = "balanced"
    iterations: int = Field(default=2, ge=1, le=5)
    max_evaluations: int = Field(default=48, alias="maxEvaluations", ge=4, le=160)
    step_scale: float = Field(default=1.0, alias="stepScale", gt=0.0, le=5.0)
    operand_weights: dict[str, float] | None = Field(default=None, alias="operandWeights")
    operand_targets: dict[str, float] | None = Field(default=None, alias="operandTargets")


class QuickOptimizeMoveDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    surface_index: int = Field(alias="surfaceIndex")
    token: str
    label: str
    before: float
    after: float
    score: float


class QuickOptimizeResultDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    status: Literal["improved", "no-change", "failed"]
    objective: str
    message: str
    baseline_score: float = Field(alias="baselineScore")
    final_score: float = Field(alias="finalScore")
    improvement: float
    evaluations: int
    iterations: int
    variable_count: int = Field(alias="variableCount")
    operand_weights: dict[str, float] = Field(alias="operandWeights")
    variables: list[str]
    moves: list[QuickOptimizeMoveDTO]


class SurfaceDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    index: int
    label: str
    type: str
    radius: float | None
    curvature: float
    thickness: float | None
    glass: str
    catalog: str
    semi_diameter: float = Field(alias="semiDiameter")
    conic: float | None
    mode: str
    is_stop: bool = Field(alias="isStop")
    variable: str


class SystemDTO(BaseModel):
    aperture: dict
    field: dict
    wavelengths: dict
    focus: dict


class ModelDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    filename: str | None
    radius_mode: bool = Field(alias="radiusMode")
    update_mode: UpdateMode = Field(alias="updateMode")
    stop_surface: int | None = Field(alias="stopSurface")
    surfaces: list[SurfaceDTO]
    system: SystemDTO


class ModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    model: ModelDTO
    warnings: list[str]
    errors: list[str]
    dirty: bool
    can_undo: bool = Field(alias="canUndo")
    can_redo: bool = Field(alias="canRedo")
    last_updated_at: str = Field(alias="lastUpdatedAt")
    workbench: dict[str, Any] | None = None


class QuickOptimizeResponse(ModelResponse):
    result: QuickOptimizeResultDTO


class ToleranceSweepRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    scope: Literal["powered", "variables"] = "powered"
    perturbation_pct: float = Field(default=0.5, alias="perturbationPct", ge=0.05, le=5.0)
    max_surfaces: int = Field(default=6, alias="maxSurfaces", ge=1, le=20)


class ToleranceSweepCaseDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    label: str
    surface_index: int = Field(alias="surfaceIndex")
    token: str
    perturbation_pct: float = Field(alias="perturbationPct")
    before: float
    after: float
    score: float = Field(ge=0.0, le=1.0)
    status: MetricStatus
    spot_rms_um: float | None = Field(alias="spotRmsUm")
    mtf50_lpmm: float | None = Field(alias="mtf50LpMm")
    throughput: float | None
    distortion_pct: float | None = Field(alias="distortionPct")
    cra_deg: float | None = Field(alias="craDeg")
    trace_failures: list[str] = Field(alias="traceFailures")


class ToleranceSweepResultDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    status: MetricStatus
    scope: str
    perturbation_pct: float = Field(alias="perturbationPct")
    baseline_score: float = Field(alias="baselineScore")
    attempted_cases: int = Field(alias="attemptedCases")
    passed_cases: int = Field(alias="passedCases")
    warned_cases: int = Field(alias="warnedCases")
    failed_cases: int = Field(alias="failedCases")
    worst_case: str | None = Field(alias="worstCase")
    worst_score: float | None = Field(alias="worstScore")
    cases: list[ToleranceSweepCaseDTO]
    warnings: list[str]


class ToleranceSweepResponse(ModelResponse):
    result: ToleranceSweepResultDTO


class CockpitMetricDTO(BaseModel):
    key: str
    label: str
    value: str
    unit: str | None = None
    target: str
    status: MetricStatus
    score: float = Field(ge=0.0, le=1.0)
    source: MetricSource
    note: str | None = None


class CockpitRiskDTO(BaseModel):
    label: str
    value: str
    status: MetricStatus
    detail: str
    source: MetricSource


class CockpitCountsDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    surface_count: int = Field(alias="surfaceCount")
    powered_surfaces: int = Field(alias="poweredSurfaces")
    wavelength_count: int = Field(alias="wavelengthCount")
    field_count: int = Field(alias="fieldCount")
    complexity: float
    max_semi_diameter: float = Field(alias="maxSemiDiameter")
    max_field: float = Field(alias="maxField")


class SensorAssumptionDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    name: str
    pixel_pitch_um: float = Field(alias="pixelPitchUm")
    quantum_efficiency: float = Field(alias="quantumEfficiency")
    read_noise_e: float = Field(alias="readNoiseE")
    dark_noise_e: float = Field(alias="darkNoiseE")
    full_well_e: float = Field(alias="fullWellE")
    exposure_ms: float = Field(alias="exposureMs")
    scene_luminance_cd_m2: float = Field(alias="sceneLuminanceCdM2")
    optical_transmission: float = Field(alias="opticalTransmission")
    microlens_cra_limit_deg: float = Field(alias="microlensCraLimitDeg")
    reference_wavelength_nm: float = Field(alias="referenceWavelengthNm")


class SensorPatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    name: str | None = None
    pixel_pitch_um: float | None = Field(default=None, alias="pixelPitchUm", ge=0.2, le=30.0)
    quantum_efficiency: float | None = Field(default=None, alias="quantumEfficiency", ge=0.01, le=1.0)
    read_noise_e: float | None = Field(default=None, alias="readNoiseE", ge=0.0, le=100.0)
    dark_noise_e: float | None = Field(default=None, alias="darkNoiseE", ge=0.0, le=1000.0)
    full_well_e: float | None = Field(default=None, alias="fullWellE", ge=100.0, le=1.0e7)
    exposure_ms: float | None = Field(default=None, alias="exposureMs", ge=0.001, le=60000.0)
    scene_luminance_cd_m2: float | None = Field(default=None, alias="sceneLuminanceCdM2", ge=0.001, le=1.0e6)
    optical_transmission: float | None = Field(default=None, alias="opticalTransmission", ge=0.01, le=1.0)
    microlens_cra_limit_deg: float | None = Field(default=None, alias="microlensCraLimitDeg", ge=0.1, le=90.0)
    reference_wavelength_nm: float | None = Field(default=None, alias="referenceWavelengthNm", ge=100.0, le=30000.0)


class AnalysisSummaryResponse(ModelResponse):
    first_order: dict[str, float | None] = Field(alias="firstOrder")
    counts: CockpitCountsDTO
    metrics: list[CockpitMetricDTO]
    risks: list[CockpitRiskDTO]
    sensor: SensorAssumptionDTO
