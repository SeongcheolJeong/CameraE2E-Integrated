import { Aperture, Focus, Gauge, Ruler, ShieldAlert, SlidersHorizontal, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useWorkbench } from "../lib/store";
import { profileLabel } from "../lib/requirements";
import type { WorkflowStage } from "../lib/types";
import { deriveMetrics, statusLabel } from "../lib/uxMetrics";

export function SystemExplorer({ activeStage }: { activeStage: WorkflowStage }) {
  const model = useWorkbench((state) => state.model);
  const selectedSurface = useWorkbench((state) => state.selectedSurface);
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);
  const summary = useWorkbench((state) => state.summary);
  const updateMode = useWorkbench((state) => state.updateMode);
  const radiusDisplay = useWorkbench((state) => state.radiusDisplay);
  const requirements = useWorkbench((state) => state.requirements);
  const requirementProfile = useWorkbench((state) => state.requirementProfile);
  const autosaveSavedAt = useWorkbench((state) => state.autosaveSavedAt);
  const draftCount = useWorkbench((state) => state.draftCount);
  const patchSystem = useWorkbench((state) => state.patchSystem);
  const metrics = useMemo(() => deriveMetrics(model, warnings, errors, summary, requirements), [errors, model, requirements, summary, warnings]);
  const selected = model?.surfaces.find((surface) => surface.index === selectedSurface);
  const spot = metrics.metrics.find((metric) => metric.key === "spot");
  const cra = metrics.metrics.find((metric) => metric.key === "cra");
  const toleranceMetric = metrics.metrics.find((metric) => metric.key === "yield");
  const traceMetric = metrics.metrics.find((metric) => metric.key === "trace");

  if (!model) {
    return <aside className="systemExplorer emptyState">No model</aside>;
  }

  const wavelengthValues = finiteNumbers(model.system.wavelengths.values);
  const wavelengthWeights = normalizedWeights(model.system.wavelengths.weights, wavelengthValues.length);
  const wavelengthReference = clampReference(model.system.wavelengths.reference, wavelengthValues.length);
  const fieldCount = Math.max(1, model.system.field.fields.length);
  const fieldXValues = normalizedSeries(
    model.system.field.fields.map((field) => field.x),
    fieldCount,
    0
  );
  const fieldYValues = normalizedSeries(
    model.system.field.fields.map((field) => field.y),
    fieldCount,
    0
  );
  const fieldWeights = normalizedWeights(
    model.system.field.fields.map((field) => field.weight ?? 1),
    fieldCount
  );
  const fieldLabel = model.system.field.isRelative ? "Rel" : "Abs";

  return (
    <div className="systemExplorer">
      <div className="paneHeader compactHeader">
        <div>
          <span className="eyebrow">Properties</span>
          <h2>{activeStage === "sensor" ? "Sensor Coupling" : activeStage === "tolerance" ? "Manufacturing" : "Optical Spec"}</h2>
        </div>
      </div>
      <section className="propertyGroup selectedSurfaceGroup">
        <h3>Selected Surface</h3>
        {selected ? (
          <>
            <div className="surfaceInspectorTitle">
              <strong>S{selected.index}</strong>
              <span>{selected.label || selected.type}</span>
            </div>
            <Property label="Radius" value={format(selected.radius)} />
            <Property label="Curvature" value={format(selected.curvature)} />
            <Property label="Thickness" value={format(selected.thickness)} />
            <Property label="Glass" value={`${selected.glass || "air"}${selected.catalog ? ` / ${selected.catalog}` : ""}`} />
            <Property label="Semi-Dia" value={format(selected.semiDiameter)} />
            <Property label="Mode" value={selected.mode} />
          </>
        ) : (
          <span className="muted">Select a row</span>
        )}
      </section>
      <section className="propertyGroup">
        <h3>
          <Gauge size={16} />
          Impact
        </h3>
        <ImpactRow
          label={`Spot RMS (${spot?.source ?? "computed"})`}
          value={spot ? `${spot.value}${spot.unit ? ` ${spot.unit}` : ""}` : "n/a"}
          status={spot?.status ?? "warn"}
        />
        <ImpactRow
          label={`CRA Margin (${cra?.source ?? "proxy"})`}
          value={cra ? `${cra.value}${cra.unit ? ` ${cra.unit}` : ""}` : "n/a"}
          status={cra?.status ?? "pass"}
        />
        <ImpactRow
          label={`${toleranceMetric?.label ?? "Tolerance"} (${toleranceMetric?.source ?? "proxy"})`}
          value={toleranceMetric ? `${toleranceMetric.value}${toleranceMetric.unit ? ` ${toleranceMetric.unit}` : ""}` : "n/a"}
          status={toleranceMetric?.status ?? "pass"}
        />
        <ImpactRow
          label="Trace State (diagnostic)"
          value={traceMetric ? `${traceMetric.value} ${traceMetric.unit ?? ""}` : `${errors.length} / ${warnings.length}`}
          status={traceMetric?.status ?? (errors.length ? "fail" : warnings.length ? "warn" : "pass")}
        />
      </section>
      <section className="propertyGroup">
        <h3>
          <Aperture size={16} />
          Aperture
        </h3>
        <Property label="Key" value={model.system.aperture.key.join(" / ")} />
        <EditableProperty
          label="Value"
          min={1.0e-12}
          value={model.system.aperture.value}
          onCommit={(value) => void patchSystem({ apertureValue: value })}
        />
      </section>
      <section className="propertyGroup">
        <h3>
          <Ruler size={16} />
          Field
        </h3>
        <Property label="Key" value={model.system.field.key.join(" / ")} />
        <EditableProperty
          label="Value"
          min={0}
          value={model.system.field.value}
          onCommit={(value) => void patchSystem({ fieldValue: value })}
        />
        <EditableListProperty
          label={`X ${fieldLabel}`}
          expectedLength={fieldCount}
          maxLength={12}
          values={fieldXValues}
          onCommit={(values) =>
            void patchSystem({
              fieldXValues: values,
              fieldYValues,
              fieldWeights
            })
          }
        />
        <EditableListProperty
          label={`Y ${fieldLabel}`}
          maxLength={12}
          values={fieldYValues}
          onCommit={(values) =>
            void patchSystem({
              fieldXValues: resizeSeries(fieldXValues, values.length, 0),
              fieldYValues: values,
              fieldWeights: resizeSeries(fieldWeights, values.length, 1)
            })
          }
        />
        <EditableListProperty
          label="Weights"
          expectedLength={fieldCount}
          maxLength={12}
          min={0}
          requirePositiveSum
          values={fieldWeights}
          onCommit={(values) =>
            void patchSystem({
              fieldXValues,
              fieldYValues,
              fieldWeights: values
            })
          }
        />
      </section>
      <section className="propertyGroup">
        <h3>
          <Waves size={16} />
          Wavelengths
        </h3>
        <ReferenceSelectProperty
          label="Ref Index"
          values={wavelengthValues}
          value={wavelengthReference}
          onCommit={(value) => void patchSystem({ wavelengthReferenceIndex: value })}
        />
        <EditableListProperty
          label="Values nm"
          min={1.0e-12}
          maxLength={7}
          values={wavelengthValues}
          strictlyIncreasing
          onCommit={(values) =>
            void patchSystem({
              wavelengthValues: values,
              wavelengthWeights: resizeSeries(wavelengthWeights, values.length, 1),
              wavelengthReferenceIndex: clampReference(wavelengthReference, values.length)
            })
          }
        />
        <EditableListProperty
          label="Weights"
          expectedLength={wavelengthValues.length}
          maxLength={7}
          min={0}
          requirePositiveSum
          values={wavelengthWeights}
          onCommit={(weights) =>
            void patchSystem({
              wavelengthValues,
              wavelengthWeights: weights,
              wavelengthReferenceIndex: wavelengthReference
            })
          }
        />
      </section>
      <section className="propertyGroup">
        <h3>
          <Focus size={16} />
          Focus
        </h3>
        <EditableProperty
          label="Shift"
          value={model.system.focus.focusShift}
          onCommit={(value) => void patchSystem({ focusShift: value })}
        />
        <EditableProperty
          label="Range"
          min={0}
          value={model.system.focus.defocusRange}
          onCommit={(value) => void patchSystem({ defocusRange: value })}
        />
      </section>
      <section className="propertyGroup">
        <h3>
          <SlidersHorizontal size={16} />
          Constraints
        </h3>
        <Property label="Update" value={updateMode} />
        <Property label="Req Profile" value={profileLabel(requirementProfile)} />
        <Property label="Autosave" value={autosaveLabel(autosaveSavedAt, draftCount)} />
        <Property label="Aperture Stop" value={model.stopSurface === null ? "n/a" : `S${model.stopSurface}`} />
        <Property label="Lens Editor Units" value={radiusDisplay === "radius" ? "Radius" : "Curvature"} />
      </section>
      <section className="propertyGroup">
        <h3>
          <ShieldAlert size={16} />
          Dock Status
        </h3>
        <div className="dockStatusGrid">
          {metrics.metrics.slice(1, 5).map((metric) => (
            <span key={metric.key} className={metric.status}>
              {metric.label}
              <strong>{statusLabel(metric.status)}</strong>
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}

function Property({ label, value }: { label: string; value: string }) {
  return (
    <div className="property">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function autosaveLabel(savedAt: string | null, draftCount: number | null) {
  if (!savedAt) return "Not written";
  const date = new Date(savedAt);
  const time = Number.isNaN(date.getTime()) ? "saved" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return draftCount === null ? time : `${time} / ${draftCount} drafts`;
}

function ReferenceSelectProperty({
  label,
  values,
  value,
  onCommit
}: {
  label: string;
  values: number[];
  value: number;
  onCommit: (value: number) => void;
}) {
  return (
    <div className="property editableProperty">
      <span>{label}</span>
      <select
        className="propertySelect"
        value={value}
        disabled={values.length === 0}
        onChange={(event) => onCommit(Number(event.target.value))}
      >
        {values.map((wavelength, index) => (
          <option key={`${wavelength}-${index}`} value={index}>
            {index} / {format(wavelength)} nm
          </option>
        ))}
      </select>
    </div>
  );
}

function EditableProperty({
  label,
  value,
  min,
  onCommit
}: {
  label: string;
  value: number | null | undefined;
  min?: number;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(value === null || value === undefined ? "" : String(value));

  useEffect(() => {
    setDraft(value === null || value === undefined ? "" : String(value));
  }, [value]);

  const commit = () => {
    const normalized = draft.trim();
    const parsed = Number(normalized);
    if (normalized === "" || !Number.isFinite(parsed) || (min !== undefined && parsed < min)) {
      setDraft(value === null || value === undefined ? "" : String(value));
      return;
    }
    if (value !== null && value !== undefined && Math.abs(parsed - value) < 1.0e-12) {
      setDraft(String(value));
      return;
    }
    onCommit(parsed);
  };

  return (
    <div className="property editableProperty">
      <span>{label}</span>
      <input
        className="propertyInput"
        inputMode="decimal"
        value={draft}
        onBlur={commit}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            setDraft(value === null || value === undefined ? "" : String(value));
            event.currentTarget.blur();
          }
        }}
      />
    </div>
  );
}

function EditableListProperty({
  label,
  values,
  expectedLength,
  maxLength = 7,
  min,
  requirePositiveSum,
  strictlyIncreasing,
  onCommit
}: {
  label: string;
  values: number[];
  expectedLength?: number;
  maxLength?: number;
  min?: number;
  requirePositiveSum?: boolean;
  strictlyIncreasing?: boolean;
  onCommit: (values: number[]) => void;
}) {
  const [draft, setDraft] = useState(formatList(values));

  useEffect(() => {
    setDraft(formatList(values));
  }, [values]);

  const revert = () => setDraft(formatList(values));
  const commit = () => {
    const parsed = parseNumberList(draft);
    if (
      parsed.length === 0 ||
      parsed.length > maxLength ||
      (expectedLength !== undefined && parsed.length !== expectedLength) ||
      parsed.some((value) => !Number.isFinite(value) || (min !== undefined && value < min)) ||
      (strictlyIncreasing && parsed.some((value, index) => index > 0 && parsed[index - 1] >= value)) ||
      (requirePositiveSum && !parsed.some((value) => value > 0))
    ) {
      revert();
      return;
    }
    if (arraysEqual(parsed, values)) {
      revert();
      return;
    }
    onCommit(parsed);
  };

  return (
    <div className="property editableProperty listProperty">
      <span>{label}</span>
      <input
        className="propertyInput listInput"
        inputMode="decimal"
        value={draft}
        onBlur={commit}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            revert();
            event.currentTarget.blur();
          }
        }}
      />
    </div>
  );
}

function ImpactRow({ label, value, status }: { label: string; value: string; status: "pass" | "warn" | "fail" }) {
  return (
    <div className={`impactRow ${status}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function format(value: number | null | undefined) {
  if (value === null || value === undefined) return "n/a";
  return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

function formatList(values: number[]) {
  return values.map(format).join(", ");
}

function parseNumberList(value: string) {
  return value
    .trim()
    .split(/[,;\s]+/)
    .filter(Boolean)
    .map((item) => Number(item));
}

function finiteNumbers(values: Array<number | null>) {
  return values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function normalizedWeights(values: Array<number | null>, count: number) {
  return Array.from({ length: count }, (_, index) => {
    const value = values[index];
    return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 1;
  });
}

function normalizedSeries(values: Array<number | null | undefined>, count: number, fallback: number) {
  return Array.from({ length: count }, (_, index) => {
    const value = values[index];
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
  });
}

function resizeSeries(values: number[], count: number, fallback: number) {
  return Array.from({ length: count }, (_, index) => values[index] ?? fallback);
}

function clampReference(index: number, count: number) {
  if (count <= 0) return 0;
  return Math.max(0, Math.min(Math.trunc(index), count - 1));
}

function arraysEqual(left: number[], right: number[]) {
  return left.length === right.length && left.every((value, index) => Math.abs(value - right[index]) < 1.0e-12);
}
