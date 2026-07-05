import { Activity, ChartSpline, Grid3X3, Map, Waves } from "lucide-react";
import { useWorkbench } from "../lib/store";
import type { AnalysisTab } from "../lib/types";
import { OpticalLayout } from "./OpticalLayout";

const tabs: Array<{ id: AnalysisTab; label: string; icon: typeof Activity }> = [
  { id: "layout", label: "Layout", icon: Map },
  { id: "ray-fan", label: "Ray Fan", icon: ChartSpline },
  { id: "opd-fan", label: "OPD Fan", icon: Waves },
  { id: "spot", label: "Spot", icon: Grid3X3 },
  { id: "wavefront", label: "Wavefront", icon: Waves },
  { id: "field-curves", label: "Field", icon: ChartSpline },
  { id: "first-order", label: "First Order", icon: Activity }
];

export function AnalysisPanel() {
  const model = useWorkbench((state) => state.model);
  const activeTab = useWorkbench((state) => state.activeTab);
  const setActiveTab = useWorkbench((state) => state.setActiveTab);
  const refreshAnalysis = useWorkbench((state) => state.refreshAnalysis);
  const svg = useWorkbench((state) => state.svg);
  const firstOrder = useWorkbench((state) => state.firstOrder);
  const sampling = useWorkbench((state) => state.sampling);
  const scale = useWorkbench((state) => state.scale);
  const analysisFieldIndex = useWorkbench((state) => state.analysisFieldIndex);
  const analysisWavelengthIndex = useWorkbench((state) => state.analysisWavelengthIndex);
  const setSampling = useWorkbench((state) => state.setSampling);
  const setScale = useWorkbench((state) => state.setScale);
  const setAnalysisFieldIndex = useWorkbench((state) => state.setAnalysisFieldIndex);
  const setAnalysisWavelengthIndex = useWorkbench((state) => state.setAnalysisWavelengthIndex);
  const canScopeAnalysis = activeTab !== "layout" && activeTab !== "first-order";

  return (
    <section className="analysisPanel">
      <div className="analysisHeader">
        <div className="tabStrip">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button key={tab.id} className={tab.id === activeTab ? "active" : ""} onClick={() => setActiveTab(tab.id)}>
                <Icon size={15} />
                {tab.label}
              </button>
            );
          })}
        </div>
        <button className="secondaryButton" onClick={() => void refreshAnalysis()}>
          Refresh View
        </button>
      </div>
      <div className="analysisControls">
        <label>
          Field
          <select
            value={analysisFieldIndex === null ? "all" : String(analysisFieldIndex)}
            disabled={!canScopeAnalysis}
            onChange={(event) => setAnalysisFieldIndex(event.target.value === "all" ? null : Number(event.target.value))}
          >
            <option value="all">All</option>
            {model?.system.field.fields.map((field, index) => (
              <option key={`${field.x}-${field.y}-${index}`} value={index}>
                F{index + 1} ({formatField(field.x)}, {formatField(field.y)})
              </option>
            ))}
          </select>
        </label>
        <label>
          Wavelength
          <select
            value={analysisWavelengthIndex === null ? "all" : String(analysisWavelengthIndex)}
            disabled={!canScopeAnalysis}
            onChange={(event) => setAnalysisWavelengthIndex(event.target.value === "all" ? null : Number(event.target.value))}
          >
            <option value="all">All</option>
            {model?.system.wavelengths.values.map((wavelength, index) => (
              <option key={`${wavelength}-${index}`} value={index}>
                W{index + 1} {wavelength === null ? "n/a" : `${wavelength.toPrecision(6)} nm`}
              </option>
            ))}
          </select>
        </label>
        <label>
          Scale
          <select value={scale} disabled={!canScopeAnalysis} onChange={(event) => setScale(event.target.value as "all" | "same")}>
            <option value="same">Same</option>
            <option value="all">Fit Each</option>
          </select>
        </label>
        <label>
          Sampling
          <input
            type="number"
            min={5}
            max={65}
            step={2}
            value={sampling}
            disabled={!canScopeAnalysis}
            onChange={(event) => setSampling(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="analysisBody">
        {activeTab === "layout" ? <OpticalLayout /> : null}
        {activeTab !== "layout" && activeTab !== "first-order" ? (
          svg ? <div className="svgFrame" dangerouslySetInnerHTML={{ __html: svg }} /> : <div className="emptyState">Run analysis to render this view.</div>
        ) : null}
        {activeTab === "first-order" ? (
          <div className="firstOrderGrid">
            {Object.entries(firstOrder).map(([name, value]) => (
              <div key={name} className="metric">
                <span>{name}</span>
                <strong>{value === null ? "n/a" : value.toPrecision(7)}</strong>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function formatField(value: number | null | undefined) {
  if (value === null || value === undefined) return "n/a";
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}
