import { useWorkbench } from "../lib/store";

function scaleValue(value: number, min: number, max: number, size: number, pad: number) {
  if (Math.abs(max - min) < 1e-9) return size / 2;
  return pad + ((value - min) / (max - min)) * (size - 2 * pad);
}

export function OpticalLayout() {
  const layout = useWorkbench((state) => state.layout);
  const selectedSurface = useWorkbench((state) => state.selectedSurface);
  const setSelectedSurface = useWorkbench((state) => state.setSelectedSurface);

  if (!layout) {
    return <div className="emptyState">No layout data</div>;
  }

  const width = 920;
  const height = 520;
  const pad = 42;
  const zValues = layout.surfaces.map((surface) => surface.z);
  const yValues = [
    ...layout.surfaces.flatMap((surface) => [-surface.semiDiameter, surface.semiDiameter]),
    ...layout.rays.flatMap((ray) => ray.map((point) => point.y))
  ];
  const minZ = Math.min(...zValues, 0);
  const maxZ = Math.max(...zValues, 1);
  const maxYAbs = Math.max(1, ...yValues.map((value) => Math.abs(value)));
  const yMin = -maxYAbs;
  const yMax = maxYAbs;

  return (
    <div className="layoutShell">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Optical layout">
        <line x1={pad} x2={width - pad} y1={height / 2} y2={height / 2} className="axisLine" />
        {layout.rays.map((ray, index) => {
          const points = ray
            .map((point) => `${scaleValue(point.z, minZ, maxZ, width, pad)},${height - scaleValue(point.y, yMin, yMax, height, pad)}`)
            .join(" ");
          return <polyline key={index} points={points} className={`rayLine ray${index % 4}`} />;
        })}
        {layout.surfaces.map((surface) => {
          const x = scaleValue(surface.z, minZ, maxZ, width, pad);
          const y1 = height - scaleValue(surface.semiDiameter, yMin, yMax, height, pad);
          const y2 = height - scaleValue(-surface.semiDiameter, yMin, yMax, height, pad);
          const selected = surface.index === selectedSurface;
          return (
            <g key={surface.index} onClick={() => setSelectedSurface(surface.index)} className="surfaceGroup">
              <line x1={x} x2={x} y1={y1} y2={y2} className={`surfaceLine ${surface.mode === "reflect" ? "mirror" : ""} ${selected ? "selected" : ""}`} />
              {surface.isStop ? <rect x={x - 7} y={height / 2 - 7} width="14" height="14" className="stopMark" /> : null}
              <text x={x} y={height - 14} className="surfaceLabel">
                {surface.label}
              </text>
            </g>
          );
        })}
      </svg>
      {layout.warnings.length ? (
        <div className="layoutWarnings">
          {layout.warnings.map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
