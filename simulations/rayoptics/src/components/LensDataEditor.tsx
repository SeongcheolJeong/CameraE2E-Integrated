import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import { Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useWorkbench } from "../lib/store";
import type { SurfaceRow } from "../lib/types";

type EditableKey = "label" | "radius" | "curvature" | "thickness" | "glass" | "catalog" | "semiDiameter" | "conic" | "mode";
type VariableToken = "R" | "T" | "SD" | "K";
const variableTokens: Array<{ token: VariableToken; label: string; title: string }> = [
  { token: "R", label: "R", title: "Radius / curvature variable" },
  { token: "T", label: "T", title: "Thickness variable" },
  { token: "SD", label: "SD", title: "Semi-diameter variable" },
  { token: "K", label: "K", title: "Conic variable" }
];

function EditableCell({
  row,
  field,
  value,
  type = "text"
}: {
  row: SurfaceRow;
  field: EditableKey;
  value: string | number | null;
  type?: "text" | "number";
}) {
  const patchSurface = useWorkbench((state) => state.patchSurface);
  const disabled = (row.index === 0 || row.mode === "dummy") && field !== "label";
  const [draft, setDraft] = useState(valueToText(value));

  useEffect(() => {
    setDraft(valueToText(value));
  }, [value]);

  function commit(raw: string) {
    if (disabled) return;
    if (type === "number") {
      const normalized = raw.trim();
      const next = Number(normalized);
      if (normalized === "" || !Number.isFinite(next) || (field === "semiDiameter" && next <= 0)) {
        setDraft(valueToText(value));
        return;
      }
      if (next === value) return;
      void patchSurface(row.index, { [field]: next });
      return;
    }
    const next = raw;
    if (next === value) return;
    void patchSurface(row.index, { [field]: next });
  }

  return (
    <input
      disabled={disabled}
      className={type === "number" ? "numInput" : ""}
      value={draft}
      type={type}
      min={field === "semiDiameter" ? 1.0e-12 : undefined}
      step={type === "number" ? "any" : undefined}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => commit(draft)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
        if (event.key === "Escape") {
          setDraft(valueToText(value));
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function valueToText(value: string | number | null) {
  return value === null ? "" : String(value);
}

function VariableCell({ row }: { row: SurfaceRow }) {
  const patchSurface = useWorkbench((state) => state.patchSurface);
  const active = new Set(row.variable.split(",").map((item) => item.trim()).filter(Boolean));
  const disabled = row.index === 0 || row.mode === "dummy";

  function toggle(token: VariableToken) {
    const next = new Set(active);
    if (next.has(token)) next.delete(token);
    else next.add(token);
    const encoded = variableTokens.map((item) => item.token).filter((item) => next.has(item)).join(",");
    void patchSurface(row.index, { variable: encoded });
  }

  return (
    <div className="variableToggles" role="group" aria-label={`S${row.index} variable flags`}>
      {variableTokens.map((item) => (
        <button
          key={item.token}
          type="button"
          className={active.has(item.token) ? "active" : ""}
          disabled={disabled}
          title={`${item.title}. Workbench metadata; optimizer not connected yet.`}
          onClick={(event) => {
            event.stopPropagation();
            toggle(item.token);
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export function LensDataEditor() {
  const model = useWorkbench((state) => state.model);
  const selectedSurface = useWorkbench((state) => state.selectedSurface);
  const setSelectedSurface = useWorkbench((state) => state.setSelectedSurface);
  const radiusDisplay = useWorkbench((state) => state.radiusDisplay);
  const setRadiusDisplay = useWorkbench((state) => state.setRadiusDisplay);
  const patchSurface = useWorkbench((state) => state.patchSurface);
  const insertSurface = useWorkbench((state) => state.insertSurface);
  const deleteSurface = useWorkbench((state) => state.deleteSurface);

  const columns = useMemo<ColumnDef<SurfaceRow>[]>(
    () => [
      { header: "Surf", accessorKey: "index", size: 56 },
      {
        header: "Label",
        accessorKey: "label",
        cell: ({ row }) => <EditableCell row={row.original} field="label" value={row.original.label} />
      },
      { header: "Type", accessorKey: "type" },
      {
        header: radiusDisplay === "radius" ? "Radius" : "Curvature",
        accessorKey: radiusDisplay,
        cell: ({ row }) => (
          <EditableCell
            row={row.original}
            field={radiusDisplay}
            value={radiusDisplay === "radius" ? row.original.radius : row.original.curvature}
            type="number"
          />
        )
      },
      {
        header: "Thickness",
        accessorKey: "thickness",
        cell: ({ row }) => <EditableCell row={row.original} field="thickness" value={row.original.thickness} type="number" />
      },
      {
        header: "Glass",
        accessorKey: "glass",
        cell: ({ row }) => <EditableCell row={row.original} field="glass" value={row.original.glass} />
      },
      {
        header: "Catalog",
        accessorKey: "catalog",
        cell: ({ row }) => <EditableCell row={row.original} field="catalog" value={row.original.catalog} />
      },
      {
        header: "Semi-Dia",
        accessorKey: "semiDiameter",
        cell: ({ row }) => <EditableCell row={row.original} field="semiDiameter" value={row.original.semiDiameter} type="number" />
      },
      {
        header: "Conic",
        accessorKey: "conic",
        cell: ({ row }) => <EditableCell row={row.original} field="conic" value={row.original.conic} type="number" />
      },
      {
        header: "Mode",
        accessorKey: "mode",
        cell: ({ row }) => (
          <select
            value={row.original.mode}
            disabled={row.original.index === 0 || row.original.index === model!.surfaces.length - 1}
            onChange={(event) => void patchSurface(row.original.index, { mode: event.target.value })}
          >
            <option value="transmit">transmit</option>
            <option value="reflect">reflect</option>
            <option value="dummy">dummy</option>
            <option value="phantom">phantom</option>
          </select>
        )
      },
      {
        header: "Stop",
        accessorKey: "isStop",
        cell: ({ row }) => (
          <input
            type="checkbox"
            checked={row.original.isStop}
            disabled={row.original.index === 0 || row.original.index === model!.surfaces.length - 1}
            onChange={(event) => void patchSurface(row.original.index, { stop: event.target.checked })}
          />
        )
      },
      {
        header: "Variable",
        accessorKey: "variable",
        cell: ({ row }) => <VariableCell row={row.original} />
      }
    ],
    [deleteSurface, insertSurface, model, patchSurface, radiusDisplay]
  );

  const table = useReactTable({
    data: model?.surfaces ?? [],
    columns,
    getCoreRowModel: getCoreRowModel()
  });

  if (!model) {
    return <div className="emptyState">Loading optical model...</div>;
  }

  const insertAfter = selectedSurface ?? Math.max(0, model.surfaces.length - 2);
  const canDelete = selectedSurface !== null && selectedSurface > 0 && selectedSurface < model.surfaces.length - 1;

  return (
    <div className="lensEditor">
      <div className="tableTools">
        <div className="segmented">
          <button className={radiusDisplay === "radius" ? "active" : ""} onClick={() => setRadiusDisplay("radius")}>
            Radius
          </button>
          <button className={radiusDisplay === "curvature" ? "active" : ""} onClick={() => setRadiusDisplay("curvature")}>
            Curvature
          </button>
        </div>
        <button title="Insert surface after selected row" onClick={() => void insertSurface(insertAfter)}>
          <Plus size={16} />
          Insert
        </button>
        <button title="Delete selected non-object/image surface" disabled={!canDelete} onClick={() => selectedSurface !== null && void deleteSurface(selectedSurface)}>
          <Trash2 size={16} />
          Delete
        </button>
      </div>
      <div className="tableWrap">
        <table className="dataTable">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={row.original.index === selectedSurface ? "selected" : ""}
                onClick={() => setSelectedSurface(row.original.index)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
