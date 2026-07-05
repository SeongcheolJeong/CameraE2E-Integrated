import { AlertTriangle } from "lucide-react";
import { useWorkbench } from "../lib/store";

export function Diagnostics() {
  const warnings = useWorkbench((state) => state.warnings);
  const errors = useWorkbench((state) => state.errors);

  return (
    <section className="diagnostics" aria-label="Diagnostics">
      <div className="diagMessages">
        {errors.map((error) => (
          <span key={error} className="errorMsg">
            <AlertTriangle size={14} />
            {error}
          </span>
        ))}
        {warnings.map((warning) => (
          <span key={warning} className="warnMsg">
            {warning}
          </span>
        ))}
        {!warnings.length && !errors.length ? <span className="muted">No diagnostics</span> : null}
      </div>
    </section>
  );
}
