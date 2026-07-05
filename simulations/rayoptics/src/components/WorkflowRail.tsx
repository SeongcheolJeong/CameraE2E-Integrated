import {
  Activity,
  Aperture,
  Cpu,
  FileText,
  GitCompare,
  Goal,
  Layers3,
  LineChart,
  Microscope,
  ShieldCheck
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { WorkflowStage } from "../lib/types";

export const workflowStages: Array<{ id: WorkflowStage; label: string; short: string; icon: LucideIcon }> = [
  { id: "project", label: "Project / Requirements", short: "Project", icon: Goal },
  { id: "optical-model", label: "Optical Model", short: "Model", icon: Aperture },
  { id: "analysis", label: "Analysis Diagnostics", short: "Analysis", icon: Activity },
  { id: "optimization", label: "Optimization", short: "Optimize", icon: LineChart },
  { id: "tolerance", label: "Tolerance & Manufacturing", short: "Tolerance", icon: ShieldCheck },
  { id: "sensor", label: "Sensor / ISP", short: "Sensor", icon: Cpu },
  { id: "scene", label: "Scene Validation", short: "Scene", icon: Microscope },
  { id: "compare", label: "Compare / Current Snapshot", short: "Compare", icon: GitCompare },
  { id: "report", label: "Report / Export", short: "Report", icon: FileText }
];

export function WorkflowRail({ activeStage, onStageChange }: { activeStage: WorkflowStage; onStageChange: (stage: WorkflowStage) => void }) {
  return (
    <nav className="workflowRail" aria-label="Optical workflow">
      <div className="railLogo" title="Optical Design Cockpit">
        <Layers3 size={19} />
      </div>
      <div className="railButtons">
        {workflowStages.map((stage) => {
          const Icon = stage.icon;
          return (
            <button
              key={stage.id}
              className={stage.id === activeStage ? "active" : ""}
              onClick={() => onStageChange(stage.id)}
              title={stage.label}
              aria-label={stage.label}
            >
              <Icon size={19} />
              <span>{stage.short}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
