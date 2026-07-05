import { useEffect, useState } from "react";
import { KpiDashboard } from "./components/KpiDashboard";
import { ProjectNavigator } from "./components/ProjectNavigator";
import { StageWorkspace } from "./components/StageWorkspace";
import { SystemExplorer } from "./components/SystemExplorer";
import { Toolbar } from "./components/Toolbar";
import { useWorkbench } from "./lib/store";
import type { WorkflowStage } from "./lib/types";
import { WorkflowRail, workflowStages } from "./components/WorkflowRail";

export function App() {
  const bootstrap = useWorkbench((state) => state.bootstrap);
  const [activeStage, setActiveStage] = useState<WorkflowStage>(() => stageFromUrl());

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  function handleStageChange(stage: WorkflowStage) {
    setActiveStage(stage);
    const url = new URL(window.location.href);
    url.searchParams.set("stage", stage);
    window.history.replaceState({}, "", url);
  }

  return (
    <main className="workbench">
      <Toolbar />
      <section className="workspace">
        <WorkflowRail activeStage={activeStage} onStageChange={handleStageChange} />
        <ProjectNavigator activeStage={activeStage} onStageChange={handleStageChange} />
        <StageWorkspace activeStage={activeStage} onStageChange={handleStageChange} />
        <aside className="sidePane">
          <SystemExplorer activeStage={activeStage} />
        </aside>
      </section>
      <KpiDashboard />
    </main>
  );
}

function stageFromUrl(): WorkflowStage {
  const stage = new URLSearchParams(window.location.search).get("stage");
  const validStages = new Set(workflowStages.map((item) => item.id));
  return validStages.has(stage as WorkflowStage) ? (stage as WorkflowStage) : "optical-model";
}
