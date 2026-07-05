import type {
  AssetStatus,
  DatasetResult,
  OptimizationResult,
  PresetsResponse,
  ReportResult,
  Settings,
  SimulationResult
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload));
  }
  return (await response.json()) as T;
}

export function fetchPresets(): Promise<PresetsResponse> {
  return request<PresetsResponse>("/api/presets");
}

export function fetchAssetStatus(): Promise<AssetStatus> {
  return request<AssetStatus>("/api/assets/status");
}

export function runSimulation(settings: Settings): Promise<SimulationResult> {
  return request<SimulationResult>("/api/simulate", {
    method: "POST",
    body: JSON.stringify({ settings })
  });
}

export function runOptimization(settings: Settings): Promise<OptimizationResult> {
  return request<OptimizationResult>("/api/optimize", {
    method: "POST",
    body: JSON.stringify({ settings, maxCandidates: settings.maxCandidates })
  });
}

export function exportDataset(settings: Settings): Promise<DatasetResult> {
  return request<DatasetResult>("/api/dataset/export", {
    method: "POST",
    body: JSON.stringify({
      settings,
      selection: settings.datasetSelection,
      caseCount: settings.datasetCaseCount
    })
  });
}

export function exportReport(settings: Settings): Promise<ReportResult> {
  return request<ReportResult>("/api/report", {
    method: "POST",
    body: JSON.stringify({ settings })
  });
}

