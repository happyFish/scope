// --- Types matching backend schema ---

export interface ScopeWorkflow {
  format: "scope-workflow";
  format_version: string;
  metadata: {
    name: string;
    description: string;
    author: string;
    created_at: string;
    scope_version: string;
  };
  pipelines: WorkflowPipeline[];
  timeline?: {
    entries: WorkflowTimelineEntry[];
  } | null;
  min_scope_version?: string | null;
}

export interface WorkflowPipeline {
  pipeline_id: string;
  pipeline_version: string;
  source: { package: string; version: string };
  loras: WorkflowLoRA[];
  params: Record<string, unknown>;
}

export interface WorkflowLoRA {
  id?: string | null;
  filename: string;
  weight: number;
  merge_mode: string;
  provenance?: {
    source: "huggingface" | "civitai" | "url" | "local";
    repo_id?: string | null;
    hf_filename?: string | null;
    model_id?: string | null;
    version_id?: string | null;
    url?: string | null;
  } | null;
  expected_sha256?: string | null;
}

export interface WorkflowTimelineEntry {
  start_time: number;
  end_time: number;
  prompts: { text: string; weight: number }[];
  transition_steps?: number;
  temporal_interpolation_method?: "linear" | "slerp";
}

export interface WorkflowExportRequest {
  name: string;
  description?: string;
  author?: string;
  frontend_params?: Record<string, Record<string, unknown>>;
  timeline?: { entries: WorkflowTimelineEntry[] };
}

export interface ResolutionItem {
  kind: "pipeline" | "plugin" | "lora";
  name: string;
  status: "ok" | "missing" | "version_mismatch";
  detail?: string | null;
  action?: string | null;
  can_auto_resolve: boolean;
}

export interface WorkflowResolutionPlan {
  can_apply: boolean;
  items: ResolutionItem[];
  settings_warnings: string[];
}

export interface WorkflowApplyRequest {
  workflow: ScopeWorkflow;
  install_missing_plugins?: boolean;
  skip_missing_loras?: boolean;
}

export interface ApplyResult {
  applied: boolean;
  pipeline_ids: string[];
  skipped_loras: string[];
  runtime_params: Record<string, unknown>;
  restart_required: boolean;
  message: string;
}

export interface LoRADownloadRequest {
  source: "huggingface" | "civitai" | "url";
  repo_id?: string | null;
  hf_filename?: string | null;
  model_id?: string | null;
  version_id?: string | null;
  url?: string | null;
  subfolder?: string | null;
  expected_sha256?: string | null;
}

export interface LoRADownloadResult {
  filename: string;
  path: string;
  sha256: string;
  size_bytes: number;
}

// --- Helpers ---

async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const errorJson = await response.json();
    if (errorJson.detail) {
      return typeof errorJson.detail === "string"
        ? errorJson.detail
        : JSON.stringify(errorJson.detail);
    }
    return JSON.stringify(errorJson);
  } catch {
    return response.statusText;
  }
}

// --- API functions ---

export async function exportWorkflow(
  req: WorkflowExportRequest
): Promise<ScopeWorkflow> {
  const response = await fetch("/api/v1/workflow/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Export failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function validateWorkflow(
  workflow: ScopeWorkflow
): Promise<WorkflowResolutionPlan> {
  const response = await fetch("/api/v1/workflow/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(workflow),
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Validation failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function applyWorkflow(
  req: WorkflowApplyRequest
): Promise<ApplyResult> {
  const response = await fetch("/api/v1/workflow/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Apply failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function downloadLora(
  req: LoRADownloadRequest
): Promise<LoRADownloadResult> {
  const response = await fetch("/api/v1/lora/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`LoRA download failed (${response.status}): ${detail}`);
  }
  return response.json();
}
