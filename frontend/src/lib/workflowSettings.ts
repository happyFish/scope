import type { SettingsState, LoRAConfig, InputMode } from "../types";
import type { WorkflowLoRA } from "./workflowApi";

/**
 * Mapping of SettingsState keys to their snake_case workflow param keys.
 * Adding a new exportable setting only requires adding one entry here.
 */
const PARAM_MAP: [keyof SettingsState, string][] = [
  ["denoisingSteps", "denoising_steps"],
  ["noiseScale", "noise_scale"],
  ["noiseController", "noise_controller"],
  ["manageCache", "manage_cache"],
  ["quantization", "quantization"],
  ["kvCacheAttentionBias", "kv_cache_attention_bias"],
  ["inputMode", "input_mode"],
];

export function settingsToWorkflowParams(
  settings: SettingsState,
  getPipelineDefaultMode: (pipelineId: string) => InputMode
): Record<string, unknown> {
  const params: Record<string, unknown> = {};

  // Flatten resolution
  if (settings.resolution) {
    params.height = settings.resolution.height;
    params.width = settings.resolution.width;
  }

  // Map simple fields
  for (const [settingsKey, paramKey] of PARAM_MAP) {
    const value = settings[settingsKey];
    if (value !== undefined) {
      params[paramKey] = value;
    }
  }

  // Apply default input_mode fallback
  if (!params.input_mode) {
    params.input_mode = getPipelineDefaultMode(settings.pipelineId);
  }

  // LoRA params
  if (settings.loras) {
    params.loras = settings.loras.map(({ path, scale, mergeMode }) => ({
      path,
      scale,
      ...(mergeMode && { merge_mode: mergeMode }),
    }));
  }
  params.lora_merge_mode = settings.loraMergeStrategy ?? "permanent_merge";

  return params;
}

export function workflowParamsToSettings(
  pipelineId: string,
  params: Record<string, unknown>,
  loras: WorkflowLoRA[],
  loraFiles: { path: string; name: string }[]
): Partial<SettingsState> {
  const mapped: Partial<SettingsState> = {
    pipelineId: pipelineId as SettingsState["pipelineId"],
  };

  // Resolution
  if (params.height != null && params.width != null) {
    mapped.resolution = {
      height: params.height as number,
      width: params.width as number,
    };
  }

  // Map simple fields
  for (const [settingsKey, paramKey] of PARAM_MAP) {
    if (params[paramKey] != null) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mapped as any)[settingsKey] = params[paramKey];
    }
  }

  // Resolve LoRAs
  if (loras.length > 0) {
    mapped.loras = loras
      .map(l => {
        const match = loraFiles.find(
          f =>
            f.path.endsWith(l.filename) ||
            f.name === l.filename.split("/").pop()
        );
        if (!match) return null;
        return {
          id: crypto.randomUUID(),
          path: match.path,
          scale: l.weight,
          mergeMode: l.merge_mode as LoRAConfig["mergeMode"],
        };
      })
      .filter((l): l is NonNullable<typeof l> => l !== null);
    mapped.loraMergeStrategy = loras[0]
      ?.merge_mode as SettingsState["loraMergeStrategy"];
  }

  return mapped;
}
