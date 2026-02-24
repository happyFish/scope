import { useState, useCallback } from "react";
import { CheckCircle2, XCircle, AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import type {
  WorkflowResolutionPlan,
  ResolutionItem,
  ScopeWorkflow,
  LoRADownloadRequest,
} from "../lib/workflowApi";
import { downloadLora, validateWorkflow } from "../lib/workflowApi";

type DownloadStatus = "idle" | "downloading" | "done" | "error";

interface WorkflowImportDialogProps {
  open: boolean;
  onClose: () => void;
  workflow: ScopeWorkflow | null;
  plan: WorkflowResolutionPlan | null;
  onPlanUpdate: (plan: WorkflowResolutionPlan) => void;
  onApply: (installMissingPlugins: boolean) => void;
  isApplying: boolean;
}

function StatusIcon({ status }: { status: ResolutionItem["status"] }) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />;
    case "missing":
      return <XCircle className="h-4 w-4 text-red-500 shrink-0" />;
    case "version_mismatch":
      return <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0" />;
  }
}

function KindBadge({ kind }: { kind: ResolutionItem["kind"] }) {
  const colors = {
    pipeline: "bg-blue-500/20 text-blue-400",
    plugin: "bg-purple-500/20 text-purple-400",
    lora: "bg-orange-500/20 text-orange-400",
  };
  return (
    <span
      className={`text-xs px-1.5 py-0.5 rounded font-medium ${colors[kind]}`}
    >
      {kind}
    </span>
  );
}

export function WorkflowImportDialog({
  open,
  onClose,
  workflow,
  plan,
  onPlanUpdate,
  onApply,
  isApplying,
}: WorkflowImportDialogProps) {
  const [installMissingPlugins, setInstallMissingPlugins] = useState(false);
  const [downloadStatuses, setDownloadStatuses] = useState<
    Record<string, DownloadStatus>
  >({});
  const [downloadErrors, setDownloadErrors] = useState<Record<string, string>>(
    {}
  );

  const isDownloading = Object.values(downloadStatuses).some(
    s => s === "downloading"
  );

  const buildDownloadRequest = useCallback(
    (item: ResolutionItem): LoRADownloadRequest | null => {
      if (!workflow) return null;
      for (const pipeline of workflow.pipelines) {
        const lora = pipeline.loras.find(l => l.filename === item.name);
        if (lora?.provenance && lora.provenance.source !== "local") {
          return {
            source: lora.provenance.source,
            repo_id: lora.provenance.repo_id,
            hf_filename: lora.provenance.hf_filename,
            model_id: lora.provenance.model_id,
            version_id: lora.provenance.version_id,
            url: lora.provenance.url,
            expected_sha256: lora.expected_sha256,
          };
        }
      }
      return null;
    },
    [workflow]
  );

  const handleDownload = useCallback(
    async (item: ResolutionItem) => {
      const req = buildDownloadRequest(item);
      if (!req || !workflow) return;

      setDownloadStatuses(prev => ({ ...prev, [item.name]: "downloading" }));
      setDownloadErrors(prev => {
        const next = { ...prev };
        delete next[item.name];
        return next;
      });

      try {
        await downloadLora(req);
        setDownloadStatuses(prev => ({ ...prev, [item.name]: "done" }));
        // Re-validate to update resolution plan
        const updatedPlan = await validateWorkflow(workflow);
        onPlanUpdate(updatedPlan);
      } catch (err) {
        setDownloadStatuses(prev => ({ ...prev, [item.name]: "error" }));
        setDownloadErrors(prev => ({
          ...prev,
          [item.name]: err instanceof Error ? err.message : "Download failed",
        }));
      }
    },
    [buildDownloadRequest, workflow, onPlanUpdate]
  );

  const handleDownloadAll = useCallback(async () => {
    if (!plan) return;
    const autoResolvable = plan.items.filter(
      i =>
        i.kind === "lora" &&
        i.status === "missing" &&
        i.can_auto_resolve &&
        downloadStatuses[i.name] !== "done" &&
        downloadStatuses[i.name] !== "downloading"
    );
    for (const item of autoResolvable) {
      await handleDownload(item);
    }
  }, [plan, downloadStatuses, handleDownload]);

  const hasAutoResolvableItems =
    plan?.items.some(
      i =>
        i.kind === "lora" &&
        i.status === "missing" &&
        i.can_auto_resolve &&
        downloadStatuses[i.name] !== "done"
    ) ?? false;

  const hasPluginItems =
    plan?.items.some(i => i.kind === "plugin" && i.status === "missing") ??
    false;

  if (!workflow || !plan) return null;

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Import Workflow: {workflow.metadata.name}</DialogTitle>
          {workflow.metadata.description && (
            <DialogDescription>
              {workflow.metadata.description}
            </DialogDescription>
          )}
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-3 my-2">
          {/* Dependency list */}
          <div className="space-y-1.5">
            {plan.items.map((item, idx) => (
              <div
                key={`${item.kind}-${item.name}-${idx}`}
                className="flex items-center gap-2 p-2 rounded-md bg-muted/50"
              >
                <StatusIcon status={item.status} />
                <KindBadge kind={item.kind} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {item.name}
                  </div>
                  {item.detail && (
                    <div className="text-xs text-muted-foreground">
                      {item.detail}
                    </div>
                  )}
                  {item.action && (
                    <div className="text-xs text-muted-foreground italic">
                      {item.action}
                    </div>
                  )}
                  {downloadErrors[item.name] && (
                    <div className="text-xs text-red-400">
                      {downloadErrors[item.name]}
                    </div>
                  )}
                </div>
                {item.kind === "lora" &&
                  item.status === "missing" &&
                  item.can_auto_resolve && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={
                        downloadStatuses[item.name] === "downloading" ||
                        downloadStatuses[item.name] === "done"
                      }
                      onClick={() => handleDownload(item)}
                      className="shrink-0"
                    >
                      {downloadStatuses[item.name] === "downloading" ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : downloadStatuses[item.name] === "done" ? (
                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                      ) : (
                        "Download"
                      )}
                    </Button>
                  )}
              </div>
            ))}
          </div>

          {/* Warnings */}
          {plan.settings_warnings.length > 0 && (
            <div className="rounded-md bg-yellow-500/10 border border-yellow-500/20 p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span className="text-sm font-medium text-yellow-500">
                  Warnings
                </span>
              </div>
              <ul className="text-xs text-yellow-400 space-y-0.5">
                {plan.settings_warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <DialogFooter className="flex items-center gap-2">
          <div className="flex-1 flex items-center gap-2">
            {hasAutoResolvableItems && (
              <Button
                size="sm"
                variant="outline"
                disabled={isDownloading}
                onClick={handleDownloadAll}
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    Downloading...
                  </>
                ) : (
                  "Download All"
                )}
              </Button>
            )}
            {hasPluginItems && (
              <label className="flex items-center gap-1.5 text-xs">
                <input
                  type="checkbox"
                  checked={installMissingPlugins}
                  onChange={e => setInstallMissingPlugins(e.target.checked)}
                  className="rounded"
                />
                Install missing plugins
              </label>
            )}
          </div>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={() => onApply(installMissingPlugins)}
            disabled={!plan.can_apply || isDownloading || isApplying}
          >
            {isApplying ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                Applying...
              </>
            ) : (
              "Apply"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
