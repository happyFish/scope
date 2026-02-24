import { useState } from "react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

interface WorkflowExportDialogProps {
  open: boolean;
  onClose: () => void;
  onExport: (name: string, description: string) => void;
}

export function WorkflowExportDialog({
  open,
  onClose,
  onExport,
}: WorkflowExportDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleExport = () => {
    if (!name.trim()) return;
    onExport(name.trim(), description.trim());
    setName("");
    setDescription("");
    onClose();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={isOpen => {
        if (!isOpen) {
          setName("");
          setDescription("");
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export Workflow</DialogTitle>
          <DialogDescription>
            Save the current session as a shareable .scope-workflow.json file.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 mt-2">
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium" htmlFor="workflow-name">
              Name
            </label>
            <input
              id="workflow-name"
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="My Workflow"
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              onKeyDown={e => {
                if (e.key === "Enter" && name.trim()) handleExport();
              }}
              autoFocus
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label
              className="text-sm font-medium"
              htmlFor="workflow-description"
            >
              Description (optional)
            </label>
            <textarea
              id="workflow-description"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Describe what this workflow does..."
              rows={3}
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleExport} disabled={!name.trim()}>
            Export
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
