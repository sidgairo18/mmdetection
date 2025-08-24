# mmdet/engine/hooks/wandb_hooks.py
from pathlib import Path
import os
import wandb
from mmengine.hooks import Hook
from mmengine.visualization import Visualizer
from mmdet.registry import HOOKS

def _current_wandb_run():
    vis = Visualizer.get_current_instance()
    run = None
    for b in getattr(vis, "_vis_backends", []):
        if b.__class__.__name__ == "WandbVisBackend":
            run = getattr(b, "experiment", None)
            if run is not None:
                break
    if run is None:
        try:
            run = wandb.run
        except Exception:
            run = None
    return run

@HOOKS.register_module()
class WandbArtifactHook(Hook):
    """Upload best/latest checkpoints as W&B Artifacts (rank-0 only)."""
    priority = 90  # VERY_LOW
    _state_fname = "wandb_artifacts_state.txt"

    def after_val_epoch(self, runner, metrics=None):
        if getattr(runner, "rank", 0) != 0:
            return
        if os.getenv("WANDB_UPLOAD_CKPTS", "1").lower() in {"0","false","no"}:
            return
        run = _current_wandb_run()
        if run is None:
            return

        work_dir = Path(runner.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        state_path = work_dir / self._state_fname
        uploaded = set(state_path.read_text().splitlines()) if state_path.exists() else set()

        best_ckpts = sorted(work_dir.glob("best_*.pth"))
        latest = sorted(work_dir.glob("epoch_*.pth"))[-1:]  # newest only
        to_upload, aliases = [], []

        if best_ckpts:
            for p in best_ckpts:
                if p.name not in uploaded:
                    to_upload.append(p); aliases.append(["best"])
        elif latest:
            p = latest[0]
            if p.name not in uploaded:
                als = ["latest"]
                try:
                    ep = int(p.stem.split("_")[-1]); als.insert(0, f"epoch-{ep}")
                except Exception:
                    pass
                to_upload.append(p); aliases.append(als)

        if not to_upload:
            return

        series_name = f"ckpt-{work_dir.name}".replace(" ", "-")
        for p, als in zip(to_upload, aliases):
            art = wandb.Artifact(name=series_name, type="model")
            art.add_file(str(p), name=p.name)
            run.log_artifact(art, aliases=als)
            uploaded.add(p.name)
        state_path.write_text("\n".join(sorted(uploaded)))

