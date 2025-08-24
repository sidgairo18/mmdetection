# mmdet/engine/hooks/wandb_helpers.py
import os
import re
import uuid
import argparse
import pathlib
from typing import Tuple, Set, Dict, List, Any
from mmengine.config import Config

# ---------- tiny utils ----------
def _camel_to_token(name: str) -> str:
    s = re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()
    return s.replace('_', '-')

_NON_AUG_TYPES: Set[str] = {
    "LoadImageFromFile", "LoadAnnotations", "LoadPanopticAnnotations",
    "FilterAnnotations", "PackDetInputs", "LoadProposals", "ResizeShortestEdge",
}
_AUG_ALIAS: Dict[str, str] = {
    "RandomResize": "rr", "Resize": "resize",
    "RandomCrop": "rcrop", "RandomFlip": "flip",
    "Mosaic": "mosaic", "MixUp": "mixup", "CopyPaste": "cp",
    "AutoAugment": "autoaug", "RandAugment": "randaug",
    "PhotoMetricDistortion": "pmd", "ColorJitter": "cj",
    "CutOut": "cutout", "CutMix": "cutmix",
    "MinIoURandomCrop": "minioircrop", "Expand": "expand",
    "RandomShift": "rshift", "RandomRotate": "rrotate",
    "YOLOXHSVRandomAug": "yoloxhsv", "RandomErasing": "rerase",
    "Albu": "albu", "Albumentations": "albu",
}
_AUG_SUBSTR_HINTS = [
    "Random", "Mosaic", "MixUp", "CopyPaste", "Aug", "Jitter",
    "CutOut", "CutMix", "Rotate", "Shear", "Translate", "Affine",
    "HSV", "Color", "Brightness", "Contrast", "Saturation", "Hue",
    "Expand", "MinIoU", "YOLO", "Erasing",
]

def _is_probably_aug(t: str) -> bool:
    if t in _NON_AUG_TYPES:
        return False
    if t in _AUG_ALIAS:
        return True
    return any(h in t for h in _AUG_SUBSTR_HINTS)

def _iter_steps(obj: Any):
    if obj is None:
        return
    if isinstance(obj, dict):
        if 'type' in obj and isinstance(obj['type'], str):
            yield obj
        for k in ('pipeline', 'pipelines', 'transforms', 'transform',
                  'ops', 'augments', 'augmentations', 'dataset'):
            if k in obj:
                yield from _iter_steps(obj[k])
        # Albumentations wrapper
        if obj.get('type') in ('Albu', 'Albumentations') and 'transforms' in obj:
            for alb in obj['transforms']:
                if isinstance(alb, dict) and 'type' in alb:
                    alb['_albu_type'] = alb['type']
                    alb['type'] = 'Albu'
                    yield alb
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_steps(x)

def _collect_aug_types(cfg: Config) -> Tuple[Set[str], Set[str]]:
    all_types: Set[str] = set()
    signal: Set[str] = set()

    # top-level
    for key in ('load_pipeline', 'train_pipeline', 'test_pipeline'):
        if key in cfg:
            for s in _iter_steps(cfg.get(key)):
                t = s.get('type')
                if not t:
                    continue
                if _is_probably_aug(t):
                    all_types.add(t)
                    signal.add(_AUG_ALIAS.get(t, _camel_to_token(t)))
                if t in ('Albu', 'Albumentations') and s.get('_albu_type'):
                    all_types.add(f"albu:{s['_albu_type']}")

    # dataloader
    for dl_key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dl = cfg.get(dl_key, {})
        if isinstance(dl, dict):
            for s in _iter_steps(dl.get('dataset')):
                t = s.get('type')
                if not t:
                    continue
                if _is_probably_aug(t):
                    all_types.add(t)
                    signal.add(_AUG_ALIAS.get(t, _camel_to_token(t)))
                if t in ('Albu', 'Albumentations') and s.get('_albu_type'):
                    all_types.add(f"albu:{s['_albu_type']}")
    return all_types, signal

def _dataset_token(cfg: Config) -> str:
    try:
        ann = cfg.train_dataloader['dataset']['dataset'].get('ann_file', '')
        if 'minicoco' in str(ann).lower():
            return 'minicoco'
    except Exception:
        pass
    d = cfg.get('train_dataloader', {}).get('dataset', {})
    if isinstance(d, dict) and 'type' in d:
        return str(d['type']).lower().replace('dataset', '') or 'dataset'
    return 'dataset'

def compose_wandb_meta(cfg: Config):
    model = cfg.get('model', {})
    mtype = model.get('type', 'Model')
    bb = model.get('backbone', {})
    bb_tok = f"{bb.get('type','')}{bb.get('depth','')}" or 'backbone'
    img_sz = cfg.get('image_size', None)
    img_tok = f"{img_sz[0]}" if (isinstance(img_sz, (list, tuple)) and img_sz) else "var"

    all_aug, sig = _collect_aug_types(cfg)
    if model.get('use_lsj', False):
        sig.add('lsj')
    aug_token = "-".join(sorted(sig)) if sig else "noaug"
    if aug_token != "noaug":
        aug_token = "-".join(aug_token.split("-")[:4])

    ds = _dataset_token(cfg)
    epochs = cfg.get('max_epochs', '?')
    lr = (cfg.get('optim_wrapper', {}) or {}).get('optimizer', {}).get('lr', None)
    per_gpu_bs = cfg.get('train_dataloader', {}).get('batch_size', None)

    parts: List[str] = [mtype, bb_tok, img_tok, aug_token, ds, f"{epochs}e"]
    if lr is not None:
        parts.append(f"lr{lr:g}")
    if per_gpu_bs:
        parts.append(f"bs{per_gpu_bs}")
    name = "-".join(str(p) for p in parts if p)
    group = f"{mtype}-{ds}"

    tags = {
        'mmdet', mtype, bb_tok, ds, f'{epochs}e', f'img{img_tok}',
        *(s for s in all_aug), aug_token
    }
    tags = sorted({t for t in tags if t})
    facets = {
        "model.type": mtype, "backbone": bb_tok, "image_size": img_sz, "dataset": ds,
        "epochs": epochs, "batch_size_per_gpu": per_gpu_bs, "lr": lr,
        "augs_all": sorted(all_aug), "aug_token": aug_token,
        "config_path": getattr(cfg, "filename", None),
    }
    return name, group, tags, facets

def ensure_run_id_file(work_dir: str, force_new: bool = False) -> Tuple[str, bool]:
    """Return (run_id, existed_before)."""
    p = pathlib.Path(work_dir) / "wandb_run_id.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    if existed and not force_new:
        return p.read_text().strip(), True
    rid = uuid.uuid4().hex
    p.write_text(rid)
    return rid, existed and not force_new

def seed_env_from_cfg_file(cfg_path: str, work_dir: str, project: str, entity: str,
                           force_new_run: bool = False) -> str:
    """Load cfg, compose meta, ensure run-id, and emit a shell file exporting W&B envs."""
    cfg = Config.fromfile(cfg_path)
    name, group, tags, facets = compose_wandb_meta(cfg)
    run_id, existed = ensure_run_id_file(work_dir, force_new=force_new_run)

    # First-time -> allow; existing id -> must
    resume_mode = "must" if existed else "allow"

    env_lines = [
        f'export WANDB_PROJECT="{project}"',
        f'export WANDB_ENTITY="{entity}"',
        f'export WANDB_NAME="{name}"',
        f'export WANDB_RUN_GROUP="{group}"',   # correct var for group
        f'export WANDB_TAGS="{",".join(tags)}"',
        f'export WANDB_RUN_ID="{run_id}"',
        f'export WANDB_RESUME="{resume_mode}"',
    ]

    facets_file = pathlib.Path(work_dir) / "wandb_facets.txt"
    facets_file.write_text("\n".join([f"{k}={v}" for k, v in facets.items()]))

    env_file = pathlib.Path(work_dir) / "_wandb_env.sh"
    env_file.write_text("\n".join(env_lines) + "\n")
    return str(env_file)

def main():
    ap = argparse.ArgumentParser("Seed W&B environment from an MMDet config")
    ap.add_argument("--cfg", required=True, help="Path to config.py")
    ap.add_argument("--work-dir", required=True, help="Work directory (same as --work-dir to train)")
    ap.add_argument("--project", required=True, help="W&B project")
    ap.add_argument("--entity", required=True, help="W&B entity (user/team)")
    ap.add_argument("--force-new-run", action="store_true", help="Generate a new run id")
    args = ap.parse_args()
    path = seed_env_from_cfg_file(args.cfg, args.work_dir, args.project, args.entity, args.force_new_run)
    print(path)

if __name__ == "__main__":
    main()

