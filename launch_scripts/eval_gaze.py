"""Standalone gaze eval: checkpoint -> ALL val episodes -> cached per-episode S3 results.

Given a trained gaze checkpoint (olmo-native: config.yaml + model_and_optim/; local path or
s3://), this runs the NATIVE molmo2 gaze eval (`gaze_video_point_eval` -> GazePointEval: L2 in
0-100 space + acc@5/10/15 + valid) on EVERY val episode and writes a self-describing cache the
gaze viewer ingests:

  s3://far-research-internal/colbhben/gaze/evals/<run_name>/step<N>/
      summary.json     aggregate metrics + run/checkpoint slugs + episode count
      results.jsonl    one record per episode (GT + predicted gaze triplets + per-episode metrics)

The val split is the same one the training run validated on: it comes from GAZE_DATA_DIR
(joint/manifest.jsonl + splits/<GAZE_SPLIT_NAME>/val.jsonl), exactly as gaze_sft.sh sets up.

Single-GPU, single-process: launch with `torchrun --nproc-per-node=1`. The sharded distcp
checkpoint is loaded onto the one GPU without FSDP (mirrors the in-loop eval's unsharded gather).

Usage (driven by training/gaze_eval.sh, but runnable directly):
  torchrun --nproc-per-node=1 launch_scripts/eval_gaze.py \
      s3://.../molmo/runs/<run>/step<N>/ \
      --bundle-s3-uri s3://.../manifests/<bundle>/ \
      [--max-examples N] [--device-batch-size N] [--overwrite] [--out-uri s3://...]
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from os.path import basename, dirname
from typing import cast

import torch.cuda
from omegaconf import OmegaConf

from olmo.eval.eval_utils import get_evaluation
from olmo.eval.evaluators import GazePointEvalCache
from olmo.eval.inf_evaluator import InfEvaluator
from olmo.eval.model_evaluator import EvalConfig
from olmo.io import file_exists, write_file
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo_point.molmo_point import MolmoPointConfig
from olmo.torch_util import get_global_rank
from olmo.util import clean_opt, prepare_torchrun_environment, resource_path, select_checkpoint

log = logging.getLogger(__name__)

EVALS_ROOT = "s3://far-research-internal/colbhben/gaze/evals"
RADII = (5.0, 10.0, 15.0)
METRIC_KEYS = ["l2", "valid"] + [f"acc@{r:g}" for r in RADII]


def _checkpoint_step(load_path: str):
    m = re.match(r".*/(?:step|bk)([0-9]+).*", load_path.rstrip("/"))
    return int(m.group(1)) if m else None


def _checkpoint_slug(load_path: str) -> str:
    step = _checkpoint_step(load_path)
    return f"step{step}" if step is not None else basename(load_path.rstrip("/")) or "checkpoint"


def _aggregate(records, eval_metrics):
    """Aggregate per-episode metrics, falling back to the native eval's aggregate."""
    agg = {}
    for k in METRIC_KEYS:
        # Prefer the native GazePointEval aggregate (identical math) when present.
        if k in eval_metrics and isinstance(eval_metrics[k], (int, float)):
            agg[k] = float(eval_metrics[k])
            continue
        vals = [r["metrics"].get(k) for r in records if r["metrics"].get(k) is not None]
        agg[k] = float(sum(vals) / len(vals)) if vals else None
    return agg


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Standalone gaze eval -> cached S3 results")
    parser.add_argument("checkpoint", help="Checkpoint dir (config.yaml + model_and_optim/); local or s3://")
    parser.add_argument("--bundle-s3-uri", required=True,
                        help="S3 URI of the manifest bundle (videos/<dataset>/<file>.mp4 live under it); "
                             "used to build each episode's video_s3_uri for the viewer.")
    parser.add_argument("--out-uri", default=None,
                        help="Output prefix; default s3://.../evals/<run_name>/<step>/.")
    parser.add_argument("--max-examples", type=int, default=-1,
                        help="Cap episodes (-1 = ALL val, the default).")
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if summary.json already exists at the output prefix.")
    args, other_args = parser.parse_known_args()

    is_rank0 = get_global_rank() == 0

    checkpoint_dir = select_checkpoint(args.checkpoint)
    run_name = basename(dirname(checkpoint_dir.rstrip("/"))) or "run"
    ckpt_slug = _checkpoint_slug(checkpoint_dir)
    out_uri = args.out_uri or f"{EVALS_ROOT}/{run_name}/{ckpt_slug}"
    summary_uri = f"{out_uri}/summary.json"

    if file_exists(summary_uri) and not args.overwrite:
        log.info(f"{summary_uri} already exists and --overwrite not set; nothing to do.")
        return

    log.info(f"Evaluating checkpoint {checkpoint_dir}")
    log.info(f"Output -> {out_uri}")

    # Model config from the checkpoint's config.yaml (bump crops/images like eval_molmo2).
    model_cfg_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")
    model_cfg = MolmoConfig.load(model_cfg_path, key="model", validate_paths=False)
    if isinstance(model_cfg, (Molmo2Config, MolmoPointConfig)):
        if model_cfg.mm_preprocessor.image is not None:
            model_cfg.mm_preprocessor.image.max_crops = max(model_cfg.mm_preprocessor.image.max_crops, 24)
            model_cfg.mm_preprocessor.image.max_images = 20
    elif isinstance(model_cfg, MolmoConfig):
        model_cfg.mm_preprocessor.max_crops = 24
        model_cfg.mm_preprocessor.max_images = 20
    model_cfg.llm.max_sequence_length = 64000

    # One gaze val evaluation, on ALL episodes (max_examples=-1).
    base = get_evaluation(
        name="gaze_video_point_eval",
        seq_len=None,
        max_examples=args.max_examples,
        num_workers=args.num_workers,
        device_batch_size=args.device_batch_size,
        num_wandb_examples=0,
    )
    from dataclasses import replace
    from olmo.eval.model_evaluator import DatasetEvaluatorConfig
    evaluation = DatasetEvaluatorConfig(
        label=base.label,
        data=replace(base.data, pad=None),
        generative_evaluator=replace(base.evaluator, n_to_log=0, num_wandb_examples=0, save_predictions=None),
        sampling=base.sampling,
        device_batch_size=args.device_batch_size,
        subset_num_batches=None,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens or base.max_new_tokens,
    )

    cfg = EvalConfig(
        evaluations=[evaluation],
        load_path=checkpoint_dir,
        console_log_interval=5,
        beaker_log_interval=-1,
        precision="amp_bf16",
        pbar=False,
        fsdp=None,            # single GPU: load the sharded distcp into the full module, no FSDP.
        load_bf16=True,
        skip_if_metrics_cached=False,
        save_dir=None,
        save_to_checkpoint_dir=False,
        model=model_cfg,
    )
    config = OmegaConf.create(cfg)
    if other_args:
        config.merge_with_dotlist([clean_opt(a) for a in other_args])
    cfg = cast(EvalConfig, OmegaConf.to_object(config))

    evaluator_runner = cfg.build()
    model, device = evaluator_runner.initialize_and_load_model()

    # Build the dataset evaluator, then swap in our caching evaluator so we recover per-episode
    # records (the native run() returns only the aggregate dict and discards predictions/metadata).
    dataset_evaluator = evaluation.build_evaluator(
        model.config, device, None, cfg.console_log_interval, cfg.include_image
    )
    cache_eval = GazePointEvalCache(n_to_log=0)
    dataset_evaluator.evaluator = InfEvaluator([cache_eval])

    metrics = dataset_evaluator.run(
        model, device,
        autocast_precision=cfg.autocast_precision,
        is_distributed=False,
        pbar=False,
    )
    torch.cuda.empty_cache()

    if not is_rank0:
        return

    records = getattr(cache_eval, "episode_records", [])
    bundle = args.bundle_s3_uri.rstrip("/")

    def _video_s3_uri(rec):
        vp = rec.get("video_path") or ""
        ds = rec.get("dataset") or ""
        return f"{bundle}/videos/{ds}/{basename(vp)}" if vp else None

    # results.jsonl (viewer contract): drop the absolute local video_path, add the S3 pointer.
    lines = []
    for rec in records:
        out = {
            "example_id": rec["example_id"],
            "dataset": rec["dataset"],
            "label": rec["label"],
            "frame_side": rec["frame_side"],
            "video_duration": rec["video_duration"],
            "clip_start_time": rec["clip_start_time"],
            "clip_end_time": rec["clip_end_time"],
            "video_s3_uri": _video_s3_uri(rec),
            "gt_triplets": rec["gt_triplets"],
            "pred_triplets": rec["pred_triplets"],
            "prediction_text": rec["prediction_text"],
            "metrics": rec["metrics"],
        }
        lines.append(json.dumps(out, sort_keys=True))
    results_jsonl = "\n".join(lines) + ("\n" if lines else "")

    summary = {
        "schema_version": 1,
        "run_name": run_name,
        "checkpoint_step": _checkpoint_step(checkpoint_dir),
        "checkpoint_uri": checkpoint_dir,
        "training_slug": run_name,
        "checkpoint_slug": ckpt_slug,
        "split_name": os.environ.get("GAZE_SPLIT_NAME", "v1_95_05"),
        "split": "validation",
        "dataset_task": "gaze_video_point_eval",
        "gaze_objective": os.environ.get("GAZE_OBJECTIVE", "first"),
        "n_episodes": len(records),
        "max_new_tokens": evaluation.max_new_tokens,
        "bundle_s3_uri": bundle,
        "metrics": _aggregate(records, metrics),
        "radii": list(RADII),
        "created": datetime.utcnow().isoformat() + "Z",
    }

    write_file(out_uri, "results.jsonl", results_jsonl, save_overwrite=True)
    write_file(out_uri, "summary.json", json.dumps(summary, indent=2, sort_keys=True), save_overwrite=True)
    log.info(f"Wrote {len(records)} episode records + summary to {out_uri}")
    log.info(f"Aggregate metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
