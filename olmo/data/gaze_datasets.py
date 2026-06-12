"""Gaze video-point dataset for Molmo2 SFT (specialize-then-rehearse).

Loads the gaze CLIP manifest produced by the `gaze` curation repo and exposes it as a
Molmo2 video-point dataset. Each clip is a short egocentric video; the per-frame target
is where the camera wearer is looking, as raw PIXEL `{x,y}` points on the padded frame
(the same convention `Molmo2VideoPoint` reads).

Data layout (under ``GAZE_DATA_DIR``, all LOCAL — never /nfs):
  joint/manifest.jsonl        the joint clip manifest (one row per clip, ABSOLUTE `video`
                              path, per-frame `points`/`timestamps`, `metadata`, ...)
  splits/<name>/<split>.jsonl lightweight pointers {id, dataset, video} selecting which
                              clips belong to this train/val split

The split pointer files (produced by `gaze curate make-splits`) decide membership; the
joint manifest (joined by `id`) supplies the actual points/timestamps/video. We keep
membership and payload separate so a trainer can pick a split without copying clip data.

Training objective (``GAZE_OBJECTIVE`` env, or ``objective=`` arg):
  "first"  -> predict ONLY the first gaze point (t0) given the full video. Each example
              emits points[:1] / timestamps[:1].
  "all"    -> predict ALL per-frame gaze points. Emits every frame's point.
Future variants (e.g. "first_chunk") add an entry to ``_OBJECTIVE_KEEP``.

The video frames the model sees are ALWAYS the full clip (the video loader samples frames
from the file independently); only the *target* point set changes with the objective.
"""
import json
import logging
import os
from os.path import join, exists, isabs

from olmo.data.dataset import DatasetBase
from olmo.util import set_example_style

log = logging.getLogger(__name__)

# Objectives: name -> how many leading frames of (points, timestamps) to keep as the target.
#   "first" = 1 (predict t0), "all" = None (keep every frame).
_OBJECTIVE_KEEP = {
    "first": 1,
    "all": None,
}


def gaze_data_dir() -> str:
    d = os.environ.get("GAZE_DATA_DIR")
    if not d:
        raise RuntimeError(
            "GAZE_DATA_DIR is not set. Point it at the local dir holding "
            "joint/manifest.jsonl and splits/<name>/<split>.jsonl."
        )
    return d


class GazeVideoPoint(DatasetBase):
    """Gaze clips as a Molmo2 video-point dataset (mirrors Molmo2VideoPoint's msg schema).

    Reads a split's pointer file (which clip ids are in this split) and joins the joint
    clip manifest (the per-clip points/timestamps/video). Builds one example `msg` dict
    per clip in the exact shape `Molmo2VideoPoint` emits, so the existing preprocessor /
    point formatter / video loader consume it unchanged.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        objective: str = None,
        split_name: str = None,
        mode="gaze_point",
        point_sort_by: str = "xy",
        sample: int = None,
    ):
        # objective/split_name come from env when the launcher sets them, so a single
        # registry entry serves both first/all modes and any split name.
        self.objective = objective or os.environ.get("GAZE_OBJECTIVE", "first")
        if self.objective not in _OBJECTIVE_KEEP:
            raise ValueError(f"unknown GAZE_OBJECTIVE {self.objective!r}; known: {list(_OBJECTIVE_KEEP)}")
        self.split_name = split_name or os.environ.get("GAZE_SPLIT_NAME", "v1_95_05")
        self.mode = mode
        self.point_sort_by = point_sort_by
        super().__init__(split, sample=sample)

    # --- loading -------------------------------------------------------------------- #
    def _split_file(self) -> str:
        # split arg: "train"/"val"/"validation" -> the pointer file name. molmo2 uses
        # "validation"; our make-splits writes "val.jsonl".
        split = {"validation": "val"}.get(self.split, self.split)
        return join(gaze_data_dir(), "splits", self.split_name, f"{split}.jsonl")

    def _manifest_file(self) -> str:
        return join(gaze_data_dir(), "joint", "manifest.jsonl")

    def _resolve_video(self, video: str) -> str:
        return video if isabs(video) else join(gaze_data_dir(), video)

    def load(self):
        split_path = self._split_file()
        manifest_path = self._manifest_file()
        if not exists(split_path):
            raise FileNotFoundError(f"gaze split pointer file not found: {split_path}")
        if not exists(manifest_path):
            raise FileNotFoundError(f"gaze joint manifest not found: {manifest_path}")

        # 1. which clip ids belong to this split
        want_ids = set()
        with open(split_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                want_ids.add(json.loads(line)["id"])

        # 2. stream the joint manifest, keep only this split's clips, build example dicts
        data = []
        kept = 0
        skipped_no_points = 0
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "error" in row or row.get("id") not in want_ids:
                    continue
                msg = self._row_to_msg(row)
                if msg is None:
                    skipped_no_points += 1
                    continue
                data.append(msg)
                kept += 1
        log.info(
            f"GazeVideoPoint[{self.split_name}/{self.split}] objective={self.objective}: "
            f"{kept} clips (of {len(want_ids)} in split); skipped {skipped_no_points} with no usable point."
        )
        return data

    def _row_to_msg(self, row: dict) -> dict | None:
        """One joint-manifest clip row -> the Molmo2VideoPoint `msg` example dict."""
        points = row.get("points") or []
        timestamps = row.get("timestamps") or []
        if not points or not timestamps or len(points) != len(timestamps):
            return None

        # REAL GAZE ONLY: frames with no/invalid gaze are stored as empty point lists
        # (frame_mask==0). The OUTPUT target must be real gaze points, so drop empty frames
        # BEFORE objective slicing -- otherwise "first" could pick an empty t0 frame and the
        # formatter would emit "There are none." instead of a point. Timestamps are already
        # clip-relative (frame j at j/fps) per the gaze manifest, so we keep them as-is.
        real = [(fp, ts) for fp, ts in zip(points, timestamps) if fp]
        if not real:
            return None
        # Objective lever: keep the leading N real-gaze frames. "first" => earliest gaze
        # point only (predict t0 given the full video); "all" => every real gaze frame.
        keep = _OBJECTIVE_KEEP[self.objective]
        if keep is not None:
            real = real[:keep]
        points = [fp for fp, _ in real]
        timestamps = [ts for _, ts in real]

        # COORDINATE SCALE: our manifest stores RAW PIXELS on a `resolution`x`resolution`
        # padded frame, but Molmo2's point formatter expects points in a 0-100 range
        # (UnifiedPointFormatter._scale_point divides by scale=100 then clamps to [0,1];
        # Molmo2VideoPoint stores "points normalized 0-100"). Convert pixel -> 0-100 here,
        # else every point clamps to the frame edge.
        side = float(row.get("resolution") or 0) or None
        if side is None:
            # No resolution recorded: can't safely normalize raw pixels -> skip the clip
            # rather than emit edge-clamped garbage.
            return None

        def _to_100(p):
            return {"x": p["x"] / side * 100.0, "y": p["y"] / side * 100.0}

        # Per-frame point sort (matches Molmo2VideoPoint.point_sort_by). We keep BOTH the
        # 0-100 form (the training target) and the raw pixel triplets (the eval GT, which the
        # GazePointEval matches against predictions parsed back into pixel space).
        md = row.get("metadata") or {}
        clip_start = md.get("clip_start_time", 0.0) or 0.0
        sorted_points = []
        rel_ts = []
        gt_abs_triplets = []  # (t, x_px, y_px) on the side x side frame -- eval GT
        for fp_px, ts in zip(points, timestamps):
            t = round(float(ts), 3)  # clip-relative in our manifest
            for p in fp_px:
                gt_abs_triplets.append((t, float(p["x"]), float(p["y"])))
            fp = [_to_100(p) for p in fp_px]
            if self.point_sort_by == "xy":
                fp = sorted(fp, key=lambda p: (p["x"], p["y"]))
            elif self.point_sort_by == "yx":
                fp = sorted(fp, key=lambda p: (p["y"], p["x"]))
            sorted_points.append(fp)
            rel_ts.append(t)

        n_points = sum(len(fp) for fp in sorted_points)
        clip_end = md.get("clip_end_time")
        clip_dur = (clip_end - clip_start) if clip_end is not None else None
        # ANNOTATION = INPUT: `label` fills the input prompt (the gaze-point style's prompt
        # template asks "given <label>, where is the wearer looking"); the OUTPUT/target is
        # gaze points only. Prefer the recipe's final bundled annotation, else the source text.
        label = md.get("final_annotation") or md.get("annotation_text") or "the current activity"

        metadata = {
            "dataset": row.get("dataset"),
            "gaze_objective": self.objective,
            "example_id": row["id"],
            "points": sorted_points,
            "timestamps": rel_ts,
            # --- GazePointEval inputs (L2 + accuracy@radius) --------------------------
            # GT in pixel space; predictions are parsed back to pixels via video dims.
            # The clip is a square `side`x`side` frame, so width == height == side.
            "gt_abs_triplets": gt_abs_triplets,
            "video_duration": clip_dur if clip_dur and clip_dur > 0 else 1.0,
            "video_height": side,
            "video_width": side,
        }
        # Only set clip bounds when the duration is real & positive. The preprocessor reads
        # clip = (clip_start_time, clip_end_time) whenever clip_start_time is not None, so a
        # missing/zero clip_end would feed (0.0, None) into the video loader -- omit instead
        # and let the loader sample the whole clip file.
        if clip_dur and clip_dur > 0:
            metadata["clip_start_time"] = 0.0
            metadata["clip_end_time"] = clip_dur

        return {
            "subset": row.get("dataset", "gaze"),
            "example_id": row["id"],
            "label": label,
            "answer": str(n_points),
            "count": n_points,
            "points": sorted_points,
            "timestamps": rel_ts,
            "video": self._resolve_video(row["video"]),
            "metadata": metadata,
        }

    # --- access --------------------------------------------------------------------- #
    def _style(self) -> str:
        return f"video_{self.mode}"  # "video_gaze_point"

    def get(self, idx, rng):
        example = dict(self.data[idx])
        return set_example_style(example, self._style())


class GazeVideoPointEval(GazeVideoPoint):
    """Held-out gaze split for inference-eval metrics (point L2 / accuracy@radius).

    Same payload as GazeVideoPoint but loads the ``val`` split by default and is used by
    the video-point inference evaluator (registered in sft.py's evaluations).
    """

    def __init__(self, split: str = "validation", **kwargs):
        super().__init__(split=split, **kwargs)
