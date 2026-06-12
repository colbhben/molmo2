"""Offline unit tests for the gaze video-point dataset's example construction.

The full molmo2 stack (datasets/torch/...) is NOT importable in a bare checkout, so we
stub the two things `gaze_datasets` imports (`DatasetBase`, `set_example_style`) and exercise
the pure row->example transform (`_row_to_msg`) + objective slicing directly. This guards the
load-bearing contract with Molmo2's point formatter:
  * points must be 0-100 (formatter divides by scale=100 then clamps to [0,1]) -- we store
    raw pixels on a `resolution` frame and must normalize pixel -> 0-100;
  * the target is REAL gaze only (empty frames dropped before objective slicing);
  * "first" yields exactly one point, "all" yields every real-gaze frame.
"""
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --- stub the heavy imports gaze_datasets pulls from olmo ------------------------------- #
_dataset_mod = types.ModuleType("olmo.data.dataset")


class _StubDatasetBase:
    def __init__(self, split, sample=None):
        self.split = split
        self.sample = sample
        data = self.load()
        if sample is not None:
            data = data[:sample]
        self.data = data

    def load(self):  # overridden
        raise NotImplementedError


_dataset_mod.DatasetBase = _StubDatasetBase
sys.modules["olmo.data.dataset"] = _dataset_mod

_util_mod = types.ModuleType("olmo.util")
_util_mod.set_example_style = lambda ex, style: {**ex, "style": style}
sys.modules["olmo.util"] = _util_mod

from olmo.data.gaze_datasets import GazeVideoPoint, _OBJECTIVE_KEEP  # noqa: E402


def _row(n_frames=6, side=378, empty_first=False, empty_idx=()):
    """Synthetic joint-manifest clip row: per-frame single pixel point on a `side` frame."""
    pts, ts = [], []
    for j in range(n_frames):
        if (empty_first and j == 0) or j in empty_idx:
            pts.append([])              # no/invalid gaze this frame
        else:
            # pixel point at (j*10, j*10+5) on the padded frame
            pts.append([{"x": float(j * 10), "y": float(j * 10 + 5)}])
        ts.append(round(j / 6.0, 3))    # fps=6 grid
    return {
        "id": f"egtea:ep#seg0", "dataset": "egtea", "video": "/abs/clip.mp4",
        "resolution": side, "points": pts, "timestamps": ts,
        "metadata": {"clip_start_time": 0.0, "clip_end_time": round(n_frames / 6.0, 3),
                     "final_annotation": "cutting a tomato", "annotation_text": "kitchen"},
    }


class _GVP(GazeVideoPoint):
    """Bypass file IO: feed rows straight into the real `_row_to_msg`/load path."""

    def __init__(self, rows, **kw):
        self._rows = rows
        super().__init__(**kw)

    def load(self):
        out = []
        for r in self._rows:
            m = self._row_to_msg(r)
            if m is not None:
                out.append(m)
        return out


class TestObjectiveSlicing(unittest.TestCase):
    def test_first_yields_single_point(self):
        ds = _GVP([_row(6)], objective="first")
        ex = ds.data[0]
        self.assertEqual(ex["count"], 1)
        self.assertEqual(len(ex["points"]), 1)
        self.assertEqual(len(ex["points"][0]), 1)
        self.assertEqual(ex["timestamps"], [0.0])  # earliest real gaze

    def test_all_yields_every_real_frame(self):
        ds = _GVP([_row(6)], objective="all")
        ex = ds.data[0]
        self.assertEqual(len(ex["points"]), 6)
        self.assertEqual(ex["count"], 6)

    def test_unknown_objective_raises(self):
        with self.assertRaises(ValueError):
            _GVP([_row(6)], objective="bogus")


class TestRealGazeOnly(unittest.TestCase):
    def test_first_skips_empty_t0(self):
        # frame 0 has no gaze -> "first" must pick frame 1's gaze (t=0.167), not "There are none."
        ds = _GVP([_row(6, empty_first=True)], objective="first")
        ex = ds.data[0]
        self.assertEqual(ex["count"], 1)
        self.assertAlmostEqual(ex["timestamps"][0], round(1 / 6.0, 3))

    def test_all_drops_empty_frames(self):
        ds = _GVP([_row(6, empty_idx=(2, 4))], objective="all")
        ex = ds.data[0]
        self.assertEqual(len(ex["points"]), 4)  # 6 - 2 empty

    def test_no_gaze_at_all_dropped(self):
        row = _row(3, empty_idx=(0, 1, 2))
        ds = _GVP([row], objective="all")
        self.assertEqual(len(ds.data), 0)


class TestCoordinateScale(unittest.TestCase):
    def test_pixels_normalized_to_0_100(self):
        # pixel (10,15) on a 378 frame -> (10/378*100, 15/378*100) ~= (2.65, 3.97)
        ds = _GVP([_row(6)], objective="all")
        # frame j=1 had pixel (10,15)
        ex = ds.data[0]
        p = ex["points"][1][0]
        self.assertAlmostEqual(p["x"], 10 / 378 * 100, places=4)
        self.assertAlmostEqual(p["y"], 15 / 378 * 100, places=4)
        # every emitted coord is within the formatter's expected 0-100 range
        for fr in ex["points"]:
            for q in fr:
                self.assertTrue(0.0 <= q["x"] <= 100.0)
                self.assertTrue(0.0 <= q["y"] <= 100.0)

    def test_missing_resolution_skips_clip(self):
        row = _row(6)
        row.pop("resolution")
        ds = _GVP([row], objective="first")
        self.assertEqual(len(ds.data), 0)


class TestEvalMetadata(unittest.TestCase):
    def test_gt_abs_triplets_are_raw_pixels(self):
        # eval GT must be pixel triplets (t, x_px, y_px), NOT the 0-100 training form
        ds = _GVP([_row(6)], objective="all")
        md = ds.data[0]["metadata"]
        self.assertEqual(len(md["gt_abs_triplets"]), 6)
        t, x, y = md["gt_abs_triplets"][1]   # frame j=1 had pixel (10,15)
        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 15.0)
        self.assertAlmostEqual(t, round(1 / 6.0, 3))
        self.assertEqual(md["video_height"], 378)
        self.assertEqual(md["video_width"], 378)
        self.assertGreater(md["video_duration"], 0)

    def test_first_objective_single_gt_triplet(self):
        ds = _GVP([_row(6)], objective="first")
        md = ds.data[0]["metadata"]
        self.assertEqual(len(md["gt_abs_triplets"]), 1)

    def test_clip_bounds_set_when_duration_known(self):
        ds = _GVP([_row(6)], objective="all")
        md = ds.data[0]["metadata"]
        self.assertEqual(md["clip_start_time"], 0.0)
        self.assertGreater(md["clip_end_time"], 0)

    def test_clip_bounds_omitted_when_duration_missing(self):
        # No clip_end_time -> must NOT emit clip_start_time (else loader gets (0.0, None)).
        row = _row(6)
        row["metadata"].pop("clip_end_time")
        ds = _GVP([row], objective="first")
        md = ds.data[0]["metadata"]
        self.assertNotIn("clip_start_time", md)
        self.assertNotIn("clip_end_time", md)
        self.assertEqual(md["video_duration"], 1.0)   # safe fallback for eval normalization


class TestExampleSchema(unittest.TestCase):
    def test_label_is_annotation_input(self):
        ds = _GVP([_row(6)], objective="first")
        ex = ds.data[0]
        self.assertEqual(ex["label"], "cutting a tomato")  # final_annotation preferred
        self.assertEqual(ex["subset"], "egtea")
        self.assertEqual(ex["video"], "/abs/clip.mp4")
        self.assertEqual(ex["metadata"]["gaze_objective"], "first")

    def test_label_fallbacks(self):
        row = _row(6)
        row["metadata"].pop("final_annotation")
        ds = _GVP([row], objective="first")
        self.assertEqual(ds.data[0]["label"], "kitchen")  # annotation_text fallback

    def test_style_routes_to_video_gaze_point(self):
        import numpy as np
        ds = _GVP([_row(6)], objective="first")
        ex = ds.get(0, np.random)
        self.assertEqual(ex["style"], "video_gaze_point")


if __name__ == "__main__":
    unittest.main()
