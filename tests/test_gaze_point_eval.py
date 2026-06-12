"""Standalone test of the GazePointEval scoring math.

`evaluators.py` pulls in torch (unavailable in a bare checkout), so we can't import the class
directly. This test re-implements the EXACT scoring kernel from `GazePointEval.__call__`
(bipartite match on normalized x,y -> L2 + accuracy@radius) and verifies its behavior on
hand-checked cases. Keep this in sync with `olmo/eval/evaluators.py::GazePointEval`.
"""
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def normalize(triplets, video_duration, video_h, video_w, upper_bound=100, nd=3):
    out = []
    for ts, x, y in triplets:
        out.append((round(ts / video_duration * upper_bound, nd),
                    round(x / video_w * upper_bound, nd),
                    round(y / video_h * upper_bound, nd)))
    return out


RADII = (5.0, 10.0, 15.0)


def score(gt_abs, pred_abs, video_duration, video_h, video_w):
    """Mirror of GazePointEval.__call__'s per-example scoring."""
    s = {"valid": float(len(pred_abs) > 0)}
    if len(gt_abs) == 0:
        return s
    gt_norm = normalize(gt_abs, video_duration, video_h, video_w)
    if len(pred_abs) == 0:
        for r in RADII:
            s[f"acc@{r:g}"] = 0.0
        return s
    pred_norm = normalize(pred_abs, video_duration, video_h, video_w)
    pred_xy = np.array([[x, y] for _, x, y in pred_norm], dtype=np.float64)
    gt_xy = np.array([[x, y] for _, x, y in gt_norm], dtype=np.float64)
    dists = cdist(pred_xy, gt_xy)
    row_ind, col_ind = linear_sum_assignment(dists)
    matched = dists[row_ind, col_ind]
    s["l2"] = [float(d) for d in matched]
    n_gt = len(gt_xy)
    for r in RADII:
        s[f"acc@{r:g}"] = int(np.sum(matched <= r)) / n_gt
    return s


class TestGazeScoring(unittest.TestCase):
    def test_perfect_prediction_zero_l2_full_acc(self):
        # GT and pred identical -> L2 == 0, acc@all == 1
        gt = [(0.0, 189.0, 189.0)]   # center of a 378 frame
        s = score(gt, list(gt), video_duration=1.0, video_h=378, video_w=378)
        self.assertAlmostEqual(s["l2"][0], 0.0)
        for r in RADII:
            self.assertEqual(s[f"acc@{r:g}"], 1.0)
        self.assertEqual(s["valid"], 1.0)

    def test_small_offset_within_radius(self):
        # pred off by ~5 normalized units (5% of 378 = 18.9 px) -> within acc@10/15, maybe not @5
        gt = [(0.0, 189.0, 189.0)]
        pred = [(0.0, 189.0 + 0.05 * 378, 189.0)]   # +5.0 normalized in x
        s = score(gt, pred, video_duration=1.0, video_h=378, video_w=378)
        self.assertAlmostEqual(s["l2"][0], 5.0, places=2)
        self.assertEqual(s["acc@5"], 1.0)    # exactly at radius (<=)
        self.assertEqual(s["acc@10"], 1.0)
        self.assertEqual(s["acc@15"], 1.0)

    def test_far_miss_zero_acc(self):
        gt = [(0.0, 10.0, 10.0)]
        pred = [(0.0, 370.0, 370.0)]   # opposite corner
        s = score(gt, pred, video_duration=1.0, video_h=378, video_w=378)
        self.assertGreater(s["l2"][0], 15.0)
        for r in RADII:
            self.assertEqual(s[f"acc@{r:g}"], 0.0)

    def test_no_prediction_misses_all(self):
        gt = [(0.0, 189.0, 189.0)]
        s = score(gt, [], video_duration=1.0, video_h=378, video_w=378)
        self.assertEqual(s["valid"], 0.0)
        for r in RADII:
            self.assertEqual(s[f"acc@{r:g}"], 0.0)
        self.assertNotIn("l2", s)   # no L2 when nothing was localized

    def test_multi_point_bipartite_matching(self):
        # 2 GT points, 2 preds swapped in order -> matching pairs them correctly, L2~0
        gt = [(0.0, 50.0, 50.0), (0.1, 300.0, 300.0)]
        pred = [(0.1, 300.0, 300.0), (0.0, 50.0, 50.0)]  # reversed
        s = score(gt, pred, video_duration=1.0, video_h=378, video_w=378)
        self.assertEqual(len(s["l2"]), 2)
        for d in s["l2"]:
            self.assertAlmostEqual(d, 0.0, places=3)
        self.assertEqual(s["acc@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
