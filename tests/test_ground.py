import numpy as np

from line_clearance.config import GroundConfig
from line_clearance.ground import classify_ground


def test_sloped_ground_with_an_object_on_top():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 60, (40_000, 2))
    slope = 100.0 + 0.08 * xy[:, 0] + 0.03 * xy[:, 1]
    ground = np.column_stack([xy, slope + rng.normal(0, 0.02, len(xy))])
    box_xy = rng.uniform(25, 31, (3_000, 2))
    box_base = 100.0 + 0.08 * box_xy[:, 0] + 0.03 * box_xy[:, 1]
    box = np.column_stack([box_xy, box_base + rng.uniform(2.0, 9.0, len(box_xy))])

    is_ground, hag, _ = classify_ground(np.vstack([ground, box]), GroundConfig())

    assert is_ground[: len(ground)].mean() > 0.98
    assert not is_ground[len(ground) :].any()
    assert np.abs(hag[: len(ground)]).mean() < 0.1
    true_height = box[:, 2] - box_base
    assert np.abs(hag[len(ground) :] - true_height).mean() < 0.25
