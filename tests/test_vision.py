"""Tests for green-marker detection.

Green markers tell the robot which way to turn at an intersection, so a false
positive sends it down the wrong branch and a missed one sends it straight past.
The two spatial filters — bottom half only, ignore side margins — exist to stop
markers on adjacent track or arena clutter from being read as instructions, and
they are the parts most worth pinning down.
"""
import numpy as np
import pytest

from vision import Vision

W, H = 320, 240


def quiet(*_args, **_kwargs):
    """Silence the module's logging inside tests."""


@pytest.fixture
def vis():
    return Vision({}, log_function=quiet)


def frame(spots, size=40):
    """BGR frame with pure-green squares at the given (x, y) centres."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for cx, cy in spots:
        img[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = (0, 255, 0)
    return img


def test_empty_frame_detects_nothing(vis):
    cents, direction = vis.detect_greens(np.zeros((H, W, 3), dtype=np.uint8))
    assert cents == []
    assert direction is None


def test_single_marker_left_of_centre(vis):
    cents, direction = vis.detect_greens(frame([(80, 200)]))
    assert len(cents) == 1
    assert direction == "left"


def test_single_marker_right_of_centre(vis):
    cents, direction = vis.detect_greens(frame([(240, 200)]))
    assert len(cents) == 1
    assert direction == "right"


def test_two_markers_are_a_uturn(vis):
    cents, direction = vis.detect_greens(frame([(80, 200), (240, 200)]))
    assert len(cents) == 2
    assert direction == "uturn"


def test_marker_in_the_top_half_is_ignored(vis):
    """Only the bottom half is acted on — green further up the frame is track
    the robot has not reached yet."""
    cents, direction = vis.detect_greens(frame([(80, 40)]))
    assert cents == []
    assert direction is None


def test_marker_inside_the_side_margin_is_ignored(vis):
    """The outer 10% of the frame is clutter, not the robot's own lane."""
    cents, direction = vis.detect_greens(frame([(12, 200)]))
    assert cents == []
    assert direction is None


def test_blob_below_min_area_is_ignored(vis):
    cents, direction = vis.detect_greens(frame([(80, 200)], size=8))
    assert cents == []
    assert direction is None


def test_at_most_three_blobs_are_returned(vis):
    img = frame([(60, 180), (110, 180), (170, 180), (220, 180), (270, 200)])
    cents, _ = vis.detect_greens(img)
    assert len(cents) <= 3


def test_update_config_changes_thresholds(vis):
    vis.update_config({"min_area": 999999})
    cents, direction = vis.detect_greens(frame([(80, 200)]))
    assert cents == []                      # nothing clears the new threshold
    assert direction is None


def test_update_config_ignores_non_dict(vis):
    before = dict(vis.cfg)
    vis.update_config(None)
    vis.update_config([1, 2, 3])
    assert vis.cfg == before


def test_calibrate_by_click_picks_up_the_clicked_pixel(vis):
    img = frame([(80, 200)])
    assert vis.calibrate_by_click(img, 80, 200, color="green") is True
    assert vis.cfg["green_h_center"] == 60          # pure green in OpenCV HSV
    assert vis.cfg["green_s_min"] >= 40
    assert vis.cfg["green_v_min"] >= 40


def test_calibrate_by_click_rejects_unknown_colour(vis):
    assert vis.calibrate_by_click(frame([(80, 200)]), 80, 200, color="blue") is False


def test_calibrate_by_click_survives_out_of_bounds(vis):
    """A click outside the frame must not raise into the control loop."""
    assert vis.calibrate_by_click(frame([(80, 200)]), 99999, 99999) is False


def test_detection_survives_recalibration_round_trip(vis):
    img = frame([(240, 200)])
    vis.calibrate_by_click(img, 240, 200, color="green")
    cents, direction = vis.detect_greens(img)
    assert len(cents) == 1
    assert direction == "right"
