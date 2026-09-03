"""Tests for the scanline / derivative line detection ported from RCJ 2014.

The robot follows the line by sampling a 1D strip of the image and finding the
pair of derivative peaks that bracket a dark band. These cover that the peaks
are found where they should be, and — as importantly — that nothing is reported
when the image holds no line.
"""
import math

import numpy as np
import pytest

import rcj2014_port as rcj


def bright_row_with_dark_band(width=120, start=50, end=70, bright=200, dark=10):
    row = np.full(width, bright, dtype=np.uint8)
    row[start:end] = dark
    return row


def frame_with_dark_band(h=80, w=120, row=40, start=50, end=70):
    img = np.full((h, w), 200, dtype=np.uint8)
    img[:, start:end] = 10
    return img


# --------------------------------------------------------------- scanline ---

def test_scanline_extracts_the_requested_row():
    img = frame_with_dark_band()
    line, x0 = rcj.scanline(img, (60, 40), 30)
    assert x0 == 30
    assert len(line) == 60
    assert line.dtype == np.int16          # int16 so the derivative cannot overflow


def test_scanline_pads_at_the_left_edge():
    img = frame_with_dark_band()
    line, x0 = rcj.scanline(img, (5, 40), 20)
    assert len(line) == 40                 # full width despite running off frame
    assert x0 == -15


def test_scanline_pads_at_the_right_edge():
    img = frame_with_dark_band(w=120)
    line, x0 = rcj.scanline(img, (115, 40), 20)
    assert len(line) == 40


# ------------------------------------------------------ derivative detect ---

def test_finds_a_dark_line_at_its_centre():
    scan = bright_row_with_dark_band(start=50, end=70).astype(np.int16)
    pt, deriv = rcj.find_line_from_scan(
        scan, 0, "line", {"center_point": (60, 40)}, min_line_width=6)
    assert pt is not None
    x, y = pt
    assert x == pytest.approx(60, abs=2)   # centre of the 50..70 band
    assert y == 40


def test_index_base_offsets_the_result_into_frame_coordinates():
    scan = bright_row_with_dark_band(start=50, end=70).astype(np.int16)
    pt, _ = rcj.find_line_from_scan(
        scan, 100, "line", {"center_point": (0, 40)}, min_line_width=6)
    assert pt[0] == pytest.approx(160, abs=2)


def test_rejects_a_band_narrower_than_min_line_width():
    """A two-pixel dark speck is noise, not the line."""
    scan = bright_row_with_dark_band(start=60, end=62).astype(np.int16)
    pt, _ = rcj.find_line_from_scan(
        scan, 0, "line", {"center_point": (60, 40)}, min_line_width=10)
    assert pt is None


def test_uniform_scan_reports_no_line():
    scan = np.full(120, 200, dtype=np.int16)
    pt, _ = rcj.find_line_from_scan(
        scan, 0, "line", {"center_point": (60, 40)}, min_line_width=6)
    assert pt is None


def test_circle_scan_returns_a_point_on_the_circle():
    n = 64
    angs = np.linspace(-math.pi, math.pi, n, endpoint=False)
    ring = np.full(n, 200, dtype=np.int16)
    ring[20:30] = 10
    details = {"center_point": (60, 40), "radius": 20}
    pt, _ = rcj.find_line_from_scan(ring, angs, "circle", details, min_line_width=4)
    assert pt is not None
    dist = math.hypot(pt[0] - 60, pt[1] - 40)
    assert dist == pytest.approx(20, abs=1.5)


# ------------------------------------------------------------ line angle ---

def test_angle_is_zero_for_a_line_straight_ahead():
    assert rcj.line_angle_from_points((10, 100), (10, 50)) == pytest.approx(0.0, abs=1e-6)


def test_angle_is_signed_by_side():
    right = rcj.line_angle_from_points((10, 100), (60, 50))
    left = rcj.line_angle_from_points((10, 100), (-40, 50))
    assert right > 0 and left < 0
    assert right == pytest.approx(-left, abs=1e-6)


def test_angle_handles_missing_points():
    assert rcj.line_angle_from_points(None, (1, 2)) == 0.0
    assert rcj.line_angle_from_points((1, 2), None) == 0.0


# ---------------------------------------------------------- green tracker ---

GREEN_BOUNDS = {
    "lower": np.array([40, 80, 80], dtype=np.uint8),
    "upper": np.array([80, 255, 255], dtype=np.uint8),
}


def frame_with_green(spots, h=200, w=320, size=30):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cx, cy in spots:
        img[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = (0, 255, 0)
    return img


def test_no_green_reports_no_direction():
    img = np.zeros((200, 320, 3), dtype=np.uint8)
    cents, direction, _ = rcj.track_green_centroids(img, GREEN_BOUNDS)
    assert cents == []
    assert direction is None


def test_single_green_left_of_centre_turns_left():
    _, direction, _ = rcj.track_green_centroids(frame_with_green([(60, 150)]), GREEN_BOUNDS)
    assert direction == "left"


def test_single_green_right_of_centre_turns_right():
    _, direction, _ = rcj.track_green_centroids(frame_with_green([(260, 150)]), GREEN_BOUNDS)
    assert direction == "right"


def test_green_on_both_sides_is_a_uturn():
    _, direction, _ = rcj.track_green_centroids(
        frame_with_green([(60, 150), (260, 150)]), GREEN_BOUNDS)
    assert direction == "uturn"


def test_green_near_the_centre_is_straight():
    """Inside the ±50 px dead band, a marker is not a turn instruction."""
    _, direction, _ = rcj.track_green_centroids(frame_with_green([(160, 150)]), GREEN_BOUNDS)
    assert direction == "straight"


def test_blob_below_area_minimum_is_ignored():
    img = frame_with_green([(60, 150)], size=6)     # 36 px, under the threshold
    cents, direction, _ = rcj.track_green_centroids(img, GREEN_BOUNDS, area_min=200)
    assert cents == []
    assert direction is None
