import numpy as np
import math

"""Utility functions mirroring the RCJ 2014 line detection logic."""


# Store runtime configuration passed in from ``main.py``.  The helpers in this
# module are mostly stateless, but ``update_config`` keeps compatibility with
# the previous API which expected a mutable configuration object.
CONFIG = {}


def update_config(new_cfg):
    """Update internal configuration with values from ``new_cfg``.

    The line detection helpers do not currently use these settings directly,
    but keeping this function allows ``main.py`` to call it without raising
    errors and makes it easy to wire configuration parameters in the future.
    """

    if not isinstance(new_cfg, dict):
        return
    CONFIG.update(new_cfg)


def scanline(gray_image, center_point, radius):
    """Scan a horizontal line centered at ``center_point``.

    Returns the grayscale values (``scandata``) and the starting x-coordinate.
    """
    y = int(center_point[1])
    x_start = int(center_point[0] - radius)
    x_end = int(center_point[0] + radius)

    img_width = gray_image.shape[1]
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(img_width, x_end)

    scandata = gray_image[y, x_start_clipped:x_end_clipped]

    pad_left = x_start_clipped - x_start
    if pad_left > 0:
        scandata = np.pad(scandata, (pad_left, 0), mode="edge")
    pad_right = x_end - x_end_clipped
    if pad_right > 0:
        scandata = np.pad(scandata, (0, pad_right), mode="edge")

    return scandata, x_start


def scancircle(gray_image, center_point, radius, look_angle_deg, width_deg):
    """Scan an entire circle around ``center_point`` and keep only an arc."""
    img_height, img_width = gray_image.shape
    cx, cy = center_point

    # Sample the full circumference so that array indexing matches the
    # original C++ implementation.
    num_points = max(1, int(2 * math.pi * radius))
    angles = np.linspace(-math.pi, math.pi, num_points, endpoint=False)

    x_coords = cx + radius * np.cos(angles)
    y_coords = cy + radius * np.sin(angles)

    x_coords = np.clip(x_coords, 0, img_width - 1).astype(int)
    y_coords = np.clip(y_coords, 0, img_height - 1).astype(int)
    scandata = gray_image[y_coords, x_coords]

    # Mask everything outside the look window by duplicating border values
    look_angle_rad = math.radians(look_angle_deg)
    width_rad = math.radians(width_deg)
    start = int((look_angle_rad + math.pi - width_rad / 2) / (2 * math.pi) * num_points)
    end = int((look_angle_rad + math.pi + width_rad / 2) / (2 * math.pi) * num_points)

    if start < 0:
        start = 0
    if end > num_points:
        end = num_points
    if start + 1 < num_points:
        scandata[:start] = scandata[start + 1]
    if end - 1 >= 0:
        scandata[end:] = scandata[end - 1]

    return scandata, angles


def find_line_from_scan(
    scandata,
    angles_or_start_x,
    scan_type,
    scan_details,
    min_line_width=0,
):
    """Locate the line center using the raw derivative approach from RCJ.

    Parameters
    ----------
    scandata : array-like
        Samples from ``scanline`` or ``scancircle``.
    angles_or_start_x : array-like or int
        Angles array for circular scans or starting ``x`` for line scans.
    scan_type : {"line", "circle"}
        Type of scan performed.
    scan_details : dict
        Extra details like the scan center and radius.
    min_line_width : int, optional
        Minimum distance between the detected left and right edges.  If the
        measured width is below this value the function returns ``None`` to
        signal that a valid line was not found.
    """

    if scandata is None or len(scandata) < 3:
        return None, None

    derivative = np.zeros_like(scandata, dtype=np.float32)
    derivative[1:-1] = scandata[:-2].astype(np.float32) - scandata[2:].astype(np.float32)

    left_idx = int(np.argmax(derivative))
    right_idx = int(np.argmin(derivative))
    if left_idx == right_idx or abs(right_idx - left_idx) < min_line_width:
        return None, derivative

    line_pos = int((left_idx + right_idx) / 2)

    if scan_type == "line":
        start_x = angles_or_start_x
        y = scan_details["center_point"][1]
        line_point = (int(start_x + line_pos), y)
    elif scan_type == "circle":
        angles = angles_or_start_x
        cx, cy = scan_details["center_point"]
        radius = scan_details["radius"]
        angle = angles[line_pos]
        line_point = (
            int(cx + radius * math.cos(angle)),
            int(cy + radius * math.sin(angle)),
        )
    else:
        line_point = None

    return line_point, derivative


def line_angle_from_points(point1, point2):
    if point1 is None or point2 is None:
        return 0.0
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

