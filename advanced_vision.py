import cv2
import numpy as np
import math

"""
This module contains the advanced line detection logic ported from the C++
project 'RCJ_2014.cpp'. It uses grayscale analysis, derivatives, and a
sequence of scans to robustly find and track a line.
"""

def scanline(gray_image, center_point, radius):
    """
    Scans a horizontal line on the image and returns the grayscale values.

    Args:
        gray_image (np.array): The grayscale input image.
        center_point (tuple): (x, y) coordinate for the center of the scan.
        radius (int): The radius of the scan line (total length is 2 * radius).

    Returns:
        tuple: (scandata, start_x)
            - scandata (np.array): 1D array of grayscale values.
            - start_x (int): The starting x-coordinate of the scan.
    """
    y = int(center_point[1])
    x_start = int(center_point[0] - radius)
    x_end = int(center_point[0] + radius)

    # Clamp coordinates to be within image boundaries
    img_width = gray_image.shape[1]
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(img_width, x_end)

    # Extract the slice from the image
    scandata = gray_image[y, x_start_clipped:x_end_clipped]

    # Pad the array if the scan went out of bounds, to maintain the expected length
    pad_left = x_start_clipped - x_start
    if pad_left > 0:
        scandata = np.pad(scandata, (pad_left, 0), 'edge')

    pad_right = x_end - x_end_clipped
    if pad_right > 0:
        scandata = np.pad(scandata, (0, pad_right), 'edge')

    return scandata, x_start


def scancircle(gray_image, center_point, radius, look_angle_deg, width_deg):
    """
    Scans a circular arc on the image and returns grayscale values and their angles.

    Args:
        gray_image (np.array): The grayscale input image.
        center_point (tuple): (x, y) coordinate for the center of the scan.
        radius (int): The radius of the circular arc.
        look_angle_deg (float): The center angle of the arc in degrees.
                                0 is to the right (East), -90 is up (North).
        width_deg (float): The total width of the arc in degrees.

    Returns:
        tuple: (scandata, angles)
            - scandata (np.array): 1D array of grayscale values from the arc.
            - angles (np.array): 1D array of the angle (in radians) for each point in scandata.
    """
    img_height, img_width = gray_image.shape
    center_x, center_y = center_point

    # Use a number of points proportional to the radius
    num_points = int(radius * math.pi * (width_deg / 360.0) * 2)
    if num_points < 20: num_points = 20 # Minimum number of points

    # Convert angles to radians and define the arc
    look_angle_rad = math.radians(look_angle_deg)
    width_rad = math.radians(width_deg)

    start_angle = look_angle_rad - width_rad / 2
    end_angle = look_angle_rad + width_rad / 2

    # Generate angles for sampling
    angles = np.linspace(start_angle, end_angle, num_points)

    # Calculate coordinates of points on the arc
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)

    # Clip coordinates to image boundaries
    x_coords = np.clip(x_coords, 0, img_width - 1).astype(int)
    y_coords = np.clip(y_coords, 0, img_height - 1).astype(int)

    # Sample pixel values. This is fast and simple.
    scandata = gray_image[y_coords, x_coords]

    return scandata, angles


def find_line_from_scan(scandata, angles_or_start_x, scan_type, scan_details):
    """
    Finds the line center from a 1D scan array by analyzing its derivative.

    Args:
        scandata (np.array): The 1D array of grayscale values from a scan.
        angles_or_start_x: For 'circle' scan, the array of angles (radians).
                           For 'line' scan, the starting x-coordinate.
        scan_type (str): 'circle' or 'line'.
        scan_details (dict): Dictionary with additional info like center_point, radius.

    Returns:
        tuple: (line_point, derivative)
            - line_point (tuple): The (x, y) coordinate of the detected line center.
            - derivative (np.array): The calculated derivative of the scan.
        Returns (None, None) if no line is found.
    """
    if scandata is None or len(scandata) < 3:
        return None, None

    # 1. Calculate derivative, matching the C++ implementation: der[i] = scan[i-1] - scan[i+1]
    derivative = np.zeros_like(scandata, dtype=np.float32)
    derivative[1:-1] = scandata[:-2].astype(np.float32) - scandata[2:].astype(np.float32)

    # 2. Find the indices of the left (max derivative) and right (min derivative) edges.
    # A transition from white (high value) to black (low value) is a positive peak.
    left_edge_index = np.argmax(derivative)
    right_edge_index = np.argmin(derivative)

    # 3. Basic validation
    # Check if peaks are significant (threshold can be tuned)
    if derivative[left_edge_index] < 20 or derivative[right_edge_index] > -20:
        return None, derivative
    # Check if edges are in the correct order
    if left_edge_index >= right_edge_index:
        return None, derivative

    # 4. Calculate line center in the 1D array
    line_pos_1d_idx = int((left_edge_index + right_edge_index) / 2.0)

    # 5. Convert 1D position back to 2D image coordinates
    line_point = None
    if scan_type == 'line':
        start_x = angles_or_start_x
        line_point = (int(start_x + line_pos_1d_idx), scan_details['center_point'][1])

    elif scan_type == 'circle':
        angles = angles_or_start_x
        center_x, center_y = scan_details['center_point']
        radius = scan_details['radius']

        # Get the angle corresponding to the line center
        angle_rad = angles[line_pos_1d_idx]

        # Convert polar to cartesian coordinates
        line_point_x = center_x + radius * np.cos(angle_rad)
        line_point_y = center_y + radius * np.sin(angle_rad)
        line_point = (int(line_point_x), int(line_point_y))

    return line_point, derivative

def line_angle_from_points(point1, point2):
    """
    Calculates the angle of the line segment between two points.
    Angle is in degrees, with 0 degrees pointing to the right (East).
    """
    if point1 is None or point2 is None:
        return 0.0
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
