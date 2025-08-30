import cv2
import numpy as np
import math

class Vision:
    """
    This class encapsulates the vision processing logic translated from the C++ file RJC_2014.cpp.
    It handles line detection, object tracking, and calculating control errors.
    """
    def __init__(self):
        # Constants translated from the C++ code
        self.FRAME_WIDTH = 320
        self.FRAME_HEIGHT = 240
        self.PI = 3.14159265

        # State variables that were global in the C++ code
        self.scandata = np.zeros(640, dtype=np.uint8)
        self.scan_w = 0
        self.scanpoint = (0, 0)
        self.line_point = (0, 0)
        self.line_points = [(0, 0)] * 7  # Equivalent to vector<Point> line_points(7)
        self.scan_radius = 0
        self.sc_strt = 0
        self.sc_end = 0
        self.last_arc_points = [] # To store points from scancircle

    def test_inimage(self, image, x, y):
        """Checks if a point (x, y) is within the image boundaries."""
        rows, cols = image.shape[:2]
        return 0 <= x < cols and 0 <= y < rows

    def draw_object(self, frame, x, y, color=(0, 255, 0)):
        """Draws crosshairs on the frame at the specified (x, y) location."""
        cv2.circle(frame, (x, y), 10, color, 1)

        # Vertical line
        if y - 15 > 0:
            cv2.line(frame, (x, y), (x, y - 15), color, 1)
        else:
            cv2.line(frame, (x, y), (x, 0), color, 1)
        if y + 15 < self.FRAME_HEIGHT:
            cv2.line(frame, (x, y), (x, y + 15), color, 1)
        else:
            cv2.line(frame, (x, y), (x, self.FRAME_HEIGHT), color, 1)

        # Horizontal line
        if x - 15 > 0:
            cv2.line(frame, (x, y), (x - 15, y), color, 1)
        else:
            cv2.line(frame, (x, y), (0, y), color, 1)
        if x + 15 < self.FRAME_WIDTH:
            cv2.line(frame, (x, y), (x + 15, y), color, 1)
        else:
            cv2.line(frame, (x, y), (self.FRAME_WIDTH, y), color, 1)

    def track_object(self, work_image):
        """
        Finds contours in the thresholded image and returns the center of the largest object.
        Returns: (objectFound, x, y)
        """
        contours, _ = cv2.findContours(work_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_found = False
        x, y = 0, 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # C++ code had a threshold of 50 for the area
            if area > 50:
                moment = cv2.moments(largest_contour)
                if moment['m00'] != 0:
                    x = int(moment['m10'] / moment['m00'])
                    y = int(moment['m01'] / moment['m00'])
                    object_found = True

        if not object_found:
            x = self.FRAME_WIDTH // 2
            y = 15 # Default value from C++

        return object_found, x, y

    def scanline(self, gray_image, mp, line_radius):
        """
        Scans a horizontal line of pixels from the gray_image.
        Equivalent to the C++ scanline function.
        """
        self.scanpoint = mp
        self.scan_radius = line_radius
        self.scan_w = line_radius * 2
        self.sc_strt = mp[0] - line_radius

        row = gray_image[mp[1]]

        scan_values = []
        for i in range(self.scan_w):
            current_col = self.sc_strt + i
            if current_col < 0:
                scan_values.append(row[0])
            elif current_col >= gray_image.shape[1]:
                scan_values.append(row[-1])
            else:
                scan_values.append(row[current_col])

        self.scandata = np.array(scan_values[:self.scan_w], dtype=np.uint8)

    def scancircle(self, gray_image, mp, radius, look_angle, width):
        """
        Scans pixels along a circular arc.
        This is a simplified and more efficient version of the C++ scancircle function.
        """
        self.scanpoint = mp
        self.scan_radius = radius

        # --- Angle Conversion ---
        # The C++ code seems to use a different angle convention.
        # 180 degrees is up, 0 is down.
        # This translates to OpenCV angles where 0 is right, 90 is up, 180 is left, 270 is down.
        # C++ angle -> CV angle: cv_angle = 270 - cpp_angle
        center_angle_cv = (270 - look_angle) % 360
        # The start and end angles for the arc must be integers for cv2.ellipse2Poly.
        angle_start = int(center_angle_cv - width / 2)
        angle_end = int(center_angle_cv + width / 2)

        # Get points for the arc using OpenCV
        axes = (radius, radius)
        # Using a delta of 1 degree for point generation
        self.last_arc_points = cv2.ellipse2Poly(mp, axes, 0, angle_start, angle_end, 1)

        scan_values = []
        for p in self.last_arc_points:
            x, y = p[0], p[1]
            if self.test_inimage(gray_image, x, y):
                scan_values.append(gray_image[y, x])
            elif scan_values: # If out of bounds, use the last valid pixel
                scan_values.append(scan_values[-1])
            else: # If the first point is out of bounds
                scan_values.append(0)

        self.scandata = np.array(scan_values, dtype=np.uint8)
        self.scan_w = len(self.scandata)

    def find_line(self, scan_mode):
        """
        Finds the line center from the scandata array.
        Calculates derivative, finds min/max peaks, and determines line position.
        Returns the detected line point and the derivative for graphing.
        """
        if self.scan_w < 3:
            return self.scanpoint, np.array([])

        # Calculate derivative using convolution, which is more robust than the C++ version
        derivative = np.convolve(self.scandata.astype(float), [-1, 0, 1], 'valid')

        left_edge_index = np.argmax(derivative)
        right_edge_index = np.argmin(derivative)

        line_pos_index = (left_edge_index + right_edge_index) / 2.0

        if scan_mode == 0:  # Linear scan
            # Convert index back to image coordinates
            # +1 to account for 'valid' convolution offset
            x = int(line_pos_index + self.sc_strt + 1)
            y = self.scanpoint[1]
            self.line_point = (x, y)
        elif scan_mode == 1:  # Circular scan
            if self.scan_w > 0 and self.last_arc_points is not None and len(self.last_arc_points) > 0:
                # Map the index from the derivative array back to the original arc points array
                point_index = int(np.clip(line_pos_index + 1, 0, len(self.last_arc_points) - 1))
                self.line_point = tuple(self.last_arc_points[point_index])
            else:
                self.line_point = self.scanpoint

        # Clamp point to be within image bounds
        self.line_point = (
            np.clip(self.line_point[0], 0, self.FRAME_WIDTH - 1),
            np.clip(self.line_point[1], 0, self.FRAME_HEIGHT - 1)
        )

        # Update history of line points
        self.line_points.pop()
        self.line_points.insert(0, self.line_point)

        return self.line_point, derivative

    def line_angle(self):
        """
        Calculates the angle of the line relative to the scan center.
        The C++ code uses atan2(dx, -dy).
        """
        p_x, p_y = self.line_points[0]
        s_x, s_y = self.scanpoint

        # In Python, math.atan2(y, x) is equivalent to C++ atan2(x, y).
        # So we use atan2(-(p_y - s_y), p_x - s_x)
        angle_rad = math.atan2(-(p_y - s_y), p_x - s_x)
        return round(math.degrees(angle_rad))
