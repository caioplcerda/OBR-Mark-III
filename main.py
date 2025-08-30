import cv2
import threading
import time
import numpy as np

# Import our custom modules.
# vision.py contains the translated C++ vision logic.
# hardware_control.py manages the motors and GPIO.
# web_stream.py provides the Flask web server and shared state.
from vision import Vision
from hardware_control import HardwareControl
from web_stream import SHARED_STATE, run_stream, log

def main_control_loop():
    """
    The main loop for robot control, vision processing, and state updates.
    This function orchestrates the different modules.
    """

    # --- Initialization ---
    log("Main control loop started. Initializing components.")

    # On a real Raspberry Pi, the camera would be initialized with:
    # cap = cv2.VideoCapture(0)
    # For development without a camera, we create a black image.
    use_real_camera = False
    if use_real_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log("Error: Cannot open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    else:
        log("Running without a camera. Using a dummy black frame.")

    # Initialize our custom modules
    vision = Vision()
    # The hardware_control class needs the config, which is loaded by web_stream
    hardware = HardwareControl(SHARED_STATE['config'])

    # These variables are from the C++ main function and are used to control the scan logic.
    # In a full implementation, these would be managed via the web UI.
    first_scanpoint = (160, 220)  # Initial scan height from scan_height_reg
    first_angle = 0
    scan_radius1_reg = 55
    scan_radius2_reg = 18
    look_angle_reg = 180
    look_width_reg = 160
    base_speed = 30 # Base motor speed

    log("Initialization complete. Waiting for start command from web interface...")
    SHARED_STATE['start_event'].wait() # Pauses until 'start' is clicked in the web UI
    log("Start command received. Beginning main processing loop.")

    try:
        while True:
            # --- 1. CAPTURE FRAME ---
            if use_real_camera:
                ret, frame = cap.read()
                if not ret:
                    log("Error: Can't receive frame (stream end?). Exiting ...")
                    break
            else:
                # Create a dummy frame for testing
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # We can draw a fake line for testing the vision code
                cv2.line(frame, (20, 0), (250, 240), (255, 255, 255), 10)


            # --- 2. VISION PROCESSING (from C++ main loop) ---

            # Create a copy of the frame for drawing overlays
            display_frame = frame.copy()

            # A. Object Tracking (Green Dot)
            # This part can be enabled/disabled or configured via the web UI in a full version.
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_config = SHARED_STATE['config']['hsv_black'] # Using black line config for now
            thresh_image = cv2.inRange(hsv_image, hsv_config['lower'], hsv_config['upper'])
            erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            eroded_image = cv2.erode(thresh_image, erode_element)

            object_found, green_x, green_y = vision.track_object(eroded_image)
            if object_found:
                vision.draw_object(display_frame, green_x, green_y)

            # B. Line Following Scans
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # This sequence of scans is a direct translation of the logic in the C++ main function.
            # It creates a "snake" of scan points that follow the line.

            # Zero Scan: A horizontal line scan to find the line initially.
            vision.scanline(gray_image, (first_scanpoint[0], 220), scan_radius1_reg)
            first_scanpoint, _ = vision.find_line(scan_mode=0)

            # First Scan: A circular scan based on the first found point.
            vision.scancircle(gray_image, vision.line_points[0], scan_radius2_reg, first_angle, look_width_reg)
            vision.find_line(scan_mode=1)
            first_angle = vision.line_angle()
            first_angle = np.clip(first_angle, -45, 45) # Clamp angle as in C++

            # Subsequent scans to refine the path
            vision.scancircle(gray_image, vision.line_points[0], scan_radius2_reg, first_angle, 180)
            vision.find_line(scan_mode=1)
            vision.scancircle(gray_image, vision.line_points[0], scan_radius2_reg, vision.line_angle(), 180)
            vision.find_line(scan_mode=1)
            vision.scancircle(gray_image, vision.line_points[0], scan_radius2_reg, vision.line_angle(), 180)
            _, derivative_data = vision.find_line(scan_mode=1)


            # --- 3. CONTROL ---

            # P_Error in C++ is the horizontal offset of the first scan point from the center.
            p_error = first_scanpoint[0] - (frame.shape[1] / 2)

            # I_Error in C++ is the angle of the line. Our PID controller in hardware_control.py
            # is more traditional and only needs the positional error (p_error).
            i_error_angle = vision.line_angle()

            # Send the error to the hardware controller to set motor speeds.
            # The PID logic is handled within the HardwareControl class.
            hardware.set_motor_speed(base_speed, p_error)


            # --- 4. UPDATE WEB STREAM STATE ---

            # This block updates the shared dictionary that the web server thread reads from.
            with SHARED_STATE['stream_lock']:
                # Draw the detected line path for visualization
                for i in range(1, len(vision.line_points)):
                    p1 = vision.line_points[i-1]
                    p2 = vision.line_points[i]
                    if p1 and p2:
                        cv2.line(display_frame, p1, p2, (255, 0, 0), 2)
                for point in vision.line_points:
                    cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

                s_data = SHARED_STATE['stream_data']
                s_data['last_frame'] = display_frame
                s_data['last_mask'] = eroded_image
                s_data['derivative_scan'] = derivative_data
                s_data['status_data'] = {'P_Error': p_error, 'Angle': i_error_angle}
                s_data['motor_speeds'] = {'left': hardware.last_left_speed, 'right': hardware.last_right_speed}

            # A small delay to yield CPU time
            time.sleep(0.01)

    except KeyboardInterrupt:
        log("Keyboard interrupt received. Shutting down.")
    finally:
        # Cleanup
        log("Stopping hardware and cleaning up GPIO.")
        if use_real_camera:
            cap.release()
        hardware.cleanup()


if __name__ == '__main__':
    # Run the Flask web server in a separate thread.
    # The 'daemon=True' ensures the thread will exit when the main program exits.
    stream_thread = threading.Thread(target=run_stream)
    stream_thread.daemon = True
    stream_thread.start()

    # Run the main robot control loop in the main thread.
    main_control_loop()
