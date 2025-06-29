import cv2
# Attempt to import GazeTracking and handle potential import error
try:
    from gaze_tracking import GazeTracking
except ImportError:
    GazeTracking = None # Placeholder if not installed
    print("Warning: GazeTracking library not found. GazeEstimator will not function.")
    print("Please install it by running: pip install gaze-tracking")

class GazeEstimator:
    def __init__(self):
        """
        Initializes the GazeEstimator.
        """
        if GazeTracking is None:
            self.gaze = None
            print("Error: GazeTracking library is not installed. Gaze estimation will be disabled.")
            return

        self.gaze = GazeTracking()
        if self.gaze is None: # Should not happen if GazeTracking() constructor worked
             print("Error: Failed to initialize GazeTracking object.")


    def estimate_gaze(self, frame_bgr, face_id="unknown"):
        """
        Estimates the gaze direction for a person in the frame.
        The GazeTracking library processes the whole frame to find a face and its gaze.
        If multiple faces are present, it typically focuses on the most prominent one.
        For more precise per-person gaze with multiple faces, each face might need to be
        processed individually (e.g., by cropping).

        Args:
            frame_bgr (numpy.ndarray): The input image/frame in BGR format.
            face_id (str): Identifier for the face being processed (for logging/debugging).

        Returns:
            dict: A dictionary containing gaze information, e.g.,
                  {
                      "text_direction": "center", "is_blinking": False,
                      "is_right": False, "is_left": False, "is_center": True,
                      "horizontal_ratio": 0.5, "vertical_ratio": 0.4,
                      "pupil_left_coords": (x,y) or None,
                      "pupil_right_coords": (x,y) or None
                  }
                  Returns None if gaze cannot be determined or GazeTracking is not available.
        """
        if self.gaze is None:
            # This check is redundant if constructor already prints, but good for safety
            # print("GazeTracking not initialized. Cannot estimate gaze.")
            return None

        if frame_bgr is None:
            print(f"[{face_id}] Error: Input frame is None for gaze estimation.")
            return None

        # The GazeTracking library internally converts to grayscale and finds landmarks.
        # We pass the BGR frame directly.
        self.gaze.refresh(frame_bgr)

        # The GazeTracking library's `refresh` method updates its internal state.
        # It tries to find a face in the frame. If it doesn't, its methods might return None or default values.
        # We should check if pupils were located, as an indicator of successful processing.
        if self.gaze.pupils_located:
            text_direction = ""
            if self.gaze.is_blinking():
                text_direction = "blinking"
            elif self.gaze.is_right():
                text_direction = "looking right"
            elif self.gaze.is_left():
                text_direction = "looking left"
            elif self.gaze.is_center():
                text_direction = "looking center"
            else:
                # This case might occur if pupils are located but direction is ambiguous
                text_direction = "unknown"


            gaze_data = {
                "text_direction": text_direction,
                "is_blinking": self.gaze.is_blinking(),
                "is_right": self.gaze.is_right(),
                "is_left": self.gaze.is_left(),
                "is_center": self.gaze.is_center(),
                "horizontal_ratio": self.gaze.horizontal_ratio(), # Left eye: 0.0-1.0 (left to right)
                "vertical_ratio": self.gaze.vertical_ratio(),     # Left eye: 0.0-1.0 (top to bottom)
                "pupil_left_coords": self.gaze.pupil_left_coords(),
                "pupil_right_coords": self.gaze.pupil_right_coords(),
                "face_detected_by_gaze_tracker": True # GazeTracking found a face
            }
        else:
            # print(f"[{face_id}] Warning: Pupils not located by GazeTracking in the frame.")
            gaze_data = {
                "text_direction": "N/A (pupils not located)",
                "is_blinking": None, "is_right": None, "is_left": None, "is_center": None,
                "horizontal_ratio": None, "vertical_ratio": None,
                "pupil_left_coords": None, "pupil_right_coords": None,
                "face_detected_by_gaze_tracker": False # GazeTracking did not find a face or pupils
            }
            # It's possible GazeTracking's internal face detector didn't find the face
            # we are interested in, especially if the image is complex or the face is small/occluded.

        return gaze_data

if __name__ == '__main__':
    print("Gaze Estimator Test")

    if GazeTracking is None:
        print("GazeTracking library is not installed. Skipping GazeEstimator tests.")
        print("To run these tests, please install it: pip install gaze-tracking")
        # Also ensure dlib is correctly installed, which gaze-tracking depends on.
        # This might require: sudo apt-get install build-essential cmake pkg-config
        #                    sudo apt-get install libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev
        #                    pip install dlib
    else:
        try:
            estimator = GazeEstimator()
            if estimator.gaze is None: # If GazeTracking init failed inside class
                 raise RuntimeError("GazeTracking object within GazeEstimator is None after init.")
            print("GazeEstimator initialized.")

            # Create a dummy image. GazeTracking needs a realistic image to work.
            # A simple black image won't allow landmark detection.
            # For a real test, use an image or video feed with a clear face.
            # e.g., image_path = "path/to/your/face_image.jpg"
            # frame = cv2.imread(image_path)

            # We'll create a blank image. GazeTracking will likely fail to find pupils.
            dummy_frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame_bgr, "Gaze Test - Look Here", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            print("Estimating gaze on dummy frame (expect pupils not located)...")
            gaze_result = estimator.estimate_gaze(dummy_frame_bgr, face_id="test_face_dummy")

            if gaze_result:
                print(f"Gaze Estimation Result: {gaze_result['text_direction']}")
                print(f"  Horizontal Ratio: {gaze_result['horizontal_ratio']}")
                print(f"  Vertical Ratio: {gaze_result['vertical_ratio']}")
                print(f"  Pupils located: {gaze_result['face_detected_by_gaze_tracker']}")
                if gaze_result['pupil_left_coords']:
                    print(f"  Left Pupil: {gaze_result['pupil_left_coords']}")
                if gaze_result['pupil_right_coords']:
                    print(f"  Right Pupil: {gaze_result['pupil_right_coords']}")
            else:
                print("Gaze estimation returned None (likely GazeTracking not initialized).")

            print("\nGaze estimation module structure is complete.")
            print("Note: For actual gaze tracking, use real images/video with clear faces.")
            print("The GazeTracking library processes the whole frame; for multiple faces,")
            print("consider processing cropped face regions if needed, though this example")
            print("uses the full frame for simplicity in this module's first version.")

            # Example of how to use with a webcam (requires a webcam)
            # print("\nAttempting webcam test (press 'q' to quit)...")
            # cap = cv2.VideoCapture(0)
            # if not cap.isOpened():
            #     print("Cannot open webcam.")
            # else:
            #     while True:
            #         ret, frame = cap.read()
            #         if not ret:
            #             print("Can't receive frame (stream end?). Exiting ...")
            #             break
            #
            #         gaze_data_webcam = estimator.estimate_gaze(frame, "webcam_face")
            #
            #         if gaze_data_webcam and gaze_data_webcam['face_detected_by_gaze_tracker']:
            #             text = gaze_data_webcam['text_direction']
            #             cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            #
            #             left_pupil = gaze_data_webcam['pupil_left_coords']
            #             right_pupil = gaze_data_webcam['pupil_right_coords']
            #             if left_pupil:
            #                 cv2.circle(frame, left_pupil, 5, (0,0,255), -1)
            #             if right_pupil:
            #                 cv2.circle(frame, right_pupil, 5, (0,0,255), -1)
            #
            #         cv2.imshow('Webcam Gaze Test', frame)
            #         if cv2.waitKey(1) & 0xFF == ord('q'):
            #             break
            #     cap.release()
            #     cv2.destroyAllWindows()

        except ImportError: # Should be caught by the class-level check already
            print("GazeTracking library not installed. Cannot run GazeEstimator tests.")
        except RuntimeError as e: # Catch init error
            print(f"RuntimeError during GazeEstimator test: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during GazeEstimator test: {e}")
            print("This might be due to dlib model files not being found by GazeTracking or other setup issues.")
            print("GazeTracking downloads 'shape_predictor_68_face_landmarks.dat' on first run if not found.")
            print("Ensure you have internet connection if it's the first time, or place the file manually.")

```
