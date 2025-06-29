import face_recognition
import numpy as np
import json
import os
import uuid

class FaceRecognizer:
    def __init__(self, known_faces_db_path="data/known_faces.json"):
        """
        Initializes the FaceRecognizer.

        Args:
            known_faces_db_path (str): Path to the JSON file storing known faces.
        """
        self.known_faces_db_path = known_faces_db_path
        self.known_face_encodings = []
        self.known_face_ids = []
        self._load_known_faces()
        self.recognition_tolerance = 0.6 # Lower is more strict

    def _load_known_faces(self):
        """Loads known face encodings and IDs from the DB file."""
        if os.path.exists(self.known_faces_db_path):
            try:
                with open(self.known_faces_db_path, 'r') as f:
                    data = json.load(f)
                    self.known_face_encodings = [np.array(enc) for enc in data.get("encodings", [])]
                    self.known_face_ids = data.get("ids", [])
                print(f"Loaded {len(self.known_face_ids)} known faces from {self.known_faces_db_path}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.known_faces_db_path}. Starting with an empty database.")
            except Exception as e:
                print(f"Warning: Could not load known faces from {self.known_faces_db_path}: {e}. Starting with an empty database.")
        else:
            print(f"No known faces database found at {self.known_faces_db_path}. Starting fresh.")

    def save_known_faces(self):
        """Saves current known face encodings and IDs to the DB file."""
        os.makedirs(os.path.dirname(self.known_faces_db_path), exist_ok=True)
        data = {
            "encodings": [enc.tolist() for enc in self.known_face_encodings],
            "ids": self.known_face_ids
        }
        with open(self.known_faces_db_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(self.known_face_ids)} known faces to {self.known_faces_db_path}")

    def register_face(self, image_rgb, face_location, person_id=None):
        """
        Registers a new face or updates an existing one.
        If person_id is provided and exists, it can be used to update the encoding (e.g., average multiple encodings).
        For simplicity now, if person_id is new or None, it registers a new face.

        Args:
            image_rgb (numpy.ndarray): The input image in RGB format.
            face_location (tuple): A tuple (top, right, bottom, left) for the face bounding box.
                                   This is the order face_recognition library expects.
            person_id (str, optional): The ID for this person. If None, a new ID is generated.

        Returns:
            str: The ID of the registered or identified person.
        """
        face_encodings = face_recognition.face_encodings(image_rgb, [face_location])

        if not face_encodings:
            print("Warning: Could not generate encoding for the given face location.")
            return None

        current_encoding = face_encodings[0]

        if person_id and person_id in self.known_face_ids:
            # Potentially update existing entry, e.g., by averaging or replacing.
            # For now, we'll just ensure it's there. If multiple encodings per ID are desired,
            # this logic would need to be more complex.
            try:
                idx = self.known_face_ids.index(person_id)
                self.known_face_encodings[idx] = current_encoding # Replace with the new encoding
                print(f"Updated encoding for existing ID: {person_id}")
                return person_id
            except ValueError:
                # Should not happen if person_id in self.known_face_ids
                pass

        # If person_id is not provided or not found, treat as new or assign new ID
        if not person_id:
            person_id = f"person_{uuid.uuid4().hex[:6]}"

        self.known_face_encodings.append(current_encoding)
        self.known_face_ids.append(person_id)
        print(f"Registered new face with ID: {person_id}")
        self.save_known_faces() # Auto-save after registration
        return person_id

    def recognize_faces(self, image_rgb, detected_face_locations_cv):
        """
        Recognizes faces in an image and registers new ones.
        The face_recognition library expects face locations in (top, right, bottom, left) order.
        OpenCV's DNN detector typically provides (startX, startY, endX, endY).

        Args:
            image_rgb (numpy.ndarray): The input image in RGB format.
                                       (face_recognition library works best with RGB)
            detected_face_locations_cv (list): List of face bounding boxes from OpenCV
                                               in (startX, startY, endX, endY) format.

        Returns:
            list: A list of tuples, where each tuple is (face_id, face_location_cv).
                  face_id can be a known ID or a newly generated one for new faces.
                  face_location_cv is the original OpenCV format box.
        """
        if image_rgb is None:
            print("Error: Input image is None.")
            return []
        if not detected_face_locations_cv:
            return []

        # Convert OpenCV (startX, startY, endX, endY) to face_recognition (top, right, bottom, left)
        face_locations_fr = []
        for (startX, startY, endX, endY) in detected_face_locations_cv:
            # (top, right, bottom, left)
            face_locations_fr.append((startY, endX, endY, startX))

        # Get encodings for all detected faces in the current image
        # This is more efficient than one by one if multiple faces
        current_face_encodings = face_recognition.face_encodings(image_rgb, face_locations_fr)

        recognized_people = []
        new_faces_registered_in_this_run = False

        for i, current_encoding in enumerate(current_face_encodings):
            original_cv_location = detected_face_locations_cv[i]
            face_id = "unknown"

            if len(self.known_face_encodings) > 0:
                # Compare this face with all known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, current_encoding, tolerance=self.recognition_tolerance)
                face_distances = face_recognition.face_distance(self.known_face_encodings, current_encoding)

                best_match_index = -1
                if True in matches:
                    # Find the best match (smallest distance)
                    # Filter distances for True matches only
                    true_match_indices = [j for j, match in enumerate(matches) if match]
                    if true_match_indices: # Should always be true if True in matches
                        # Get distances corresponding to true matches
                        distances_of_true_matches = face_distances[true_match_indices]
                        if len(distances_of_true_matches) > 0:
                            best_match_index_in_true_matches = np.argmin(distances_of_true_matches)
                            best_match_index = true_match_indices[best_match_index_in_true_matches]

                if best_match_index != -1:
                    face_id = self.known_face_ids[best_match_index]
                else:
                    # No match found, register as a new face
                    # Use the original OpenCV location to register, but it needs to be converted
                    # for the register_face function if it were to call face_encodings again.
                    # However, we already have the encoding.
                    new_id = f"person_{uuid.uuid4().hex[:6]}"
                    self.known_face_encodings.append(current_encoding)
                    self.known_face_ids.append(new_id)
                    face_id = new_id
                    new_faces_registered_in_this_run = True
                    print(f"New face detected and registered with ID: {face_id}")
            else:
                # No known faces yet, register this as the first one
                new_id = f"person_{uuid.uuid4().hex[:6]}"
                self.known_face_encodings.append(current_encoding)
                self.known_face_ids.append(new_id)
                face_id = new_id
                new_faces_registered_in_this_run = True
                print(f"First face registered with ID: {face_id}")

            recognized_people.append((face_id, original_cv_location))

        if new_faces_registered_in_this_run:
            self.save_known_faces() # Save if any new faces were added

        return recognized_people

if __name__ == '__main__':
    # This is a placeholder for a simple test.
    # To run this, you'll need an image with faces and the face_detector.py
    # and its model files.
    from face_detector import FaceDetector # Assuming face_detector.py is in the same directory
    import cv2

    print("Face Recognizer Test")
    # Ensure dummy model files for FaceDetector exist for this test to run structurally
    os.makedirs("models/face_detection", exist_ok=True)
    if not os.path.exists("models/face_detection/deploy.prototxt.txt"):
        with open("models/face_detection/deploy.prototxt.txt", "w") as f: f.write("")
    if not os.path.exists("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"):
        with open("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel", "w") as f: f.write("")

    # 0. Cleanup previous DB if exists for a clean test run
    DB_PATH = "data/test_known_faces.json"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # 1. Initialize FaceDetector and FaceRecognizer
    try:
        face_detector = FaceDetector() # Uses default model paths
        face_recognizer = FaceRecognizer(known_faces_db_path=DB_PATH)
        print("FaceDetector and FaceRecognizer initialized.")
    except FileNotFoundError as e:
        print(f"Error initializing detectors/recognizers: {e}")
        print("Please ensure face detection model files are available.")
        exit()
    except Exception as e:
        print(f"An unexpected error during initialization: {e}")
        exit()

    # 2. Create a dummy image (e.g., a 600x400 black image)
    # For a real test, use an image with actual faces.
    # e.g., image_path = "path/to/your/image_with_faces.jpg"
    # test_image = cv2.imread(image_path)
    # For this script, we'll simulate by creating a dummy image.
    # Actual face_recognition requires real faces.

    # Create a dummy image with some colored squares that FaceDetector (with dummy models) won't find.
    # The test for face_recognition itself is harder without actual face images.
    # We'll focus on the structural integrity and DB saving/loading.

    dummy_image_bgr = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(dummy_image_bgr, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Since we can't easily "draw" a face that face_recognition will identify without real image data
    # and dlib models, the following lines will likely not find/encode anything.
    # This test primarily checks the flow and DB saving/loading.

    print("\n--- Simulating First Run (No Known Faces) ---")
    # Convert BGR to RGB for face_recognition library
    dummy_image_rgb = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2RGB)

    # Simulate face detection (will be empty with dummy detector models)
    # To test properly, you'd replace this with actual detections from a real image
    # For this test, let's manually provide some dummy bounding boxes
    # These are (startX, startY, endX, endY)
    simulated_detections = [
        (50, 100, 150, 200), # A dummy box where a face might be
        (200, 150, 300, 250) # Another dummy box
    ]

    # If using real face detector with dummy models:
    # detected_face_locations = face_detector.detect_faces(dummy_image_bgr)
    # print(f"FaceDetector found {len(detected_face_locations)} faces.")

    # We will use simulated_detections.
    # The face_recognition.face_encodings call will likely return empty list for non-face image regions.
    # This means no new faces will actually be "registered" in terms of valid encodings.

    print(f"Simulating {len(simulated_detections)} detected faces.")
    # The following call will try to get encodings. If dummy_image_rgb has no real faces at
    # simulated_detections, encodings will be empty, and thus no registration.
    recognized_people_run1 = face_recognizer.recognize_faces(dummy_image_rgb, simulated_detections)

    if not recognized_people_run1:
        print("No faces were processed by recognizer (likely no valid encodings from dummy data).")
    for person_id, loc in recognized_people_run1:
        print(f"Run 1: Recognized {person_id} at {loc}")
        # (startX, startY, endX, endY) = loc
        # cv2.rectangle(dummy_image_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.putText(dummy_image_bgr, person_id, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"\nKnown faces after run 1: {len(face_recognizer.known_face_ids)}")

    # To properly test registration, one would need to use `register_face` with a real face image patch.
    # For example:
    # if a real face image `face_image.png` exists:
    # try:
    #   img = face_recognition.load_image_file("face_image.png") # This is a real image of a face
    #   # Assuming the whole image is a single, clear face
    #   h, w, _ = img.shape
    #   face_loc_fr = (0, w, h, 0) # top, right, bottom, left
    #   generated_id = face_recognizer.register_face(img, face_loc_fr, person_id="test_person_01")
    #   print(f"Manually registered face with ID: {generated_id}")
    # except Exception as e:
    #   print(f"Could not run manual registration example: {e}")
    #   print("Skipping manual registration. You need a 'face_image.png' for this.")


    print("\n--- Simulating Second Run (After Potential Registrations) ---")
    # Re-initialize recognizer to test loading from DB
    face_recognizer_run2 = FaceRecognizer(known_faces_db_path=DB_PATH)
    print(f"FaceRecognizer (run 2) loaded {len(face_recognizer_run2.known_face_ids)} known faces.")

    recognized_people_run2 = face_recognizer_run2.recognize_faces(dummy_image_rgb, simulated_detections)
    if not recognized_people_run2:
        print("No faces were processed by recognizer in run 2.")
    for person_id, loc in recognized_people_run2:
        print(f"Run 2: Recognized {person_id} at {loc}")

    print(f"\nFace recognition module implementation structure is complete.")
    print(f"Note: For actual face recognition and registration, use real images with detectable faces.")
    print(f"Known faces are stored in: {DB_PATH}")

    # cv2.imshow("Recognized Faces", dummy_image_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Cleanup dummy db
    # if os.path.exists(DB_PATH):
    #     os.remove(DB_PATH)
    # if os.path.exists("models/face_detection/deploy.prototxt.txt"): # remove dummy models
    #    os.remove("models/face_detection/deploy.prototxt.txt")
    # if os.path.exists("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"):
    #    os.remove("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")

```
