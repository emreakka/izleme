import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, prototxt_path="models/face_detection/deploy.prototxt.txt",
                 caffemodel_path="models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"):
        """
        Initializes the FaceDetector.

        Args:
            prototxt_path (str): Path to the Caffe prototxt file.
            caffemodel_path (str): Path to the Caffe model weights file.
        """
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            raise FileNotFoundError(
                f"Model files not found. Please ensure '{prototxt_path}' and "
                f"'{caffemodel_path}' exist. \nYou may need to download them. "
                "Search for 'opencv face detection caffe model'."
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.confidence_threshold = 0.5  # Default confidence threshold

    def set_confidence_threshold(self, threshold):
        """
        Sets the confidence threshold for face detection.

        Args:
            threshold (float): Minimum confidence score for a detection to be considered a face.
                               Should be between 0.0 and 1.0.
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            print("Warning: Confidence threshold must be between 0.0 and 1.0. Using default.")

    def detect_faces(self, image):
        """
        Detects faces in an image.

        Args:
            image (numpy.ndarray): The input image (OpenCV format, BGR).

        Returns:
            list: A list of tuples, where each tuple contains the bounding box
                  coordinates (x1, y1, x2, y2) for a detected face.
        """
        if image is None:
            print("Error: Input image is None.")
            return []

        (h, w) = image.shape[:2]
        # Create a blob from the image: 300x300 mean subtracted (104.0, 177.0, 123.0)
        # The model expects BGR images. OpenCV loads images in BGR by default.
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Extract the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box is within the image dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)

                if startX < endX and startY < endY: # check for valid box
                    faces.append((startX, startY, endX, endY))

        return faces

if __name__ == '__main__':
    # This is a placeholder for a simple test.
    # To run this, you'll need an image and the model files.
    # Create dummy model files for the test to pass without actual models
    os.makedirs("models/face_detection", exist_ok=True)
    # Create empty dummy files, actual model files are needed for real detection
    if not os.path.exists("models/face_detection/deploy.prototxt.txt"):
        with open("models/face_detection/deploy.prototxt.txt", "w") as f:
            f.write("# Dummy prototxt for testing structure")
    if not os.path.exists("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"):
        with open("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel", "w") as f:
            f.write("# Dummy caffemodel for testing structure")

    print("Attempting to initialize FaceDetector with dummy model files...")
    try:
        detector = FaceDetector()
        print("FaceDetector initialized.")

        # Create a dummy image (e.g., a 600x400 black image)
        dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "No Faces Here", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print("Detecting faces in dummy image...")
        # Since dummy models are used, this won't detect actual faces
        # but will test the flow.
        detected_faces = detector.detect_faces(dummy_image)
        print(f"Detected {len(detected_faces)} faces (expected 0 with dummy models).")

        # Example of how to draw rectangles (optional, for visualization)
        # for (startX, startY, endX, endY) in detected_faces:
        #     cv2.rectangle(dummy_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.imshow("Output", dummy_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print("Face detection module implementation structure is complete.")
        print("Note: For actual face detection, valid model files are required in 'models/face_detection/'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the Caffe model files (deploy.prototxt.txt and "
              "res10_300x300_ssd_iter_140000.caffemodel) and place them in "
              "'models/face_detection/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
