import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self,
                 config_path="models/object_detection/yolov3.cfg",
                 weights_path="models/object_detection/yolov3.weights",
                 names_path="models/object_detection/coco.names",
                 confidence_thresh=0.5,
                 nms_thresh=0.4):
        """
        Initializes the ObjectDetector using YOLO.

        Args:
            config_path (str): Path to YOLO configuration file.
            weights_path (str): Path to YOLO pre-trained weights file.
            names_path (str): Path to file containing class names.
            confidence_thresh (float): Minimum probability to filter weak detections.
            nms_thresh (float): Threshold for non-maxima suppression.
        """
        if not all(os.path.exists(p) for p in [config_path, weights_path, names_path]):
            raise FileNotFoundError(
                f"Model files not found. Please ensure '{config_path}', '{weights_path}', and "
                f"'{names_path}' exist. \nYou may need to download them (e.g., YOLOv3 weights and cfg, coco.names)."
            )

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        # Try to set preferable backend and target for better performance if available
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # or DNN_BACKEND_OPENCV
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # or DNN_TARGET_CPU
            print("ObjectDetector: Set preferable backend to CUDA.")
        except Exception as e:
            print(f"ObjectDetector: Could not set preferable backend/target (e.g. CUDA not available or OpenCV not built with CUDA support). Using CPU. Error: {e}")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


        with open(names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        # Get Unconnected Out Layers directly if possible (OpenCV 4.x+)
        try:
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        except AttributeError: # For older OpenCV versions (e.g. 3.x)
             self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.input_width = 416 # YOLO default input size
        self.input_height = 416

    def detect_objects(self, image):
        """
        Detects objects in an image using YOLO.

        Args:
            image (numpy.ndarray): The input image (OpenCV format, BGR).

        Returns:
            list: A list of tuples, where each tuple is
                  (class_id, class_name, confidence, x, y, w, h)
                  representing a detected object. (x,y) is top-left corner.
        """
        if image is None:
            print("Error: Input image is None.")
            return []

        (H, W) = image.shape[:2]

        # Create a blob from the image and perform a forward pass of YOLO
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_width, self.input_height),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_thresh:
                    # Scale bounding box coordinates back relative to image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use center (x, y)-coordinates to derive the top-left corner
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.nms_thresh)

        detected_objects = []
        if len(indices) > 0:
            # Ensure indices is iterable and then flatten if it's a 2D array
            final_indices = indices.flatten() if hasattr(indices, 'flatten') else indices

            for i in final_indices:
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                class_name = self.class_names[class_ids[i]]
                confidence = confidences[i]
                detected_objects.append((class_ids[i], class_name, confidence, x, y, w, h))

        return detected_objects

if __name__ == '__main__':
    # This is a placeholder for a simple test.
    # To run this, you'll need an image and the YOLO model files.
    # Create dummy model files for the test to pass without actual models
    model_dir = "models/object_detection"
    os.makedirs(model_dir, exist_ok=True)
    dummy_files = {
        "yolov3.cfg": "# Dummy YOLO cfg",
        "yolov3.weights": "# Dummy YOLO weights",
        "coco.names": "person\nbicycle\ncar\n# etc."
    }
    for fname, content in dummy_files.items():
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(content)

    print("Attempting to initialize ObjectDetector with dummy model files...")
    try:
        detector = ObjectDetector(
            config_path=os.path.join(model_dir, "yolov3.cfg"),
            weights_path=os.path.join(model_dir, "yolov3.weights"),
            names_path=os.path.join(model_dir, "coco.names")
        )
        print("ObjectDetector initialized.")

        # Create a dummy image
        dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Object Detection Test", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print("Detecting objects in dummy image...")
        # Since dummy models are used, this won't detect actual objects
        # but will test the flow.
        detected_objects = detector.detect_objects(dummy_image)
        print(f"Detected {len(detected_objects)} objects (expected 0 with dummy models).")

        # Example of how to draw boxes (optional, for visualization)
        # for (class_id, class_name, confidence, x, y, w, h) in detected_objects:
        #     color = (0, 255, 0) # Example color
        #     cv2.rectangle(dummy_image, (x, y), (x + w, y + h), color, 2)
        #     text = f"{class_name}: {confidence:.2f}"
        #     cv2.putText(dummy_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, color, 2)
        # cv2.imshow("Detected Objects", dummy_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print("Object detection module implementation structure is complete.")
        print("Note: For actual object detection, valid YOLO model files (cfg, weights, names) are required in 'models/object_detection/'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download YOLOv3 model files (yolov3.cfg, yolov3.weights, coco.names) "
              "and place them in 'models/object_detection/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Clean up dummy files if you want
    # for fname in dummy_files:
    #     fpath = os.path.join(model_dir, fname)
    #     if os.path.exists(fpath) and "# Dummy" in open(fpath).readline():
    #         os.remove(fpath)

```
