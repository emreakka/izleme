import cv2
import numpy as np
import os
import argparse
import json # For structured output

# Import custom modules
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from object_detector import ObjectDetector
from gaze_estimator import GazeEstimator # Will check for GazeTracking library internally
from attention_analyzer_logic import AttentionAnalyzer # Added Attention Analyzer

# --- Configuration ---
FACE_DETECTOR_PROTOTXT = "models/face_detection/deploy.prototxt.txt"
FACE_DETECTOR_CAFFEMODEL = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"

OBJECT_DETECTOR_CFG = "models/object_detection/yolov3.cfg"
OBJECT_DETECTOR_WEIGHTS = "models/object_detection/yolov3.weights"
OBJECT_DETECTOR_NAMES = "models/object_detection/coco.names"

KNOWN_FACES_DB = "data/known_faces.json"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def draw_face_box(image, face_id, box_cv, attention_text=""):
    (startX, startY, endX, endY) = box_cv
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    id_text = f"ID: {face_id}"

    y_id = startY - 10 if startY - 10 > 10 else startY + 20
    cv2.putText(image, id_text, (startX, y_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if attention_text:
        y_attention = y_id - 20 # Position attention text above ID text
        if y_attention < 10 : # If too close to top, position below ID text
             y_attention = y_id + 15 if y_id + 15 < endY -5 else endY - 5
        cv2.putText(image, f"Attn: {attention_text}", (startX, y_attention),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)


def draw_object_box(image, label, confidence, box_yolo, color=(0, 0, 255)):
    (x, y, w, h) = box_yolo
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_face_crop(image, face_box_cv, padding_factor=0.2):
    (startX, startY, endX, endY) = face_box_cv
    h_img, w_img = image.shape[:2]

    box_w = endX - startX
    box_h = endY - startY
    padding_w = int(box_w * padding_factor)
    padding_h = int(box_h * padding_factor)

    crop_startX = max(0, startX - padding_w)
    crop_startY = max(0, startY - padding_h)
    crop_endX = min(w_img, endX + padding_w)
    crop_endY = min(h_img, endY + padding_h)

    if crop_startX >= crop_endX or crop_startY >= crop_endY:
        cropped_face = image[startY:endY, startX:endX]
    else:
        cropped_face = image[crop_startY:crop_endY, crop_startX:crop_endX]

    if cropped_face.size == 0:
         return None
    return cropped_face

# --- Main Analysis Function ---
def analyze_classroom_image(image_path, output_filename="analyzed_image.jpg"):
    print(f"Analyzing image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    annotated_image = image_bgr.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_dimensions = image_bgr.shape[:2] # (height, width)

    print("Initializing modules...")
    gaze_estimator = None
    attention_analyzer = None
    try:
        face_detector = FaceDetector(prototxt_path=FACE_DETECTOR_PROTOTXT, caffemodel_path=FACE_DETECTOR_CAFFEMODEL)
        face_recognizer = FaceRecognizer(known_faces_db_path=KNOWN_FACES_DB)
        object_detector = ObjectDetector(config_path=OBJECT_DETECTOR_CFG,
                                         weights_path=OBJECT_DETECTOR_WEIGHTS,
                                         names_path=OBJECT_DETECTOR_NAMES)
        try:
            gaze_estimator = GazeEstimator()
            if gaze_estimator.gaze is None:
                print("Warning: GazeEstimator initialized, but GazeTracking component is not available. Gaze estimation will be skipped.")
                gaze_estimator = None
        except ImportError:
            print("Warning: GazeTracking library not found. Gaze estimation will be skipped.")
            gaze_estimator = None
        except Exception as e_gaze:
            print(f"Warning: GazeEstimator failed to initialize ({e_gaze}). Gaze estimation will be skipped.")
            gaze_estimator = None

        attention_analyzer = AttentionAnalyzer()

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found for a detector. {e}")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not initialize modules: {e}")
        return None, None

    analysis_results = {"image_path": image_path, "people": [], "objects": []}

    print("Detecting objects...")
    detected_objects_raw = object_detector.detect_objects(image_bgr)
    processed_objects_list = []
    for class_id, class_name, confidence, x, y, w, h in detected_objects_raw:
        obj_info = {
            "label": class_name, "confidence": confidence,
            "location_yolo": (x,y,w,h),
            "location_cv": (x, y, x + w, y + h)
        }
        processed_objects_list.append(obj_info)
        color = (int(class_id * 20 % 255), int(class_id * 50 % 255), int(class_id * 80 % 255))
        draw_object_box(annotated_image, class_name, confidence, (x,y,w,h), color=color)
    analysis_results["objects"] = processed_objects_list
    print(f"Found {len(processed_objects_list)} object(s).")

    print("Detecting faces...")
    detected_face_locations_cv = face_detector.detect_faces(image_bgr)
    print(f"Found {len(detected_face_locations_cv)} face(s).")

    people_pipeline_data = []
    if detected_face_locations_cv:
        recognized_faces_output = face_recognizer.recognize_faces(image_rgb, detected_face_locations_cv)

        for face_id, face_loc_cv in recognized_faces_output:
            person_data = {"id": face_id, "location_cv": face_loc_cv, "gaze": None, "attention_target": "N/A", "attention_details": {}}

            if gaze_estimator:
                face_crop_bgr = get_face_crop(image_bgr, face_loc_cv, padding_factor=0.3)
                if face_crop_bgr is not None and face_crop_bgr.shape[0] > 10 and face_crop_bgr.shape[1] > 10:
                    gaze_info = gaze_estimator.estimate_gaze(face_crop_bgr, face_id=face_id)
                    person_data["gaze"] = gaze_info
                else:
                    print(f"  Skipping gaze for {face_id} due to invalid crop.")
                    person_data["gaze"] = {"text_direction": "crop_error", "face_detected_by_gaze_tracker": False}
            else:
                person_data["gaze"] = {"text_direction": "N/A (Gaze Disabled)", "face_detected_by_gaze_tracker": False}

            people_pipeline_data.append(person_data)

    print("Analyzing attention...")
    final_people_results = []
    if attention_analyzer: # Check if attention_analyzer was initialized
        for person_data_item in people_pipeline_data:
            attention_target_desc, attention_details = attention_analyzer.determine_attention(
                person_data_item,
                people_pipeline_data,
                analysis_results["objects"],
                image_dimensions
            )
            person_data_item["attention_target"] = attention_target_desc
            person_data_item["attention_details"] = attention_details
            final_people_results.append(person_data_item)

            draw_face_box(annotated_image, person_data_item["id"], person_data_item["location_cv"],
                          attention_text=attention_target_desc)
    else: # If attention_analyzer is None, just use the data without attention analysis
        print("Warning: AttentionAnalyzer not available. Skipping attention analysis.")
        for person_data_item in people_pipeline_data:
            # Draw with basic gaze info if available, or just ID
            gaze_text = "N/A"
            if person_data_item.get("gaze") and person_data_item["gaze"].get("text_direction"):
                gaze_text = person_data_item["gaze"]["text_direction"]
            draw_face_box(annotated_image, person_data_item["id"], person_data_item["location_cv"], attention_text=f"Gaze: {gaze_text}")
            final_people_results.append(person_data_item)


    analysis_results["people"] = final_people_results

    output_image_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Annotated image saved to: {output_image_path}")

    output_json_path = os.path.join(OUTPUT_DIR, os.path.splitext(output_filename)[0] + ".json")

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
        if obj is None: return None
        return obj

    serializable_results = convert_numpy_types(analysis_results)
    with open(output_json_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Analysis results saved to: {output_json_path}")

    return serializable_results, output_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classroom Attention Analyzer")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-o", "--output", default=None, help="Filename for the output annotated image (e.g., result.jpg). Default is <input_name>_analyzed.<ext>")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image path does not exist: {args.image}")
        exit(1)

    if args.output:
        output_fname = args.output
    else:
        base = os.path.basename(args.image)
        name, ext = os.path.splitext(base)
        output_fname = f"{name}_analyzed{ext}" if ext else f"{name}_analyzed.jpg"

    dummy_model_dirs = {
        "models/face_detection": {
            "deploy.prototxt.txt": "#Dummy prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": "#Dummy caffemodel"
        },
        "models/object_detection": {
            "yolov3.cfg": "#Dummy yolo cfg",
            "yolov3.weights": "#Dummy yolo weights",
            "coco.names": "person\nbook\nlaptop\ncell phone\ntvmonitor\nchair\ndesk\nbackpack"
        }
    }
    for dir_path, files_dict in dummy_model_dirs.items(): # Renamed files to files_dict
        os.makedirs(dir_path, exist_ok=True)
        for file_name, content in files_dict.items(): # Use files_dict here
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f: f.write(content)
                print(f"Created dummy model file: {file_path} (for structural run only)")

    data_dir = os.path.dirname(KNOWN_FACES_DB)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory: {data_dir}")
    if not os.path.exists(KNOWN_FACES_DB) and data_dir:
        with open(KNOWN_FACES_DB, "w") as f:
            json.dump({"encodings": [], "ids": []}, f)
        print(f"Created empty known faces DB: {KNOWN_FACES_DB}")

    # Define final_json_path here to ensure it's always defined for the print statement
    final_json_path = os.path.join(OUTPUT_DIR, os.path.splitext(output_fname)[0] + ".json")
    results, returned_json_path = analyze_classroom_image(args.image, output_filename=output_fname)

    if returned_json_path: # If analysis was successful, update final_json_path
        final_json_path = returned_json_path

    if results:
        print("\n--- Summary of Analysis ---")
        print(f"Processed Image: {results['image_path']}")
        print(f"Detected Objects: {len(results['objects'])}")
        print(f"Detected People: {len(results['people'])}")
        for i, person in enumerate(results['people']):
            gaze_info_text = "N/A"
            if person.get('gaze') and person['gaze'].get('text_direction'):
                gaze_info_text = person['gaze']['text_direction']

            attention_target_text = person.get('attention_target', "N/A")

            print(f"  Person {i+1}: ID={person['id']}, Gaze='{gaze_info_text}', Attention='{attention_target_text}'")

        print("--- End of Summary ---")
        output_image_full_path = os.path.join(OUTPUT_DIR, output_fname)
        print(f"Annotated image is available at: {output_image_full_path}")
        print(f"JSON results are available at: {final_json_path}")
    else:
        print("Analysis failed or returned no results.")
        # output_image_full_path might not exist if analysis failed early
        output_image_full_path = os.path.join(OUTPUT_DIR, output_fname)


    # To view the output image (optional, if running in a GUI environment)
    # This part requires a display environment.
    # if os.path.exists(output_image_full_path):
    #     try:
    #         img_display = cv2.imread(output_image_full_path)
    #         if img_display is not None:
    #             # cv2.imshow("Annotated Classroom Image", img_display)
    #             # print("\nDisplaying annotated image. Press any key to close window.")
    #             # cv2.waitKey(0)
    #             # cv2.destroyAllWindows()
    #             # print(f"Annotated image is available at: {output_image_full_path}") # Already printed
    #             pass
    #         else:
    #             print(f"Could not load annotated image for display from {output_image_full_path}")
    #     except Exception as e:
    #         print(f"Could not display image (may require a GUI environment): {e}")
    #         # print(f"Annotated image saved at: {output_image_full_path}") # Already printed
    #         pass

```
The previous syntax error related to markdown backticks should now be resolved as I'm providing the raw code content.

Now, let's run the test command again. This time, I'll also make the dummy image creation more robust within the `run_in_bash_session` by creating a small Python script to generate it, ensuring it uses the same environment.
