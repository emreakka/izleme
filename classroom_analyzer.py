import cv2
import numpy as np
import os
import argparse
import json

# Import custom modules
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from object_detector import ObjectDetector
from gaze_estimator import GazeEstimator
from attention_analyzer_logic import AttentionAnalyzer
# Import HeadPoseEstimator (assuming it's created and in the same directory)
try:
    from head_pose_estimator import HeadPoseEstimator
    HEAD_POSE_ENABLED = True
except ImportError:
    print("Warning: head_pose_estimator.py not found or HeadPoseEstimator class cannot be imported. Head pose estimation will be disabled.")
    HEAD_POSE_ENABLED = False
    HeadPoseEstimator = None # Define it as None to avoid NameError

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
def draw_face_box(image, face_id, box_cv, attention_text="", head_pose_text=""):
    (startX, startY, endX, endY) = box_cv
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    text_y_offset = 0
    line_height = 15 # Approximate height of a text line

    # Display ID
    id_text = f"ID: {face_id}"
    y_pos_id = startY - 5 - text_y_offset
    if y_pos_id < 10: y_pos_id = startY + line_height
    cv2.putText(image, id_text, (startX, y_pos_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    current_y_offset = y_pos_id # Keep track of the last text y-position

    # Display Attention Text
    if attention_text:
        # text_y_offset += line_height # This logic was a bit off
        y_pos_attn = current_y_offset - line_height # Try to place above
        if y_pos_attn < 10: # If it goes above image, place below previous text
            y_pos_attn = current_y_offset + line_height
        cv2.putText(image, f"Attn: {attention_text}", (startX, y_pos_attn),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        current_y_offset = y_pos_attn # Update for next text

    # Display Head Pose Text
    if head_pose_text:
        y_pos_pose = current_y_offset - line_height # Try to place above previous
        if y_pos_pose < 10 : # If it goes above image or too close to previous
            y_pos_pose = current_y_offset + line_height
            if y_pos_pose > startY + (endY - startY) - 5 : # If it goes into the box, adjust
                 y_pos_pose = startY + line_height * 2 # Fallback inside if too crowded
        cv2.putText(image, f"Pose: {head_pose_text}", (startX, y_pos_pose),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)


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

    if box_w <= 0 or box_h <= 0:
        return None

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
def analyze_classroom_image(image_path, output_filename="analyzed_image.jpg",
                            face_detector_backend='mtcnn', # Added backend option
                            face_conf_thresh=0.9, # Adjusted default for MTCNN
                            obj_conf_thresh=0.5,
                            obj_nms_thresh=0.4,
                            yolo_input_size=416):
    print(f"Analyzing image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    annotated_image = image_bgr.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_dimensions = image_bgr.shape[:2]

    print("Initializing modules...")
    gaze_estimator = None
    attention_analyzer_instance = None
    face_detector_instance = None # Renamed for clarity
    face_recognizer_instance = None # Renamed for clarity
    object_detector_instance = None # Renamed for clarity
    head_pose_estimator_instance = None

    try:
        face_detector_instance = FaceDetector(backend=face_detector_backend, confidence_threshold=face_conf_thresh)
        face_recognizer_instance = FaceRecognizer(known_faces_db_path=KNOWN_FACES_DB)
        object_detector_instance = ObjectDetector(config_path=OBJECT_DETECTOR_CFG,
                                         weights_path=OBJECT_DETECTOR_WEIGHTS,
                                         names_path=OBJECT_DETECTOR_NAMES,
                                         confidence_thresh=obj_conf_thresh,
                                         nms_thresh=obj_nms_thresh)
        object_detector_instance.input_width = yolo_input_size
        object_detector_instance.input_height = yolo_input_size

        if HEAD_POSE_ENABLED and HeadPoseEstimator is not None:
            # Prepare default camera matrix (simple approximation)
            focal_length = image_dimensions[1]
            center = (image_dimensions[1]/2, image_dimensions[0]/2)
            default_camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "double"
                             )
            head_pose_estimator_instance = HeadPoseEstimator(camera_matrix=default_camera_matrix)
            print("HeadPoseEstimator initialized.")
        else:
            print("Head pose estimation disabled or module not found.")

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

        attention_analyzer_instance = AttentionAnalyzer()

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found for a detector. {e}")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not initialize modules: {e}")
        return None, None

    analysis_results = {"image_path": image_path, "people": [], "objects": [], "detected_bodies": []}

    if object_detector_instance:
        print("Detecting objects...")
        # Pass input_size to detect_objects method
        detected_objects_raw = object_detector_instance.detect_objects(image_bgr, input_width=yolo_input_size, input_height=yolo_input_size)
        processed_objects_list = []
        person_detections_from_yolo = []

        for class_id, class_name, confidence, x, y, w, h in detected_objects_raw:
            obj_info = {
                "label": class_name, "confidence": confidence,
                "location_yolo": (x,y,w,h),
                "location_cv": (x, y, x + w, y + h)
            }
            if class_name == "person":
                person_detections_from_yolo.append(obj_info)
                draw_object_box(annotated_image, f"Body", confidence, (x,y,w,h), color=(255,100,0))
            else:
                processed_objects_list.append(obj_info)
                color = (int(class_id * 50 % 255), int(class_id * 100 % 255), int(class_id * 20 % 255))
                draw_object_box(annotated_image, class_name, confidence, (x,y,w,h), color=color)

        analysis_results["objects"] = processed_objects_list
        analysis_results["detected_bodies"] = person_detections_from_yolo # Store person/body detections from YOLO
        print(f"Found {len(processed_objects_list)} non-person object(s) and {len(person_detections_from_yolo)} person/body detection(s).")
    else:
        print("Object detector not initialized. Skipping object detection.")


    people_pipeline_data = []
    if face_detector_instance:
        print("Detecting faces...")
        # detect_faces now returns list of dicts: {'box': (x1,y1,x2,y2), 'landmarks': {}, 'confidence': float}
        face_detections_data = face_detector_instance.detect_faces(image_bgr)
        print(f"Found {len(face_detections_data)} face(s) via face detector.")

        if face_detections_data and face_recognizer_instance:
            # Extract just boxes for face_recognizer
            opencv_format_boxes = [det['box'] for det in face_detections_data]

            recognized_faces_output = face_recognizer_instance.recognize_faces(image_rgb, opencv_format_boxes)

            for i, (face_id, face_loc_cv) in enumerate(recognized_faces_output):
                original_det_data = {}
                # Find the detection dict that corresponds to this face_loc_cv
                # This assumes order is maintained. A more robust match would be by IoU if order is not guaranteed.
                # For now, we assume the recognizer processes boxes in the order they were given from face_detections_data
                if i < len(face_detections_data) and face_detections_data[i]['box'] == face_loc_cv:
                    original_det_data = face_detections_data[i]
                else: # Fallback if matching is tricky, though less ideal
                    original_det_data = {'box': face_loc_cv, 'landmarks': None, 'confidence': None}


                person_data_item = {
                    "id": face_id,
                    "location_cv": face_loc_cv,
                    "landmarks": original_det_data.get('landmarks'),
                    "confidence_face": original_det_data.get('confidence'),
                    "gaze": None,
                    "head_pose": None,
                }

                # Head Pose Estimation
                if HEAD_POSE_ENABLED and head_pose_estimator_instance and person_data_item["landmarks"]:
                    pose = head_pose_estimator_instance.estimate_head_pose_from_landmarks(image_dimensions, person_data_item["landmarks"])
                    person_data_item["head_pose"] = pose

                # Gaze Estimation
                if gaze_estimator:
                    face_crop_bgr = get_face_crop(image_bgr, face_loc_cv, padding_factor=0.3)
                    if face_crop_bgr is not None and face_crop_bgr.shape[0] > 10 and face_crop_bgr.shape[1] > 10:
                        gaze_info = gaze_estimator.estimate_gaze(face_crop_bgr, face_id=face_id)
                        person_data_item["gaze"] = gaze_info
                    else:
                        person_data_item["gaze"] = {"text_direction": "crop_error", "face_detected_by_gaze_tracker": False, "horizontal_ratio": None, "vertical_ratio": None}
                else:
                    person_data_item["gaze"] = {"text_direction": "N/A (Gaze Disabled)", "face_detected_by_gaze_tracker": False, "horizontal_ratio": None, "vertical_ratio": None}

                people_pipeline_data.append(person_data_item)
        elif not face_recognizer_instance:
            print("Face recognizer not initialized. Skipping recognition and further person-specific analysis.")
            # If faces were detected but not recognized, still add them for attention analysis with generic IDs
            if face_detections_data:
                print("Faces detected, but recognizer not available. Using generic IDs for attention analysis.")
                for i, det_data in enumerate(face_detections_data):
                    face_id = f"face_{i}"
                    face_loc_cv = det_data['box']
                    person_data_item = {
                        "id": face_id,
                        "location_cv": face_loc_cv,
                        "landmarks": det_data.get('landmarks'),
                        "confidence_face": det_data.get('confidence'),
                        "gaze": None, "head_pose": None
                    }
                    if HEAD_POSE_ENABLED and head_pose_estimator_instance and person_data_item["landmarks"]:
                        pose = head_pose_estimator_instance.estimate_head_pose_from_landmarks(image_dimensions, person_data_item["landmarks"])
                        person_data_item["head_pose"] = pose
                    if gaze_estimator:
                        face_crop_bgr = get_face_crop(image_bgr, face_loc_cv, padding_factor=0.3)
                        if face_crop_bgr is not None and face_crop_bgr.shape[0] > 10 and face_crop_bgr.shape[1] > 10:
                             person_data_item["gaze"] = gaze_estimator.estimate_gaze(face_crop_bgr, face_id=face_id)
                        else:
                            person_data_item["gaze"] = {"text_direction": "crop_error", "face_detected_by_gaze_tracker": False, "horizontal_ratio": None, "vertical_ratio": None}
                    else:
                        person_data_item["gaze"] = {"text_direction": "N/A (Gaze Disabled)", "face_detected_by_gaze_tracker": False, "horizontal_ratio": None, "vertical_ratio": None}
                    people_pipeline_data.append(person_data_item)

    else:
        print("Face detector not initialized or no faces detected by detector. Skipping face-related processing.")

    print("Analyzing attention...")
    final_people_results = []
    if attention_analyzer_instance and people_pipeline_data:
        for person_data_item in people_pipeline_data:
            attention_target_desc, attention_details = attention_analyzer_instance.determine_attention(
                person_data_item,
                people_pipeline_data,
                analysis_results["objects"] + analysis_results.get("detected_bodies", []),
                image_dimensions
            )
            person_data_item["attention_target"] = attention_target_desc
            person_data_item["attention_details"] = attention_details
            final_people_results.append(person_data_item)

            head_pose_str = ""
            if person_data_item.get("head_pose"):
                hp = person_data_item["head_pose"]
                if hp and isinstance(hp, dict):
                    roll = hp.get('roll', 'N/A')
                    pitch = hp.get('pitch', 'N/A')
                    yaw = hp.get('yaw', 'N/A')
                    try:
                        head_pose_str = f"R:{float(roll):.0f} P:{float(pitch):.0f} Y:{float(yaw):.0f}"
                    except (ValueError, TypeError):
                        head_pose_str = "PoseDataError"
                elif hp is None and person_data_item["landmarks"] is not None :
                     head_pose_str = "PoseEstFail" # Landmarks were there, but PnP failed
                elif person_data_item["landmarks"] is None and HEAD_POSE_ENABLED:
                     head_pose_str = "NoLndmrks"


            draw_face_box(annotated_image, person_data_item["id"], person_data_item["location_cv"],
                          attention_text=attention_target_desc, head_pose_text=head_pose_str)

    elif not people_pipeline_data:
        print("No people data with face detections to analyze attention for.")
    else:
        print("Warning: AttentionAnalyzer not available or no people data. Skipping detailed attention analysis and using basic drawing.")
        for person_data_item in people_pipeline_data:
            gaze_text_for_draw = "N/A"
            if person_data_item.get("gaze") and isinstance(person_data_item["gaze"], dict) and person_data_item["gaze"].get("text_direction"):
                gaze_text_for_draw = person_data_item["gaze"]["text_direction"]

            head_pose_str = ""
            if person_data_item.get("head_pose"):
                hp = person_data_item["head_pose"]
                if hp and isinstance(hp, dict):
                    roll = hp.get('roll', 'N/A')
                    pitch = hp.get('pitch', 'N/A')
                    yaw = hp.get('yaw', 'N/A')
                    try:
                        head_pose_str = f"R:{float(roll):.0f} P:{float(pitch):.0f} Y:{float(yaw):.0f}"
                    except (ValueError, TypeError):
                        head_pose_str = "PoseDataError"
                elif hp is None and person_data_item["landmarks"] is not None :
                     head_pose_str = "PoseEstFail"
                elif person_data_item["landmarks"] is None and HEAD_POSE_ENABLED:
                     head_pose_str = "NoLndmrks"

            draw_face_box(annotated_image, person_data_item["id"], person_data_item["location_cv"],
                          attention_text=f"Gaze: {gaze_text_for_draw}", head_pose_text=head_pose_str)
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
    parser.add_argument("--face_backend", type=str, default="mtcnn", choices=['mtcnn', 'opencv_dnn'], help="Face detection backend ('mtcnn' or 'opencv_dnn'). Default: mtcnn")
    parser.add_argument("--face_conf", type=float, default=0.9, help="Confidence threshold for face detection. Default 0.9 for MTCNN, recommend ~0.5 for OpenCV DNN.")
    parser.add_argument("--obj_conf", type=float, default=0.5, help="Confidence threshold for object detection (0.0-1.0).")
    parser.add_argument("--obj_nms", type=float, default=0.4, help="NMS threshold for object detection (0.0-1.0).")
    parser.add_argument("--yolo_size", type=int, default=416, choices=[320, 416, 608], help="Input size for YOLO (e.g., 320, 416, 608).")

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

    # Create dummy model files if they don't exist
    dummy_model_dirs = {
        "models/face_detection": {
            os.path.basename(FACE_DETECTOR_PROTOTXT): "#Dummy prototxt for OpenCV DNN fallback",
            os.path.basename(FACE_DETECTOR_CAFFEMODEL): "#Dummy caffemodel for OpenCV DNN fallback"
        },
        "models/object_detection": {
            os.path.basename(OBJECT_DETECTOR_CFG): f"#Dummy yolo cfg",
            os.path.basename(OBJECT_DETECTOR_WEIGHTS): "#Dummy yolo weights",
            os.path.basename(OBJECT_DETECTOR_NAMES): "person\nbook\nlaptop\ncell phone\ntvmonitor\nchair\ndesk\nbackpack\nwhiteboard\nblackboard" # Added whiteboard/blackboard
        }
    }
    for dir_path, files_dict in dummy_model_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        for file_name, content in files_dict.items():
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f: f.write(content)
                print(f"Created dummy model file: {file_path} (for structural run only)")

    data_dir = os.path.dirname(KNOWN_FACES_DB)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory: {data_dir}")
    if not os.path.exists(KNOWN_FACES_DB):
        db_dir_to_check = os.path.dirname(KNOWN_FACES_DB)
        if not db_dir_to_check:
            db_dir_to_check = "."
        os.makedirs(db_dir_to_check, exist_ok=True)
        with open(KNOWN_FACES_DB, "w") as f:
            json.dump({"encodings": [], "ids": []}, f)
        print(f"Created empty known faces DB: {KNOWN_FACES_DB}")

    final_json_path_for_summary = os.path.join(OUTPUT_DIR, os.path.splitext(output_fname)[0] + ".json")

    results, returned_json_path_from_func = analyze_classroom_image(
        args.image,
        output_filename=output_fname,
        face_detector_backend=args.face_backend,
        face_conf_thresh=args.face_conf,
        obj_conf_thresh=args.obj_conf,
        obj_nms_thresh=args.obj_nms,
        yolo_input_size=args.yolo_size
    )

    if returned_json_path_from_func:
        final_json_path_for_summary = returned_json_path_from_func

    if results:
        print("\n--- Summary of Analysis ---")
        print(f"Processed Image: {results['image_path']}")

        print(f"Detected Objects (non-person): {len(results['objects'])}")
        # for obj in results['objects']:
        #     print(f"  - {obj['label']} (Conf: {obj['confidence']:.2f}) at {obj['location_yolo']}")

        print(f"Detected Bodies (YOLO 'person'): {len(results.get('detected_bodies', []))}")
        # for body in results.get('detected_bodies', []):
        #    print(f"  - Body (Conf: {body['confidence']:.2f}) at {body['location_yolo']}")

        print(f"Detected & Processed People (from faces): {len(results['people'])}")
        for i, person in enumerate(results['people']):
            gaze_info_text = "N/A"
            if person.get('gaze') and isinstance(person['gaze'], dict) and person['gaze'].get('text_direction'):
                gaze_info_text = person['gaze']['text_direction']

            attention_target_text = person.get('attention_target', "N/A")

            head_pose_display = "N/A"
            if person.get("head_pose") and isinstance(person["head_pose"], dict):
                hp = person["head_pose"]
                try:
                    roll_val = hp.get('roll', 'N/A')
                    pitch_val = hp.get('pitch', 'N/A')
                    yaw_val = hp.get('yaw', 'N/A')

                    roll_str = f"{float(roll_val):.0f}" if isinstance(roll_val, (int, float)) else "N/A"
                    pitch_str = f"{float(pitch_val):.0f}" if isinstance(pitch_val, (int, float)) else "N/A"
                    yaw_str = f"{float(yaw_val):.0f}" if isinstance(yaw_val, (int, float)) else "N/A"
                    head_pose_display = f"R:{roll_str},P:{pitch_str},Y:{yaw_str}"
                except (ValueError, TypeError):
                    head_pose_display = "PoseDataError"
            elif person.get("head_pose") is None and person.get("landmarks") is None and HEAD_POSE_ENABLED:
                 head_pose_display = "NoLndmrks"
            elif person.get("head_pose") is None and person.get("landmarks") is not None and HEAD_POSE_ENABLED: # Explicitly check for landmarks being present but pose failing
                 head_pose_display = "PoseEstFail"


            face_conf_display = "N/A"
            if person.get('confidence_face') is not None: # Check if key exists and is not None
                try:
                    face_conf_display = f"{float(person['confidence_face']):.2f}"
                except (ValueError, TypeError):
                    face_conf_display = "N/A"


            print(f"  Person {i+1}: ID={person['id']}, FaceConf={face_conf_display}, Gaze='{gaze_info_text}', HeadPose='{head_pose_display}', Attention='{attention_target_text}'")

        print("--- End of Summary ---")
        output_image_full_path = os.path.join(OUTPUT_DIR, output_fname)
        print(f"Annotated image is available at: {output_image_full_path}")
        print(f"JSON results are available at: {final_json_path_for_summary}")
    else:
        print("Analysis failed or returned no results.")

```

Now, running the test command again with the cleaned `classroom_analyzer.py`.
