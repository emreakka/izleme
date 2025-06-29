import numpy as np

class AttentionAnalyzer:
    def __init__(self, teacher_zone=None):
        """
        Initializes the AttentionAnalyzer.

        Args:
            teacher_zone (tuple, optional): A bounding box (x1, y1, x2, y2) representing the teacher's area.
                                            Defaults to None. If None, "looking center" from gaze is primary cue.
        """
        self.teacher_zone = teacher_zone

    def _is_looking_towards_box(self, person_face_box, target_box, gaze_horizontal_ratio, gaze_vertical_ratio, horizontal_threshold=0.3, vertical_threshold=0.3):
        """
        A very simplified heuristic to check if gaze ratios suggest looking towards a target box.
        This is NOT a true geometric gaze projection.
        Assumes person_face_box and target_box are (x1,y1,x2,y2).
        gaze_horizontal_ratio: 0 (left) to 1 (right) from person's perspective.
        gaze_vertical_ratio: 0 (top) to 1 (bottom) from person's perspective.
        """
        if gaze_horizontal_ratio is None or gaze_vertical_ratio is None:
            return False

        person_center_x = (person_face_box[0] + person_face_box[2]) / 2
        person_center_y = (person_face_box[1] + person_face_box[3]) / 2
        target_center_x = (target_box[0] + target_box[2]) / 2
        target_center_y = (target_box[1] + target_box[3]) / 2

        # Horizontal check
        looks_horizontally_towards = False
        if target_center_x < person_center_x and gaze_horizontal_ratio < (0.5 - horizontal_threshold/2): # Target is to person's left
            looks_horizontally_towards = True
        elif target_center_x > person_center_x and gaze_horizontal_ratio > (0.5 + horizontal_threshold/2): # Target is to person's right
            looks_horizontally_towards = True
        elif abs(target_center_x - person_center_x) < (person_face_box[2] - person_face_box[0]): # Target is somewhat aligned horizontally
            if 0.5 - horizontal_threshold < gaze_horizontal_ratio < 0.5 + horizontal_threshold : # Looking somewhat center
                 looks_horizontally_towards = True


        # Vertical check
        looks_vertically_towards = False
        if target_center_y < person_center_y and gaze_vertical_ratio < (0.5 - vertical_threshold/2): # Target is above person
            looks_vertically_towards = True
        elif target_center_y > person_center_y and gaze_vertical_ratio > (0.5 + vertical_threshold/2): # Target is below person
            looks_vertically_towards = True
        elif abs(target_center_y - person_center_y) < (person_face_box[3] - person_face_box[1]): # Target is somewhat aligned vertically
            if 0.5 - vertical_threshold < gaze_vertical_ratio < 0.5 + vertical_threshold: # Looking somewhat center
                looks_vertically_towards = True

        # For now, let's simplify: if horizontal matches, it's a candidate.
        # A more robust check would require both to align well, or use a proper projection.
        return looks_horizontally_towards # and looks_vertically_towards


    def _get_box_center(self, box):
        """ Box can be (x,y,w,h) for yolo or (x1,y1,x2,y2) for cv_locs """
        if len(box) == 4:
            if 'w' in dir(box) or 'width' in dir(box) or len(box) == 4 and type(box[2]) in [int,float] and type(box[3]) in [int,float] and box[2] < box[0]+1000 and box[3] < box[1]+1000 : # Heuristic for (x,y,w,h)
                 # Check if it's (x,y,w,h) by typical yolo output characteristic or if w,h are smallish
                if isinstance(box, tuple) and hasattr(box, '_fields'): # namedtuple
                    x,y,w,h = box
                elif isinstance(box, dict):
                    x,y,w,h = box['x'],box['y'],box['w'],box['h']
                else: # simple tuple
                    x,y,w,h = box

                return (x + w / 2, y + h / 2)
            else: # Assuming (x1,y1,x2,y2)
                x1,y1,x2,y2 = box
                return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None


    def _calculate_proximity_score(self, person_box, target_box, image_diag):
        """
        Calculates a proximity score. Lower is closer.
        person_box: (x1,y1,x2,y2)
        target_box: (x1,y1,x2,y2) or (x,y,w,h)
        image_diag: diagonal of the image for normalization
        """
        person_center = self._get_box_center(person_box)
        target_center = self._get_box_center(target_box)

        if person_center is None or target_center is None:
            return float('inf')

        distance = np.sqrt((person_center[0] - target_center[0])**2 + (person_center[1] - target_center[1])**2)
        return distance / image_diag # Normalize by image diagonal


    def determine_attention(self, current_person_info, all_people_info, detected_objects_info, image_dimensions):
        """
        Determines the attention target for a single person.

        Args:
            current_person_info (dict): Info for the person whose attention is being analyzed.
                                     {'id': str, 'location_cv': (x1,y1,x2,y2), 'gaze': dict}
            all_people_info (list): List of dicts for all people in the scene.
            detected_objects_info (list): List of dicts for detected objects.
                                       {'label': str, 'location_yolo': (x,y,w,h), ...}
            image_dimensions (tuple): (height, width) of the image.

        Returns:
            str: A string describing the attention target.
            dict: Details about the target, e.g. {"type": "peer", "id": "peer_id", "obj_label": None}
        """
        person_id = current_person_info['id']
        person_loc_cv = current_person_info['location_cv'] # (x1,y1,x2,y2)
        gaze_data = current_person_info.get('gaze')

        img_h, img_w = image_dimensions
        img_diag = np.sqrt(img_h**2 + img_w**2)

        if not gaze_data or not gaze_data.get('face_detected_by_gaze_tracker'):
            return "gaze unclear (no face/pupils in crop)", {"type": "unclear"}

        gaze_text_direction = gaze_data.get('text_direction', 'unknown').lower()
        h_ratio = gaze_data.get('horizontal_ratio') # 0 (left) to 1 (right)
        v_ratio = gaze_data.get('vertical_ratio')   # 0 (top) to 1 (bottom)

        # 1. Check for "looking center" / Teacher
        if "center" in gaze_text_direction:
            if self.teacher_zone:
                # TODO: Add logic to check if person_loc_cv + gaze vector points towards teacher_zone
                # This is complex. For now, "center" implies teacher if teacher_zone is generally forward.
                pass # Placeholder for more specific teacher_zone check
            return "teacher/camera", {"type": "teacher"}

        if "blinking" in gaze_text_direction:
            return "blinking", {"type": "self_action"}

        # Candidate targets: (score, type, description, details)
        # Score is proximity based, lower is better. For gaze direction, this is a weak proxy.
        candidate_targets = []

        # 2. Check for Peers
        for other_person in all_people_info:
            if other_person['id'] == person_id:
                continue

            other_loc_cv = other_person['location_cv'] # (x1,y1,x2,y2)
            proximity_score = self._calculate_proximity_score(person_loc_cv, other_loc_cv, img_diag)

            # Simple directional check based on relative positions and gaze category
            person_center_x = (person_loc_cv[0] + person_loc_cv[2]) / 2
            other_center_x = (other_loc_cv[0] + other_loc_cv[2]) / 2

            is_directionally_plausible = False
            if "left" in gaze_text_direction and other_center_x < person_center_x:
                is_directionally_plausible = True
            elif "right" in gaze_text_direction and other_center_x > person_center_x:
                is_directionally_plausible = True
            # Could also use _is_looking_towards_box if ratios are reliable.
            # For now, simpler directional check.

            if is_directionally_plausible and proximity_score < 0.5: # Max normalized distance of 0.5
                 # Adjust score based on how well gaze ratios align (if available)
                if h_ratio is not None:
                    # If looking left (h_ratio low) and target is left, good.
                    # If looking right (h_ratio high) and target is right, good.
                    # This is a very rough penalty/bonus.
                    if (other_center_x < person_center_x and h_ratio > 0.5) or \
                       (other_center_x > person_center_x and h_ratio < 0.5):
                        proximity_score *= 1.5 # Penalize if gaze ratio contradicts position

                candidate_targets.append({
                    "score": proximity_score,
                    "description": f"peer ({other_person['id']})",
                    "details": {"type": "peer", "id": other_person['id']}
                })

        # 3. Check for Objects
        # Relevant classroom objects to prioritize for "downward" or specific gaze
        relevant_objects = ["book", "laptop", "screen", "monitor", "tablet", "cell phone", "paper", "notebook"]

        for obj in detected_objects_info:
            obj_label = obj['label']
            obj_loc_yolo = obj['location_yolo'] # (x,y,w,h)
            # Convert YOLO box to (x1,y1,x2,y2) for consistent proximity calculation
            obj_loc_cv = (obj_loc_yolo[0], obj_loc_yolo[1], obj_loc_yolo[0] + obj_loc_yolo[2], obj_loc_yolo[1] + obj_loc_yolo[3])

            proximity_score = self._calculate_proximity_score(person_loc_cv, obj_loc_cv, img_diag)

            person_center_x = (person_loc_cv[0] + person_loc_cv[2]) / 2
            obj_center_x = (obj_loc_cv[0] + obj_loc_cv[2]) / 2

            is_directionally_plausible = False
            if "left" in gaze_text_direction and obj_center_x < person_center_x:
                is_directionally_plausible = True
            elif "right" in gaze_text_direction and obj_center_x > person_center_x:
                is_directionally_plausible = True

            # Check for downward gaze towards relevant objects
            is_downward_relevant = False
            if v_ratio is not None and v_ratio > 0.65 and obj_label in relevant_objects: # Looking somewhat down
                # Check if object is generally below the person's face
                person_face_bottom_y = person_loc_cv[3]
                obj_top_y = obj_loc_cv[1]
                if obj_top_y > person_face_bottom_y - (person_loc_cv[3]-person_loc_cv[1])*0.2 : # Object starts below a bit above face bottom
                    is_downward_relevant = True
                    is_directionally_plausible = True # Override horizontal for downward relevant
                    proximity_score *= 0.8 # Bonus for downward gaze on relevant object

            if is_directionally_plausible and proximity_score < 0.4: # Max normalized distance
                if h_ratio is not None and not is_downward_relevant : # Adjust score if not downward case
                    if (obj_center_x < person_center_x and h_ratio > 0.6) or \
                       (obj_center_x > person_center_x and h_ratio < 0.4):
                        proximity_score *= 1.5 # Penalize if gaze ratio contradicts position

                candidate_targets.append({
                    "score": proximity_score,
                    "description": f"object ({obj_label})",
                    "details": {"type": "object", "label": obj_label, "obj_loc": obj_loc_cv}
                })

        # 4. Decide final target
        if candidate_targets:
            # Sort by score (lower is better)
            candidate_targets.sort(key=lambda x: x['score'])
            best_target = candidate_targets[0]
            # Add a threshold: if best score is too high, it's still "looking away"
            if best_target['score'] < 0.35: # Max normalized distance for a "confident" target
                return best_target['description'], best_target['details']

        # 5. Default if no specific target found or ratios are ambiguous
        if h_ratio is not None and (h_ratio < 0.3 or h_ratio > 0.7): # Looking significantly to sides
             return "looking away (side)", {"type": "away_side"}
        if v_ratio is not None and v_ratio > 0.7 and not any(c['details']['type'] == 'object' and c['details']['label'] in relevant_objects for c in candidate_targets if c['score'] < 0.5) : # Looking significantly down but not at a relevant object
             return "looking down (desk/lap?)", {"type": "down_general"}

        return "looking away/ambient", {"type": "ambient"}


if __name__ == '__main__':
    analyzer = AttentionAnalyzer()

    # Dummy data for testing
    dummy_person = {
        'id': 'person_1',
        'location_cv': (100, 100, 200, 220), # x1,y1,x2,y2
        'gaze': {
            'text_direction': 'looking right',
            'horizontal_ratio': 0.8, # Far right
            'vertical_ratio': 0.5,   # Center vertically
            'face_detected_by_gaze_tracker': True
        }
    }
    dummy_peer = {
        'id': 'person_2',
        'location_cv': (300, 100, 400, 220), # To the right of person_1
        'gaze': None # Not relevant for this test of person_1's attention
    }
    dummy_book = {
        'label': 'book',
        'location_yolo': (100, 250, 80, 50), # x,y,w,h - below person_1
        'confidence': 0.9
    }
    dummy_laptop = {
        'label': 'laptop',
        'location_yolo': (300,100,100,70), # (x,y,w,h) - to the right of person 1 (same as peer)
        'confidence': 0.8
    }

    all_people = [dummy_person, dummy_peer]
    all_objects = [dummy_book, dummy_laptop]
    image_dims = (600, 800)

    print("--- Test Case 1: Looking Right towards Peer/Laptop ---")
    target_desc, target_details = analyzer.determine_attention(dummy_person, all_people, all_objects, image_dims)
    print(f"Person 1 ({dummy_person['gaze']['text_direction']}) attention: {target_desc}, Details: {target_details}")
    # Expected: peer or laptop, depending on proximity score and logic. Peer is slightly closer by center.

    dummy_person['gaze']['text_direction'] = 'looking left'
    dummy_person['gaze']['horizontal_ratio'] = 0.2 # Far left
    print("\n--- Test Case 2: Looking Left (nothing there) ---")
    target_desc, target_details = analyzer.determine_attention(dummy_person, all_people, all_objects, image_dims)
    print(f"Person 1 ({dummy_person['gaze']['text_direction']}) attention: {target_desc}, Details: {target_details}")
    # Expected: looking away (side)

    dummy_person['gaze']['text_direction'] = 'looking center'
    dummy_person['gaze']['horizontal_ratio'] = 0.5
    print("\n--- Test Case 3: Looking Center ---")
    target_desc, target_details = analyzer.determine_attention(dummy_person, all_people, all_objects, image_dims)
    print(f"Person 1 ({dummy_person['gaze']['text_direction']}) attention: {target_desc}, Details: {target_details}")
    # Expected: teacher/camera

    dummy_person['gaze']['text_direction'] = 'looking down' # Not a direct output of GazeTracking, but we use v_ratio
    dummy_person['gaze']['vertical_ratio'] = 0.8 # Looking down
    dummy_person['gaze']['horizontal_ratio'] = 0.5 # Center horizontally
    print("\n--- Test Case 4: Looking Down towards Book ---")
    target_desc, target_details = analyzer.determine_attention(dummy_person, all_people, all_objects, image_dims)
    print(f"Person 1 (v_ratio={dummy_person['gaze']['vertical_ratio']}) attention: {target_desc}, Details: {target_details}")
    # Expected: object (book)

    dummy_person['gaze']['face_detected_by_gaze_tracker'] = False
    print("\n--- Test Case 5: Gaze Not Detected ---")
    target_desc, target_details = analyzer.determine_attention(dummy_person, all_people, all_objects, image_dims)
    print(f"Person 1 (gaze not detected) attention: {target_desc}, Details: {target_details}")
    # Expected: gaze unclear

```
