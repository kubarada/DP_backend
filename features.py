import numpy as np
import cv2

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def calculate_scale_factor(result_det):
    desired_classes = ['feeder_full', 'feeder_empty']
    class_ids = [model.CLASSES.index(cls) for cls in desired_classes if cls in model.CLASSES]

    filtered_result = []
    for i, bbox in enumerate(result_det):
        if i in class_ids:
            filtered_result.append(bbox)

    # Show the result using mmcv
    #if filtered_result:
    #    show_result_pyplot(model, img, filtered_result)

    corners = []

    for bboxes in filtered_result:
        if isinstance(bboxes, np.ndarray):
            for bbox in bboxes:
                if len(bbox) < 5:  # ensure bbox has at least 5 elements [x1, y1, x2, y2, score]
                    continue
                # Upper right corner (x2, y1)
                upper_right = (bbox[2], bbox[1])
                # Lower right corner (x2, y2)
                lower_right = (bbox[2], bbox[3])
                corners.append((upper_right, lower_right))
        elif isinstance(bboxes, list):  # In case the bboxes are stored as a list of tensors
            for bbox_tensor in bboxes:
                bbox = bbox_tensor.cpu().numpy()  # Convert tensor to numpy array if necessary
                # Upper right corner (x2, y1)
                upper_right = (bbox[2], bbox[1])
                # Lower right corner (x2, y2)
                lower_right = (bbox[2], bbox[3])
                corners.append((upper_right, lower_right))

    distances = []

    for upper_right, lower_right in corners:
        # Calculate the Euclidean distance between the upper right and lower right corners
        distance = np.sqrt((upper_right[0] - lower_right[0]) ** 2 + (upper_right[1] - lower_right[1]) ** 2)
        distances.append(distance)

    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    scale_factor = 77/mean_distance # koryto má 77cm

    return scale_factor

def is_daytime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)

    threshold = 100

    if mean_brightness > threshold:
        return True
    else:
        return False