import cv2
import numpy as np
import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model
from mmdet.apis import inference_detector, init_detector
from features import calculate_distance, calculate_scale_factor, is_daytime

# Initialize MOT model
mot_config = 'mmtracking/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
mot_model = init_model(mot_config, device='cuda:0')

# Initialize detection model for the first frame
det_config = 'mmdetection/configs/faster_rcnn/DP_faster_rcnn_r50_multiclass_detector.py'
det_checkpoint = 'mmdetection/checkpoints/epoch_10.pth'
det_model = init_detector(det_config, det_checkpoint, device='cuda:0')

# Path to the input video
input_video = 'mmtracking/demo/1.mp4'
imgs = mmcv.VideoReader(input_video)

# Create a temporary directory to store images
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name

# Initialize dictionary to store the trajectories and distances
trajectories = {}
distances_walked = {}

# Read scale factor from an external source or calculate it
scale_factor = None  # Placeholder: Set your scale factor here

prog_bar = mmcv.ProgressBar(len(imgs))
unique_pigs = set()
shape = (0,0)

for i, img in enumerate(imgs):
    shape = img.shape
    result = inference_mot(mot_model, img, frame_id=i)
    img_with_tracks = img.copy()  # Make a copy to draw on

    # Perform detection and calculate scale factor for the first frame
    if i == 0:
        result_det = inference_detector(det_model, img)
        scale_factor = calculate_scale_factor(result_det)

    light_status = is_daytime(img)
    if light_status:
      light_status_text = 'ON'
    else:
      ight_status_text = 'OFF'



    if 'track_bboxes' in result and result['track_bboxes']:
        track_bboxes = result['track_bboxes'][0]  # Assuming there's only one array in 'track_bboxes'
        for track in track_bboxes:
            object_id = int(track[0])  # Object ID
            bbox = track[1:5]  # Bounding box coordinates
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)  # Calculate the centroid of the bounding box
            unique_pigs.add(object_id)

            if object_id not in trajectories:
                trajectories[object_id] = [center]
                distances_walked[object_id] = 0
            else:
                # Calculate the distance between the current center and the last center
                distance = calculate_distance(center, trajectories[object_id][-1])
                # Scale the distance to real-world measurements and add to the total
                if distance > 1.5:
                  distances_walked[object_id] += distance * scale_factor / 100 # na metry
                trajectories[object_id].append(center)

            # Draw bounding box
            cv2.rectangle(img_with_tracks, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(img_with_tracks, f'PIG: {object_id}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw trajectories and add text for distance walked
    for object_id, points in trajectories.items():
        for j in range(1, len(points)):
            cv2.line(img_with_tracks, (int(points[j - 1][0]), int(points[j - 1][1])),
                     (int(points[j][0]), int(points[j][1])), (255, 0, 0), 2)  # Draw line

        # Display the distance traveled for each object
        text = f'Pig {object_id}: {distances_walked[object_id]:.2f} m'
        cv2.putText(img_with_tracks, text, (10, 30 + 30 * object_id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the frame with drawn bounding boxes and trajectories
    cv2.putText(img_with_tracks, f'Lights status: {light_status_text}', (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img_with_tracks, f'Number of Pigs: {len(unique_pigs)}', (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(f'{out_path}/{i:06d}.jpg', img_with_tracks)
    prog_bar.update()

# Generate the output video
output_video = 'output/pig_demo_deepsort_info_demo.mp4'
mmcv.frames2video(out_path, output_video, fps=imgs.fps, fourcc='mp4v')

# Cleanup the temporary directory
out_dir.cleanup()

print(f'Output video saved as {output_video}')