import cv2
import os

# Root directory containing the video folders
root_directory = "hap_folder"

# Directory to save extracted images
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

# Function to extract frames
def extract_frames(video_path, output_dir, frame_interval= 30 * 12):
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # print('total frames: ', total_frames)

    while success:
        success, frame = cap.read()
        if frame_count % frame_interval == 0 and success:
            
            frame_fraction = int((frame_count / total_frames) * 100)


            if frame_fraction > 60:
                break

            output_path = os.path.join(output_dir, f'{video_name}_frameproportion_{frame_fraction}.jpg')
            cv2.imwrite(output_path, frame)

        frame_count += 1
    
    cap.release()

# Walk through the directory tree
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.mov'):
            print(os.path.join(root, file))
            video_path = os.path.join(root, file)
            extract_frames(video_path, output_dir)