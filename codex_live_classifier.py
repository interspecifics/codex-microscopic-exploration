import cv2
import numpy as np
import faiss
import os
import time
from pythonosc import udp_client

from orb_feature_extraction import nfeatures, descriptor_size, fixed_num_descriptors

# Path to the FAISS index file
index_file = 'frames_orb_index.faiss'
# Directory containing the extracted frames
frames_dir = 'extracted_frames'
# Directory containing the test images
test_images_dir = 'imagenes micro'

# Test mode flag
test_mode = True

# Time to sleep parameter
time_to_sleep = 10

# Number of ORB features to extract per image
# nfeatures = 500
# descriptor_size = 32  # ORB descriptor size
# fixed_num_descriptors = 100  # Fixed number of descriptors to use

# Function to compute ORB descriptors
def compute_orb_descriptors(image):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        if descriptors.shape[0] < fixed_num_descriptors:
            # Pad with zeros if not enough descriptors
            padding = np.zeros((fixed_num_descriptors - descriptors.shape[0], descriptor_size))
            descriptors = np.vstack((descriptors, padding))
        else:
            # Take only the first fixed_num_descriptors
            descriptors = descriptors[:fixed_num_descriptors]
    else:
        # If no descriptors found, pad with zeros
        descriptors = np.zeros((fixed_num_descriptors, descriptor_size))
    return descriptors

# Load the FAISS index
index = faiss.read_index(index_file)

# Load image names
image_names = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)


if test_mode:
    test_image_names = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    test_image_index = 0
else:
    # Open the camera
    cap = cv2.VideoCapture(0)

import random

if __name__ == "__main__":
    last_selected_images = []

    while True:
        if test_mode:
            if test_image_index >= len(test_image_names):
                break
            frame = cv2.imread(os.path.join(test_images_dir, test_image_names[test_image_index]))
            test_image_index += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Compute ORB descriptors
        orb_descriptors = compute_orb_descriptors(frame)

        # Flatten ORB descriptors
        orb_flattened = orb_descriptors.flatten()
        combined_features = orb_flattened.astype(np.float32)

        # Search for the top six most similar images in the FAISS index
        distances, indices = index.search(np.array([combined_features]), k=5)
        most_similar_indices = indices[0]
        most_similar_image_names = [image_names[i] for i in most_similar_indices]
        most_similar_image_paths = [os.path.join(frames_dir, name) for name in most_similar_image_names]

        # Filter out images that are in the last 18 selected images
        filtered_image_names = [name for name in most_similar_image_names if name not in last_selected_images[-18:]]

        # Randomly select one image from the remaining images
        if len(filtered_image_names) >= 1:
            selected_image_name = random.choice(filtered_image_names)
        else:
            selected_image_name = filtered_image_names[0] if filtered_image_names else None

        if selected_image_name:
            # Load and display the selected image
            selected_image = cv2.imread(os.path.join(frames_dir, selected_image_name))

            selected_frame = selected_image_name.split('frameproportion_')[1].split('.')[0]
            file_short_name = selected_image_name.split('_frameproportion_')[0]
            
            # Send selected_frame and file_short_name over OSC
            osc_client.send_message("/selected_frame", selected_frame)
            osc_client.send_message("/file_short_name", file_short_name)

            cv2.imshow('Selected Image', selected_image)

            # Update the list of last selected images
            last_selected_images.append(selected_image_name)
            if len(last_selected_images) > 18:
                last_selected_images = last_selected_images[-18:]

        cv2.imshow('Camera Frame' if not test_mode else 'Test Image Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep for one second
        time.sleep(time_to_sleep)

    if not test_mode:
        cap.release()
    cv2.destroyAllWindows()
