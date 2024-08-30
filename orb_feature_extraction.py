import os
import cv2
import numpy as np
import faiss

# Directory containing the extracted frames
frames_dir = 'extracted_frames'

# Number of ORB features to extract per image
nfeatures = 250
descriptor_size = 32  # ORB descriptor size
fixed_num_descriptors = 200  # Fixed number of descriptors to use

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


if __name__ == "__main__":

    # List to store ORB descriptors
    orb_descriptors_list = []

    # Process each image in the frames directory
    for filename in os.listdir(frames_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(frames_dir, filename)
            print(f"Processing: {filename}")
            image = cv2.imread(image_path)
            
            # Compute ORB descriptors
            orb_descriptors = compute_orb_descriptors(image)
            
            if orb_descriptors is not None:
                # Flatten ORB descriptors
                orb_flattened = orb_descriptors.flatten()
                orb_descriptors_list.append(orb_flattened)

    # Combine all ORB descriptors into a single NumPy array
    orb_descriptors_array = np.vstack(orb_descriptors_list).astype(np.float32)

    # Create a FAISS index
    d = orb_descriptors_array.shape[1]
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    index.add(orb_descriptors_array)

    # Save the FAISS index to a file
    faiss.write_index(index, 'frames_orb_index.faiss')

    print(f"FAISS index saved to frames_orb_index.faiss with {fixed_num_descriptors} ORB features per image")
