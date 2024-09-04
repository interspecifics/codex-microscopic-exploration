import cv2
import numpy as np
import faiss
import os
import time
import random
import pygame
from pythonosc import udp_client

from orb_feature_extraction import nfeatures, descriptor_size, fixed_num_descriptors

# Constants
INDEX_FILE = 'frames_orb_index.faiss'
FRAMES_DIR = 'extracted_frames'
TEST_IMAGES_DIR = 'imagenes micro'
TEST_MODE = False
TIME_TO_SLEEP = 10
DISPLAY_MODEL_SELECTION = True
STILL_THRESHOLD = 5  # seconds
CAPTURE_INTERVAL = 20  # seconds
OSC_IP = "127.0.0.1"
OSC_PORT = 5005
CANVAS_SIZE = (1920, 1080)
FPS = 30
STILLNESS_THRESHOLD = 500000  # or 500000
MAX_SELECTED_IMAGES = 18

DISPLAY_INDEX = 0

# Function to compute ORB descriptors
def compute_orb_descriptors(image):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        if descriptors.shape[0] < fixed_num_descriptors:
            padding = np.zeros((fixed_num_descriptors - descriptors.shape[0], descriptor_size))
            descriptors = np.vstack((descriptors, padding))
        else:
            descriptors = descriptors[:fixed_num_descriptors]
    else:
        descriptors = np.zeros((fixed_num_descriptors, descriptor_size))
    return descriptors

# Function to resize and center image
def resize_and_center_image(image, canvas_size=CANVAS_SIZE):
    canvas_width, canvas_height = canvas_size
    image_height, image_width = image.shape[:2]
    scale = min(canvas_width / image_width, canvas_height / image_height)
    new_width, new_height = int(image_width * scale), int(image_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    x_offset, y_offset = (canvas_width - new_width) // 2, (canvas_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas

# Function to display message on canvas
def display_message_on_canvas(canvas, message, position=(10, 10), font_scale=1, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Flip the canvas vertically to correct the orientation
    flipped_canvas = cv2.flip(canvas, 0)
    cv2.putText(flipped_canvas, message, position, font, font_scale, color, 2, cv2.LINE_AA)
    # Flip the canvas back to its original orientation
    corrected_canvas = cv2.flip(flipped_canvas, 0)
    return corrected_canvas

# Main function
def main():
    last_selected_images = []
    last_capture_time = time.time()
    last_still_time = None
    prev_frame = None
    still_message_displayed = False
    standby_mode = False

    # Load the FAISS index
    index = faiss.read_index(INDEX_FILE)
    image_names = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1080, 1920), pygame.NOFRAME, display=DISPLAY_INDEX)

    if TEST_MODE:
        test_image_names = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(test_image_names)
        test_image_index = 0
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, FPS)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if TEST_MODE:
            if test_image_index >= len(test_image_names):
                break
            frame = cv2.imread(os.path.join(TEST_IMAGES_DIR, test_image_names[test_image_index]))
            test_image_index += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

            if standby_mode:
                if time.time() - last_capture_time >= CAPTURE_INTERVAL:
                    last_capture_time = time.time()
                    standby_mode = False
                    continue
                else:
                    canvas = resize_and_center_image(frame)
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    canvas = np.rot90(canvas, 2)  # Rotate the canvas 180 degrees
                    canvas = pygame.surfarray.make_surface(canvas)
                    screen.blit(canvas, (0, 0))
                    pygame.display.flip()
                    continue  # Skip the rest of the loop while in standby mode
            else:
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    non_zero_count = np.count_nonzero(diff)
                    if non_zero_count < STILLNESS_THRESHOLD:
                        if last_still_time is None:
                            last_still_time = time.time()
                        elif time.time() - last_still_time >= STILL_THRESHOLD:
                            still_message_displayed = True
                            frame = display_message_on_canvas(frame, "_reading", position=(10, frame.shape[0] - 30))
                            orb_descriptors = compute_orb_descriptors(frame)
                            combined_features = orb_descriptors.flatten().astype(np.float32)
                            distances, indices = index.search(np.array([combined_features]), k=5)
                            most_similar_indices = indices[0]
                            most_similar_image_names = [image_names[i] for i in most_similar_indices]
                            filtered_image_names = [name for name in most_similar_image_names if name not in last_selected_images[-MAX_SELECTED_IMAGES:]]

                            selected_image_name = random.choice(filtered_image_names) if filtered_image_names else None

                            if selected_image_name:
                                selected_image = cv2.imread(os.path.join(FRAMES_DIR, selected_image_name))
                                selected_frame = selected_image_name.split('frameproportion_')[1].split('.')[0]
                                file_short_name = selected_image_name.split('_frameproportion_')[0]
                                osc_client.send_message("/selected_frame", selected_frame)
                                osc_client.send_message("/file_short_name", file_short_name)

                                if DISPLAY_MODEL_SELECTION:
                                    cv2.imshow('Selected Image', selected_image)

                                last_selected_images.append(selected_image_name)
                                if len(last_selected_images) > MAX_SELECTED_IMAGES:
                                    last_selected_images = last_selected_images[-MAX_SELECTED_IMAGES:]

                            standby_mode = True
                            still_message_displayed = False
                            last_still_time = None
                            last_capture_time = time.time()  # Enter standby mode for CAPTURE_INTERVAL seconds
                    else:
                        last_still_time = None
                        standby_mode = False
                    still_message_displayed = non_zero_count < STILLNESS_THRESHOLD
                else:
                    last_still_time = None
                    still_message_displayed = False

                prev_frame = frame.copy()

                # Compute ORB keypoints and descriptors
                orb = cv2.ORB_create(nfeatures=nfeatures)
                keypoints, descriptors = orb.detectAndCompute(frame, None)

                # Draw keypoints on the frame
                overlay = frame.copy()
                frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 0), flags=0)
                cv2.addWeighted(frame_with_keypoints, 0.7, overlay, 0.3, 0, frame_with_keypoints)

                if still_message_displayed:
                    frame_with_keypoints = display_message_on_canvas(frame_with_keypoints, "_reading", position=(10, frame.shape[0] - 30))
                if time.time() - last_capture_time >= CAPTURE_INTERVAL and still_message_displayed:
                    last_capture_time = time.time()
                    orb_descriptors = compute_orb_descriptors(frame)
                    combined_features = orb_descriptors.flatten().astype(np.float32)
                    distances, indices = index.search(np.array([combined_features]), k=5)
                    most_similar_indices = indices[0]
                    most_similar_image_names = [image_names[i] for i in most_similar_indices]
                    filtered_image_names = [name for name in most_similar_image_names if name not in last_selected_images[-MAX_SELECTED_IMAGES:]]

                    selected_image_name = random.choice(filtered_image_names) if filtered_image_names else None

                    if selected_image_name:
                        selected_image = cv2.imread(os.path.join(FRAMES_DIR, selected_image_name))
                        selected_frame = selected_image_name.split('frameproportion_')[1].split('.')[0]
                        file_short_name = selected_image_name.split('_frameproportion_')[0]
                        osc_client.send_message("/selected_frame", selected_frame)
                        osc_client.send_message("/file_short_name", file_short_name)

                        if DISPLAY_MODEL_SELECTION:
                            cv2.imshow('Selected Image', selected_image)

                        last_selected_images.append(selected_image_name)
                        if len(last_selected_images) > MAX_SELECTED_IMAGES:
                            last_selected_images = last_selected_images[-MAX_SELECTED_IMAGES:]

        canvas = resize_and_center_image(frame_with_keypoints)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = np.rot90(canvas, 2)  # Rotate the canvas 180 degrees
        canvas = pygame.surfarray.make_surface(canvas)
        screen.blit(canvas, (0, 0))
        pygame.display.flip()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(TIME_TO_SLEEP if TEST_MODE else 1 / FPS)
    if not TEST_MODE:
        cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()