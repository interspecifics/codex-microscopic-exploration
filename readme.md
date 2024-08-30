# Project README

## Project Overview

This project involves using ORB (Oriented FAST and Rotated BRIEF) feature extraction and FAISS (Facebook AI Similarity Search) to find similar images. The `codex_live_classifier.py` script can run in two modes: test mode (using pre-extracted test images) and live mode (using a webcam feed).

### Files and Directories

- **`codex_live_classifier.py`**: Main script to run the image classification.
- **`lista_final_hap.txt`**: List of video files used in the project.
- **`frames_orb_index.faiss`**: FAISS index file containing ORB descriptors for the extracted frames.
- **`extracted_frames`**: Directory containing the extracted frames from videos.

## Setup Instructions

### 1. Create a Virtual Environment

First, create a virtual environment to manage dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate.ps1
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Requirements

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Running the Script

#### Test Mode

To run the script in test mode, ensure the `test_mode` flag is set to `True` in `codex_live_classifier.py`:


```18:18:codex_live_classifier.py
test_mode = True
```


Then, execute the script:

```bash
python codex_live_classifier.py
```

#### Live Mode

To run the script in live mode, set the `test_mode` flag to `False`:


```18:18:codex_live_classifier.py
test_mode = True
```


Ensure your webcam is connected and execute the script:

```bash
python codex_live_classifier.py
```

## Description of Key Components

### `codex_live_classifier.py`

This script performs the following tasks:

1. Loads the FAISS index from `frames_orb_index.faiss`.
2. Sets up an OSC client for communication.
3. In test mode, iterates through test images; in live mode, captures frames from the webcam.
4. Computes ORB descriptors for each frame.
5. Searches for the most similar images in the FAISS index.
6. Displays the selected images and sends information via OSC.

### `lista_final_hap.txt`

This file contains a list of video files used in the project. Each line represents a video file name.

### `frames_orb_index.faiss`

This is the FAISS index file that stores the ORB descriptors for the extracted frames. It is used to quickly find similar images based on their features.

### `extracted_frames`

This directory contains the frames extracted from the videos listed in `lista_final_hap.txt`. These frames are used to build the FAISS index and for comparison during the classification process.
