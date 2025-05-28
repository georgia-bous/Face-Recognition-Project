# Attendance Marking System using Face Recognition

This repository contains the code for a simple yet effective attendance marking system.  
It uses face recognition technology to identify individuals through the computer’s camera and automatically logs their attendance.

## Features

- 📸 Real-time face detection and recognition
- 🧠 Compares detected faces with a set of known individuals
- 📂 Logs attendance in a CSV file, avoiding duplicate entries
- ⚙️ Modular code structure for easy testing and extension

## Project Structure

├── Images Basic/ # Contains images of known individuals
├── basic.py # Face recognition functionality and testing
├── Attendance.py # Main application for real-time attendance marking

### 🔹 `Images Basic/`

This folder contains images of already identified individuals.  
The live camera feed is compared against these images to recognize the person in front of the camera.

- Each image should be named with the person's name (e.g., `JohnDoe.jpg`).
- These images serve as the reference dataset for the recognition system.

### 🔹 `basic.py`

This script contains the core face recognition logic.  
It includes basic functionality for detecting and comparing faces, along with testing routines to experiment with the recognition library.

Use this file to:
- Test image-to-image face matching
- Understand how the recognition logic works before integrating into the live system

### 🔹 `AttendanceProject.py`

This is the main application script.  
It continuously captures frames from the webcam and performs the following:

1. Detects faces in the video stream
2. Compares detected faces with known faces from `Images Basic/`
3. If a match is found and the person is not already marked present:
   - Logs their name and timestamp into `attendance.csv`

---

## ⚙️ Requirements

Before running the project, ensure you have the required Python packages installed:

```bash
pip install face_recognition opencv-python numpy dlib
```

cmake is also required.


## 📝 Credits & Inspiration

This project was inspired by and based on the following resources:

- 📖 [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).
- 🎥 [FACE RECOGNITION + ATTENDANCE PROJECT](https://www.youtube.com/watch?v=sz25xxF_AVE).

Special thanks to the authors for sharing their work!


