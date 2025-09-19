# Pose Detection and Bicep Curl Counter with MediaPipe
## Project Metadata

| | |
| :--- | :--- |
| **Developer** | Dibyendu Karmahapatra |
| **Role** | AI/ML and Data Science Engineer |
| **Email** | dibyendukarmahapatra@gmail.com |
| **GitHub** | [https://github.com/Dibyendu17122003/](https://github.com/Dibyendu17122003/) |
| **LinkedIn** | [https://www.linkedin.com/in/dibyendu-karmahapatra-17d2004/](https://www.linkedin.com/in/dibyendu-karmahapatra-17d2004/) |
| **Status** | Solo Developer |
| **Technology Stack** | Python, OpenCV, MediaPipe, NumPy |

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8%2B-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A comprehensive computer vision application that uses pose estimation to track and count bicep curl exercises in real-time. This system leverages Google's MediaPipe framework for accurate human pose detection and OpenCV for video processing and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technical Architecture](#technical-architecture)
4. [Installation Guide](#installation-guide)
5. [Usage Instructions](#usage-instructions)
6. [How It Works](#how-it-works)
7. [Customization Options](#customization-options)
8. [Extending the Application](#extending-the-application)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)
11. [Future Enhancements](#future-enhancements)
12. [References](#references)
13. [License](#license)

## Overview

This project implements a real-time human pose detection system that specifically focuses on tracking and counting bicep curl exercises. By utilizing machine learning-powered pose estimation, the application can accurately identify key body landmarks, calculate joint angles, and determine when a complete repetition has been performed.

The system is built with Python and leverages several powerful computer vision libraries:
- **MediaPipe**: Google's framework for perceptual AI tasks
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing library for mathematical operations

## Features

- **Real-time Pose Detection**: Processes webcam feed at high frame rates with low latency
- **33-Point Body Landmark Tracking**: Identifies key joints and body parts with high precision
- **Angle Calculation**: Computes elbow joint angles using trigonometric functions
- **Intelligent Rep Counting**: Uses state-based logic to accurately count repetitions
- **Visual Feedback**: Displays real-time information including:
  - Current joint angle
  - Repetition count
  - Exercise stage (up/down)
  - Body landmark connections
- **Customizable Parameters**: Adjustable detection confidence thresholds and angle parameters
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux systems

## Technical Architecture

### System Components

1. **Video Capture Module**: Handles webcam input using OpenCV's VideoCapture
2. **Pose Estimation Engine**: MediaPipe Pose processes frames to detect body landmarks
3. **Angle Calculation Module**: Uses vector mathematics to compute joint angles
4. **Repetition Counter**: State machine that tracks exercise progress
5. **Visualization Engine**: Renders landmarks, angles, and UI elements

### Data Flow

```
Webcam Input → Frame Capture → Color Conversion → Pose Detection → 
Landmark Extraction → Angle Calculation → Rep Counting → 
Visualization → Display Output
```

### Key Landmarks Used

The system focuses on these specific body landmarks for bicep curl tracking:
- LEFT_SHOULDER (Landmark #11)
- LEFT_ELBOW (Landmark #13)
- LEFT_WRIST (Landmark #15)

## Installation Guide

### Prerequisites

- Python 3.7 or higher
- Webcam connected to your system
- Internet connection (for initial package downloads)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dibyendu17122003/pose-detection-curl-counter.git
   cd pose-detection-curl-counter
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv pose-env
   source pose-env/bin/activate  # On Windows: pose-env\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

4. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, numpy; print('All packages installed successfully')"
   ```

### Alternative Installation Methods

**Using conda**:
```bash
conda create -n pose-env python=3.8
conda activate pose-env
conda install -c conda-forge opencv
pip install mediapipe numpy
```

**Using requirements.txt**:
```bash
pip install -r requirements.txt
```

## Usage Instructions

### Basic Usage

1. **Run the application**:
   ```bash
   python pose_detection.py
   ```

2. **Position yourself in front of the camera**:
   - Ensure your entire upper body is visible
   - Stand approximately 4-6 feet from the webcam
   - Make sure you have adequate lighting

3. **Perform bicep curls**:
   - The system will automatically detect your pose
   - Your elbow angle will be displayed in real-time
   - Repetitions will be counted as you complete each curl

4. **Exit the application**:
   - Press the 'q' key to quit

### Advanced Usage

The application can be customized with command line arguments:

```bash
python pose_detection.py --cam_index 1 --min_detection_confidence 0.7 --min_tracking_confidence 0.7
```

Available parameters:
- `--cam_index`: Camera device index (default: 0)
- `--min_detection_confidence`: Minimum confidence value for detection (default: 0.5)
- `--min_tracking_confidence`: Minimum confidence value for tracking (default: 0.5)

## How It Works

### Pose Detection Pipeline

1. **Frame Capture**: The webcam feed is captured using OpenCV's VideoCapture
2. **Color Conversion**: Frames are converted from BGR to RGB format for MediaPipe processing
3. **Pose Processing**: MediaPipe Pose analyzes the frame to detect body landmarks
4. **Landmark Extraction**: Key points (shoulder, elbow, wrist) are extracted from results
5. **Angle Calculation**: The elbow angle is computed using vector mathematics
6. **Rep Counting Logic**: A state machine tracks movement patterns to count repetitions
7. **Visualization**: Results are annotated on the output frame
8. **Display**: The processed frame is shown to the user

### Angle Calculation

The elbow angle is calculated using the following trigonometric approach:

```python
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    
    Parameters:
    a (list): First point [x, y] (shoulder)
    b (list): Mid point [x, y] (elbow)
    c (list): End point [x, y] (wrist)
    
    Returns:
    float: Angle in degrees
    """
    a = np.array(a)  # Convert to numpy array for vector operations
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors from the mid point to the first and end points
    ba = a - b
    bc = c - b
    
    # Calculate the cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clamp value to avoid numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle
```

### Repetition Counting Logic

The application uses a state machine to accurately count repetitions:

```python
# State variables
counter = 0
stage = None

# For each frame with detected landmarks:
angle = calculate_angle(shoulder, elbow, wrist)

if angle > 160:  # Arm is extended
    stage = "down"
if angle < 30 and stage == "down":  # Arm is curled and was previously extended
    stage = "up"
    counter += 1
    print(f"Rep count: {counter}")
```

## Customization Options

### Adjusting Angle Thresholds

You can modify the angle thresholds in the rep counting logic to match your exercise form:

```python
# Original values
if angle > 160:
    stage = "down"
if angle < 30 and stage == "down":

# Modified values for different form
if angle > 170:  # Require fuller extension
    stage = "down"
if angle < 45 and stage == "down":  # Allow less flexion
```

### Changing Visual Styles

Customize the appearance of landmarks and connections:

```python
# Custom drawing specifications
landmark_drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Green landmarks
    thickness=3, 
    circle_radius=5
)

connection_drawing_spec = mp_drawing.DrawingSpec(
    color=(255, 0, 0),  # Blue connections
    thickness=4, 
    circle_radius=2
)

# Use custom specs when drawing
mp_drawing.draw_landmarks(
    image, 
    results.pose_landmarks, 
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec, 
    connection_drawing_spec
)
```

### Tracking Different Body Parts

The system can be adapted to track other exercises by modifying the landmark indices:

```python
# For shoulder press (tracking both arms)
left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
```

## Extending the Application

### Adding Exercise Recognition

You can extend the system to recognize multiple exercises:

```python
# Exercise recognition class
class ExerciseRecognizer:
    def __init__(self):
        self.current_exercise = None
        self.exercise_models = {}
        
    def recognize_exercise(self, landmarks):
        # Implement exercise recognition logic
        # Based on landmark positions, determine the exercise being performed
        pass
        
    def track_exercise(self, landmarks):
        exercise = self.recognize_exercise(landmarks)
        if exercise != self.current_exercise:
            self.current_exercise = exercise
            self.reset_counter()
        
        if self.current_exercise == "bicep_curl":
            return self.track_bicep_curl(landmarks)
        elif self.current_exercise == "shoulder_press":
            return self.track_shoulder_press(landmarks)
        # Add more exercises as needed
```

### Adding Repetition History

Track and display historical data:

```python
# Rep history tracking
rep_history = {
    'timestamps': [],
    'angles': [],
    'completion_times': []
}

def update_rep_history(counter, angle):
    if counter > len(rep_history['timestamps']):
        rep_history['timestamps'].append(time.time())
        rep_history['angles'].append(angle)
        if len(rep_history['timestamps']) > 1:
            completion_time = rep_history['timestamps'][-1] - rep_history['timestamps'][-2]
            rep_history['completion_times'].append(completion_time)
```

### Adding Form Feedback

Provide real-time feedback on exercise form:

```python
def check_form(angle, previous_angles):
    # Check for consistent range of motion
    if max(previous_angles) - min(previous_angles) < 50:
        return "Increase your range of motion"
    
    # Check for jerky movements (rapid angle changes)
    angle_changes = np.diff(previous_angles)
    if max(np.abs(angle_changes)) > 30:
        return "Slow down your movements"
    
    return "Good form!"
```

## Troubleshooting

### Common Issues and Solutions

1. **No pose detection**:
   - Ensure you have adequate lighting
   - Check that your entire upper body is visible
   - Try increasing the detection confidence threshold

2. **Poor performance/low FPS**:
   - Reduce the resolution of your webcam
   - Close other applications using the camera
   - Use a more powerful computer if possible

3. **Inaccurate angle calculations**:
   - Ensure you're facing the camera directly
   - Maintain a consistent distance from the camera
   - Avoid wearing loose clothing that might obscure joint positions

4. **Webcam not found**:
   - Check if the correct camera index is being used
   - Verify that no other application is using the camera

### Debug Mode

Enable debug mode for more detailed output:

```python
# Add debug parameter
debug = True

# In the processing loop
if debug and landmarks:
    print(f"Landmarks detected: {len(landmarks)}")
    print(f"Shoulder: {shoulder}, Elbow: {elbow}, Wrist: {wrist}")
    print(f"Angle: {angle}, Stage: {stage}, Counter: {counter}")
```

## Performance Considerations

### Optimization Techniques

1. **Frame resizing**:
   ```python
   # Resize frame for faster processing
   frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
   ```

2. **Processing every nth frame**:
   ```python
   frame_count = 0
   process_every_n_frames = 2  # Process every 2nd frame
   
   while cap.isOpened():
       ret, frame = cap.read()
       frame_count += 1
       
       if frame_count % process_every_n_frames == 0:
           # Process frame
           # ...
   ```

3. **Selective landmark processing**:
   ```python
   # Only process upper body landmarks for upper body exercises
   with mp_pose.Pose(
       static_image_mode=False,
       model_complexity=1,  # Use lighter model
       enable_segmentation=False,
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5,
       upper_body_only=True  # Only detect upper body
   ) as pose:
   ```

### Hardware Recommendations

- **Minimum**: Dual-core CPU, 4GB RAM, integrated graphics
- **Recommended**: Quad-core CPU, 8GB RAM, dedicated GPU
- **Optimal**: 6+ core CPU, 16GB RAM, NVIDIA GPU with CUDA support

## Future Enhancements

### Planned Features

1. **Multi-person tracking**: Track multiple people simultaneously
2. **Exercise library**: Support for various exercises (squats, lunges, etc.)
3. **Form analysis**: Detailed feedback on exercise technique
4. **Cloud synchronization**: Save workout data to the cloud
5. **Mobile application**: iOS and Android versions
6. **AI personal trainer**: Personalized workout recommendations
7. **Social features**: Share progress with friends
8. **Integration with fitness apps**: Export data to popular fitness platforms

### Technical Improvements

1. **3D pose estimation**: Add depth information for more accurate tracking
2. **Temporal filtering**: Smooth landmark detection across frames
3. **Custom ML models**: Train specialized models for specific exercises
4. **Edge deployment**: Optimize for resource-constrained devices
5. **Web version**: Browser-based implementation using TensorFlow.js

## References

### Academic Papers

1. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). BlazePose: On-device Real-time Body Pose tracking. arXiv preprint arXiv:2006.10204.

2. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv preprint arXiv:1906.08172.

### Documentation

- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)

### Related Projects

- [TensorFlow.js Pose Detection](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

- Google MediaPipe team for the excellent pose estimation solution
- OpenCV community for computer vision tools
- Contributors to the NumPy library for numerical computing

---

For questions, support, or contributions, please open an issue on the GitHub repository or contact the development team at dibyendukarmahapatra@gmail.com.
