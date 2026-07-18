Dashcam Traffic Violation Monitoring System

An AI-based traffic monitoring system that analyzes dashcam videos to detect vehicles, track their identities, identify road lanes, estimate vehicle distance, and detect potentially dangerous driving behaviors.

The project combines YOLO11, BYTETrack, Kalman Filter, Ultra-Fast Lane Detection (UFLD), monocular distance estimation, and temporal rule-based verification. A Streamlit dashboard is provided for uploading videos, visualizing results, and reviewing detected violations.

Project Overview

Traditional object detection only identifies vehicles independently in each frame. This project extends the detection pipeline by combining vehicle tracking, lane understanding, distance estimation, and temporal reasoning.

The system can:

Detect cars, trucks, buses, and motorcycles from dashcam footage.
Maintain stable vehicle IDs across consecutive frames.
Detect lane boundaries using UFLD.
Estimate the distance between the ego vehicle and surrounding vehicles.
Detect unsafe following distance.
Detect dangerous vehicle cut-ins.
Detect ego-lane departure.
Detect vehicles moving outside or across lane boundaries.
Confirm violations across multiple frames to reduce false alarms.
Save annotated videos, evidence images, and CSV violation logs.
Display processing results through a Streamlit dashboard.
System Pipeline
flowchart LR
    A[Dashcam Video] --> B[Frame Preprocessing]
    B --> C[YOLO11 Vehicle Detection]
    C --> D[BYTETrack + Kalman Filter]
    B --> E[UFLD Lane Detection]
    D --> F[Monocular Distance Estimation]
    E --> G[Lane Geometry Analysis]
    F --> H[Violation Rule Engine]
    G --> H
    H --> I[Temporal Confirmation]
    I --> J[Annotated Video]
    I --> K[Evidence Images]
    I --> L[CSV Violation Report]
    J --> M[Streamlit Dashboard]
    K --> M
    L --> M
Detected Events
Event	Description
UNSAFE_DISTANCE	A tracked vehicle remains closer than the configured safe-distance threshold.
DANGEROUS_CUT_IN	A vehicle moves laterally into the ego lane at a potentially unsafe distance.
EGO_LANE_DEPARTURE	The dashcam vehicle appears to leave its current lane.
OUTSIDE_LANE	Another vehicle crosses or moves outside the detected lane boundaries.

A violation is only confirmed when its conditions remain valid for multiple consecutive detection frames. A cooldown mechanism prevents the same tracked vehicle from being counted repeatedly.

Technical Highlights
Vehicle Detection

YOLO11 is fine-tuned for vehicle detection in dashcam scenes. The detector filters the output to retain relevant road vehicles and supports low-confidence recovery for large or nearby vehicles.

Multi-object Tracking

BYTETrack is combined with a Kalman Filter to:

Maintain a unique ID for each detected vehicle.
Recover vehicles after temporary occlusion.
Reduce duplicate counting.
Provide motion history for cut-in detection.
Lane Detection

UFLD is used to detect lane points efficiently. The lane model was fine-tuned on the CurveLanes dataset to improve performance on curved roads and complex dashcam scenes.

Distance Estimation

Vehicle distance is estimated using monocular camera geometry and bounding-box information.

To reduce unstable distance values, the pipeline combines:

Per-class vehicle height assumptions.
Bounding-box and horizon-based estimation.
Median filtering.
Exponential moving average smoothing.
Maximum frame-to-frame distance-change constraints.
Temporal Violation Confirmation

Instead of reporting a violation from a single frame, suspicious events must persist across consecutive frames before being confirmed.

This approach helps reduce false detections caused by:

Unstable bounding boxes.
Temporary lane-detection errors.
Short occlusions.
Sudden camera movement.
Noisy distance estimates.
Asynchronous Processing

Vehicle detection, lane detection, video writing, and interface rendering are separated into asynchronous processing stages.

The implementation uses small queues that keep the newest frame and discard outdated frames, helping the interface remain responsive during video processing.

Low-light Enhancement

Optional CLAHE and gamma correction are applied to the detector input to improve vehicle visibility in dark or backlit dashcam footage.

Model Results
YOLO11 Vehicle Detection

The included training run contains 50 epochs.

Metric	Final epoch
Precision	80.03%
Recall	59.51%
mAP@0.5	67.81%
mAP@0.5:0.95	38.08%

Training curves and evaluation images are available in:

runs/detect/fast_adas_model/
anh_thong_so/




Lane Detection

Comparison between the original TuSimple model and the CurveLanes fine-tuned model:

Metric	TuSimple model	CurveLanes fine-tuned
Strict Accuracy	7.25%	30.41%
Precision	36.88%	81.89%
Recall	27.32%	81.45%
F1-score	31.38%	81.67%
Validation Loss	2.4799	1.5260




Technology Stack
Category	Technology
Programming language	Python
Deep learning	PyTorch
Vehicle detection	Ultralytics YOLO11
Object tracking	BYTETrack, Kalman Filter
Lane detection	UFLD
Computer vision	OpenCV
Numerical processing	NumPy, SciPy, Pandas
Visualization	Matplotlib
Web interface	Streamlit
Project Structure
detect_car/
├── src/
│   ├── app.py
│   ├── main_pipeline.py
│   ├── lane_detector.py
│   ├── violation_classifier.py
│   ├── byte_tracker.py
│   ├── basetrack.py
│   ├── kalman_filter.py
│   ├── matching.py
│   ├── train_lane_curvelanes.py
│   ├── train_lane_tusimple.py
│   ├── eval_lane_model.py
│   ├── eval_old_vs_new.py
│   └── plot_evaluation_charts.py
│
├── models/
│   ├── curvelanes_scratch_best.json
│   ├── accuracy_comparison_curve.png
│   └── loss_comparison_curve.png
│
├── runs/detect/fast_adas_model/
│   ├── weights/
│   │   └── best.pt
│   ├── results.csv
│   ├── results.png
│   └── confusion_matrix.png
│
├── anh_thong_so/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   └── BoxF1_curve.png
│
├── detect_video.py
├── train_fast.py
├── requirements.txt
└── README.md
Installation
Requirements
Python 3.9 or newer
Git
NVIDIA GPU with CUDA support is recommended
CPU execution is supported but will be slower
Clone the Repository
git clone https://github.com/kuuhaku2012/detect_car.git
cd detect_car
Create a Virtual Environment

Windows CMD:

python -m venv .venv
.venv\Scripts\activate

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

Linux or macOS:

python3 -m venv .venv
source .venv/bin/activate
Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Model Weights

The YOLO vehicle-detection weight is included at:

runs/detect/fast_adas_model/weights/best.pt

The following UFLD weights are not stored directly in the repository because of their file sizes:

models/curvelanes_finetuned_best.pth
models/tusimple_18.pth

Download the required lane weight and place it inside the models/ directory before running the complete system.

Expected structure:

models/
├── curvelanes_finetuned_best.pth
└── tusimple_18.pth
Running the Application

Start the Streamlit dashboard:

streamlit run src/app.py

Open the local address displayed by Streamlit, normally:

http://localhost:8501

From the dashboard, users can:

Select a provided test video or upload a new video.
Start and stop video processing.
View annotated frames.
Monitor FPS, vehicle count, and violation count.
Review recent violation notifications.
Lane Detection CLI

Run the UFLD lane detector directly on a video:

python detect_video.py ^
    --source path/to/video.mp4 ^
    --model models/curvelanes_finetuned_best.pth

Linux or macOS:

python detect_video.py \
    --source path/to/video.mp4 \
    --model models/curvelanes_finetuned_best.pth

Use a webcam:

python detect_video.py --source 0 --model models/curvelanes_finetuned_best.pth

Run without saving the processed video:

python detect_video.py \
    --source path/to/video.mp4 \
    --model models/curvelanes_finetuned_best.pth \
    --no-save
Training
Train the Vehicle Detector

Update the dataset path inside train_fast.py, then run:

python train_fast.py

The best model is saved by Ultralytics under:

runs/detect/fast_adas_model/weights/best.pt
Train the CurveLanes Lane Detector
python src/train_lane_curvelanes.py
Train the TuSimple Lane Detector
python src/train_lane_tusimple.py
Evaluate Lane Models
python src/eval_lane_model.py

Compare two lane-model versions:

python src/eval_old_vs_new.py
Output

After processing, generated files are stored in the output/ directory.

output/
├── output_annotated.avi
├── violation_log.csv
└── violations/
    ├── violation_001.jpg
    ├── violation_002.jpg
    └── ...

The CSV report contains information such as:

Timestamp
Frame ID
Track ID
Violation type
Estimated distance
Evidence image path
Current Limitations
Monocular distance estimation is approximate and depends on camera position and calibration.
Lane detection performance may decrease when lane markings are hidden, faded, or heavily occluded.
Night scenes, rain, motion blur, and extreme camera angles can affect detection quality.
Rule thresholds may need to be recalibrated for different cameras, resolutions, and vehicle speeds.
The system is intended for research, education, and portfolio demonstration rather than legal enforcement.
Future Improvements
Calibrate distance estimation using real-world measured reference points.
Evaluate tracking with MOTA, MOTP, IDF1, and ID-switch metrics.
Add automatic camera calibration.
Export models to ONNX or TensorRT.
Add support for RTSP and live dashcam streams.
Add Docker deployment.
Add automated tests for violation rules.
Build a labeled violation-event evaluation dataset.
Improve night and adverse-weather performance.
Deploy an online demonstration.