# Deep-Fake-detection
# Deepfake Video Detection System

## 📌 Project Overview
The **Deepfake Video Detection System** is an end-to-end machine learning project designed to identify whether a video is real or AI-generated (deepfake).  
The system leverages **EfficientNet** for spatial feature extraction and **LSTM** for temporal sequence modeling, with preprocessing steps that include frame extraction and face cropping.  
The trained model is integrated into a **Flask web application** for real-time video upload and detection.

---

## 🚀 Features
- **Frame Extraction & Preprocessing**: Extracts multiple frames per video and detects faces for analysis.
- **Deep Learning Architecture**:
  - **EfficientNet** for spatial features.
  - **LSTM** for temporal sequence modeling.
- **Real-Time Detection**: Upload videos via the Flask web interface and get prediction results instantly.
- **Modular Code Structure**: Organized to avoid cascading errors in Jupyter/Kaggle environments.
- **Custom Dataset Handling**: Trained on a balanced subset of the **FaceForensics++ C23 dataset**.

---

## 📂 Project Structure
plaintext
Deepfake-Detection/
│
├── app.py                 # Flask backend
├── requirements.txt       # Project dependencies
├── static/                # CSS, JS, and frontend assets
├── templates/             # HTML frontend templates
├── model/                 # Saved trained model files
├── preprocessing/         # Frame extraction and face detection scripts
├── utils/                 # Helper functions
├── README.md              # Project documentation
└── dataset/               # FaceForensics++ data (not included in repo)

## 🛠️ Tech Stack
Programming Language: Python

ML/DL Frameworks: TensorFlow, Keras

Frontend: HTML, CSS

Backend: Flask

Computer Vision: OpenCV

Model Architecture: EfficientNetB0 + LSTM

Dataset: FaceForensics++ (C23 version)

##📊 Model Performance
Metric	Value
Precision- 0.50
Recall- 0.53
F1-Score- 0.51

## 📌 How It Works
Video Upload: Users upload a video via the Flask frontend.

Frame Extraction: Key frames are extracted and preprocessed.

Face Detection & Cropping: Only facial regions are used for analysis.

Feature Extraction: EfficientNet extracts spatial features from frames.

Temporal Modeling: LSTM processes frame sequences to capture motion patterns.

Prediction: Aggregated predictions determine if the video is real or deepfake.

## 👨‍💻 Author
Abhishek Soni
B.Tech (AI & ML), Manipal University Jaipur
📧 Email: abhisheksoni0074@gmail.com
🔗 LinkedIn- https://www.linkedin.com/in/abhisheksoni0074/


