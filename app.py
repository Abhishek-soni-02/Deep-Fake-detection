import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

# === Initialize App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load Trained Model ===
MODEL_PATH = 'model/deepfake_model.h5'
model = load_model(MODEL_PATH)

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, frame_count=10, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // frame_count, 1)

    frames = []
    count = 0
    while len(frames) < frame_count and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            image = cv2.resize(frame, target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        count += 1
    cap.release()

    # If less frames, pad with last frame
    while len(frames) < frame_count:
        frames.append(frames[-1])
    
    frames = np.array(frames, dtype=np.float32)
    frames = preprocess_input(frames)
    return np.expand_dims(frames, axis=0)

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        frames = extract_frames(filepath)
        prediction = model.predict(frames)[0][0]
        result = "Fake" if prediction >= 0.46 else "Real"

        return render_template('result.html', result=result, confidence=round(float(prediction)*100, 2), video=filename)
    else:
        return "Invalid file format."

# === Run App ===
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
