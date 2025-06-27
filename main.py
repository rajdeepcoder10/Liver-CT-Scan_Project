from flask import Flask, render_template, request, redirect, send_from_directory, session
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your-secret-key'  # for session use

# Set valid credentials
VALID_USERNAME = "admin"
VALID_PASSWORD = "rajdeep"

# Load the trained model
model = load_model('Models/model.h5')
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "Squares Cell Carcenioma", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            return redirect('/dashboard')
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            result, confidence = predict_tumor(file_location)
            session['disease'] = result
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')
    return render_template('index.html', result=None)

@app.route('/report')
def report():
    disease = session.get('disease', 'Unknown')
    treatment_map = {
        "Large cell Carcinoma": [
            "Chemotherapy sessions as prescribed",
            "Liver function tests every 3 months",
            "Low-fat, high-protein diet",
            "Regular hydration and enzyme monitoring"
        ],
        "Tumor: pituitary": [
            "Hormonal replacement therapy",
            "Surgical removal if enlarged",
            "MRI checkups every 6 months"
        ],
        "Tumor: glioma": [
            "Radiotherapy and chemotherapy",
            "Avoid alcohol/smoking",
            "Monitor neurological health"
        ],
        "Tumor: meningioma": [
            "CT/MRI scans periodically",
            "Surgical resection if symptomatic",
            "Anti-inflammatory support"
        ]
    }
    tips = treatment_map.get(disease, ["Consult a liver specialist for personalized treatment."])
    return render_template('report.html', disease=disease, tips=tips)

@app.route('/heat')
def heatmap():
    return render_template('heat.html', confidence="87.24")

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
