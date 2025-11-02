# 🧠 Emotion Detector Web App  

An interactive web application that detects human emotions from facial images using a **Convolutional Neural Network (CNN)** model.  
Built with **Flask**, **TensorFlow**, and a modern **HTML/CSS/JavaScript** frontend, the app allows users to either **upload an image** or **use their webcam** to get instant emotion predictions.

---

## 🚀 Features
- 🎥 Real-time emotion detection via webcam or image upload  
- 🧩 CNN-based model trained on facial emotion datasets  
- 🎨 Beautiful blue-glass UI with smooth user interaction  
- 💾 SQLite database to log predictions with timestamps  
- ☁️ Fully deployable on Render or other cloud platforms  

---

## 🧰 Tech Stack
**Backend:** Flask, TensorFlow, SQLite  
**Frontend:** HTML5, CSS3, JavaScript  
**Libraries:** Flask-CORS, Pillow, NumPy  
**Deployment:** Gunicorn + Render  

---

## 📸 How It Works
1. Launch the web app.  
2. Upload a facial image or activate your webcam.  
3. The model analyzes the image and returns the detected emotion (e.g., *Happy, Sad, Angry, Neutral*, etc.).  
4. Each prediction is stored in a local SQLite database for record-keeping and analysis.  

---

## 🛠️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/Atinukecodes/emotion-detector.git
cd emotion-detector
