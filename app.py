from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64, sqlite3, os, io
from PIL import Image
import numpy as np
from model import EmotionModel

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'emotion_cnn_v1.h5')
CLASS_NAMES = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
model = EmotionModel(MODEL_PATH)

def init_db():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'database.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uses (
                 id INTEGER PRIMARY KEY,
                 name TEXT,
                 label TEXT,
                 confidence REAL,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_b64 = data.get('image', '')
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]

    # Decode and preprocess image
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run model prediction
    res = model.predict(img, CLASS_NAMES) if hasattr(model, 'predict') else model.predict_bytes(img_bytes, CLASS_NAMES)

    # Ensure consistent response structure
    if isinstance(res, str):
      label = res.get('label', res.get('prediction', 'Unknown'))
      confidence = res.get('confidence', 0.0)
    else:
      label = str(res)
      confidence = 0.0
    
    return jsonify({
    'prediction': label,
    'confidence': confidence
})
    # Store result in database
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'database.db'))
    c = conn.cursor()
    c.execute("INSERT INTO uses (name,label,confidence) VALUES (?,?,?)",
              (data.get('name','anon'),
               res.get('label','Unknown'),
               res.get('confidence',0.0)))
    conn.commit()
    conn.close()

    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
