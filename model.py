import tensorflow as tf
import numpy as np
from PIL import Image
import io

class EmotionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = (48, 48)

    def preprocess_bytes(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('L').resize(self.input_shape)
        arr = np.array(img).astype('float32')/255.0
        arr = arr.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        return arr

    def predict_bytes(self, image_bytes, class_names=None):
        x = self.preprocess_bytes(image_bytes)
        preds = self.model.predict(x)[0]
        idx = int(preds.argmax())
        conf = float(preds[idx])
        label = class_names[idx] if class_names else idx
        return {"label": label, "index": idx, "confidence": conf}
