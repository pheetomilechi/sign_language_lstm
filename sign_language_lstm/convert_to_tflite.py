"""Convert the trained Keras model to TensorFlow Lite."""

import os

import tensorflow as tf

from config import MODEL_DIR, MODEL_PATH


model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

try:
    tflite_model = converter.convert()
except Exception:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

output_path = os.path.join(MODEL_DIR, "sign_lstm_model.tflite")
with open(output_path, "wb") as model_file:
    model_file.write(tflite_model)

print(f"TensorFlow Lite model saved to: {output_path}")
