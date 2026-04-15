from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["GET"])
def predict():
    # TEST tạm input (sau sẽ nối Firebase)
    data = [30, 80, 1012]

    x = np.array(data).reshape(1, -1)
    pred = model.predict(x).tolist()

    return jsonify({
        "input": data,
        "prediction": pred
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
