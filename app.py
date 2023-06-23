import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = None
def load_model_instance():
    global model
    if model is None:
        model = load_model("mnist_model.h5")
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/classify", methods=["POST"])
def classify():
    load_model_instance()  # Load the model instance if it's not already loaded
    data = request.get_json()
    points = data["points"]
    canvas = np.zeros((28, 28), dtype=np.uint8)
    for point in points:
        x = int(point["x"])
        y = int(point["y"])
        canvas[y, x] = 255
    image = Image.fromarray(canvas)
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions[0])
    image_data = image.tobytes()
    return jsonify({
        "predicted_label": int(predicted_label),
        "image_data": image_data.decode("latin1")  # Convert bytes to string for JSON serialization
    })
def save_image(canvas, image_path):
    image = Image.fromarray(canvas)
    image.save(image_path)
if __name__ == "__main__":
    load_model_instance()  # Load the model instance when the script is run
    app.run()
