from flask import Flask, render_template, request, jsonify
import os, io
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# -----------------------------
# 1) Load model
# -----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_original.keras")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Infer input size (e.g., InceptionV3 is usually 299x299)
if isinstance(model.input_shape, list):
    h, w = model.input_shape[0][1], model.input_shape[0][2]
else:
    h, w = model.input_shape[1], model.input_shape[2]
TARGET_SIZE = (w, h)  # PIL expects (width, height)
NUM_CLASSES = model.output_shape[-1]

# -----------------------------
# 2) Class names (your list)
# -----------------------------
RAW_CLASSES = [
    'Tomato_Bacterial_spot',
    'Potato___Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Early_blight',
    'Tomato__Tomato_mosaic_virus',
    'Pepper__bell___Bacterial_spot',
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato_Late_blight',
    'Tomato_healthy',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Septoria_leaf_spot',
    'Tomato__Target_Spot',
    'Pepper__bell___healthy',
]
assert len(RAW_CLASSES) == NUM_CLASSES, (
    f"Class count ({len(RAW_CLASSES)}) must match model output ({NUM_CLASSES})."
)

# Prettify for display only
def pretty(name: str) -> str:
    return name.replace("___", " ").replace("__", " ").replace("_", " ")

CLASS_NAMES = [pretty(n) for n in RAW_CLASSES]

# -----------------------------
# 3) Utilities
# -----------------------------
def preprocess_pil_image(pil_img, target_size=TARGET_SIZE):
    img = pil_img.convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# -----------------------------
# 4) Routes
# -----------------------------
@app.route("/")
def index():
    # List sample images in static/samples
    samples_dir = os.path.join(app.static_folder, "samples")
    sample_files = []
    if os.path.isdir(samples_dir):
        for fn in sorted(os.listdir(samples_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_files.append(fn)
    return render_template("index.html", sample_files=sample_files, class_names=CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    x = preprocess_pil_image(pil_img, TARGET_SIZE)
    logits = model.predict(x, verbose=0)
    probs = softmax(logits)[0]

    topk = min(3, NUM_CLASSES)
    top_idx = probs.argsort()[-topk:][::-1]
    top = [{"label": CLASS_NAMES[i], "index": int(i), "prob": float(probs[i])} for i in top_idx]

    return jsonify({
        "top": top,
        "best": top[0],
        "all_len": int(NUM_CLASSES),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4021, debug=True)
