from flask import Flask, render_template, request, jsonify
import os
import random

app = Flask(__name__)

# Mock class names for testing
CLASS_NAMES = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']

@app.route("/")
def index():
    # List sample images in static/samples
    samples_dir = os.path.join(app.static_folder, "samples")
    sample_files = []
    if os.path.isdir(samples_dir):
        for fn in sorted(os.listdir(samples_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_files.append(fn)
    
    print(f"Found sample files: {sample_files}")
    return render_template("index.html", sample_files=sample_files, class_names=CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Simulate processing time
    import time
    time.sleep(0.1)
    
    # Generate mock predictions
    # Pick a random "best" class
    best_index = random.randint(0, len(CLASS_NAMES) - 1)
    
    # Generate probabilities for all classes (make them sum to 1.0)
    probs = [random.uniform(0.01, 0.3) for _ in range(len(CLASS_NAMES))]
    probs[best_index] = random.uniform(0.4, 0.8)  # Make best class more confident
    
    # Normalize to sum to 1.0
    total = sum(probs)
    probs = [p / total for p in probs]
    
    # Get top 3 predictions
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top = [{"label": CLASS_NAMES[i], "index": int(i), "prob": float(probs[i])} for i in top_indices]
    
    print(f"Mock prediction: {CLASS_NAMES[best_index]} ({probs[best_index]:.3f})")
    
    return jsonify({
        "top": top,
        "best": top[0],
        "all_len": int(len(CLASS_NAMES)),
    })

if __name__ == "__main__":
    print("Starting test server...")
    print("Sample files should be displayed in the left panel")
    print("Click on samples to see mock predictions!")
    app.run(host="0.0.0.0", port=4022, debug=True) 