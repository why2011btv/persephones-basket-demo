import os
import tensorflow_datasets as tfds
import tensorflow as tf

# === Load dataset ===
ds, info = tfds.load('plant_village', split='train', with_info=True)

# === Create output directory ===
output_dir = './plant_examples'
os.makedirs(output_dir, exist_ok=True)

# === Get class label names ===
label_names = info.features['label'].names
num_classes = len(label_names)
print(f"Number of categories: {num_classes}")

# === Prepare dict to track if we already saved one example per class ===
saved = {i: False for i in range(num_classes)}

# === Iterate through dataset ===
for example in tfds.as_numpy(ds):
    label = example['label']
    if not saved[label]:
        img = example['image']
        label_name = label_names[label]

        # Save image to file
        filename = f"{label}_{label_name.replace(' ', '_')}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Convert to TensorFlow image and write
        tf.io.write_file(filepath, tf.image.encode_jpeg(tf.convert_to_tensor(img)))
        saved[label] = True
        print(f"Saved example for: {label_name}")

    # Stop when all saved
    if all(saved.values()):
        break

print(f"✅ Saved one image for each of the {num_classes} categories in '{output_dir}'")

