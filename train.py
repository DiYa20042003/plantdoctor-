import tensorflow as tf
import os
import json

# ---------------- Config ----------------
IMG_SIZE = (128, 128)   # smaller image size to reduce memory usage
BATCH_SIZE = 8          # reduced batch size (adjust to your RAM/GPU)
EPOCHS = 20
DATA_DIR = "."    # your dataset folder
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Load Dataset ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Extract class names before caching/prefetch
class_names = train_ds.class_names
print(f"📂 Classes: {class_names}")

# Save class names
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

# Optimize datasets
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ---------------- Build Model ----------------
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base model

global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dropout = tf.keras.layers.Dropout(0.3)(global_avg)
output = tf.keras.layers.Dense(len(class_names), activation="softmax")(dropout)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- Train Model ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ---------------- Save Model ----------------
model.save(os.path.join(MODEL_DIR, "plant_disease_model.keras"))
print("✅ Model training complete & saved to 'models/plant_disease_model.keras'")
