"""
Training pipeline for plant disease detection using EfficientNetB0 transfer learning.
"""
import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 plant disease classifier")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to organized image dataset directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory where model and metadata are saved")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--img-size", type=int, default=128, help="Square image resolution (width & height)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)

    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Dataset directory '{args.data_dir}' not found.")
        print("Please download an organized dataset (e.g. PlantVillage) with subfolders for each disease category.")
        sys.exit(1)

    subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    if len(subdirs) < 2:
        print(f"❌ Error: Dataset directory '{args.data_dir}' must contain at least 2 class subdirectories.")
        sys.exit(1)

    import tensorflow as tf

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"📂 Loading dataset from '{args.data_dir}' ({len(subdirs)} classes detected)...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=args.val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=args.batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=args.val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=args.batch_size
    )

    class_names = train_ds.class_names
    print(f"✅ Discovered {len(class_names)} classes: {class_names}")

    class_names_path = os.path.join(args.model_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"💾 Saved class metadata to '{class_names_path}'")

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("🧠 Building EfficientNetB0 transfer learning architecture...")
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    # EfficientNet preprocess handles proper RGB scaling
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PlantDoctor_EfficientNetB0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print(f"🚀 Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model_save_path = os.path.join(args.model_dir, "plant_disease_model.keras")
    model.save(model_save_path)
    print(f"🎉 Model training successfully complete! Saved to '{model_save_path}'")


if __name__ == "__main__":
    main()
