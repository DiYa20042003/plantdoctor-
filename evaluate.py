"""
Evaluation pipeline for Plant Doctor.
Computes test metrics (accuracy, macro precision, recall, F1-score) and generates evaluation summary.
"""
import argparse
import json
import os
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plant Doctor model on a test/validation split")
    parser.add_argument("--data-dir", type=str, default="data/test", help="Path to evaluation image directory")
    parser.add_argument("--model-path", type=str, default="models/plant_disease_model.keras", help="Path to saved .keras model")
    parser.add_argument("--class-file", type=str, default="models/class_names.json", help="Path to saved class names JSON")
    parser.add_argument("--img-size", type=int, default=128, help="Input image dimension")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Path to output metrics JSON")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Compute per-class and macro classification metrics without external heavy deps"""
    accuracy = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "total_evaluated_samples": int(len(y_true))
    }


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model weights not found at '{args.model_path}'. Train model first using train.py.")
        sys.exit(1)

    if not os.path.exists(args.class_file):
        print(f"❌ Error: Class metadata file not found at '{args.class_file}'.")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Evaluation dataset directory not found at '{args.data_dir}'.")
        sys.exit(1)

    with open(args.class_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    import tensorflow as tf

    print(f"📦 Loading model from '{args.model_path}'...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"📂 Loading test dataset from '{args.data_dir}'...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False
    )

    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(pred_labels)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = compute_metrics(y_true, y_pred, num_classes=len(class_names))

    print("\n================ EVALUATION RESULTS ================")
    print(f"Total Samples:    {metrics['total_evaluated_samples']}")
    print(f"Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:  {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:     {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score:   {metrics['macro_f1'] * 100:.2f}%")
    print("====================================================\n")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to '{args.output_json}'")


if __name__ == "__main__":
    main()
