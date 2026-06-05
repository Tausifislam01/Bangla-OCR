import os
import json
import argparse
from pathlib import Path

import cv2
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import layers, models, callbacks


IMAGE_SIZE = 64
MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"


def load_dataset(dataset_dir, max_images_per_class=None):
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    class_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

    if len(class_folders) == 0:
        raise ValueError("No class folders found. Check your dataset path.")

    class_folders = sorted(
        class_folders,
        key=lambda folder: int(folder.name) if folder.name.isdigit() else folder.name
    )

    class_names = [folder.name for folder in class_folders]

    images = []
    labels = []

    print(f"Found {len(class_names)} classes.")
    print("Loading dataset...")

    for label_index, class_folder in enumerate(class_folders):
        image_files = [
            image_path for image_path in class_folder.glob("*")
            if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ]

        image_files = sorted(image_files)

        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]

        print(f"Class {class_folder.name}: {len(image_files)} images")

        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )

            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(label_index)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("No valid images loaded. Check dataset path and image files.")

    X = np.expand_dims(X, axis=-1)

    print(f"Total images loaded: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y, class_names


def build_model(num_classes, learning_rate):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation"
    )

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        data_augmentation,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.20),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.30),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.35),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_labels(class_names):
    labels = {str(index): label for index, label in enumerate(class_names)}

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=4, ensure_ascii=False)

    print(f"Saved labels to {LABELS_PATH}")


def plot_training_curves(history, output_path):
    accuracy = history.history.get("accuracy", [])
    val_accuracy = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def check_overfitting(history, test_accuracy):
    train_accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    accuracy_gap = train_accuracy - val_accuracy
    loss_gap = val_loss - train_loss

    print("\nOverfitting Check")
    print("-----------------")
    print(f"Final train accuracy: {train_accuracy:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Accuracy gap train-val: {accuracy_gap:.4f}")
    print(f"Loss gap val-train: {loss_gap:.4f}")

    if accuracy_gap > 0.15:
        status = "High overfitting risk"
        print("Warning: Training accuracy is much higher than validation accuracy.")
    elif accuracy_gap > 0.08:
        status = "Moderate overfitting risk"
        print("Warning: Some overfitting may be present.")
    else:
        status = "No major overfitting detected"
        print("No major overfitting detected.")

    return {
        "final_train_accuracy": float(train_accuracy),
        "final_val_accuracy": float(val_accuracy),
        "final_train_loss": float(train_loss),
        "final_val_loss": float(val_loss),
        "accuracy_gap": float(accuracy_gap),
        "loss_gap": float(loss_gap),
        "overfitting_status": status
    }


def train(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/mlflow_artifacts", exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment_name = "Bangla OCR Character Recognition"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location="artifacts/mlflow_artifacts"
        )

    mlflow.set_experiment(experiment_name)

    X, y, class_names = load_dataset(
        args.dataset_dir,
        args.max_images_per_class
    )

    if len(class_names) < 2:
        raise ValueError("At least 2 classes are required for training.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("dataset_dir", args.dataset_dir)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("max_images_per_class", args.max_images_per_class)
        mlflow.log_param("model_type", "Improved Custom CNN")
        mlflow.log_param("augmentation", "rotation_zoom_translation")
        mlflow.log_param("preprocessing", "grayscale_resize_64_normalize")
        mlflow.log_param("overfit_check", "train_val_accuracy_gap_and_loss_gap")

        model = build_model(
            num_classes=len(class_names),
            learning_rate=args.learning_rate
        )

        model.summary()

        checkpoint = callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )

        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(f"\nTest loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))

        for epoch in range(len(history.history["accuracy"])):
            mlflow.log_metric(
                "train_accuracy",
                float(history.history["accuracy"][epoch]),
                step=epoch
            )
            mlflow.log_metric(
                "val_accuracy",
                float(history.history["val_accuracy"][epoch]),
                step=epoch
            )
            mlflow.log_metric(
                "train_loss",
                float(history.history["loss"][epoch]),
                step=epoch
            )
            mlflow.log_metric(
                "val_loss",
                float(history.history["val_loss"][epoch]),
                step=epoch
            )

            if "learning_rate" in history.history:
                mlflow.log_metric(
                    "learning_rate",
                    float(history.history["learning_rate"][epoch]),
                    step=epoch
                )

        overfit_info = check_overfitting(history, test_accuracy)

        mlflow.log_metric("final_train_accuracy", overfit_info["final_train_accuracy"])
        mlflow.log_metric("final_val_accuracy", overfit_info["final_val_accuracy"])
        mlflow.log_metric("accuracy_gap_train_val", overfit_info["accuracy_gap"])
        mlflow.log_metric("loss_gap_val_train", overfit_info["loss_gap"])
        mlflow.log_param("overfitting_status", overfit_info["overfitting_status"])

        training_curve_path = "artifacts/training_curves.png"
        plot_training_curves(history, training_curve_path)
        mlflow.log_artifact(training_curve_path)

        y_pred_prob = model.predict(X_test, batch_size=args.batch_size, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        confusion_matrix_path = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, class_names, confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)

        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )

        report_path = "artifacts/classification_report.txt"

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

        mlflow.log_artifact(report_path)

        save_labels(class_names)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(LABELS_PATH)

        print("\nClassification Report")
        print("---------------------")
        print(report)

        print(f"Model saved to {MODEL_PATH}")
        print(f"Labels saved to {LABELS_PATH}")
        print(f"Training curves saved to {training_curve_path}")
        print(f"Confusion matrix saved to {confusion_matrix_path}")
        print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/BanglaLekha-Isolated/Images",
        help="Path to BanglaLekha-Isolated Images folder"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="improved_cnn_overfit_checked",
        help="MLflow run name"
    )

    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=None,
        help="Limit images per class for faster training"
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=6,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience"
    )

    args = parser.parse_args()
    train(args)