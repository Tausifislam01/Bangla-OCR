import os
import json
import argparse
from pathlib import Path

import cv2
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
        image_files = list(class_folder.glob("*"))

        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]

        print(f"Class {class_folder.name}: {len(image_files)} images")

        for image_path in image_files:
            if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(label_index)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    X = np.expand_dims(X, axis=-1)

    print(f"Total images loaded: {len(X)}")
    print(f"Image shape: {X.shape}")

    return X, y, class_names


def build_model(num_classes, learning_rate):
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

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


def train(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts/mlflow_artifacts", exist_ok=True)

    # MLflow 3.x works better with SQLite than file-based tracking
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

        model = build_model(
            num_classes=len(class_names),
            learning_rate=args.learning_rate
        )

        checkpoint = callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint, early_stop]
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))

        for epoch in range(len(history.history["accuracy"])):
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)

        save_labels(class_names)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(LABELS_PATH)

        print(f"Model saved to {MODEL_PATH}")
        print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/BanglaLekha-Isolated/Images"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="cnn_baseline"
    )

    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=None
    )

    args = parser.parse_args()
    train(args)