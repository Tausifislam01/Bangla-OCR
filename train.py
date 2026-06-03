import os
import json
import argparse
from pathlib import Path

import cv2
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks


IMAGE_SIZE = 64
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
LABELS_PATH = "labels.json"


def load_dataset(dataset_dir):
    """
    Expected dataset structure:
    dataset/BanglaLekha-Isolated/
        class_1/
            image1.png
            image2.png
        class_2/
            image1.png
            image2.png
    """

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Please place the BanglaLekha-Isolated dataset inside the dataset folder."
        )

    images = []
    labels = []

    class_folders = sorted(
        [folder for folder in dataset_path.iterdir() if folder.is_dir()],
        key=lambda x: x.name
    )

    if len(class_folders) == 0:
        raise ValueError("No class folders found inside dataset directory.")

    class_names = [folder.name for folder in class_folders]
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes.")

    for class_folder in class_folders:
        class_name = class_folder.name
        class_index = class_to_index[class_name]

        image_files = list(class_folder.glob("*"))

        for image_path in image_files:
            if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = preprocess_character_image(image)
            images.append(image)
            labels.append(class_index)

    if len(images) == 0:
        raise ValueError("No valid images found in dataset directory.")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    X = np.expand_dims(X, axis=-1)

    return X, y, class_names


def preprocess_character_image(image):
    """
    Converts an input character image to normalized 64x64 grayscale.
    """

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    return image


def build_model(num_classes, learning_rate):
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_labels(class_names):
    index_to_label = {str(index): label for index, label in enumerate(class_names)}

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(index_to_label, file, ensure_ascii=False, indent=4)

    print(f"Saved label mapping to {LABELS_PATH}")


def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("artifacts/mlflow", exist_ok=True)

    mlflow.set_tracking_uri("file:artifacts/mlflow")
    mlflow.set_experiment("Bangla OCR Character Recognition")

    X, y, class_names = load_dataset(args.dataset_dir)

    num_classes = len(class_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("model_type", "CNN")

        model = build_model(num_classes, args.learning_rate)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        checkpoint = callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop, checkpoint],
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        for epoch, accuracy in enumerate(history.history["accuracy"]):
            mlflow.log_metric("train_accuracy", accuracy, step=epoch)

        for epoch, val_accuracy in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("train_loss", loss, step=epoch)

        for epoch, val_loss in enumerate(history.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        save_labels(class_names)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(LABELS_PATH)

        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bangla OCR character recognition model")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/BanglaLekha-Isolated",
        help="Path to BanglaLekha-Isolated dataset folder",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="cnn_baseline",
        help="MLflow run name",
    )

    args = parser.parse_args()
    train(args)