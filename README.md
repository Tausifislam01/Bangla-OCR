# Bangla Handwritten Word Recognition System

## README Documentation

**Model Training | Streamlit UI | Docker | MLflow**

---

## 1. Project Overview

This project implements a Bangla handwritten word recognition system using the BanglaLekha-Isolated dataset. The system trains a character-level image classifier and provides a Streamlit web interface where users can draw a Bangla word or separated Bangla characters.

The drawn input is preprocessed, segmented into individual character regions, classified using the trained CNN model, and the predictions are combined to produce the final recognized Bangla character sequence.

The project includes:

* Model training
* MLflow experiment tracking
* Streamlit prediction UI
* Docker support
* Trained model artifacts
* Character mapping
* Experiment screenshots

---

## 2. Dataset

Dataset used: **BanglaLekha-Isolated** from Mendeley Data.

The dataset contains isolated Bangla handwritten character images. Since the dataset is character-level, this project approaches word recognition by segmenting a drawn word into individual characters and classifying each character separately.

The dataset contains 84 character classes. The folders are numbered from 1 to 84.

---

## 3. Project Structure

```text
bangla-ocr-assignment/
|-- train.py
|-- app.py
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- labels.json
|-- character_map.json
|-- github_link.txt
|-- mlflow.db
|
|-- models/
|   `-- model.keras
|
|-- artifacts/
|   |-- training_curves.png
|   |-- confusion_matrix.png
|   `-- classification_report.txt
|
`-- screenshots/
    |-- streamlit_app.png
    |-- mlflow_experiment.png
    |-- mlflow_runs.png
    |-- mlflow_run_overview.png
    `-- mlflow_artifacts.png
```

---

## 4. Dataset Preparation

The BanglaLekha-Isolated dataset was downloaded and extracted. The extracted dataset contains one folder per class inside the `Images` directory. Each class folder contains handwritten samples for one Bangla character class.

Expected dataset path during training:

```text
dataset/BanglaLekha-Isolated/Images
```

The dataset folder is not included in the final submission zip because it is large.

---

## 5. Preprocessing

Each training image is processed using the following steps:

1. Read the image in grayscale.
2. Resize the image to 64 x 64 pixels.
3. Normalize pixel values to the range 0 to 1.
4. Expand dimensions to match the CNN input shape: `(64, 64, 1)`.

For the Streamlit prediction app, the canvas image is processed using the following steps:

1. Convert RGBA canvas image to grayscale.
2. Apply binary inverse thresholding.
3. Remove noise using morphological opening.
4. Apply slight dilation to connect broken strokes.
5. Detect contours.
6. Sort segmented character regions from left to right.
7. Crop each character region.
8. Add padding around each character.
9. Resize each character image to 64 x 64.
10. Predict each character using the trained CNN model.

---

## 6. Model Architecture

The model is an improved custom Convolutional Neural Network trained from scratch.

The model uses:

* Data augmentation
* Convolutional layers
* Batch normalization
* Max pooling
* Dropout
* Global average pooling
* Dense layer
* Softmax output layer

| Item              | Details                     |
| ----------------- | --------------------------- |
| Input             | 64 x 64 grayscale image     |
| Output            | 84-class softmax prediction |
| Model type        | Improved Custom CNN         |
| Saved model       | models/model.keras          |
| Label mapping     | labels.json                 |
| Character mapping | character_map.json          |

---

## 7. Data Augmentation

The training pipeline uses light augmentation to improve generalization.

Augmentation methods:

* Small random rotation
* Random zoom
* Random translation

These transformations help the model handle variations in handwriting style.

---

## 8. Training Process

The dataset was split into training, validation, and test sets. The model was trained using Adam optimization and sparse categorical crossentropy loss.

| Item                     | Details                         |
| ------------------------ | ------------------------------- |
| Optimizer                | Adam                            |
| Loss function            | Sparse Categorical Crossentropy |
| Metric                   | Accuracy                        |
| Checkpointing            | Best validation accuracy        |
| Early stopping           | Enabled                         |
| Learning rate scheduling | ReduceLROnPlateau               |

Final result:

```text
Test Accuracy: 95.27%
Test Loss: 0.1845
Overfitting Check: No major overfitting detected
```

The trained model was saved as:

```text
models/model.keras
```

The label mapping was saved as:

```text
labels.json
```

---

## 9. MLflow Tracking

MLflow was used to track experiment parameters, metrics, and artifacts.

Tracked parameters include:

* Dataset path
* Image size
* Epochs
* Batch size
* Learning rate
* Number of classes
* Model type
* Preprocessing details
* Data augmentation details

Tracked metrics include:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Test accuracy
* Test loss
* Overfitting gap

Logged artifacts include:

* Trained model
* Label mapping
* Training curves
* Confusion matrix
* Classification report

To start MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 10. Streamlit UI

The Streamlit app allows the user to draw a Bangla word or separated Bangla characters using an interactive canvas.

The app performs the following steps:

1. Captures the canvas image.
2. Converts the canvas image to grayscale.
3. Applies thresholding and preprocessing.
4. Segments the drawing into character regions.
5. Predicts each character using the trained CNN model.
6. Converts predicted dataset class IDs to Bangla characters using `character_map.json`.
7. Displays the final recognized Bangla character sequence.
8. Shows the predicted dataset class sequence.
9. Shows per-character confidence scores.
10. Shows top-3 predictions for each segmented character.

Run the app locally:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 11. Character Mapping

The dataset folders are numeric class IDs from 1 to 84. The model predicts dataset class IDs. To display actual Bangla characters, this project uses `character_map.json`.

Example:

```json
{
  "id": 42,
  "character": "স"
}
```

The Streamlit app uses this mapping to display both:

* Bangla Output
* Predicted Dataset Class Sequence

This makes the prediction result easier to understand.

---

## 12. Word Segmentation Strategy

The word segmentation strategy is based on image processing and contour detection.

Steps:

1. Convert the drawn canvas image to grayscale.
2. Apply binary inverse thresholding.
3. Remove small noise.
4. Detect external contours.
5. Filter out small noise regions.
6. Sort bounding boxes from left to right.
7. Crop each character region.
8. Pad and resize each character image.
9. Predict characters one by one.
10. Combine predicted characters into a final sequence.

This simple segmentation works best when characters are drawn clearly with slight spacing between them.

---

## 13. Docker Usage

Build the Docker image:

```bash
docker build -t bangla-ocr-app:0.1 .
```

Run the Docker container:

```bash
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open the app:

```text
http://localhost:8501
```

---

## 14. Screenshots

Required screenshots:

```text
screenshots/streamlit_app.png
screenshots/mlflow_experiment.png
```

Additional MLflow screenshots included:

```text
screenshots/mlflow_runs.png
screenshots/mlflow_run_overview.png
screenshots/mlflow_artifacts.png
```

---

## 15. Limitations

* The model is trained on isolated character images, not full handwritten word images.
* Word recognition depends on successful character segmentation.
* Connected Bangla handwriting may be difficult to segment correctly.
* The app performs character-by-character recognition instead of using a sequence model.
* The segmentation strategy works best when characters are drawn with slight spacing.
* The system is not a full production-level OCR engine.
* Recognition quality depends on handwriting clarity and canvas input quality.

---

## 16. Possible Improvements

* Use a better segmentation algorithm for connected Bangla handwriting.
* Train on actual Bangla word-level datasets.
* Use CRNN, CTC, or Transformer-based OCR for full word recognition.
* Improve preprocessing for different stroke thicknesses and writing styles.
* Add support for uploaded image prediction.
* Add line segmentation and word segmentation for paragraph-level OCR.
* Improve character mapping validation using official dataset documentation.
* Deploy the app online using a cloud service.

---

## 17. Submission Contents

The final submission includes:

* Public GitHub repository link
* Clean source code zip
* `train.py`
* `app.py`
* `requirements.txt`
* `Dockerfile`
* `README.md`
* `labels.json`
* `character_map.json`
* `models/model.keras`
* `screenshots/streamlit_app.png`
* `screenshots/mlflow_experiment.png`
* `github_link.txt`

The final zip excludes:

* Dataset folder
* Virtual environment
* Cache files
* Large temporary files
* MLflow run folders
* Python cache folders

---

## 18. Suggested Commands

Train the model:

```bash
python train.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Build Docker image:

```bash
docker build -t bangla-ocr-app:0.1 .
```

Run Docker container:

```bash
docker run -p 8501:8501 bangla-ocr-app:0.1
```

---

## 19. Final Notes

This project satisfies the main assignment requirements by training a Bangla handwritten character recognition model, tracking experiments with MLflow, providing a Streamlit drawing interface, attempting word recognition through character segmentation, displaying confidence scores, saving the trained model and label mapping, and supporting Docker-based execution.
