# Bangla Handwritten Word Recognition System

## README Documentation

**Model Training | Streamlit UI | Docker | MLflow**

---

## 1. Project Overview

This project implements a Bangla handwritten word recognition system using the BanglaLekha-Isolated dataset. The system trains a character-level image classifier and provides a Streamlit web interface where users can draw a Bangla character, selected jukto borno, or Bangla word.

The drawn input is preprocessed, segmented into safe base character or jukto borno units, classified using the trained CNN model, and the predictions are combined to produce the final recognized Bangla output.

The project includes:

* Model training
* MLflow experiment tracking
* Streamlit prediction UI
* Docker support
* Trained model artifacts
* Character mapping
* Matra-aware segmentation
* Experiment screenshots

---

## 2. Dataset

Dataset used: **BanglaLekha-Isolated** from Mendeley Data.

The dataset contains isolated Bangla handwritten character images. Since the dataset is character-level, this project approaches word recognition by segmenting a drawn word into base character or selected jukto borno units and classifying each unit separately.

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
3. Remove small noise using morphological opening.
4. Apply slight dilation to connect broken strokes.
5. Detect the full ink region.
6. Estimate and handle the Bangla matra/headline region during segmentation.
7. Find safe vertical cut points using projection-based analysis.
8. Remove tiny noise and headline-only fragments.
9. Segment the drawing into base character or jukto borno units.
10. Add padding around each segmented unit.
11. Resize each unit image to 64 x 64.
12. Predict each unit using the trained CNN model.

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

Note: `mlflow.db` is generated after training. If it is not included in the submitted repository, the MLflow UI can be regenerated by running `train.py`.

---

## 10. Streamlit UI

The Streamlit app allows the user to draw a Bangla character, selected jukto borno, or Bangla word using an interactive canvas.

The app performs the following steps:

1. Captures the canvas image.
2. Converts the canvas image to grayscale.
3. Applies thresholding and preprocessing.
4. Uses matra-aware safe segmentation to split the drawing into base/jukto units.
5. Removes headline-only fragments and very small noise components.
6. Predicts each segmented unit using the trained CNN model.
7. Converts predicted dataset class IDs to Bangla characters using `character_map.json`.
8. Displays the final recognized Bangla output.
9. Shows the predicted dataset class sequence.
10. Shows per-unit confidence scores.
11. Shows top-3 predictions for each segmented unit.

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

The word segmentation strategy is based on image processing, matra-aware segmentation, projection analysis, and connected component filtering.

The goal is to segment Bangla words into base character or selected jukto borno units, not into separate vowel signs or কার চিহ্ন.

For example:

```text
ওজন = ও + জ + ন
ক্ষমা = ক্ষ + ম
```

Here, `া` is not counted as a separate unit because the model was not trained to classify কার চিহ্ন independently.

### Segmentation Steps

1. Convert the drawn canvas image to grayscale.
2. Apply binary inverse thresholding.
3. Remove small noise using morphological opening.
4. Apply slight dilation to connect broken strokes inside characters.
5. Detect the full word or character ink region.
6. Estimate the Bangla matra/headline region.
7. Temporarily remove the matra/headline only for finding safe cut points.
8. Use vertical projection to find low-ink gaps between base/jukto units.
9. Avoid unsafe cuts through connected handwriting or jukto borno.
10. Remove tiny noise and headline-only fragments.
11. Crop each safe base/jukto unit.
12. Pad and resize each unit image.
13. Predict units one by one using the CNN model.
14. Combine predicted units into the final Bangla output.

### Expected Base/Jukto Unit Count

The Streamlit sidebar includes an optional expected unit count. This helps the segmentation process when the user knows how many base or jukto units are present.

Examples:

| Word/Input | Expected Units | Explanation |
| ---------- | -------------- | ----------- |
| অ | 1 | Single character |
| ক্ষ | 1 | Single jukto borno |
| ওজন | 3 | ও + জ + ন |
| ক্ষমা | 2 | ক্ষ + ম, ignoring আ-কার |
| ক্ষতি | 2 | ক্ষ + ত, ignoring ই-কার |

The expected count should not include separate কার চিহ্ন.

---

## 13. Jukto Borno and Kar Chinho Handling

The model supports the classes available in the BanglaLekha-Isolated dataset and the provided `character_map.json`.

Some selected jukto borno classes are included, such as:

```text
ক্ষ, ব্দ, ঙ্গ, স্ক, স্ফ, স্থ, চ্ছ, ক্ত, ম্ন, ষ্ণ, ম্প, হ্ম, প্ত, ম্ব, ন্ড, দ্ভ, ত্থ, ষ্ঠ, ল্প, ষ্প, ন্দ, ন্ধ, ম্ম, ণ্ঠ
```

These should be treated as one recognition unit if they appear in a word.

The app does not separately recognize vowel signs or কার চিহ্ন such as:

```text
া, ি, ী, ু, ূ, ে, ৈ, ো, ৌ
```

This is because these signs were not trained as independent output classes in the current model. Therefore, the app focuses on base characters and selected jukto borno units.

---

## 14. Docker Usage

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

## 15. Screenshots

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

## 16. Limitations

* The model is trained on isolated character images, not full handwritten word images.
* Word recognition depends on successful segmentation.
* Fully connected Bangla handwriting may still be difficult to segment correctly.
* The app performs unit-by-unit recognition instead of using an end-to-end sequence model.
* The app does not separately recognize কার চিহ্ন as independent symbols.
* The segmentation strategy works best when base or jukto units have slight visual separation.
* The system is not a full production-level OCR engine.
* Recognition quality depends on handwriting clarity and canvas input quality.
* If a jukto borno is not present in the trained 84 classes, the model may not recognize it correctly.

---

## 17. Possible Improvements

* Train on actual Bangla word-level datasets.
* Use CRNN, CTC, or Transformer-based OCR for full word recognition.
* Add a dedicated word-level OCR model.
* Add support for separate vowel signs and কার চিহ্ন recognition.
* Improve segmentation for fully connected handwriting.
* Add uploaded image prediction.
* Add line segmentation and word segmentation for paragraph-level OCR.
* Improve character mapping validation using official dataset documentation.
* Deploy the app online using a cloud service.

---

## 18. Submission Contents

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

## 19. Suggested Commands

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

## 20. Final Notes

This project satisfies the main assignment requirements by training a Bangla handwritten character recognition model, tracking experiments with MLflow, providing a Streamlit drawing interface, attempting word recognition through matra-aware segmentation, displaying confidence scores, saving the trained model and label mapping, and supporting Docker-based execution.

The final system recognizes isolated Bangla characters, selected jukto borno, and simple Bangla word sequences by segmenting the drawn input into safe base/jukto units and combining the predictions.
