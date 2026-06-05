# Bangla Handwritten Word Recognition System

## Project Overview

This project implements a Bangla handwritten word recognition system using the BanglaLekha-Isolated dataset. The system trains a character-level image classifier and provides a Streamlit web interface where users can draw a Bangla word. The drawn word is segmented into individual character regions, each character is classified using the trained CNN model, and the predictions are combined to produce the final recognized sequence.

The project includes model training, MLflow experiment tracking, a Streamlit prediction UI, Docker support, trained model artifacts, and experiment screenshots.

## Dataset

Dataset used: BanglaLekha-Isolated from Mendeley Data.

The dataset contains isolated Bangla handwritten character images. Since the dataset is character-level, this project approaches word recognition by segmenting a drawn word into individual characters and classifying each character separately.

## Project Structure

```text
bangla-ocr-assignment/
│-- train.py
│-- app.py
│-- requirements.txt
│-- Dockerfile
│-- README.md
│-- labels.json
│-- github_link.txt
│-- mlflow.db
│
│-- models/
│   └-- model.keras
│
│-- artifacts/
│   ├-- training_curves.png
│   ├-- confusion_matrix.png
│   └-- classification_report.txt
│
│-- screenshots/
│   ├-- streamlit_app.png
│   └-- mlflow_experiment.png