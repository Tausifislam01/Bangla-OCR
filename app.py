import json
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas


IMAGE_SIZE = 64
MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"
CHARACTER_MAP_PATH = "character_map.json"


st.set_page_config(
    page_title="Bangla Handwritten Word Recognition",
    page_icon="✍️",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found: {LABELS_PATH}")
        st.stop()

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_character_map():
    if not os.path.exists(CHARACTER_MAP_PATH):
        return {}

    with open(CHARACTER_MAP_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    mapping = {}

    for item in data.get("bengali_character_sheet", []):
        class_id = str(item["id"])
        character = item["character"]
        mapping[class_id] = character

    return mapping


def preprocess_character(character_image):
    if character_image is None or character_image.size == 0:
        return None

    character_image = cv2.resize(
        character_image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    character_image = character_image.astype("float32") / 255.0
    character_image = np.expand_dims(character_image, axis=-1)
    character_image = np.expand_dims(character_image, axis=0)

    return character_image


def segment_characters(canvas_image):
    gray = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)

    _, binary = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    noise_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel)

    dilate_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w >= 8 and h >= 8:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda item: item[0])

    character_images = []

    for x, y, w, h in boxes:
        character = binary[y:y + h, x:x + w]

        square_size = max(w, h) + 30
        square_image = np.zeros((square_size, square_size), dtype=np.uint8)

        x_offset = (square_size - w) // 2
        y_offset = (square_size - h) // 2

        square_image[
            y_offset:y_offset + h,
            x_offset:x_offset + w
        ] = character

        character_images.append(square_image)

    return character_images, boxes, binary


def predict_character(model, labels, character_map, character_image):
    processed_image = preprocess_character(character_image)

    if processed_image is None:
        return None

    probabilities = model.predict(processed_image, verbose=0)[0]

    best_index = int(np.argmax(probabilities))

    # labels.json maps model output index to dataset folder ID, for example:
    # "41" -> "42"
    dataset_class_id = labels.get(str(best_index), str(best_index))

    # character_map.json maps dataset folder ID to Bangla Unicode character, for example:
    # "42" -> "স"
    bangla_character = character_map.get(str(dataset_class_id), str(dataset_class_id))

    confidence = float(probabilities[best_index])

    top_indices = probabilities.argsort()[-3:][::-1]

    top_predictions = []

    for index in top_indices:
        index = int(index)

        top_class_id = labels.get(str(index), str(index))
        top_character = character_map.get(str(top_class_id), str(top_class_id))

        top_predictions.append({
            "class_id": top_class_id,
            "character": top_character,
            "confidence": float(probabilities[index])
        })

    return dataset_class_id, bangla_character, confidence, top_predictions


def main():
    st.title("Bangla Handwritten Word Recognition System")

    st.write(
        "Draw a Bangla word or separated Bangla characters on the canvas. "
        "The system segments the drawing into character regions, predicts each "
        "character using the trained CNN model, and combines the predictions."
    )

    model = load_model()
    labels = load_labels()
    character_map = load_character_map()

    with st.sidebar:
        st.header("Model Information")
        st.write("Model: Improved Custom CNN")
        st.write("Dataset: BanglaLekha-Isolated")
        st.write("Input size: 64 × 64 grayscale")
        st.write(f"Number of classes: {len(labels)}")
        st.write("Test accuracy: 95.27%")

        if len(character_map) > 0:
            st.write(f"Character mapping loaded: {len(character_map)} classes")
        else:
            st.write("Character mapping not found. Showing numeric class IDs.")

        st.header("Drawing Tips")
        st.write("- Draw clearly using black stroke.")
        st.write("- Keep small spacing between characters.")
        st.write("- Draw larger if segmentation fails.")
        st.write("- For connected Bangla words, segmentation may be imperfect.")
        st.write("- Refresh the page to clear the canvas.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Drawing Canvas")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=12,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=760,
            drawing_mode="freedraw",
            key="canvas"
        )

        recognize_button = st.button("Recognize Word", type="primary")

    with col2:
        st.subheader("Output")
        st.write("The app will display:")
        st.write("1. Segmented character images")
        st.write("2. Bangla character output")
        st.write("3. Dataset class sequence")
        st.write("4. Per-character confidence scores")
        st.write("5. Top-3 predictions")

    if recognize_button:
        if canvas_result.image_data is None:
            st.warning("Please draw a Bangla word or character first.")
            return

        canvas_image = canvas_result.image_data.astype(np.uint8)

        character_images, boxes, binary_image = segment_characters(canvas_image)

        if len(character_images) == 0:
            st.warning("No character detected. Please draw larger and darker characters.")
            return

        st.subheader("Preprocessed Canvas")
        st.image(binary_image, caption="Binary image after preprocessing", width=500)

        predicted_class_sequence = ""
        predicted_bangla_sequence = ""
        prediction_rows = []

        st.subheader("Segmented Characters")

        preview_columns = st.columns(min(len(character_images), 6))

        for index, character_image in enumerate(character_images):
            prediction_result = predict_character(
                model,
                labels,
                character_map,
                character_image
            )

            if prediction_result is None:
                continue

            class_id, bangla_character, confidence, top_predictions = prediction_result

            predicted_class_sequence += str(class_id)
            predicted_bangla_sequence += str(bangla_character)

            prediction_rows.append({
                "Character No.": index + 1,
                "Bangla Prediction": bangla_character,
                "Dataset Class ID": class_id,
                "Confidence": f"{confidence:.2%}",
                "Top 3 Predictions": ", ".join(
                    [
                        f"{item['character']} / Class {item['class_id']} ({item['confidence']:.2%})"
                        for item in top_predictions
                    ]
                )
            })

            with preview_columns[index % len(preview_columns)]:
                st.image(
                    character_image,
                    caption=f"Char {index + 1}: {bangla_character}",
                    width=110
                )

        st.subheader("Recognized Bangla Word / Character Sequence")

        st.write("Bangla Output:")
        st.success(predicted_bangla_sequence)

        st.write("Predicted Dataset Class Sequence:")
        st.info(predicted_class_sequence)

        st.subheader("Confidence Scores")
        st.dataframe(prediction_rows, use_container_width=True)

        st.info(
            "Note: The model was trained on isolated Bangla character images. "
            "Word recognition is performed by segmenting the drawn word into "
            "characters and predicting each character separately."
        )


if __name__ == "__main__":
    main()