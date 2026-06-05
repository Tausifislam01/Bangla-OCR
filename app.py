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


def make_square_image(character_image, padding=30):
    h, w = character_image.shape

    square_size = max(w, h) + padding
    square_image = np.zeros((square_size, square_size), dtype=np.uint8)

    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2

    square_image[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = character_image

    return square_image


def crop_to_ink(image):
    coords = cv2.findNonZero(image)

    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)

    if w < 4 or h < 4:
        return None, None

    cropped = image[y:y + h, x:x + w]
    box = (x, y, w, h)

    return cropped, box


def get_gap_ranges(gap_columns, minimum_gap_width=3):
    gap_ranges = []
    start = None

    for i, is_gap in enumerate(gap_columns):
        if is_gap and start is None:
            start = i
        elif not is_gap and start is not None:
            if i - start >= minimum_gap_width:
                gap_ranges.append((start, i))
            start = None

    if start is not None and len(gap_columns) - start >= minimum_gap_width:
        gap_ranges.append((start, len(gap_columns)))

    return gap_ranges


def estimate_matra_band(word_image):
    h, w = word_image.shape

    if h < 20 or w < 20:
        return None

    horizontal_projection = np.sum(word_image > 0, axis=1).astype(np.float32)

    if np.max(horizontal_projection) == 0:
        return None

    upper_limit = max(1, int(h * 0.45))
    upper_projection = horizontal_projection[:upper_limit]

    matra_y = int(np.argmax(upper_projection))
    matra_strength = upper_projection[matra_y]

    if matra_strength < 0.25 * w:
        return None

    band_top = max(0, matra_y - 2)
    band_bottom = min(h, matra_y + 5)

    return band_top, band_bottom


def remove_matra_for_segmentation(word_image):
    segmentation_image = word_image.copy()

    matra_band = estimate_matra_band(word_image)

    if matra_band is None:
        return segmentation_image, None

    band_top, band_bottom = matra_band

    segmentation_image[band_top:band_bottom, :] = 0

    return segmentation_image, matra_band


def safe_vertical_cuts_from_projection(segmentation_image, expected_count=None):
    h, w = segmentation_image.shape

    if w < 25:
        return []

    projection = np.sum(segmentation_image > 0, axis=0).astype(np.float32)

    if len(projection) == 0 or np.max(projection) == 0:
        return []

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    smooth_projection = np.convolve(projection, kernel, mode="same")

    max_projection = np.max(smooth_projection)

    gap_threshold = max(1, 0.10 * max_projection)
    gap_columns = smooth_projection <= gap_threshold

    gap_ranges = get_gap_ranges(
        gap_columns,
        minimum_gap_width=3
    )

    candidate_cuts = []

    for gap_start, gap_end in gap_ranges:
        midpoint = (gap_start + gap_end) // 2

        if 8 < midpoint < w - 8:
            candidate_cuts.append(midpoint)

    candidate_cuts = sorted(candidate_cuts)

    if expected_count is not None and expected_count <= 1:
        return []

    if expected_count is None:
        return candidate_cuts

    required_cuts = expected_count - 1

    if required_cuts <= 0:
        return []

    if len(candidate_cuts) <= required_cuts:
        return candidate_cuts

    selected_cuts = []

    for i in range(1, expected_count):
        ideal_cut = int((w * i) / expected_count)

        available_cuts = [
            cut for cut in candidate_cuts
            if cut not in selected_cuts
        ]

        if len(available_cuts) == 0:
            break

        nearest_cut = min(
            available_cuts,
            key=lambda cut: abs(cut - ideal_cut)
        )

        max_distance = w / expected_count * 0.55

        if abs(nearest_cut - ideal_cut) <= max_distance:
            selected_cuts.append(nearest_cut)

    return sorted(selected_cuts)


def build_segments_from_cuts(word_image, cut_points, word_x, word_y):
    h, w = word_image.shape

    raw_segments = []
    previous_x = 0

    for cut_x in cut_points + [w]:
        segment_x1 = previous_x
        segment_x2 = cut_x
        previous_x = cut_x

        if segment_x2 - segment_x1 < 6:
            continue

        segment = word_image[:, segment_x1:segment_x2]

        cropped, local_box = crop_to_ink(segment)

        if cropped is None:
            continue

        sx, sy, sw, sh = local_box

        if sw < 4 or sh < 6:
            continue

        ink_area = cv2.countNonZero(cropped)

        if ink_area < 15:
            continue

        global_box = (
            word_x + segment_x1 + sx,
            word_y + sy,
            sw,
            sh
        )

        raw_segments.append((cropped, global_box))

    raw_segments = sorted(raw_segments, key=lambda item: item[1][0])

    return raw_segments


def whole_image_as_single_segment(binary):
    cropped, box = crop_to_ink(binary)

    if cropped is None:
        return []

    x, y, w, h = box

    if w < 4 or h < 4:
        return []

    return [(cropped, box)]


def matra_aware_segmentation(binary, expected_count=None):
    coords = cv2.findNonZero(binary)

    if coords is None:
        return []

    word_x, word_y, word_w, word_h = cv2.boundingRect(coords)

    if word_w < 4 or word_h < 4:
        return []

    word_image = binary[word_y:word_y + word_h, word_x:word_x + word_w]

    if expected_count is not None and expected_count <= 1:
        return [(word_image, (word_x, word_y, word_w, word_h))]

    segmentation_image, matra_band = remove_matra_for_segmentation(word_image)

    cut_points = safe_vertical_cuts_from_projection(
        segmentation_image,
        expected_count=expected_count
    )

    raw_segments = build_segments_from_cuts(
        word_image,
        cut_points,
        word_x,
        word_y
    )

    if len(raw_segments) == 0:
        raw_segments = [(word_image, (word_x, word_y, word_w, word_h))]

    return raw_segments


def contour_based_segmentation(binary):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if w >= 6 and h >= 8 and area >= 20:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda item: item[0])

    raw_segments = []

    for x, y, w, h in boxes:
        character = binary[y:y + h, x:x + w]
        raw_segments.append((character, (x, y, w, h)))

    return raw_segments


def is_matra_only_segment(segment, box):
    x, y, w, h = box
    ink_area = cv2.countNonZero(segment)

    if w == 0 or h == 0:
        return True

    aspect_ratio = w / h

    if h <= 14 and aspect_ratio >= 2.0:
        return True

    if ink_area < 30:
        return True

    horizontal_projection = np.sum(segment > 0, axis=1)

    if len(horizontal_projection) > 0:
        max_row_ink = np.max(horizontal_projection)

        if max_row_ink > 0.60 * w and h <= 18:
            return True

    return False


def merge_tiny_components(segments):
    if len(segments) <= 1:
        return segments

    cleaned_segments = []

    for segment, box in segments:
        x, y, w, h = box
        area = cv2.countNonZero(segment)

        if w < 5 or h < 5 or area < 20:
            continue

        if is_matra_only_segment(segment, box):
            continue

        cleaned_segments.append((segment, box))

    if len(cleaned_segments) == 0:
        return segments

    return sorted(cleaned_segments, key=lambda item: item[1][0])


def segment_characters(canvas_image, expected_count=None, use_expected_count=True):
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

    effective_expected_count = expected_count if use_expected_count else None

    if effective_expected_count is not None and effective_expected_count <= 1:
        raw_segments = whole_image_as_single_segment(binary)
    else:
        raw_segments = matra_aware_segmentation(
            binary,
            expected_count=effective_expected_count
        )

    if len(raw_segments) == 0:
        raw_segments = contour_based_segmentation(binary)

    raw_segments = merge_tiny_components(raw_segments)

    raw_segments = sorted(raw_segments, key=lambda item: item[1][0])

    character_images = []
    boxes = []

    for cropped_character, box in raw_segments:
        square_image = make_square_image(cropped_character, padding=30)
        character_images.append(square_image)
        boxes.append(box)

    return character_images, boxes, binary


def predict_character(model, labels, character_map, character_image):
    processed_image = preprocess_character(character_image)

    if processed_image is None:
        return None

    probabilities = model.predict(processed_image, verbose=0)[0]

    best_index = int(np.argmax(probabilities))

    dataset_class_id = labels.get(str(best_index), str(best_index))

    bangla_character = character_map.get(
        str(dataset_class_id),
        str(dataset_class_id)
    )

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
        "Draw a Bangla character, jukto borno, or word on the canvas. "
        "The system segments the drawing into safe base/jukto units, predicts each "
        "unit using the trained CNN model, and combines the predictions."
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

        st.header("Segmentation Settings")

        use_expected_count = st.checkbox(
            "Use expected base/jukto unit count",
            value=True,
            help=(
                "Turn this on when you know how many base/jukto units are in the word. "
                "For a single character or single jukto borno, set the count to 1."
            )
        )

        expected_count = st.number_input(
            "Expected number of base/jukto units",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help=(
                "Single character = 1. "
                "ওজন = 3 units: ও + জ + ন. "
                "ক্ষমা = 2 units if ignoring আ-কার: ক্ষ + ম."
            )
        )

        st.header("Drawing Tips")
        st.write("- For a single character, set expected count to 1.")
        st.write("- For a single jukto borno, set expected count to 1.")
        st.write("- For ওজন, set expected count to 3: ও + জ + ন.")
        st.write("- For ক্ষমা, set expected count to 2: ক্ষ + ম, ignoring আ-কার.")
        st.write("- Count jukto borno as one unit if it exists in the trained classes.")
        st.write("- Do not count কার চিহ্ন separately.")
        st.write("- Keep slight spacing between base/jukto units for word recognition.")
        st.write("- Fully connected handwriting may still be difficult without a word-level OCR model.")
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

        recognize_button = st.button("Recognize", type="primary")

    with col2:
        st.subheader("Output")
        st.write("The app will display:")
        st.write("1. Preprocessed binary image")
        st.write("2. Segmented base/jukto unit images")
        st.write("3. Bangla output")
        st.write("4. Dataset class sequence")
        st.write("5. Per-unit confidence scores")
        st.write("6. Top-3 predictions")

    if recognize_button:
        if canvas_result.image_data is None:
            st.warning("Please draw a Bangla character, jukto borno, or word first.")
            return

        canvas_image = canvas_result.image_data.astype(np.uint8)

        character_images, boxes, binary_image = segment_characters(
            canvas_image,
            expected_count=expected_count,
            use_expected_count=use_expected_count
        )

        if len(character_images) == 0:
            st.warning("No character detected. Please draw larger and darker.")
            return

        st.subheader("Preprocessed Canvas")
        st.image(binary_image, caption="Binary image after preprocessing", width=500)

        if use_expected_count and len(character_images) != expected_count:
            st.warning(
                f"Expected {expected_count} base/jukto unit(s), but safely detected "
                f"{len(character_images)}. The app avoided unsafe cuts through connected characters."
            )

        predicted_class_sequence = []
        predicted_bangla_sequence = ""
        prediction_rows = []

        st.subheader("Segmented Base/Jukto Units")

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

            predicted_class_sequence.append(str(class_id))
            predicted_bangla_sequence += str(bangla_character)

            prediction_rows.append({
                "Unit No.": index + 1,
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
                    caption=f"Unit {index + 1}: {bangla_character}",
                    width=110
                )

        st.subheader("Recognized Bangla Output")

        st.write("Bangla Output:")
        st.success(predicted_bangla_sequence)

        st.write("Predicted Dataset Class Sequence:")
        st.info(" → ".join(predicted_class_sequence))

        st.subheader("Confidence Scores")
        st.dataframe(prediction_rows, use_container_width=True)

        st.info(
            "Note: The model was trained on isolated Bangla characters and selected "
            "jukto borno classes. The app uses matra-aware safe segmentation and removes "
            "headline-only fragments. It does not separately recognize কার চিহ্ন as "
            "independent symbols."
        )


if __name__ == "__main__":
    main()