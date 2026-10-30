
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# 1. Page setup
st.set_page_config(
    page_title="SketchMatch",
    layout="centered"
)

st.title("Sketch - Match")
st.write("Sketch something and find similar matches!")


# 2. Import the model and search functions from search_core.py
from search_core import (
    load_feature_extractor,
    preprocess_query,
    embed,
    search
)


# 3. Load model once
@st.cache_resource
def load_model():
    return load_feature_extractor()


# 4. Load index once
@st.cache_resource
def load_index():
    embeddings = np.load("index/embeddings.npy")
    gallery_images = np.load("index/gallery_images.npy")

    with open("index/config.json", "r") as f:
        config = json.load(f)

    return embeddings, gallery_images, config


# 5. Choose input method
mode = st.radio(
    "Choose your input method:",
    ["Draw", "Upload", "Camera"],
    horizontal=True
)

query_image = None

# Changing the canvas key gives the user a fresh, empty canvas.
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0


def clear_canvas():
    st.session_state.canvas_version += 1


if mode == "Draw":
    st.subheader("Draw your doodle")

    draw_column, preview_column = st.columns([3, 2])

    with draw_column:
        canvas = st_canvas(
            stroke_width=10,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=350,
            width=350,
            drawing_mode="freedraw",
            update_streamlit=True,
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_version}"
        )

        if canvas.image_data is not None:
            query_image = np.asarray(
                canvas.image_data, dtype=np.uint8
            )

        st.button("Clear Canvas", on_click=clear_canvas)

    with preview_column:
        preview_placeholder = st.empty()

elif mode == "Upload":
    input_column, preview_column = st.columns([3, 2])

    with input_column:
        uploaded_file = st.file_uploader(
            "Upload your drawing",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            query_image = np.array(
                Image.open(uploaded_file).convert("RGBA")
            )

    with preview_column:
        preview_placeholder = st.empty()

elif mode == "Camera":
    input_column, preview_column = st.columns([3, 2])

    with input_column:
        camera_photo = st.camera_input(
            "Take a photo of your drawing"
        )

        if camera_photo is not None:
            query_image = np.array(
                Image.open(camera_photo).convert("RGBA")
            )

    with preview_column:
        preview_placeholder = st.empty()


# Find button: search only when clicked
left_space, find_column, right_space = st.columns([1, 2, 1])

with find_column:
    find_clicked = st.button(
        "🔍 Find",
        type="primary",
        use_container_width=True
    )


# 6. Process the drawing
if find_clicked and query_image is not None:

    processed_image = preprocess_query(query_image)

    if processed_image is None:
        st.warning("Please draw something first.")
        st.stop()

    processed_image = np.asarray(processed_image)

    # Display the processed image beside the input, only after Find is clicked.
    with preview_placeholder.container():
        st.write("Preprocessed Drawing (28×28)")
        st.image(processed_image, width=150, clamp=True)

    # 7. Check whether index files are available
    required_files = [
        "index/embeddings.npy",
        "index/gallery_images.npy",
        "index/config.json"
    ]

    missing = [
        f for f in required_files
        if not Path(f).exists()
    ]

    if missing:
        st.info(
            "Waiting for the index and threshold files."
        )
        st.stop()


    # 8. Load model and index
    try:
        model = load_model()
        embeddings, gallery_images, config = load_index()

        # Convert query to an embedding vector
        query_batch = np.expand_dims(
            processed_image, axis=0
        )

        query_vector = embed(model, query_batch, invert=True)[0]

        # Search for the top 25 matches
        indices, scores = search(
            query_vector,
            embeddings,
            k=25
        )

        indices = np.asarray(indices).reshape(-1)
        scores = np.asarray(scores).reshape(-1)

        threshold = float(config["threshold"])

        # Show all 25 nearest matches, or none if the best score is below the threshold
        if len(indices) < 25:
            st.error("The index returned fewer than 25 matches.")

        elif float(scores[0]) < threshold:
            st.warning("Nothing similar found")

        else:
            st.subheader("Top 25 Similar Doodles")

            # Display 25 matches, five columns per row
            for row in range(0, 25, 5):
                cols = st.columns(5)

                for col, position in enumerate(range(row, row + 5)):
                    image_index = int(indices[position])
                    score = float(scores[position])
                    image = gallery_images[image_index]
                    image = image.reshape(28, 28)

                    # Invert colors for display only
                    image = 255 - image

                    with cols[col]:
                        st.image(
                            image,
                            width=120,
                            clamp=True
                        )
                        st.caption(
                            f"Similarity: {score:.3f}"
                        )

    except Exception as e:
        st.error(f"Search could not run: {e}")

if find_clicked and query_image is None:
    st.warning("Please draw or upload an image first.")

# Dataset attribution
st.caption(
    "Dataset: Google Quick, Draw! — Data made available by Google, Inc. "
    "under the Creative Commons Attribution 4.0 International (CC BY 4.0) license."
)
