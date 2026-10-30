# SketchMatch — Doodle Similarity Search

## Overview
SketchMatch is a Streamlit application that finds drawings similar to a user's sketch using the Google Quick, Draw! dataset and a pretrained MobileNetV2 model. It displays similar doodles instead of predicting a category label.

## Objective
Build an image similarity search engine using pretrained image embeddings and cosine similarity, without training or fine-tuning a model.

## Features
- Draw directly on the canvas, upload a PNG/JPG image, or take a photo with a camera.
- Clear the drawing canvas and search by clicking **Find**.
- Display the **top 25 similar doodles** with their similarity scores.
- Show **"Nothing similar found"** when the best match is below the similarity threshold.
- Keep category labels hidden from the app's results.

## Technologies Used
- Python
- Streamlit
- TensorFlow / Keras (MobileNetV2)
- NumPy
- OpenCV
- Pillow

## Dataset & Model
- **Dataset:** Google Quick, Draw! bitmap drawings (28 × 28 pixels).
- **Current gallery:** 4,050 drawings across 9 indexed categories.
- **Model:** MobileNetV2 pretrained on ImageNet, used only as a feature extractor.
- **Model input:** 128 × 128 pixels.
- **Embeddings:** 1,280 features per drawing, L2-normalized.
- **Search:** Cosine similarity with a rejection threshold stored in `index/config.json`.
- **Training:** No training or fine-tuning was performed.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `search_core.py` and the required files in the `index/` folder alongside `app.py`. The pretrained model weights may download on first use.

## Limitations
- Results are limited to the categories included in the indexed dataset.
- Similar-looking drawings may produce incorrect matches.
- Image quality, lighting, and line thickness can affect uploaded or camera-based drawings.
- MobileNetV2 was pretrained on natural images, not specifically on doodles.
- The similarity threshold may reject valid drawings or accept unrelated ones.

## Future Improvements
- Expand the number of indexed categories.
- Evaluate and refine the similarity threshold using held-out categories.
- Improve preprocessing for photographed drawings.

## Dataset Credit
Google Quick, Draw! data is made available by Google, Inc. under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
