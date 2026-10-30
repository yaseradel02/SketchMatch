# SketchMatch — Findings

> **Review status (30 October 2026):** This document summarizes what can be verified from the files committed to the `main` branch. Items labeled **Not measured / To complete** are deliberately not presented as experimental results.

## 1. Project Objective

SketchMatch is an image-retrieval application using Google's Quick, Draw! bitmap dataset. Users draw on a Streamlit canvas, upload a PNG/JPG, or photograph a paper drawing. The system returns the **25 closest gallery doodles** and their similarity scores, or **"Nothing similar found"** when the best result is below the configured threshold. The application does **not** show category predictions or labels.

## 2. Dataset Preparation

**Source:** Google Quick, Draw! NumPy bitmap drawings, originally 28 × 28 pixels.

The committed data-preparation notebook (`notebooks/prepare_data.ipynb`) documents a fixed random seed of **42** and the following sampling scheme:

| Split | Number of categories | Images per category | Total |
|---|---:|---:|---:|
| Indexed gallery | 9 | 450 | **4,050** |
| In-index queries | 9 | 50 | **450** |
| Held-out queries | 2 | 50 | **100** |

**Indexed categories:** aircraft carrier, airplane, ambulance, angel, ant, anvil, basket, basketball, bed.

**Held-out categories:** alarm clock, animal migration.

The gallery is balanced at **450 doodles per indexed category**, based on the notebook's saved output. The `index/config.json` file likewise records `gallery_size: 4050`.

**Assignment gap:** The lab requires **10–15 indexed categories**, but this saved index contains **9**. A tenth category must be sampled, the gallery index rebuilt, and evaluation repeated. The `categories.txt` file lists candidate category names; its entries are not evidence that those categories are currently indexed.

## 3. Model Choice and Feature Extraction

The project uses **MobileNetV2 with ImageNet pretrained weights**, `include_top=False`, and global average pooling. Its weights are frozen; **no training or fine-tuning** is performed.

| Property | Implementation |
|---|---|
| Feature extractor | MobileNetV2 (Keras / TensorFlow) |
| Model input | 128 × 128 × 3 |
| Feature vector | 1,280 values per image |
| Gallery preprocessing | Resize 28 × 28 grayscale images, repeat to RGB, **invert colors**, apply MobileNetV2 `preprocess_input` |
| Embedding normalization | L2 normalization |
| Similarity | Cosine similarity, computed as a dot product between normalized embeddings |
| Gallery embeddings | `index/embeddings.npy` |
| Gallery images | `index/gallery_images.npy` |

**Reason for selection:** MobileNetV2 is designed as a relatively lightweight image model, making it a practical pretrained feature extractor for a CPU-based demonstration. Global average pooling produces a compact fixed-length representation suitable for similarity search. **Actual CPU embedding time and runtime memory use have not been benchmarked in the committed files**, so no numeric speed claim is made.

The current `index/config.json` specifies **`invert: true`**, and `app.py` matches this by calling `embed(model, query_batch, invert=True)`. Inverting result images for *display* does not change their stored embeddings.

## 4. Search Quality — Precision@k

The lab requires evaluating all in-index query doodles against the gallery and calculating mean **Precision@5** and **Precision@25**, where a retrieved doodle is relevant when its category equals the query's category. It also requires a **raw-pixel baseline**, using flattened, L2-normalized 784-pixel vectors.

| Retrieval method | Mean Precision@5 | Mean Precision@25 |
|---|---|---|
| Raw pixels (baseline) | **Not measured** | **Not measured** |
| MobileNetV2, inverted colors (deployed index) | **Not measured** | **Not measured** |
| MobileNetV2, normal colors (optional comparison) | **Not measured** | **Not measured** |

**Important:** The repository did not contain `evaluate.py` or saved precision results at review time. These table entries must be filled by actually running the evaluation on the saved query/gallery split; they must not be inferred from visually plausible search results. The optional normal-color experiment should use a separately built index and consistent query preprocessing.

## 5. Similarity Threshold and Unknown-Category Rejection

The currently committed `index/config.json` contains:

```json
{
  "model_name": "MobileNetV2",
  "gallery_size": 4050,
  "embedding_dim": 1280,
  "invert": true,
  "top_k": 25,
  "threshold": 0.75
}
```

The app checks the **highest** cosine similarity score. If it is **below 0.75**, the app displays **"Nothing similar found"**. Otherwise, it displays **all 25** nearest doodles, including individual results whose scores may fall below 0.75.

**Threshold evaluation required:**

| Measurement | Result |
|---|---|
| Best-score distribution for 450 indexed queries | **Not measured** |
| Best-score distribution for 100 held-out queries | **Not measured** |
| In-index queries wrongly rejected at 0.75 | **Not measured** |
| Held-out queries wrongly accepted at 0.75 | **Not measured** |
| Evidence supporting selection of 0.75 | **Not documented** |

Person 1 should compute the two top-1 score distributions, plot their histograms, report the false rejection/acceptance counts, and explain the trade-off in the chosen threshold. A configured number alone does not establish that the threshold is calibrated.

## 6. Visual Quality Review

The lab requests **three good** and **three bad** query-result examples, with a one-line explanation of each bad case. The repository contains `results.png`, but the image's contents and six distinct examples were not independently verified in this review.

| Example | Query / observed results | Assessment |
|---|---|---|
| Good 1 | **To complete from actual testing** | — |
| Good 2 | **To complete from actual testing** | — |
| Good 3 | **To complete from actual testing** | — |
| Bad 1 | **To complete from actual testing** | **Cause to document** |
| Bad 2 | **To complete from actual testing** | **Cause to document** |
| Bad 3 | **To complete from actual testing** | **Cause to document** |

Potential explanations to investigate **only when supported by a test** include lookalike shapes, preprocessing errors, an unsuitable threshold, or limitations of an ImageNet-pretrained model on sketches.

## 7. Streamlit App and Task 6 Testing

**Implemented in `app.py`:**
- Draw canvas, file upload and camera input.
- **Find** button, Clear Canvas button and cached model/index loading.
- Shared query preprocessing through `preprocess_query()`.
- Processed 28 × 28 preview beside the drawing canvas **after clicking Find**.
- Top 25 results (five columns per row), similarity scores and threshold rejection.
- Category labels hidden from the user interface.
- Dataset attribution to Google, Inc., under CC BY 4.0.

**Task 6 testing evidence still required:**
1. **Drawing race:** record the tested indexed category and whether at least 13 of 25 results were visually appropriate.
2. **Fool the search:** record a high-scoring but incorrect match, including the drawing and returned result.
3. **Held-out and out-of-index tests:** record what happens for a held-out drawing, initials or a squiggle.
4. **Three actual failures:** for each, write down the input, what returned and the likely cause. **Not yet documented.**

**Differences from the lab specification:** Results currently appear on clicking **Find**, not automatically while drawing. The processed-image preview currently appears in **Draw** mode only, rather than next to uploaded/captured images as the lab requests. These differences should be resolved or approved before grading.

## 8. Limitations

- Retrieval can only draw from the categories actually included in the index.
- Doodles with similar shapes can be confused; similarity scores are not probabilities of a correct category.
- The pretrained model learned from natural images, not Quick, Draw! doodles.
- Camera lighting, shadows and faint strokes may change preprocessing quality.
- An unknown drawing may receive a high similarity score; threshold errors need to be quantified.
- Current evaluation metrics and threshold error rates have not been published.

## 9. Reproducibility and Remaining Deliverables

Current repository files include `app.py`, `search_core.py`, `build_index.py`, `data/subset.npz`, the saved gallery/index, `index/config.json`, `notebooks/prepare_data.ipynb`, `requirements.txt`, `README.md` and `results.png`.

**Not found at the time of this review:** the assignment's required `prepare_data.py` and `evaluate.py` scripts.

To rebuild the current *inverted* index after updating the dataset:

```bash
python build_index.py --invert
```

**Caution:** `build_index.py` writes `"threshold": null` when rebuilding. Re-evaluate and set a numerical threshold in `index/config.json` before launching the app again.

## 10. Conclusion

The repository contains a functioning Streamlit similarity-search implementation built around an ImageNet-pretrained MobileNetV2 feature extractor and cosine similarity. It uses a 4,050-image gallery, currently configured with an inverted-color preprocessing path and a top-1 similarity cutoff of 0.75.

**A quantitative claim that MobileNetV2 outperforms raw pixels, or that 0.75 reliably rejects unknown drawings, cannot yet be made** without running and reporting the prescribed precision, threshold and visual evaluations.

---

**Data credit:** Google Quick, Draw! data made available by Google, Inc. under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
