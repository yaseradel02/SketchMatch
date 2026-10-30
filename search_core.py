
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SIZE = (128, 128)
DOODLE_SIZE = 28


# ============================================================
# 1. LOAD PRETRAINED FEATURE EXTRACTOR
# ============================================================

def load_feature_extractor():
    """
    Load pretrained MobileNetV2 as a feature extractor.

    No training or fine-tuning is performed.
    Each image produces a 1280-dimensional feature vector.
    """

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(128, 128, 3)
    )

    # Freeze pretrained weights
    model.trainable = False

    return model


# ============================================================
# 2. CONVERT STANDARDIZED DOODLES TO MODEL INPUT
# ============================================================

def to_model_input(images_28, invert=False):
    """
    Convert 28x28 grayscale drawings into MobileNetV2 inputs.

    Accepted:
        (28, 28)       - single drawing
        (N, 28, 28)    - batch of drawings

    Returns:
        Tensor with shape (N, 128, 128, 3)
        Pixel values scaled to approximately -1 to +1.
    """

    images = np.asarray(
        images_28,
        dtype=np.float32
    )

    # Add batch dimension for a single doodle
    if images.shape == (DOODLE_SIZE, DOODLE_SIZE):
        images = images[np.newaxis, ...]

    # Validate shape
    if (
        images.ndim != 3
        or images.shape[1:] != (DOODLE_SIZE, DOODLE_SIZE)
    ):
        raise ValueError(
            "Expected (28,28) or (N,28,28) grayscale doodles."
        )

    if images.shape[0] == 0:
        raise ValueError("No doodles provided.")

    # Validate pixel values
    if not np.isfinite(images).all():
        raise ValueError("Doodles contain NaN or infinity.")

    if images.min() < 0 or images.max() > 255:
        raise ValueError(
            "Doodle pixel values must be between 0 and 255."
        )

    # Add grayscale channel
    # (N, 28, 28) -> (N, 28, 28, 1)
    images = tf.convert_to_tensor(
        images[..., np.newaxis]
    )

    # Resize to MobileNetV2 input size
    # (N, 28, 28, 1) -> (N, 128, 128, 1)
    images = tf.image.resize(
        images,
        MODEL_SIZE,
        method="bilinear"
    )

    # Repeat grayscale into three channels
    # (N, 128, 128, 1) -> (N, 128, 128, 3)
    images = tf.repeat(
        images,
        repeats=3,
        axis=-1
    )

    # Optional color inversion experiment
    if invert:
        images = 255.0 - images

    # Official MobileNetV2 preprocessing
    # Converts 0..255 into -1..1
    images = preprocess_input(images)

    return images


# ============================================================
# 3. EXTRACT AND NORMALIZE EMBEDDINGS
# ============================================================

def embed(model, images_28, batch_size=64, invert=False):
    """
    Generate embeddings from one or more 28x28 doodles.

    Images are processed in small batches to save memory.

    Returns:
        (N, 1280) L2-normalized embeddings.
    """

    images = np.asarray(images_28)

    # Handle one image
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    # Validate input shape
    if (
        images.ndim != 3
        or images.shape[1:] != (28, 28)
    ):
        raise ValueError(
            "Expected (28,28) or (N,28,28) grayscale doodles."
        )

    if len(images) == 0:
        raise ValueError("No doodles provided.")

    batch_size = int(batch_size)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    all_vectors = []

    # Process the gallery in small batches
    for start in range(0, len(images), batch_size):

        batch = images[start:start + batch_size]

        # Apply shared model preprocessing
        model_inputs = to_model_input(
            batch,
            invert=invert
        )

        # Extract pretrained CNN features
        batch_vectors = model.predict(
            model_inputs,
            verbose=0
        )

        all_vectors.append(
            batch_vectors.astype(np.float32)
        )

    # Combine batch outputs
    vectors = np.concatenate(
        all_vectors,
        axis=0
    )

    # Calculate L2 norm of each vector
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    # Normalize vectors to length 1
    vectors = vectors / np.maximum(
        norms,
        1e-12
    )

    return vectors


# ============================================================
# 4. PREPROCESS A USER QUERY
# ============================================================

def preprocess_query(image):
    """
    Convert a canvas drawing, uploaded image, or camera photo
    into the Quick Draw bitmap format.

    Accepts:
        PIL Image
        NumPy grayscale image
        NumPy RGB image
        NumPy RGBA image

    For NumPy color images, channel order must be RGB/RGBA.

    Returns:
        (28, 28) uint8 image:
        white strokes on a black background.

        Returns None if the image is blank.
    """

    if image is None:
        return None

    # Convert PIL Image into NumPy
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    arr = np.asarray(image)

    # Ensure 8-bit pixel values
    if arr.dtype != np.uint8:
        arr = np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)

    # --------------------------------------------------------
    # STEP A: CONVERT TO GRAYSCALE
    # --------------------------------------------------------

    if arr.ndim == 2:

        # Already grayscale
        gray = arr

    elif arr.ndim == 3 and arr.shape[-1] == 1:

        gray = arr[..., 0]

    elif arr.ndim == 3 and arr.shape[-1] == 3:

        # RGB -> Grayscale
        gray = cv2.cvtColor(
            arr,
            cv2.COLOR_RGB2GRAY
        )

    elif arr.ndim == 3 and arr.shape[-1] == 4:

        # RGBA canvas:
        # Flatten transparent pixels onto white first
        alpha = (
            arr[..., 3:4].astype(np.float32) / 255.0
        )

        rgb = (
            arr[..., :3].astype(np.float32) * alpha
            + 255.0 * (1.0 - alpha)
        )

        gray = cv2.cvtColor(
            rgb.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

    else:

        raise ValueError(
            "Expected grayscale, RGB, or RGBA image."
        )

    # --------------------------------------------------------
    # STEP B: DETECT BLANK INPUT
    # --------------------------------------------------------

    if gray.size == 0 or min(gray.shape) < 2:
        return None

    # Almost uniform image means no meaningful drawing
    if int(gray.max()) - int(gray.min()) < 15:
        return None

    # Reduce small pixel noise
    if min(gray.shape) >= 32:

        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

    else:

        blurred = gray

    # --------------------------------------------------------
    # STEP C: DETERMINE BACKGROUND COLOR
    # --------------------------------------------------------

    # Estimate background from border pixels
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    light_background = np.median(border) >= 128

    # --------------------------------------------------------
    # STEP D: SEPARATE STROKES FROM BACKGROUND
    # --------------------------------------------------------

    if light_background:

        # Dark strokes on white paper
        if min(gray.shape) >= 100:

            # Adaptive threshold handles uneven lighting
            mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                12
            )

        else:

            # Otsu threshold for smaller images
            _, mask = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV
                + cv2.THRESH_OTSU
            )

    else:

        # White strokes on black background
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

    # --------------------------------------------------------
    # STEP E: REMOVE SMALL NOISE
    # --------------------------------------------------------

    component_count, component_ids, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    min_area = max(
        2,
        int(gray.size * 0.00002)
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, component_count):

        component_area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if component_area >= min_area:

            cleaned[component_ids == label] = 255

    # --------------------------------------------------------
    # STEP F: FIND DRAWING BOUNDING BOX
    # --------------------------------------------------------

    coords = cv2.findNonZero(cleaned)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the drawing
    cropped = cleaned[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # STEP G: RESIZE WHILE PRESERVING ASPECT RATIO
    # --------------------------------------------------------

    # Leave a two-pixel margin on each side
    max_content = 24

    scale = max_content / max(w, h)

    new_width = max(
        1,
        int(round(w * scale))
    )

    new_height = max(
        1,
        int(round(h * scale))
    )

    # INTER_AREA is suited for shrinking images
    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=interpolation
    )

    # --------------------------------------------------------
    # STEP H: CENTER ON 28x28 BLACK BACKGROUND
    # --------------------------------------------------------

    result = np.zeros(
        (DOODLE_SIZE, DOODLE_SIZE),
        dtype=np.uint8
    )

    left = (DOODLE_SIZE - new_width) // 2
    top = (DOODLE_SIZE - new_height) // 2

    result[
        top:top + new_height,
        left:left + new_width
    ] = resized

    # --------------------------------------------------------
    # STEP I: THICKEN VERY THIN STROKES IF NECESSARY
    # --------------------------------------------------------

    if (
        max(gray.shape) >= 100
        and np.count_nonzero(result > 30) < 20
    ):

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8
        )

        result = cv2.dilate(
            result,
            kernel,
            iterations=1
        )

    return result


# ============================================================
# 5. COSINE SIMILARITY SEARCH
# ============================================================

def search(query_vector, embeddings, k=25):
    """
    Find the k most similar gallery doodles.

    Both query and gallery vectors must be L2-normalized.

    Returns:
        indices: gallery positions
        scores: cosine similarity scores
    """

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    ).reshape(-1)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Validate gallery
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(
            "Embeddings must be a nonempty 2D array."
        )

    # Validate embedding dimensions
    if embeddings.shape[1] != query_vector.shape[0]:
        raise ValueError(
            "Embedding dimensions do not match."
        )

    if (
        not np.isfinite(query_vector).all()
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(
            "Embeddings must contain finite numbers."
        )

    # Cosine similarity = dot product
    # because embeddings are L2-normalized
    scores = embeddings @ query_vector

    # Ensure k is valid
    k = min(
        max(int(k), 1),
        len(scores)
    )

    # Retrieve top-k indices efficiently
    indices = np.argpartition(
        -scores,
        k - 1
    )[:k]

    # Sort highest similarity first
    indices = indices[
        np.argsort(scores[indices])[::-1]
    ]

    return indices, scores[indices]
