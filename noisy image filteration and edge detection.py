# ============================================================
# DIGITAL IMAGE PROCESSING LAB
# NOISE + SPATIAL FILTERS + EDGE DETECTION
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Program Started...")


# ============================================================
# STEP 1: LOAD IMAGE (GRAYSCALE)
# ============================================================

# cv2.imread(path, 0) → load grayscale image
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# Resize image → make processing faster and consistent
img = cv2.resize(img, (256, 256))

# Convert to float → safer mathematical operations
img = img.astype(np.float32)


# ============================================================
# STEP 2: PADDING FUNCTION
# ============================================================
# Padding adds border pixels around image.
# Needed for convolution near edges.

def pad(image, k=3):
    p = k // 2  # amount of padding based on kernel size
    return np.pad(image, p, mode='edge')
    # mode='edge' → replicate boundary pixels


# ============================================================
# STEP 3: SPATIAL FILTERS (Noise Removal)
# ============================================================

# ------------------------------------------------------------
# 1. Min Filter → removes salt noise (bright noise)
# ------------------------------------------------------------
def min_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.min(padded[i:i+k, j:j+k])

    return out


# ------------------------------------------------------------
# 2. Max Filter → removes pepper noise (dark noise)
# ------------------------------------------------------------
def max_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.max(padded[i:i+k, j:j+k])

    return out


# ------------------------------------------------------------
# 3. Median Filter → best for salt & pepper noise
# ------------------------------------------------------------
def median_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.median(padded[i:i+k, j:j+k])

    return out


# ------------------------------------------------------------
# 4. Arithmetic Mean Filter → smoothing / averaging
# ------------------------------------------------------------
def arithmetic_mean(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.mean(padded[i:i+k, j:j+k])

    return out


# ------------------------------------------------------------
# 5. Geometric Mean Filter → preserves detail better
# ------------------------------------------------------------
def geometric_mean(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+k, j:j+k] + 1  # avoid log(0)
            out[i, j] = np.exp(np.mean(np.log(region)))

    return out


# ============================================================
# STEP 4: EDGE DETECTION FILTERS
# ============================================================

# ------------------------------------------------------------
# Sobel Edge Detector → detects gradient edges
# ------------------------------------------------------------
def sobel_edge(image):

    # Horizontal gradient kernel
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Vertical gradient kernel
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    padded = pad(image, 3)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]

            # Gradient magnitude
            out[i, j] = np.sqrt((region*gx).sum()**2 +
                                (region*gy).sum()**2)

    return out


# ------------------------------------------------------------
# Laplacian Edge Detector → detects rapid intensity change
# ------------------------------------------------------------
def laplacian_edge(image):

    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    padded = pad(image, 3)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)

    return np.abs(out)


# ============================================================
# STEP 5: LIST OF FILTERS
# ============================================================

filters = [
    ("Min", min_filter),
    ("Max", max_filter),
    ("Median", median_filter),
    ("Arithmetic", arithmetic_mean),
    ("Geometric", geometric_mean)
]


# ============================================================
# STEP 6: NOISE FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# Gaussian Noise → random normal distribution noise
# ------------------------------------------------------------
def gaussian_noise(image, mean=0, var=20):
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    return np.clip(image + noise, 0, 255)


# ------------------------------------------------------------
# Salt & Pepper Noise → random white & black pixels
# ------------------------------------------------------------
def salt_and_pepper_noise(image, p=0.02):
    out = image.copy()
    rand = np.random.rand(*image.shape)

    out[rand < p] = 0       # pepper
    out[rand > 1 - p] = 255 # salt

    return out


noises = [
    ("Gaussian Noise", gaussian_noise),
    ("Salt & Pepper Noise", salt_and_pepper_noise)
]


# ============================================================
# STEP 7: APPLY NOISE → FILTER → EDGE DETECTION
# ============================================================

for noise_name, noise_func in noises:

    # Add noise to original image
    noise_image = noise_func(img)

    # Create figure
    plt.figure(figsize=(20, 10))

    total_cols = len(filters) + 2

    # --------------------------------------------------------
    # Show Original Image
    # --------------------------------------------------------
    plt.subplot(3, total_cols, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # --------------------------------------------------------
    # Show Noisy Image
    # --------------------------------------------------------
    plt.subplot(3, total_cols, 2)
    plt.imshow(noise_image, cmap='gray')
    plt.title(noise_name)
    plt.axis('off')

    # --------------------------------------------------------
    # Apply each filter
    # --------------------------------------------------------
    for i, (name, func) in enumerate(filters):

        filtered = func(noise_image)
        filtered = np.clip(filtered, 0, 255)

        # ----- Filtered Image -----
        plt.subplot(3, total_cols, i + 3)
        plt.imshow(filtered, cmap='gray')
        plt.title(f"{name} Filter")
        plt.axis('off')

        # ----- Sobel Edge -----
        sobel = np.clip(sobel_edge(filtered), 0, 255)
        plt.subplot(3, total_cols, total_cols + i + 3)
        plt.imshow(sobel, cmap='gray')
        plt.title(f"Sobel ({name})")
        plt.axis('off')

        # ----- Laplacian Edge -----
        lap = np.clip(laplacian_edge(filtered), 0, 255)
        plt.subplot(3, total_cols, 2*total_cols + i + 3)
        plt.imshow(lap, cmap='gray')
        plt.title(f"Laplacian ({name})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()