# ============================================================
# LAPLACIAN FILTER FOR IMAGE SHARPENING (MANUAL CONVOLUTION)
# DIGITAL IMAGE PROCESSING LAB
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import cv2                      # OpenCV → image reading and processing
import numpy as np              # NumPy → numerical operations and matrices
import matplotlib.pyplot as plt # Matplotlib → display images


# ============================================================
# STEP 1: LOAD IMAGE IN GRAYSCALE
# ============================================================

# Load image in grayscale mode
# 0 → grayscale flag (pixel values from 0 to 255)
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)


# ------------------------------------------------------------
# Convert image to integer type
# ------------------------------------------------------------
# Why?
# Because convolution may produce negative values
# uint8 cannot store negative numbers safely
# So we convert to int for safe mathematical operations

img_int = img.astype(int)


# ============================================================
# STEP 2: DEFINE LAPLACIAN KERNEL
# ============================================================

# Laplacian operator detects edges by measuring
# second-order intensity change (rate of change of gradient).

# 4-neighbour Laplacian kernel:
#
#   0  -1   0
#  -1   4  -1
#   0  -1   0
#
# Center pixel minus its four neighbors.

kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

# Laplacian kernel (4-neighbour Laplacian operator) used in image processing for edge detection and sharpening.
# 4-neighbour Laplacian uses only horizontal and vertical neighbours for edge detection, while 8-neighbour Laplacian also includes diagonal neighbours, giving stronger but more noise-sensitive results.

# ============================================================
# STEP 3: IMAGE PADDING
# ============================================================

# Convolution requires neighbors around each pixel.
# Border pixels do not have full neighbors.
#
# So we add padding around image.
#
# np.pad(image, pad_width, mode='edge')
# mode='edge' → copy border values outward.

padded = np.pad(img_int, 1, mode='edge')


# ============================================================
# STEP 4: CREATE OUTPUT IMAGE FOR LAPLACIAN RESULT
# ============================================================

# Create empty array with same size as input image
# This will store Laplacian filtered values.

laplacian = np.zeros_like(img_int)


# ============================================================
# STEP 5: MANUAL CONVOLUTION PROCESS
# ============================================================
# Convolution = slide kernel over image
# Multiply kernel with neighborhood pixels
# Sum results → output pixel value

for i in range(img_int.shape[0]):      # loop over rows
    for j in range(img_int.shape[1]):  # loop over columns

        # Extract 3×3 region from padded image
        # This is neighborhood around current pixel
        region = padded[i:i+3, j:j+3]

        # Element-wise multiplication with kernel
        # then sum all values
        laplacian[i, j] = np.sum(region * kernel)


# ============================================================
# STEP 6: CLIP VALUES TO VALID INTENSITY RANGE
# ============================================================

# Pixel values must be in range [0,255]
# Negative → 0
# Above 255 → 255

laplacian = np.clip(laplacian, 0, 255)


# ============================================================
# STEP 7: IMAGE SHARPENING USING LAPLACIAN
# ============================================================

# Sharpening formula:
#
#   Sharpened Image = Original + Laplacian
#
# Laplacian highlights edges.
# Adding it back enhances details.

sharpened = img_int + laplacian

# Again clip to valid range
sharpened = np.clip(sharpened, 0, 255)


# ------------------------------------------------------------
# Convert results back to unsigned 8-bit image
# ------------------------------------------------------------
# Required for proper display and image storage

laplacian = laplacian.astype(np.uint8)
sharpened = sharpened.astype(np.uint8)
# uint8 properties:
#   - Stores only positive values
#   - Range: 0 to 255
#   - Standard format for image display and saving

# Convert Laplacian result to uint8 format
# This ensures:
#   ✓ valid pixel range



# ============================================================
# STEP 8: DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(12,4))

# ---- Original Image ----
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# ---- Laplacian Filter Output ----
plt.subplot(1,3,2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Image (Manual)")
plt.axis('off')

# ---- Sharpened Image ----
plt.subplot(1,3,3)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.show()