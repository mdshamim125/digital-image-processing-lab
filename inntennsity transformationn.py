# ============================================================
# INTENSITY TRANSFORMATIONS
# DIGITAL IMAGE PROCESSING LAB
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                      # OpenCV → image loading and processing
import matplotlib.pyplot as plt # Matplotlib → image visualization
import numpy as np              # NumPy → numerical operations


# ============================================================
# STEP 1: LOAD GRAYSCALE IMAGE
# ============================================================

# cv2.imread(path, 0) loads image in grayscale mode
# Pixel values range from 0 (black) to 255 (white)

img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# img → 2D array (height × width)
# Each value represents intensity level of pixel


# ============================================================
# STEP 2: DISPLAY RGB CHANNELS (OPTIONAL FOR VISUALIZATION)
# ============================================================

# Since the image is grayscale, it has only 1 channel.
# To visualize R, G, B channels separately, we convert it to RGB.

# Convert grayscale → RGB (3 identical channels)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Split into Red, Green, Blue channels
R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

# ------------------------------------------------------------
# Display the three channels
# ------------------------------------------------------------
# Note: Since original image is grayscale,
# all three channels will look identical.

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 3: INTENSITY TRANSFORMATIONS
# ============================================================
# Intensity transformation modifies pixel values to improve
# contrast, brightness, or visibility of image features.
# ============================================================


# ------------------------------------------------------------
# 1. NEGATIVE TRANSFORMATION
# ------------------------------------------------------------
# Formula:
# s = L - 1 - r
# where L = 256 for 8-bit image

# Effect:
# - Bright areas become dark
# - Dark areas become bright
# - Useful in medical imaging and enhancement

negative = 255 - img


# ------------------------------------------------------------
# 2. LOG TRANSFORMATION
# ------------------------------------------------------------
# Formula:
# s = c * log(1 + r)

# Effect:
# - Expands dark pixel values
# - Compresses bright pixel values
# - Enhances details in darker regions

# Compute scaling constant
c = 255 / (np.log(1 + np.max(img)))

# Apply logarithmic transformation
log_image = c * (np.log(1 + img))

# Convert back to integer type
log_image = np.array(log_image, dtype=np.uint8)


# ------------------------------------------------------------
# 3. GAMMA (POWER LAW) TRANSFORMATION
# ------------------------------------------------------------
# Formula:
# s = c * r^gamma

# Effect:
# gamma < 1 → image becomes brighter
# gamma > 1 → image becomes darker

gamma = 0.5   # adjust brightness

gamma_image = np.array(
    255 * ((img / 255) ** gamma),
    dtype=np.uint8
)


# ------------------------------------------------------------
# 4. CONTRAST STRETCHING
# ------------------------------------------------------------
# Expands intensity range to full [0–255]
# Improves image contrast

# Formula:
# s = ((r - rmin) / (rmax - rmin)) * 255

# Find minimum and maximum intensity
rmin, rmax = np.min(img), np.max(img)

# Apply contrast stretching
contrast_stretch = ((img - rmin) / (rmax - rmin)) * 255

# Convert to integer
contrast_stretch = np.array(contrast_stretch, dtype=np.uint8)


# ============================================================
# STEP 4: DISPLAY ALL RESULTS
# ============================================================

# Show original and transformed images together
# for comparison.

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(negative, cmap='gray')
plt.title("Negative Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(log_image, cmap='gray')
plt.title("Log Transformation")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gamma_image, cmap='gray')
plt.title("Gamma Transformation")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(contrast_stretch, cmap='gray')
plt.title("Contrast Stretching")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 5: PRINT IMAGE INFORMATION
# ============================================================

# ------------------------------------------------------------
# Image Size → total number of pixels
# ------------------------------------------------------------
print("Original Image Size:", img.size)
print("Negative Image Size:", negative.size)
print("Log Transform Image Size:", log_image.size)
print("Gamma Transform Image Size:", gamma_image.size)
print("Contrast Stretch Image Size:", contrast_stretch.size)

# ------------------------------------------------------------
# Image Shape → (height, width)
# ------------------------------------------------------------
print("Original Image Shape:", img.shape)
print("Negative Image Shape:", negative.shape)
print("Log Transform Image Shape:", log_image.shape)
print("Gamma Transform Image Shape:", gamma_image.shape)
print("Contrast Stretch Image Shape:", contrast_stretch.shape)