import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load grayscale image
# -----------------------------
img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg", 0)

# -----------------------------
# Display RGB channels (optional)
# -----------------------------
# Convert grayscale to RGB for visualization of channels
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

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

# -----------------------------
# Intensity Transformations
# -----------------------------

# 1. Negative Transformation
negative = 255 - img

# 2. Log Transformation
c = 255 / (np.log(1 + np.max(img)))
log_image = c * (np.log(1 + img))
log_image = np.array(log_image, dtype=np.uint8)

# 3. Gamma Transformation
gamma = 0.5
gamma_image = np.array(255 * ((img / 255) ** gamma), dtype=np.uint8)

# 4. Contrast Stretching
rmin, rmax = np.min(img), np.max(img)
contrast_stretch = ((img - rmin) / (rmax - rmin)) * 255
contrast_stretch = np.array(contrast_stretch, dtype=np.uint8)

# -----------------------------
# Display all results together
# -----------------------------
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

# -----------------------------
# Print size & shape
# -----------------------------
print("Original Image Size:", img.size)
print("Negative Image Size:", negative.size)
print("Log Transform Image Size:", log_image.size)
print("Gamma Transform Image Size:", gamma_image.size)
print("Contrast Stretch Image Size:", contrast_stretch.size)

print("Original Image Shape:", img.shape)
print("Negative Image Shape:", negative.shape)
print("Log Transform Image Shape:", log_image.shape)
print("Gamma Transform Image Shape:", gamma_image.shape)
print("Contrast Stretch Image Shape:", contrast_stretch.shape)
