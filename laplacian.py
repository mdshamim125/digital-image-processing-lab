import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread(r"C:\Users\Arup\Desktop\dip\car.png", 0)

# Convert to int for safe calculation
img_int = img.astype(int)

# Laplacian kernel (4-neighbour)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Padding image manually
padded = np.pad(img_int, 1, mode='edge')

# Output image
laplacian = np.zeros_like(img_int)

# -------- Manual Convolution --------
for i in range(img_int.shape[0]):
    for j in range(img_int.shape[1]):
        region = padded[i:i+3, j:j+3]
        laplacian[i, j] = np.sum(region * kernel)

# Clip values to valid range
laplacian = np.clip(laplacian, 0, 255)

# -------- Image Sharpening --------
sharpened = img_int + laplacian
sharpened = np.clip(sharpened, 0, 255)

# Convert back to uint8
laplacian = laplacian.astype(np.uint8)
sharpened = sharpened.astype(np.uint8)

# -------- Display Results --------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Image (Manual)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.show()
