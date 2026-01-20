import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Program Started...")

# -------------------------------
# Load image (Grayscale)
# -------------------------------
img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\DIP Lab\fb profile small.jpg", 0)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32)

# -------------------------------
# Padding Function
# -------------------------------
def pad(image, k=3):
    p = k // 2
    return np.pad(image, p, mode='edge')

# -------------------------------
# Filters (Only 5)
# -------------------------------
def min_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.min(padded[i:i+k, j:j+k])
    return out

def max_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.max(padded[i:i+k, j:j+k])
    return out

def median_filter(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.median(padded[i:i+k, j:j+k])
    return out

def arithmetic_mean(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.mean(padded[i:i+k, j:j+k])
    return out

def geometric_mean(image, k=3):
    padded = pad(image, k)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+k, j:j+k] + 1  # avoid log(0)
            out[i, j] = np.exp(np.mean(np.log(region)))
    return out

# -------------------------------
# Edge Detection Filters
# -------------------------------
def sobel_edge(image):
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    padded = pad(image, 3)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = padded[i:i+3, j:j+3]
            out[i, j] = np.sqrt((r*gx).sum()**2 + (r*gy).sum()**2)
    return out

def laplacian_edge(image):
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    padded = pad(image, 3)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    return out

# -------------------------------
# Filter List
# -------------------------------
filters = [
    ("Min", min_filter),
    ("Max", max_filter),
    ("Median", median_filter),
    ("Arithmetic", arithmetic_mean),
    ("Geometric", geometric_mean)
]

# -------------------------------
# Noise Functions
# -------------------------------
def gaussian_noise(image, mean=0, var=20):
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    return np.clip((image + noise), 0, 255)

def salt_and_pepper_noise(image, p=0.01):
    out = image.copy()
    rand = np.random.rand(*image.shape)
    out[rand < p] = 0
    out[rand > 1 - p] = 255
    return out

noises = [
    ("Gaussian Noise", gaussian_noise),
    ("Salt & Pepper Noise", salt_and_pepper_noise)
]

# -------------------------------
# Apply Filters + Edge Detection
# -------------------------------
for noise_name, noise_func in noises:
    noise_image = noise_func(img)

    plt.figure(figsize=(24, 12))

    # 1) Original
    plt.subplot(3, len(filters) + 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 2) Noisy Image
    plt.subplot(3, len(filters) + 2, 2)
    plt.imshow(noise_image, cmap='gray')
    plt.title(f"Noisy ({noise_name})")
    plt.axis('off')

    # 3) Filtered Images
    for i, (name, func) in enumerate(filters):
        filtered = func(noise_image)

        # Filtered output
        plt.subplot(3, len(filters) + 2, i + 3)
        plt.imshow(np.clip(filtered, 0, 255), cmap='gray')
        plt.title(f"{name} Filter")
        plt.axis('off')

        # Sobel Edge after filtering
        plt.subplot(3, len(filters) + 2, (len(filters) + 2) + i + 3)
        plt.imshow(np.clip(sobel_edge(filtered), 0, 255), cmap='gray')
        plt.title(f"Sobel ({name})")
        plt.axis('off')

        # Laplacian Edge after filtering
        plt.subplot(3, len(filters) + 2, 2*(len(filters) + 2) + i + 3)
        plt.imshow(np.clip(laplacian_edge(filtered), 0, 255), cmap='gray')
        plt.title(f"Laplacian ({name})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
