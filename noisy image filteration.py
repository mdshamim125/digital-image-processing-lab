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

def spike_noise(image, var=0.05):
    noise = np.random.randn(*image.shape) * np.sqrt(var)
    return np.clip((image + image * noise), 0, 255)

def uniform_noise(image, low=30, high=30):
    noise = np.random.uniform(-low, high, image.shape)
    return np.clip((image + noise), 0, 255)

noises = [
    ("Gaussian Noise", gaussian_noise),
    ("Salt & Pepper Noise", salt_and_pepper_noise),
    ("Spike Noise", spike_noise),
    ("Uniform Noise", uniform_noise)
]

# -------------------------------
# Apply Filters on Noisy Images
# -------------------------------
for noise_name, noise_func in noises:
    noise_image = noise_func(img)

    # Create figure
    plt.figure(figsize=(22, 10))

    # 1) Original Image
    plt.subplot(1, len(filters) + 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 2) Noisy Image
    plt.subplot(1, len(filters) + 2, 2)
    plt.imshow(noise_image, cmap='gray')
    plt.title(f"Noisy ({noise_name})")
    plt.axis('off')

    # 3) Filtered Images
    for i, (name, func) in enumerate(filters):
        filtered = func(noise_image)
        plt.subplot(1, len(filters) + 2, i + 3)
        plt.imshow(np.clip(filtered, 0, 255), cmap='gray')
        plt.title(f"{name} Filter")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
