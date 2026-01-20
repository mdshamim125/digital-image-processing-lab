# Import required libraries
import cv2                      # For image reading and processing
import matplotlib.pyplot as plt  # For displaying images
import numpy as np               # For matrix and numerical operations

# Read image in grayscale mode
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# Get height and width of the image
h, w = img.shape[:2]


# ----------------------------------------------------
# Function to apply a geometric transformation
# ----------------------------------------------------
def apply_transform(img, T):
    h, w = img.shape              # Image dimensions
    out = np.zeros_like(img)      # Output image initialized with zeros

    # Find center of the image
    cx, cy = w // 2, h // 2

    # Loop through each pixel of the image
    for x in range(h):
        for y in range(w):

            # Convert pixel into homogeneous coordinates
            # Shift origin to image center
            p = np.array([
                [y - cx],
                [x - cy],
                [1]
            ])

            # Apply transformation matrix
            q = T @ p

            # Convert back to image coordinates
            yy = int(q[0, 0] + cx)
            xx = int(q[1, 0] + cy)

            # Check if transformed pixel is inside image boundary
            if 0 <= xx < h and 0 <= yy < w:
                out[xx, yy] = img[x, y]

    return out


# ----------------------------------------------------
# Transformation Matrices
# ----------------------------------------------------

# Translation (shift right by 50, down by 30)
T_translate = np.array([
    [1, 0, 50],
    [0, 1, 30],
    [0, 0, 1]
])

# Scaling (zoom in by 1.3 times)
T_scale = np.array([
    [1.3, 0,   0],
    [0,   1.3, 0],
    [0,   0,   1]
])

# Rotation (30 degrees anticlockwise)
theta = np.deg2rad(30)
T_rotate = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Horizontal Shearing
T_shear_h = np.array([
    [1, 0.5, 0],
    [0, 1,   0],
    [0, 0,   1]
])

# Vertical Shearing
T_shear_v = np.array([
    [1,   0, 0],
    [0.5, 1, 0],
    [0,   0, 1]
])

# Reflection along X-axis
T_reflect_x = np.array([
    [1,  0, 0],
    [0, -1, 0],
    [0,  0, 1]
])

# Reflection along Y-axis
T_reflect_y = np.array([
    [-1, 0, 0],
    [0,  1, 0],
    [0,  0, 1]
])

# Reflection about Origin
T_reflect_origin = np.array([
    [-1,  0, 0],
    [0,  -1, 0],
    [0,   0, 1]
])

# Reflection on line (Identity – no change)
T_reflect_line = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Combined affine transformation (Translation + Scaling + Rotation)
affine_transform = T_translate @ T_scale @ T_rotate


# ----------------------------------------------------
# Apply Transformations
# ----------------------------------------------------
img_trans = apply_transform(img, T_translate)
img_scale = apply_transform(img, T_scale)
img_rotate = apply_transform(img, T_rotate)
img_shear_h = apply_transform(img, T_shear_h)
img_shear_v = apply_transform(img, T_shear_v)
img_reflect_x = apply_transform(img, T_reflect_x)
img_reflect_y = apply_transform(img, T_reflect_y)
img_reflect_origin = apply_transform(img, T_reflect_origin)
img_reflect_line = apply_transform(img, T_reflect_line)
img_affine = apply_transform(img, affine_transform)


# ----------------------------------------------------
# Titles and Images for Display
# ----------------------------------------------------
titles = [
    "Original Image",
    "Translated Image",
    "Scaled Image",
    "Rotated Image",
    "Sheared Image Horizontally",
    "Sheared Image Vertically",
    "Reflected on X-Axis",
    "Reflected on Y-Axis",
    "Reflected on Origin",
    "Identity Transformation",
    "Affine Transformed Image"
]

images = [
    img,
    img_trans,
    img_scale,
    img_rotate,
    img_shear_h,
    img_shear_v,
    img_reflect_x,
    img_reflect_y,
    img_reflect_origin,
    img_reflect_line,
    img_affine
]


# ----------------------------------------------------
# Display all images in one figure
# ----------------------------------------------------
plt.figure(figsize=(18, 10))

for i in range(len(images)):
    plt.subplot(3, 4, i + 1)          # 3 rows, 4 columns
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# ----------------------------------------------------
# Image Information
# ----------------------------------------------------
print("Original Image Shape:", img.shape)   # (height, width)
print("Original Image Size:", img.size)     # Total number of pixels
