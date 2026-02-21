# ============================================================
# GEOMETRIC SPATIAL TRANSFORMATION IN DIGITAL IMAGE PROCESSING
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                        # OpenCV → used for reading and processing images
import matplotlib.pyplot as plt  # Used to display images
import numpy as np               # Used for matrix and numerical operations


# ------------------------------------------------------------
# Read Image in Grayscale Mode
# ------------------------------------------------------------

# cv2.imread(path, flag)
# flag = 0 → read image as grayscale (single channel image)

img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# ------------------------------------------------------------
# Get Image Dimensions
# ------------------------------------------------------------

# img.shape returns (height, width) for grayscale image
h, w = img.shape[:2]


# ============================================================
# FUNCTION: APPLY GEOMETRIC TRANSFORMATION
# ============================================================
# This function applies a 3×3 transformation matrix to an image
# using homogeneous coordinates.
#
# Parameters:
#   img → input grayscale image
#   T   → 3×3 transformation matrix
#
# Returns:
#   transformed image
# ============================================================

def apply_transform(img, T):

    # Get image height and width
    h, w = img.shape

    # Create empty output image with same size as input
    # All pixel values initialized to 0 (black image)
    out = np.zeros_like(img)

    # --------------------------------------------------------
    # Find center of image
    # --------------------------------------------------------
    # Transformations like rotation are applied around origin.
    # So we shift origin to image center.
    cx, cy = w // 2, h // 2

    # --------------------------------------------------------
    # Loop through each pixel of input image
    # --------------------------------------------------------
    for x in range(h):       # x → row index (height direction)
        for y in range(w):   # y → column index (width direction)

            # ------------------------------------------------
            # Convert pixel position to homogeneous coordinate
            # ------------------------------------------------
            # Subtract center → shift origin to image center
            #
            # Homogeneous form:
            # [x]
            # [y]
            # [1]
            #
            p = np.array([
                [y - cx],   # x-coordinate relative to center
                [x - cy],   # y-coordinate relative to center
                [1]         # homogeneous coordinate
            ])

            # ------------------------------------------------
            # Apply transformation matrix
            # ------------------------------------------------
            # Matrix multiplication: new_position = T × p
            q = T @ p

            # ------------------------------------------------
            # Convert back to normal image coordinates
            # ------------------------------------------------
            # Add center back to restore original coordinate system
            yy = int(q[0, 0] + cx)
            xx = int(q[1, 0] + cy)

            # ------------------------------------------------
            # Check boundary condition
            # ------------------------------------------------
            # Ensure transformed pixel lies inside image
            if 0 <= xx < h and 0 <= yy < w:

                # Assign pixel intensity to new location
                out[xx, yy] = img[x, y]

    return out


# ============================================================
# TRANSFORMATION MATRICES (3×3 HOMOGENEOUS MATRICES)
# ============================================================

# ------------------------------------------------------------
# 1. Translation Matrix
# Moves image right by 50 pixels and down by 30 pixels
#
# x' = x + 50
# y' = y + 30
# ------------------------------------------------------------
T_translate = np.array([
    [1, 0, 50],
    [0, 1, 30],
    [0, 0, 1]
])


# ------------------------------------------------------------
# 2. Scaling Matrix
# Enlarges image by 1.3 times (zoom in)
# ------------------------------------------------------------
T_scale = np.array([
    [1.3, 0,   0],
    [0,   1.3, 0],
    [0,   0,   1]
])


# ------------------------------------------------------------
# 3. Rotation Matrix (30° Anticlockwise)
# ------------------------------------------------------------

# Convert degrees to radians
theta = np.deg2rad(30)

T_rotate = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])


# ------------------------------------------------------------
# 4. Horizontal Shearing Matrix
# Tilts image horizontally
# ------------------------------------------------------------
T_shear_h = np.array([
    [1, 0.5, 0],
    [0, 1,   0],
    [0, 0,   1]
])


# ------------------------------------------------------------
# 5. Vertical Shearing Matrix
# Tilts image vertically
# ------------------------------------------------------------
T_shear_v = np.array([
    [1,   0, 0],
    [0.5, 1, 0],
    [0,   0, 1]
])


# ------------------------------------------------------------
# 6. Reflection along X-axis
# Flips image upside down
# ------------------------------------------------------------
T_reflect_x = np.array([
    [1,  0, 0],
    [0, -1, 0],
    [0,  0, 1]
])


# ------------------------------------------------------------
# 7. Reflection along Y-axis
# Mirror image left-right
# ------------------------------------------------------------
T_reflect_y = np.array([
    [-1, 0, 0],
    [0,  1, 0],
    [0,  0, 1]
])


# ------------------------------------------------------------
# 8. Reflection about Origin
# Rotates image 180°
# ------------------------------------------------------------
T_reflect_origin = np.array([
    [-1,  0, 0],
    [0,  -1, 0],
    [0,   0, 1]
])


# ------------------------------------------------------------
# 9. Identity Matrix (No change)
# ------------------------------------------------------------
T_reflect_line = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


# ------------------------------------------------------------
# 10. Combined Affine Transformation
# Applies rotation → scaling → translation
# Matrix multiplication order matters
# ------------------------------------------------------------
affine_transform = T_translate @ T_scale @ T_rotate


# ============================================================
# APPLY ALL TRANSFORMATIONS
# ============================================================

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


# ============================================================
# STORE TITLES AND IMAGES FOR DISPLAY
# ============================================================

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


# ============================================================
# DISPLAY ALL IMAGES IN SINGLE FIGURE
# ============================================================

# Create large display window
plt.figure(figsize=(18, 10))

# Loop through all images and display
for i in range(len(images)):

    # Create subplot grid → 3 rows × 4 columns
    plt.subplot(3, 4, i + 1)

    # Display image in grayscale
    plt.imshow(images[i], cmap='gray')

    # Show title
    plt.title(titles[i])

    # Hide axis numbers
    plt.axis('off')

# Adjust layout automatically
plt.tight_layout()

# Show figure
plt.show()


# ============================================================
# PRINT IMAGE INFORMATION
# ============================================================

# Shape → (height, width)
print("Original Image Shape:", img.shape)

# Total number of pixels
print("Original Image Size:", img.size)