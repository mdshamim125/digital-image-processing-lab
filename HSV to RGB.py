# ============================================================
# RGB ↔ HSV COLOR SPACE CONVERSION
# DIGITAL IMAGE PROCESSING LAB
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                     # OpenCV → image loading & processing
import numpy as np             # NumPy → array and numerical operations
import matplotlib.pyplot as plt  # Matplotlib → image visualization


# ------------------------------------------------------------
# Program Start Indicator
# ------------------------------------------------------------
print("Program started...")   # Helps confirm program execution


# ============================================================
# STEP 1: LOAD COLOR IMAGE
# ============================================================

# OpenCV reads image in BGR format by default
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg"
)

# ------------------------------------------------------------
# Convert BGR → RGB
# ------------------------------------------------------------
# Because matplotlib displays images in RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# Normalize Pixel Values
# ------------------------------------------------------------
# Convert pixel range from [0,255] → [0,1]
# Required for HSV mathematical computation

img = img.astype(np.float32) / 255.0


# ------------------------------------------------------------
# Get Image Dimensions
# ------------------------------------------------------------
# h = height, w = width, 3 = RGB channels
h, w, _ = img.shape


# ============================================================
# STEP 2: INITIALIZE HSV IMAGE
# ============================================================

# Create empty array same size as image to store HSV values
HSV = np.zeros_like(img)


# ============================================================
# STEP 3: RGB → HSV CONVERSION (PIXEL BY PIXEL)
# ============================================================
# HSV = Hue, Saturation, Value
#
# Hue → color type (0–360°)
# Saturation → color purity
# Value → brightness
# ============================================================

for i in range(h):          # Loop over rows
    for j in range(w):      # Loop over columns

        # Extract RGB values of current pixel
        r, g, b = img[i, j]

        # ----------------------------------------------------
        # Find maximum and minimum values among R,G,B
        # ----------------------------------------------------
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin   # difference


        # ====================================================
        # HUE CALCULATION
        # ====================================================
        # Hue depends on which color channel is maximum

        if delta == 0:
            H = 0

        elif cmax == r:
            H = 60 * (((g - b) / delta) % 6)

        elif cmax == g:
            H = 60 * (((b - r) / delta) + 2)

        else:  # cmax == b
            H = 60 * (((r - g) / delta) + 4)


        # ====================================================
        # SATURATION CALCULATION
        # ====================================================
        # Saturation measures color intensity

        if cmax == 0:
            S = 0
        else:
            S = delta / cmax


        # ====================================================
        # VALUE CALCULATION
        # ====================================================
        # Value represents brightness
        V = cmax


        # ----------------------------------------------------
        # Store HSV values
        # ----------------------------------------------------
        # Hue normalized to [0,1] by dividing by 360
        HSV[i, j] = [H / 360, S, V]


# ============================================================
# STEP 4: SEPARATE HSV CHANNELS
# ============================================================

H = HSV[:, :, 0]   # Hue channel
S = HSV[:, :, 1]   # Saturation channel
V = HSV[:, :, 2]   # Value channel


# ============================================================
# STEP 5: SCALE HSV FOR DISPLAY
# ============================================================
# Convert values to 0–255 range for visualization

H_disp = (H / H.max()) * 255 if H.max() != 0 else H
S_disp = S * 255
V_disp = V * 255


# ============================================================
# STEP 6: DISPLAY HSV COMPONENTS
# ============================================================

plt.figure(figsize=(12, 4))

# Original RGB image
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

# Hue channel
plt.subplot(1, 4, 2)
plt.imshow(H_disp, cmap='gray')
plt.title("Hue")
plt.axis('off')

# Saturation channel
plt.subplot(1, 4, 3)
plt.imshow(S_disp, cmap='gray')
plt.title("Saturation")
plt.axis('off')

# Value channel
plt.subplot(1, 4, 4)
plt.imshow(V_disp, cmap='gray')
plt.title("Value")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 7: HSV → RGB RECONSTRUCTION
# ============================================================
# Convert HSV back to RGB to verify conversion accuracy
# ============================================================

rgb_rec = np.zeros_like(img)   # empty image for reconstructed RGB

for i in range(h):
    for j in range(w):

        # Convert normalized hue back to degrees
        H_ = H[i, j] * 360
        S_ = S[i, j]
        V_ = V[i, j]

        # ----------------------------------------------------
        # Compute Chroma
        # ----------------------------------------------------
        C = V_ * S_

        # Intermediate value
        X = C * (1 - abs((H_ / 60) % 2 - 1))

        # Adjustment factor
        m = V_ - C


        # ----------------------------------------------------
        # Determine RGB sector based on Hue angle
        # ----------------------------------------------------
        if 0 <= H_ < 60:
            r1, g1, b1 = C, X, 0

        elif 60 <= H_ < 120:
            r1, g1, b1 = X, C, 0

        elif 120 <= H_ < 180:
            r1, g1, b1 = 0, C, X

        elif 180 <= H_ < 240:
            r1, g1, b1 = 0, X, C

        elif 240 <= H_ < 300:
            r1, g1, b1 = X, 0, C

        else:
            r1, g1, b1 = C, 0, X


        # Add m to match original brightness
        rgb_rec[i, j] = [r1 + m, g1 + m, b1 + m]


# ============================================================
# STEP 8: DISPLAY RECONSTRUCTED IMAGE
# ============================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(rgb_rec, 0, 1))
# np.clip ensures pixel values remain between 0 and 1
plt.title("Reconstructed RGB")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 9: ERROR & STATISTICS
# ============================================================

# Mean Absolute Error → difference between original and reconstructed image
error = np.mean(np.abs(img - rgb_rec))

print("Image Shape:", img.shape)
print("Image Size:", img.size)
print("Hue Range:", H.min(), H.max())
print("Saturation Range:", S.min(), S.max())
print("Value Range:", V.min(), V.max())
print("Mean Reconstruction Error:", error)