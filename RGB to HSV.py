import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Program Started...")

# ----------------------------------------------------
# Step 1: Read the image and convert to RGB
# ----------------------------------------------------
img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg")

# OpenCV reads image in BGR format by default
# Convert BGR to RGB for correct color display
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize the pixel values to range [0, 1]
# This makes calculation easier and prevents overflow
img = img.astype(np.float32) / 255.0


# ----------------------------------------------------
# Step 2: Separate R, G, B channels
# ----------------------------------------------------
R = img[:, :, 0]   # Red channel
G = img[:, :, 1]   # Green channel
B = img[:, :, 2]   # Blue channel


# ----------------------------------------------------
# Step 3: Display R, G, B channels
# ----------------------------------------------------
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

plt.show()


# ----------------------------------------------------
# Step 4: Prepare for RGB → HSV conversion
# ----------------------------------------------------
h, w, _ = img.shape

# Create an empty array with same shape as original image
# This will store HSV values
HSV = np.zeros_like(img)


# ----------------------------------------------------
# Step 5: Convert RGB to HSV (pixel by pixel)
# ----------------------------------------------------
for i in range(h):
    for j in range(w):
        r, g, b = img[i, j]

        # Find maximum and minimum of R, G, B
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # ----------------------------
        # Calculate Hue (H)
        # ----------------------------
        if delta == 0:
            H = 0
        elif cmax == r:
            H = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            H = 60 * (((b - r) / delta) + 2)
        else:
            H = 60 * (((r - g) / delta) + 4)

        # ----------------------------
        # Calculate Saturation (S)
        # ----------------------------
        if cmax == 0:
            S = 0
        else:
            S = delta / cmax

        # ----------------------------
        # Calculate Value (V)
        # ----------------------------
        V = cmax

        # Store normalized H, S, V values
        # H is divided by 360 to make it between 0 and 1
        HSV[i, j] = [H / 360, S, V]


# ----------------------------------------------------
# Step 6: Extract H, S, V channels
# ----------------------------------------------------
H = HSV[:, :, 0]
S = HSV[:, :, 1]
V = HSV[:, :, 2]


# ----------------------------------------------------
# Step 7: Convert H, S, V to displayable format
# ----------------------------------------------------
# H values are between 0 and 1, so multiply by 255 for display
H_disp = (H / H.max()) * 255

# S and V are already in [0,1], so multiply by 255 for display
S_disp = S * 255
V_disp = V * 255


# ----------------------------------------------------
# Step 8: Display RGB and HSV channels
# ----------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(H_disp, cmap='gray')
plt.title("Hue Channel")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(S_disp, cmap='gray')
plt.title("Saturation Channel")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(V_disp, cmap='gray')
plt.title("Value Channel")
plt.axis('off')

plt.tight_layout()
plt.show()


# ----------------------------------------------------
# Step 9: Print image properties
# ----------------------------------------------------
print("Image Shape:", img.shape)   # (height, width, channels)
print("Image Size:", img.size)     # total number of pixels

# Print range of values for H, S, V
print("H range:", H.min(), H.max())
print("S range:", S.min(), S.max())
print("V range:", V.min(), V.max())
