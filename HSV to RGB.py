import cv2                     # OpenCV library for image processing
import numpy as np             # NumPy for array operations
import matplotlib.pyplot as plt  # Matplotlib for plotting images

# -------------------------------
# Program start indicator
# -------------------------------
print("Program started...")    # Print message to confirm program execution

# -------------------------------
# Load color image (BGR format)
# -------------------------------
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg"
)
# cv2.imread loads image in BGR color format by default

# Convert BGR (OpenCV default) to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.cvtColor changes color space from BGR to RGB

# Convert pixel range from [0,255] to [0,1] for HSV computation
img = img.astype(np.float32) / 255.0
# Convert image to float32 then normalize by dividing by 255

# Get image dimensions
h, w, _ = img.shape
# h = height, w = width, _ = number of channels (3 for RGB)

# -------------------------------
# Initialize HSV image
# -------------------------------
HSV = np.zeros_like(img)
# Create an empty array of same shape as img to store HSV values

# -------------------------------
# RGB → HSV Conversion (Pixel-wise)
# -------------------------------
for i in range(h):              # Loop over each row
    for j in range(w):          # Loop over each column
        r, g, b = img[i, j]     # Get RGB values of current pixel

        # Maximum, minimum and difference
        cmax = max(r, g, b)     # maximum of r,g,b
        cmin = min(r, g, b)     # minimum of r,g,b
        delta = cmax - cmin     # difference between max and min

        # -------- Hue Calculation --------
        if delta == 0:
            H = 0
        elif cmax == r:
            H = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            H = 60 * (((b - r) / delta) + 2)
        else:
            H = 60 * (((r - g) / delta) + 4)
        # Hue calculation based on which channel is maximum

        # -------- Saturation Calculation --------
        if cmax == 0:
            S = 0
        else:
            S = delta / cmax
        # Saturation is the color purity. If max=0, saturation=0

        # -------- Value Calculation --------
        V = cmax
        # Value is just the maximum value among r,g,b

        # Store normalized Hue
        HSV[i, j] = [H / 360, S, V]
        # H normalized to [0,1] by dividing by 360

# Separate HSV channels
H = HSV[:, :, 0]   # Hue channel
S = HSV[:, :, 1]   # Saturation channel
V = HSV[:, :, 2]   # Value channel

# Scale HSV components for display
H_disp = (H / H.max()) * 255 if H.max() != 0 else H
# Normalize hue for display (0-255)

S_disp = S * 255   # Saturation scale to 0-255
V_disp = V * 255   # Value scale to 0-255

# -------------------------------
# Display HSV components
# -------------------------------
plt.figure(figsize=(12, 4))  # Figure size in inches

plt.subplot(1, 4, 1)         # 1 row, 4 columns, position 1
plt.imshow(img)              # Show original RGB image
plt.title("Original RGB")    # Title
plt.axis('off')              # Hide axis values

plt.subplot(1, 4, 2)         # Position 2
plt.imshow(H_disp, cmap='gray')  # Show Hue in gray
plt.title("Hue")
plt.axis('off')

plt.subplot(1, 4, 3)         # Position 3
plt.imshow(S_disp, cmap='gray')  # Show Saturation
plt.title("Saturation")
plt.axis('off')

plt.subplot(1, 4, 4)         # Position 4
plt.imshow(V_disp, cmap='gray')  # Show Value
plt.title("Value")
plt.axis('off')

plt.tight_layout()           # Adjust spacing between subplots
plt.show()                   # Display the plot

# -------------------------------
# HSV → RGB Reconstruction
# -------------------------------
rgb_rec = np.zeros_like(img)
# Create empty RGB image for reconstruction

for i in range(h):            # Loop over each row
    for j in range(w):        # Loop over each column
        H_ = H[i, j] * 360    # Convert hue back to degrees
        S_ = S[i, j]          # Saturation value
        V_ = V[i, j]          # Value

        C = V_ * S_           # Chroma
        X = C * (1 - abs((H_ / 60) % 2 - 1))
        m = V_ - C

        # Determine sector of hue
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

        # Add m to match actual RGB values
        rgb_rec[i, j] = [r1 + m, g1 + m, b1 + m]

# -------------------------------
# Display Reconstruction Result
# -------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(rgb_rec, 0, 1))
# clip values to ensure valid display range
plt.title("Reconstructed RGB")
plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Error & Statistics
# -------------------------------
error = np.mean(np.abs(img - rgb_rec))
# Calculate mean absolute error between original and reconstructed image

print("Image Shape:", img.shape)
print("Image Size:", img.size)
print("Hue Range:", H.min(), H.max())
print("Saturation Range:", S.min(), S.max())
print("Value Range:", V.min(), V.max())
print("Mean Reconstruction Error:", error)
