# ============================================================
# BASIC IMAGE PROCESSING OPERATIONS USING OPENCV
# (Display, Downsampling, RGB Channels, Cropping)
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import cv2                     # OpenCV → image processing
import matplotlib.pyplot as plt  # Matplotlib → image display


# ============================================================
# STEP 1: LOAD IMAGE IN GRAYSCALE
# ============================================================

# cv2.imread(path, 0) → loads image in grayscale (single channel)
img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# ------------------------------------------------------------
# Convert grayscale image → RGB
# ------------------------------------------------------------
# Why?
# Matplotlib expects RGB format for displaying images properly
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# STEP 2: DISPLAY ORIGINAL IMAGE
# ============================================================

plt.imshow(img)               # Show image
plt.title("Original Image")   # Title of image
plt.axis('off')               # Hide axis values
plt.show()

# Print total number of pixel values
print("Original Image Size:", img.size)


# ============================================================
# STEP 3: IMAGE DOWNSAMPLING (RESIZING)
# ============================================================
# Downsampling → reduce image size
# This decreases resolution and number of pixels

scale = 0.5   # Reduce image to 50% of original size

downsampled = cv2.resize(
    img,                   # input image
    None,                  # output size auto calculated
    fx=scale,              # scale factor along x-axis (width)
    fy=scale,              # scale factor along y-axis (height)
    interpolation=cv2.INTER_NEAREST  # nearest neighbor method
)

# INTER_NEAREST → fastest interpolation method
# Picks nearest pixel value (no averaging)

# Display downsampled image
plt.imshow(downsampled)
plt.title("Downsampled Image")
plt.axis('off')
plt.show()

print("Downsampled Image Size:", downsampled.size)


# ============================================================
# STEP 4: EXTRACT RGB COLOR CHANNELS
# ============================================================
# A color image contains 3 channels:
# R → Red intensity
# G → Green intensity
# B → Blue intensity

R = img[:, :, 0]   # Extract Red channel
G = img[:, :, 1]   # Extract Green channel
B = img[:, :, 2]   # Extract Blue channel

# Display each channel separately
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap='gray')   # Show Red channel in grayscale
plt.axis('off')
plt.title("Red Channel")

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='gray')   # Show Green channel
plt.axis('off')
plt.title("Green Channel")

plt.subplot(1, 3, 3)         # Correct position for Blue channel
plt.imshow(B, cmap='gray')   # Show Blue channel
plt.axis('off')
plt.title("Blue Channel")

plt.show()


# ============================================================
# STEP 5: IMAGE CROPPING
# ============================================================
# Cropping → extract a specific region of interest (ROI)

# Syntax:
# image[start_row:end_row, start_col:end_col]

crop = img[50:250, 50:250]   # Extract region from image

plt.imshow(crop)
plt.title("Cropped Image")
plt.axis('off')
plt.show()

print("Cropped Image Size:", crop.size)


# ============================================================
# STEP 6: DISPLAY IMAGE INFORMATION
# ============================================================

# shape → (height, width, channels)
print("Original Image Shape:", img.shape)
print("Downsampled Image Shape:", downsampled.shape)
print("Cropped Image Shape:", crop.shape)


# ============================================================
# STEP 7: DISPLAY ALL RESULTS TOGETHER
# ============================================================

plt.figure(figsize=(10, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

# Downsampled image
plt.subplot(1, 3, 2)
plt.imshow(downsampled)
plt.title("Downsampled Image")
plt.axis('off')

# Cropped image
plt.subplot(1, 3, 3)
plt.imshow(crop)
plt.title("Cropped Image")
plt.axis('off')

plt.show()