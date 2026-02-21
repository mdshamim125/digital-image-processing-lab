# ============================================================
# HUFFMAN CODING FOR IMAGE COMPRESSION (DIGITAL IMAGE PROCESSING)
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                    # OpenCV → for reading image
import numpy as np           # Numerical operations and arrays
import matplotlib.pyplot as plt  # Display images


# ------------------------------------------------------------
# Load Grayscale Image
# ------------------------------------------------------------

# cv2.imread(path, 0)
# 0 → read image as grayscale (pixel values 0–255)

img = cv2.imread(
    r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg",
    0
)

# Get image height and width
h, w = img.shape


# ============================================================
# STEP 1: COUNT PIXEL FREQUENCIES
# ============================================================
# Huffman coding works based on frequency of symbols.
# Here pixel values (0–255) are treated as symbols.

freq = {}   # dictionary → {pixel_value : frequency}

# Loop through every pixel in image
for i in range(h):
    for j in range(w):

        # Get pixel value
        p = img[i, j]

        # Count frequency
        if p in freq:
            freq[p] += 1
        else:
            freq[p] = 1


# ============================================================
# STEP 2: PREPARE SYMBOL LIST AND CODE STORAGE
# ============================================================

# symbols → list storing [symbol, frequency]
symbols = []

# Convert frequency dictionary into list format
for k in freq:
    symbols.append([k, freq[k]])

# codes → dictionary storing Huffman code for each pixel value
# Initially empty strings
codes = {}
for k in freq:
    codes[k] = ""


# ============================================================
# STEP 3: BUILD HUFFMAN TREE AND GENERATE CODES
# ============================================================
# Process:
# 1. Sort symbols by frequency
# 2. Take two smallest frequencies
# 3. Assign 0 and 1
# 4. Merge them
# 5. Repeat until one node remains

while len(symbols) > 1:

    # Sort symbols based on frequency (ascending order)
    symbols.sort(key=lambda x: x[1])

    # Remove two smallest frequency symbols
    a = symbols.pop(0)
    b = symbols.pop(0)

    # Assign codes
    for k in codes:
        if k == a[0]:
            codes[k] = "0" + codes[k]
        elif k == b[0]:
            codes[k] = "1" + codes[k]

    # Merge the two nodes
    # (symbol values combined, frequencies added)
    symbols.append([a[0] + b[0], a[1] + b[1]])


# ============================================================
# STEP 4: ENCODE IMAGE USING HUFFMAN CODES
# ============================================================

encoded_list = []   # store encoded bit strings

# Replace each pixel with its Huffman code
for i in range(h):
    for j in range(w):
        encoded_list.append(codes[img[i, j]])

# Join all codes into one long bit string
encoded_bits = "".join(encoded_list)


# ============================================================
# STEP 5: PAD BITS TO FORM COMPLETE BYTES
# ============================================================
# Computer stores data in 8-bit units (1 byte).
# So we add extra 0s if needed.

while len(encoded_bits) % 8 != 0:
    encoded_bits += "0"


# ------------------------------------------------------------
# Convert bit string to byte values
# ------------------------------------------------------------

byte_values = []

# Take 8 bits at a time and convert to decimal
for i in range(0, len(encoded_bits), 8):
    byte_values.append(int(encoded_bits[i:i+8], 2))

# Store as numpy array (compressed data)
encoded_array = np.array(byte_values, dtype=np.uint8)


# ============================================================
# STEP 6: VISUALIZE HUFFMAN BIT STREAM AS IMAGE
# ============================================================
# This is just for visualization, not required in real compression.

# Create square image size
enc_h = int(np.sqrt(len(encoded_bits)))

# Convert bits to array (0 or 1)
bit_img = np.array(list(encoded_bits[:enc_h * enc_h]), dtype=np.uint8)

# Reshape into square and scale:
# 0 → black (0)
# 1 → white (255)
bit_img = bit_img.reshape(enc_h, enc_h) * 255


# ============================================================
# STEP 7: DECODE HUFFMAN CODE (RECONSTRUCT IMAGE)
# ============================================================

# Reverse dictionary → {code : pixel_value}
reverse_codes = {v: k for k, v in codes.items()}

# Create empty image for decoded output
decoded = np.zeros((h, w), dtype=np.uint8)

temp = ""   # temporary string to store bits
idx = 0     # pixel index

# Read bit stream one bit at a time
for bit in encoded_bits:

    temp += bit

    # If bit sequence matches a Huffman code
    if temp in reverse_codes:

        # Assign decoded pixel value
        decoded[idx // w, idx % w] = reverse_codes[temp]

        idx += 1
        temp = ""

        # Stop when all pixels decoded
        if idx >= h * w:
            break


# ============================================================
# STEP 8: DISPLAY ORIGINAL, COMPRESSED BITS, AND DECODED IMAGE
# ============================================================

plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

# Huffman encoded bits visualization
plt.subplot(1, 3, 2)
plt.imshow(bit_img)
plt.title("Huffman Bits")
plt.axis('off')

# Decoded image
plt.subplot(1, 3, 3)
plt.imshow(decoded, cmap='gray')
plt.title("Decoded")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 9: CALCULATE COMPRESSION STATISTICS
# ============================================================

# Original image bits
# Each pixel = 8 bits
original_bits = h * w * 8

# Compressed size
compressed_bits = len(encoded_bits)

# Compression ratio
compression_ratio = original_bits / compressed_bits

print("Original bits:", original_bits)
print("Compressed bits:", compressed_bits)
print("Compression Ratio:", compression_ratio)




# What This Code Does (Simple Flow)
# Image → Count pixel frequency → Build Huffman codes
# → Encode image → Convert to bits → Decode → Compare size