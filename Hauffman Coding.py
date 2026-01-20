import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load grayscale image
# -----------------------------
img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg", 0)
h, w = img.shape

# -----------------------------
# Count pixel frequencies
# -----------------------------
freq = {}
for i in range(h):
    for j in range(w):
        p = img[i, j]
        if p in freq:
            freq[p] += 1
        else:
            freq[p] = 1

# -----------------------------
# Prepare symbols and codes
# -----------------------------
symbols = []
for k in freq:
    symbols.append([k, freq[k]])

codes = {}
for k in freq:
    codes[k] = ""

# -----------------------------
# Build Huffman codes
# -----------------------------
while len(symbols) > 1:
    symbols.sort(key=lambda x: x[1])
    a = symbols.pop(0)
    b = symbols.pop(0)
    for k in codes:
        if k == a[0]:
            codes[k] = "0" + codes[k]
        elif k == b[0]:
            codes[k] = "1" + codes[k]
    symbols.append([a[0]+b[0], a[1] + b[1]])

# -----------------------------
# Encode image using Huffman codes
# -----------------------------
encoded_list = []
for i in range(h):
    for j in range(w):
        encoded_list.append(codes[img[i, j]])

encoded_bits = "".join(encoded_list)

# -----------------------------
# Pad bits to full bytes
# -----------------------------
while len(encoded_bits) % 8 != 0:
    encoded_bits += "0"

# Convert to bytes
byte_values = []
for i in range(0, len(encoded_bits), 8):
    byte_values.append(int(encoded_bits[i:i+8], 2))

encoded_array = np.array(byte_values, dtype=np.uint8)

# -----------------------------
# Visualize Huffman bits
# -----------------------------
enc_h = int(np.sqrt(len(encoded_bits)))
bit_img = np.array(list(encoded_bits[:enc_h*enc_h]), dtype=np.uint8)
bit_img = bit_img.reshape(enc_h, enc_h) * 255  # 0 -> 0, 1 -> 255

# -----------------------------
# Decode Huffman
# -----------------------------
reverse_codes = {v: k for k, v in codes.items()}
decoded = np.zeros((h, w), dtype=np.uint8)
temp = ""
idx = 0

for bit in encoded_bits:
    temp += bit
    if temp in reverse_codes:
        decoded[idx // w, idx % w] = reverse_codes[temp]
        idx += 1
        temp = ""
        if idx >= h * w:
            break

# -----------------------------
# Plot images
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(bit_img)
plt.title("Huffman Bits")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(decoded)
plt.title("Decoded")
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Compression statistics
# -----------------------------
original_bits = h * w * 8
compressed_bits = len(encoded_bits)
compression_ratio = original_bits / compressed_bits

print("Original bits:", original_bits)
print("Compressed bits:", compressed_bits)
print("Compression Ratio:", compression_ratio)
