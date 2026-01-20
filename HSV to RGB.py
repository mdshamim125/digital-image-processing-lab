import cv2
import numpy as np
import matplotlib.pyplot as plt


print("Program started...")

img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg")  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

h, w, _ = img.shape


R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

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
plt.tight_layout()
plt.show()

HSV = np.zeros_like(img)

for i in range(h):
    for j in range(w):
        r, g, b = img[i, j]

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Hue
        if delta == 0:
            H = 0
        elif cmax == r:
            H = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            H = 60 * (((b - r) / delta) + 2)
        else:
            H = 60 * (((r - g) / delta) + 4)

        # Saturation
        if cmax == 0:
            S = 0
        else:
            S = delta / cmax

        # Value
        V = cmax

        HSV[i, j] = [H / 360, S, V]

H = HSV[:, :, 0]
S = HSV[:, :, 1]
V = HSV[:, :, 2]

H_disp = (H / H.max()) * 255 if H.max() != 0 else H
S_disp = S * 255
V_disp = V * 255

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(H_disp, cmap='gray')
plt.title("Hue")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(S_disp, cmap='gray')
plt.title("Saturation")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(V_disp, cmap='gray')
plt.title("Value")
plt.axis('off')
plt.tight_layout()
plt.show()


rgb_rec = np.zeros_like(img)

for i in range(h):
    for j in range(w):
        H_ = H[i, j] * 360
        S_ = S[i, j]
        V_ = V[i, j]

        C = V_ * S_
        X = C * (1 - abs((H_ / 60) % 2 - 1))
        m = V_ - C

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

        rgb_rec[i, j] = [r1 + m, g1 + m, b1 + m]


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.clip(HSV, 0, 1))
plt.title("HSV Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.clip(rgb_rec, 0, 1))
plt.title("Reconstructed RGB")
plt.axis('off')
plt.tight_layout()
plt.show()

error = np.mean(np.abs(img - rgb_rec))

print("Image Shape:", img.shape)
print("Image Size:", img.size)
print("Hue Range:", H.min(), H.max())
print("Saturation Range:", S.min(), S.max())
print("Value Range:", V.min(), V.max())
print("Mean Reconstruction Error:", error)
