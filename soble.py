import numpy as np 
import cv2

def laplacian_filter(img):
    h, w, c = img.shape
    filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    pad = 1
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
 
    filtered = np.zeros_like(img, dtype=np.float32)
 
    # convolution
    for i in range(h):
        for j in range(w):
            for k in range(c):
                window = padded[i : i + 3, j : j + 3, k]
                res = np.sum(window * filter)
                filtered[i, j, k] = res
    sharp = img.astype(np.float32) + filtered
    # sharp = custom_clip(img)
    return sharp
 
 
img = cv2.imread("fb profile small.jpg")
 
smoothed = laplacian_filter(img)
cv2.imshow("original", img)
cv2.imshow("smooth", smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows(0)