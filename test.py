import cv2
import matplotlib.pyplot as plt
from skimage import filters

img = cv2.imread(r"E:\Hons 3rd Year 2nd semester books\Digital Image Processing\DIP Lab\fb profile small.jpg", 0)

edges = filters.sobel(img)

plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis("off")
plt.show()
