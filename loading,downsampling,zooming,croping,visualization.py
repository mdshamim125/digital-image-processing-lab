import cv2
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Arup\Desktop\dip\image.jpg",0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()
print("Original Image Size:",img.size)
scale=0.5
downsampled=cv2.resize(
    img,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST

)
plt.imshow(downsampled)
plt.title("Downsampled Impage")
plt.axis('off')
plt.show()
print("Downsampled Image Size:",downsampled.size)
R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(R,cmap='gray')
plt.axis('off')
plt.title("Red Channel")
plt.subplot(1,3,2)
plt.imshow(G,cmap='gray')
plt.axis('off')
plt.title("Green Channel")
plt.subplot(1,3,2)
plt.imshow(B,cmap='gray')
plt.axis('off')
plt.show()
plt.title("Blue Channel")
plt.show()
crop=img[50:250,50:250]
plt.imshow(crop)
plt.title("Cropped Image")
plt.axis('off')
plt.show()
print("Cropped Image Size",crop.size)
print("Original Image Shape:",img.shape)
print("Downsampled Image Shape",downsampled.shape)
print("Cropped Image Shape",crop.shape)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1,3,2)
plt.title("Downsampled image")
plt.imshow(downsampled)
plt.axis('off')
plt.subplot(1,3,3)
plt.title("Cropped Image")
plt.imshow(crop)
plt.axis('off')
plt.show()
