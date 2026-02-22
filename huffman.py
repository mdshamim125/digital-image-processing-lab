import cv2
import numpy as np

img = cv2.imread("fb profile small.jpg",0)
print(img.shape)

class Node:
    def __init__(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None


def calculate_frequency(img):
    freq = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = int(img[i, j]) 
            if pixel in freq:
                freq[pixel] += 1
            else:
                freq[pixel] = 1
    return freq


res = calculate_frequency(img)


def build_huffman_tree(freq):
    nodes = []
    for pixel in freq:
        nodes.append(Node(pixel, freq[pixel]))

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)

        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        nodes.append(merged)
    return nodes[0]


root = build_huffman_tree(res)


def generate_codes(node, current_code, codes):
    if node is None:
        return

    if node.pixel is not None:
        codes[node.pixel] = current_code
        return

    generate_codes(node.left, current_code + "0", codes)
    generate_codes(node.right, current_code + "1", codes)


huffman_codes = {}
generate_codes(root, "", huffman_codes)


def encode_image(img, codes):
    encoded = ""
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            encoded += codes[img[i, j]]
    return encoded


encoded_data = encode_image(img, huffman_codes)


def decode_image(encoded_data, root, shape):
    decoded = []
    node = root

    for bit in encoded_data:
        if bit == "0":
            node = node.left
        else:
            node = node.right

        if node.pixel is not None:
            decoded.append(node.pixel)
            node = root

    decoded_img = np.array(decoded, dtype=np.uint8).reshape(shape)
    return decoded_img


decoded_img = decode_image(encoded_data, root, img.shape)


###########
cv2.imshow("Original Image", img)
cv2.imshow("Decoded Image", decoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()







# IMAGE
#   ↓
# READ PIXELS
#   ↓
# CALCULATE FREQUENCY
#   ↓
# BUILD HUFFMAN TREE
#   ↓
# GENERATE CODES (0/1)
#   ↓
# ENCODE IMAGE → COMPRESSED DATA
#   ↓
# DECODE USING TREE
#   ↓
# RECONSTRUCT IMAGE