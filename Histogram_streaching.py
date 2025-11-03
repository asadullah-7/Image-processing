import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

img = skio.imread("Pollen.tif", as_gray=True)
img = (img * 255).astype('float')

rows, cols = img.shape

r_min = np.min(img)
r_max = np.max(img)

print("r_min =", r_min, " r_max =", r_max)

stretched = np.zeros((rows, cols), dtype='float')

for r in range(rows):
    for c in range(cols):
        s = (img[r, c] - r_min) / (r_max - r_min)
        stretched[r, c] = np.clip(s * 255, 0, 255)

stretched = stretched.astype('uint8')
img = img.astype('uint8')

hist_orig = np.zeros(256, dtype=int)
hist_stretched = np.zeros(256, dtype=int)

for r in range(rows):
    for c in range(cols):
        hist_orig[img[r, c]] += 1
        hist_stretched[stretched[r, c]] += 1

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(stretched, cmap='gray', vmin=0, vmax=255)
plt.title("Contrast Stretched Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(hist_orig, color='blue')
plt.title("Original Histogram")
plt.xlim([0, 255])

plt.subplot(2, 2, 4)
plt.plot(hist_stretched, color='green')
plt.title("Stretched Histogram")
plt.xlim([0, 255])

plt.tight_layout()
plt.show()
