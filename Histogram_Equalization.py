import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read grayscale image
img = skio.imread("Pollen.tif", as_gray=True)
img = (img * 255).astype(np.uint8)
rows, cols = img.shape

# Step 2: Manual histogram calculation
hist = np.zeros(256, dtype=int)
for r in range(rows):
    for c in range(cols):
        hist[img[r, c]] += 1

# Step 3: Normalize histogram (PDF)
pdf = hist / (rows * cols)

# Step 4: Compute CDF (Cumulative Distribution Function)
cdf = np.zeros(256, dtype=float)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]

# Step 5: Normalize CDF to 0–255 (mapping)
cdf_min = cdf[np.nonzero(cdf)][0]   # first non-zero value
equalized_map = np.round(((cdf - cdf_min) / (1 - cdf_min)) * 255)

# Step 6: Apply new intensity mapping to create equalized image
equalized_img = np.zeros_like(img)
for r in range(rows):
    for c in range(cols):
        equalized_img[r, c] = equalized_map[img[r, c]]

# Step 7: Compute equalized histogram
hist_eq = np.zeros(256, dtype=int)
for r in range(rows):
    for c in range(cols):
        hist_eq[equalized_img[r, c]] += 1

# Step 8: Display results
plt.figure(figsize=(12, 8))

# 1. Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# 2. Equalized Image
plt.subplot(2, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image (Manual)")
plt.axis('off')

# 3. Original Histogram
plt.subplot(2, 2, 3)
plt.plot(hist, color='blue')
plt.title("Original Histogram")
plt.xlim([0, 255])

# 4. Equalized Histogram
plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='green')
plt.title("Equalized Histogram")
plt.xlim([0, 255])

plt.tight_layout()
plt.show()
