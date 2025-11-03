import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read two grayscale images
src = skio.imread("source.tif", as_gray=True)
ref = skio.imread("referance.png", as_gray=True)

src = (src * 255).astype(np.uint8)
ref = (ref * 255).astype(np.uint8)

rows, cols = src.shape

# Step 2: Compute histogram of both images
hist_src = np.zeros(256, dtype=int)
hist_ref = np.zeros(256, dtype=int)

for r in range(rows):
    for c in range(cols):
        hist_src[src[r, c]] += 1

rows_r, cols_r = ref.shape
for r in range(rows_r):
    for c in range(cols_r):
        hist_ref[ref[r, c]] += 1

# Step 3: Normalize (PDF)
pdf_src = hist_src / (rows * cols)
pdf_ref = hist_ref / (rows_r * cols_r)

# Step 4: Compute CDFs
cdf_src = np.cumsum(pdf_src)
cdf_ref = np.cumsum(pdf_ref)

# Step 5: Create mapping from source → reference
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    diff = np.abs(cdf_ref - cdf_src[i])
    mapping[i] = np.argmin(diff)

# Step 6: Apply mapping to source image
matched = np.zeros_like(src)
for r in range(rows):
    for c in range(cols):
        matched[r, c] = mapping[src[r, c]]

# Step 7: Compute matched histogram
hist_matched = np.zeros(256, dtype=int)
for r in range(rows):
    for c in range(cols):
        hist_matched[matched[r, c]] += 1

# Step 8: Show results
plt.figure(figsize=(12, 10))

# 1. Source Image
plt.subplot(3, 3, 1)
plt.imshow(src, cmap='gray')
plt.title("Source Image")
plt.axis('off')

# 2. Reference Image
plt.subplot(3, 3, 2)
plt.imshow(ref, cmap='gray')
plt.title("Reference Image")
plt.axis('off')

# 3. Matched Image
plt.subplot(3, 3, 3)
plt.imshow(matched, cmap='gray')
plt.title("Histogram Matched Image")
plt.axis('off')

# 4. Source Histogram
plt.subplot(3, 3, 4)
plt.plot(hist_src, color='blue')
plt.title("Source Histogram")
plt.xlim([0, 255])

# 5. Reference Histogram
plt.subplot(3, 3, 5)
plt.plot(hist_ref, color='red')
plt.title("Reference Histogram")
plt.xlim([0, 255])

# 6. Matched Histogram
plt.subplot(3, 3, 6)
plt.plot(hist_matched, color='green')
plt.title("Matched Histogram")
plt.xlim([0, 255])

plt.tight_layout()
plt.show()
