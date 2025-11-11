import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Padding Function
# -----------------------------
def padImage(image, filter_shape, padding_type):
    pad_r = filter_shape[0] // 2
    pad_c = filter_shape[1] // 2
    new_r = image.shape[0] + 2 * pad_r
    new_c = image.shape[1] + 2 * pad_c

    if padding_type == '0':
        padded_img = np.zeros((new_r, new_c), dtype=np.uint8)
    elif padding_type == '255':
        padded_img = 255 * np.ones((new_r, new_c), dtype=np.uint8)
    else:
        return image

    padded_img[pad_r:pad_r + image.shape[0], pad_c:pad_c + image.shape[1]] = image
    return padded_img


# -----------------------------
# Get Window Function
# -----------------------------
def get_window(pad_img, filter_shape, row, col):
    half_r = filter_shape[0] // 2
    half_c = filter_shape[1] // 2
    return pad_img[row - half_r:row + half_r + 1, col - half_c:col + half_c + 1]


# -----------------------------
# Linear Filter (Correlation / Convolution)
# -----------------------------
def applyFilter(image, filter, padding_type='0', convolve=False):
    pad_img = padImage(image, filter.shape, padding_type)
    out_img = np.zeros_like(image, dtype=np.float32)

    if convolve:
        filter = np.flipud(np.fliplr(filter))

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = get_window(pad_img, filter.shape, r + filter.shape[0] // 2, c + filter.shape[1] // 2)
            out_img[r, c] = np.sum(window * filter)

    out_img = np.clip(out_img, 0, 255)
    return out_img.astype(np.uint8)


# -----------------------------
# Statistical Filters
# -----------------------------
def applyStatisticalFilter(image, filter_size, padding_type='0', method='Average'):
    pad_img = padImage(image, filter_size, padding_type)
    out_img = np.zeros_like(image, dtype=np.uint8)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = get_window(pad_img, filter_size, r + filter_size[0] // 2, c + filter_size[1] // 2)

            if method == 'Min':
                out_img[r, c] = np.min(window)
            elif method == 'Max':
                out_img[r, c] = np.max(window)
            elif method == 'Median':
                out_img[r, c] = np.median(window)
            elif method == 'Average':
                out_img[r, c] = np.mean(window)

    return out_img


# -----------------------------
# Show Images Side by Side
# -----------------------------
def show_comparison(original, filtered, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.show()


# -----------------------------
# Main Menu
# -----------------------------
if __name__ == "__main__":
    # Read image
    path = input("Enter image path (e.g., image.jpg): ")
    image = skio.imread(path, as_gray=True)
    image = (image * 255).astype(np.uint8)

    # Step 1: Choose Padding
    print("\nChoose Padding Type:")
    print("1. 0-Padding")
    print("2. 255-Padding")
    pad_choice = input("Enter 0 or 255: ")

    # Step 2: Choose Filter
    print("\nChoose Filter:")
    print("1. Mean Filter")
    print("2. Gaussian Filter")
    print("3. Edge Detection (Convolution)")
    print("4. Median Filter")
    print("5. Min Filter")
    print("6. Max Filter")
    print("7. Average (Statistical) Filter")

    choice = input("Enter your choice: ")

    # Filters
    mean_filter = np.ones((3, 3)) / 9
    gaussian_filter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16
    edge_filter = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

    # Step 3: Apply Selected Filter
    if choice == '1':
        result = applyFilter(image, mean_filter, pad_choice)
        show_comparison(image, result, "Mean Filter")
    elif choice == '2':
        result = applyFilter(image, gaussian_filter, pad_choice)
        show_comparison(image, result, "Gaussian Filter")
    elif choice == '3':
        result = applyFilter(image, edge_filter, pad_choice, convolve=True)
        show_comparison(image, result, "Edge Detection (Convolve)")
    elif choice == '4':
        result = applyStatisticalFilter(image, (3, 3), pad_choice, 'Median')
        show_comparison(image, result, "Median Filter")
    elif choice == '5':
        result = applyStatisticalFilter(image, (3, 3), pad_choice, 'Min')
        show_comparison(image, result, "Min Filter")
    elif choice == '6':
        result = applyStatisticalFilter(image, (3, 3), pad_choice, 'Max')
        show_comparison(image, result, "Max Filter")
    elif choice == '7':
        result = applyStatisticalFilter(image, (3, 3), pad_choice, 'Average')
        show_comparison(image, result, "Average (Statistical) Filter")
    else:
        print("Invalid choice!")
