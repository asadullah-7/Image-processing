import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

# ---------------- PADDING FUNCTION ----------------
def padImage(image, filter_shape, padding_type):
    pad_rows = filter_shape[0] // 2
    pad_cols = filter_shape[1] // 2

    new_row_count = image.shape[0] + 2 * pad_rows
    new_col_count = image.shape[1] + 2 * pad_cols

    if padding_type == '0':
        Padded_img = np.zeros((new_row_count, new_col_count), dtype=np.uint8)
    elif padding_type == '255':
        Padded_img = 255 * np.ones((new_row_count, new_col_count), dtype=np.uint8)
    else:
        return image

    # Insert original image in center
    Padded_img[pad_rows:pad_rows + image.shape[0],
               pad_cols:pad_cols + image.shape[1]] = image

    return Padded_img


# ---------------- WINDOW EXTRACTION FUNCTION ----------------
def get_window(pad_img, filter_shape, row, col):
    half_r = filter_shape[0] // 2
    half_c = filter_shape[1] // 2
    window = pad_img[row - half_r: row + half_r + 1,
                     col - half_c: col + half_c + 1]
    return window


# ---------------- FILTER APPLY FUNCTION (LINEAR) ----------------
def applyFilter(image, filter, padding_type, convolve=False):
    padded_img = padImage(image, filter.shape, padding_type)
    out_img = np.zeros_like(image)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = get_window(padded_img, filter.shape, r + filter.shape[0]//2, c + filter.shape[1]//2)

            if convolve:
                filter_used = np.flipud(np.fliplr(filter))
            else:
                filter_used = filter

            out_img[r, c] = np.clip(np.sum(window * filter_used), 0, 255)

    return out_img


# ---------------- STATISTICAL FILTER FUNCTION ----------------
def applyStatisticalFilter(image, filter_size, padding_type, method):
    padded_img = padImage(image, filter_size, padding_type)
    out_img = np.zeros_like(image)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = get_window(padded_img, filter_size, r + filter_size[0]//2, c + filter_size[1]//2)
            if method == 'min':
                out_img[r, c] = np.min(window)
            elif method == 'max':
                out_img[r, c] = np.max(window)
            elif method == 'median':
                out_img[r, c] = np.median(window)
            elif method == 'average':
                out_img[r, c] = np.mean(window)

    return out_img


# ---------------- SHOW COMPARISON FUNCTION ----------------
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


# ---------------- MAIN PROGRAM (MENU) ----------------
image = skio.imread('livingroom.tif', as_gray=True)
image = (image * 255).astype(np.uint8)

while True:
    print("\n=== Image Processing Menu ===")
    print("1. Apply Mean Filter")
    print("2. Apply Gaussian Filter")
    print("3. Apply Median Filter")
    print("4. Apply Min Filter")
    print("5. Apply Max Filter")
    print("6. Apply Convolution Filter (Edge Detection)")
    print("0. Exit")

    choice = input("Enter your choice: ")

    if choice == '0':
        break

    pad_choice = input("Enter padding type (0 or 255 or none): ").lower()

    if choice == '1':
        mean_filter = (1/9) * np.ones((3,3))
        filtered = applyFilter(image, mean_filter, pad_choice)
        show_comparison(image, filtered, "Mean Filter")

    elif choice == '2':
        gaussian_filter = (1/16) * np.array([[1,2,1],
                                             [2,4,2],
                                             [1,2,1]])
        filtered = applyFilter(image, gaussian_filter, pad_choice)
        show_comparison(image, filtered, "Gaussian Filter")

    elif choice == '3':
        filtered = applyStatisticalFilter(image, (3,3), pad_choice, 'median')
        show_comparison(image, filtered, "Median Filter")

    elif choice == '4':
        filtered = applyStatisticalFilter(image, (3,3), pad_choice, 'min')
        show_comparison(image, filtered, "Min Filter")

    elif choice == '5':
        filtered = applyStatisticalFilter(image, (3,3), pad_choice, 'max')
        show_comparison(image, filtered, "Max Filter")

    elif choice == '6':
        edge_filter = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
        filtered = applyFilter(image, edge_filter, pad_choice, convolve=True)
        show_comparison(image, filtered, "Edge Detection")

    else:
        print("Invalid choice. Try again!")

print("Program ended.")
