## L1F23BSCS0225 ASAD ULLAH
import skimage.io as skio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import os
import math


######################### FUNCTIONS ######################
#=================== 1 NEGATIVE ==============================

def doNegative(img):
    img_uint = (img * 255).astype('uint8')
    
    rows, cols = img_uint.shape
    img_neg = np.zeros((rows, cols), dtype='uint8')
    
    for r in range(rows):
        for c in range(cols):
            img_neg[r, c] = 255 - img_uint[r, c]


    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img_uint, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_neg, cmap='gray', vmin=0, vmax=255)
    plt.title("Negative Image , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.show()

    # Step 6: Save
    skio.imsave("Negative.png", img_neg)
    print("Negative image saved as Negative.png")


# ===============================2 THRESHOLD ======================================

def doThreshold(img):
    img_255 = (img * 255).astype(np.uint8)

    height, width = img_255.shape

    threshold_img = np.zeros((height, width), dtype=np.uint8)

    T = int(input("Enter threshold value (0–255): "))
    direction = input("Enter direction (up/down): ").lower()

    for y in range(height):
        for x in range(width):
            pixel = img_255[y, x]

            if direction == "up":
                if pixel >= T:
                    threshold_img[y, x] = 255
                else:
                    threshold_img[y, x] = 0

            elif direction == "down":
                if pixel <= T:
                    threshold_img[y, x] = 255
                else:
                    threshold_img[y, x] = 0

            else:
                print("Invalid direction! Use 'up' or 'down'.")
                exit()

    skio.imsave("Manual_Threshold_Image.png", threshold_img)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_255, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(threshold_img, cmap='gray')
    plt.title(f"Manual Thresholded Image ({direction})  , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Thresholding Done! Image saved as Manual_Threshold_Image.png")
    
# ============================= SCALING ===================================
def doScaled(img):
    rows = len(img)
    cols = len(img[0])

    img_scaled = [[0 for c in range(cols)] for r in range(rows)]

    scale = float(input("Enter scaling factor (e.g. 0.5 for darker, 1.5 for brighter): "))

    for r in range(rows):
        for c in range(cols):
            old_val = img[r][c] * 255
            
            new_val = old_val * scale
            
            if new_val > 255:
                new_val = 255
            elif new_val < 0:
                new_val = 0
            
            img_scaled[r][c] = new_val / 255  

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_scaled, cmap='gray')
    plt.title(f"Scaled Image (Factor = {scale})  , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.show()

    skio.imsave("Scaled_Image.png", img_as_ubyte(img_scaled))

    print("Scaled Image saved as Scaled_Image.png")
    

#========================= 4 LOGARITHM IMAGE =============================

def doLog(img):
    rows = len(img)
    cols = len(img[0])

    img_log = [[0 for c in range(cols)] for r in range(rows)]

    base = float(input("Enter logarithm base (e.g., 10 or 2): "))
    c = float(input("Enter scaling constant (e.g., 40 or 60): "))

    for r in range(rows):
        for col in range(cols):
            pixel = img[r][col] * 255

            safe_pixel = max(pixel, 0)
            s = c * math.log(1 + safe_pixel + 1e-10, base)

            if s > 255:
                s = 255
            elif s < 0:
                s = 0

            img_log[r][col] = s / 255.0 

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_log, cmap='gray')
    plt.title("Logarithmic Transformed Image , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

   
    skio.imsave("Logarithmic_Image.png", img_as_ubyte(img_log))
    print("Logarithmic transform done successfully!")

#========================= 5 ANTI LOGARITHM IMAGE =============================


def doAntiLog(img):
    rows = len(img)
    cols = len(img[0])

    img_antilog = [[0 for c in range(cols)] for r in range(rows)]

    base = float(input("Enter base for Anti-Log (e.g. 10 or 2): "))
    c = float(input("Enter scaling constant (e.g. 10, 20, 50): "))

    for r in range(rows):
        for col in range(cols):
            pixel = img[r][col] * 255      

            normalized = pixel / 255.0

            s = c * (math.pow(base, normalized) - 1)

            if s > 255:
                s = 255
            elif s < 0:
                s = 0

            img_antilog[r][col] = s / 255.0  


    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_antilog, cmap='gray')
    plt.title(f"Anti-Log Image (base={base}, c={c}) , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.show()

    skio.imsave("AntiLog_Image.png", img_as_ubyte(img_antilog))
    print("Anti-Logarithmic Transform done and saved as AntiLog_Image.png")



#========================= 6 POWER LAW IMAGE =============================

def doPowerLaw(img):
    rows = len(img)
    cols = len(img[0])

    img_gamma = [[0 for c in range(cols)] for r in range(rows)]

    c = float(input("Enter scaling constant (e.g., 1.0, 2.0, etc.): "))
    gamma = float(input("Enter gamma value (e.g., 0.5 for bright, 1 for same, >1 for dark): "))

    for r in range(rows):
        for col in range(cols):
            pixel = img[r][col]
            s = c * math.pow(pixel, gamma)

            # Clamp values
            if s > 1:
                s = 1
            elif s < 0:
                s = 0

            img_gamma[r][col] = s

    img_gamma = np.array(img_gamma, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_gamma, cmap='gray')
    plt.title(f"Power-Law Image (γ={gamma}, c={c}) , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.show()

    img_gamma_ubyte = img_as_ubyte(img_gamma)
    skio.imsave("PowerLaw_Image.png", img_gamma_ubyte)
    print("Power-Law (Gamma) Image saved as PowerLaw_Image.png")


#========================= 7 CONTRAST STREATCHING =============================


def doContrastStreaching(img):
    rows, cols = img.shape

    img_cs = np.zeros((rows, cols), dtype=float)

    print("Enter points for piecewise linear transformation:")
    r1 = float(input("Enter r1 (0-255): "))
    s1 = float(input("Enter s1 (0-255): "))
    r2 = float(input("Enter r2 (0-255): "))
    s2 = float(input("Enter s2 (0-255): "))

    for r in range(rows):
        for c in range(cols):
            pixel = img[r][c] * 255

            if pixel < r1:
                s = (s1 / r1) * pixel
            elif pixel < r2:
                s = ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
            else:
                s = ((255 - s2) / (255 - r2)) * (pixel - r2) + s2

          
            img_cs[r][c] = np.clip(s / 255.0, 0, 1)


    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_cs, cmap='gray')
    plt.title("Piecewise Linear (Contrast Stretched) , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    skio.imsave("Contrast_Stretching.png", img_as_ubyte(img_cs))
    print("Contrast Stretching done and saved as Contrast_Stretching.png")
    
#============================ 8 GRAY LEVEL SLICING =========================================    

def doGraySlice(img):
    rows, cols = img.shape
    img_slice = np.zeros((rows, cols), dtype=float)

    A = float(input("Enter starting gray level (A): "))
    B = float(input("Enter ending gray level (B): "))
    if A > B:
        A, B = B, A  # swap if user entered in reverse

    mode = input("Show background? (yes/no): ").lower()

    for r in range(rows):
        for c in range(cols):
            pixel = img[r][c] * 255
            if A <= pixel <= B:
                img_slice[r][c] = 1.0
            else:
                if mode == "yes":
                    img_slice[r][c] = img[r][c]
                else:
                    img_slice[r][c] = 0.0

    img_slice = np.clip(img_slice, 0, 1)  # avoid float >1 or <0

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"Gray Level Slicing ({A}-{B}) , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    skio.imsave("Gray_Level_Slicing.png", img_as_ubyte(img_slice))
    print("Gray Level Slicing done and saved as Gray_Level_Slicing.png")

#============================ 9 BIT PLANE SLICING ========================================= 


def doBitPlaneSlicing(img):
 
    img_ubyte = img_as_ubyte(img)
    rows, cols = img_ubyte.shape

    bit_planes = np.zeros((8, rows, cols), dtype=np.uint8)

    for k in range(8):
        bit_planes[k] = ((img_ubyte >> k) & 1) * 255

    plt.figure(figsize=(12, 8))
    for k in range(8):
        plt.subplot(2, 4, k + 1)
        plt.imshow(bit_planes[k], cmap='gray')
        plt.title(f'Bit Plane {k}  , ASAD ULLAH - L1F23BSCS0225')
        plt.axis('off')

    plt.suptitle("Bit Plane Slicing - Asad Ullah")
    plt.tight_layout()
    plt.show()

    for k in range(8):
        skio.imsave(f"BitPlane_{k}.png", bit_planes[k])

    print("Bit Plane Slicing done — 8 images saved (BitPlane_0.png ... BitPlane_7.png)")


#============================ 10 INTERPOLATED ========================================= 

def doNearestNeighbor(img):
    rows, cols = img.shape
    new_rows = int(input("Enter new number of rows: "))
    new_cols = int(input("Enter new number of cols: "))

    img_nn = np.zeros((new_rows, new_cols), dtype=float)
    row_scale = rows / new_rows
    col_scale = cols / new_cols

    for r in range(new_rows):
        for c in range(new_cols):
            old_r = min(int(round(r * row_scale)), rows - 1)
            old_c = min(int(round(c * col_scale)), cols - 1)
            img_nn[r, c] = img[old_r, old_c] 

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_nn, cmap='gray')
    plt.title("Nearest Neighbor Interpolation , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')
    plt.show()

    img_nn = np.clip(img_nn, 0, 1)
    skio.imsave("NearestNeighbor.png", img_as_ubyte(img_nn))
    print("Nearest Neighbor Interpolation saved as NearestNeighbor.png")


def doBi_linear(img):
    rows, cols = img.shape

    new_rows = int(input("Enter new number of rows: "))
    new_cols = int(input("Enter new number of cols: "))

    img_bi = np.zeros((new_rows, new_cols), dtype=float)
    row_scale = (rows - 1) / (new_rows - 1)
    col_scale = (cols - 1) / (new_cols - 1)

    for r in range(new_rows):
        for c in range(new_cols):
            old_r = r * row_scale
            old_c = c * col_scale

            r1 = int(math.floor(old_r))
            r2 = min(r1 + 1, rows - 1)
            c1 = int(math.floor(old_c))
            c2 = min(c1 + 1, cols - 1)

            a = old_r - r1
            b = old_c - c1

            P1 = (1 - a) * (1 - b) * img[r1, c1]
            P2 = a * (1 - b) * img[r2, c1]
            P3 = (1 - a) * b * img[r1, c2]
            P4 = a * b * img[r2, c2]

            img_bi[r, c] = P1 + P2 + P3 + P4

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_bi, cmap='gray')
    plt.title("Bilinear Interpolation , ASAD ULLAH - L1F23BSCS0225")
    plt.axis('off')
    plt.show()

    img_bi = np.clip(img_bi, 0, 1)
    skio.imsave("Bilinear.png", img_as_ubyte(img_bi))
    print("Bilinear Interpolation saved as Bilinear.png")

    
#============================= MAIN BODY =============================================

choice = 0

img_name = input("Write your image file name (e.g., img.tif): ")


if not os.path.exists(img_name):
    print("Error: Image file not found! Please check the file name or path.")
else:
    img = skio.imread(img_name, as_gray=True)
    img = img.astype(float) / img.max()  # normalize again between 0–1

    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.show()

    while choice != 11:
        print(
            "\n----- IMAGE PROCESSING MENU -----\n"
            "1. Negative\n"
            "2. Threshold image\n"
            "3. Scaled Image\n"
            "4. Logarithm Image\n"
            "5. Anti-Log Image\n"
            "6. Power Law Image\n"
            "7. Piece Wise linear transform for Contrast Stretching\n"
            "8. Gray Level Slicing\n"
            "9. Bit Plane Slicing\n"
            "10. Interpolated Images using Nearest Neighbor and Bi-Linear methods\n"
            "11. Exit\n"
        )

        choice = int(input("CHOICE = "))

        if choice == 1:
            print("You selected Negative Image")
            doNegative(img)

        elif choice == 2:
            print("You selected Threshold Image")
            doThreshold(img)

        elif choice == 3:
             print("You selected scaled Image")
             doScaled(img)
             
        elif choice == 4:
            print("You selected Log Image") 
            doLog(img)
            
        elif choice == 5:
             print("You selected Anti_Log Image")
             doAntiLog(img)
       
        elif choice == 6:
             print("You selected Power Law Image")     
             doPowerLaw(img)
             
        elif choice == 7:
             print("You selected Contrast streatch Image")     
             doContrastStreaching(img)
             
        elif choice == 8:
             print("You selected gray level Image")
             doGraySlice(img)
             
        elif choice == 9:
             print("You selected bit plane Image")     
             doBitPlaneSlicing(img)
             
        elif choice == 10:
             print("You selected interploted Image")     
             choice2 = 0
             print("1. Using nearest neighbour\n"
                   "2. Using Bi-Linear methods")
             choice2 = int(input("choic = "))
             if(choice2 == 1):
                 doNearestNeighbor(img)
             elif(choice2 == 2):
                 doBi_linear(img)
             else:
                 print("Invalid choice")
             
        elif choice == 11:
            print("Exiting program...")
            break
        else:
            print("Invalid option, try again.")
            
            
            
# ASAD ULLAH L1F23BSCS0225            
