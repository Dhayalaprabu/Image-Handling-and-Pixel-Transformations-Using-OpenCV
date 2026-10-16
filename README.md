# 
# EX. NO. 01: IMAGE HANDLING AND PIXEL TRANSFORMATIONS USING OPENCV
# 

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 
# 1. READ EAGLE IMAGE AS GRAYSCALE
# 

img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Eagle_in_Flight.jpg not found")

# 
# 2. PRINT IMAGE WIDTH, HEIGHT AND CHANNEL

height, width = img.shape
print("1. Eagle Image Dimensions")
print("Width:", width)
print("Height:", height)
print("Channels:", 1)

# 
# 3. DISPLAY GRAYSCALE IMAGE

plt.figure(figsize=(6, 5))
plt.imshow(img, cmap='gray')
plt.title("Eagle in Flight - Grayscale")
plt.axis('off')
plt.show()

# 4. SAVE IMAGE AS PNG

cv2.imwrite('Eagle_Grayscale.png', img)
print("Grayscale image saved as Eagle_Grayscale.png")

# 
# 5. READ SAVED IMAGE AS COLOR IMAGE

img_color = cv2.imread('Eagle_Grayscale.png', cv2.IMREAD_COLOR)

if img_color is None:
    raise FileNotFoundError("Eagle_Grayscale.png not found")

img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# 
# 6. DISPLAY COLOR IMAGE AND DIMENSIONS
# 

height, width, channel = img_color.shape

print("\n2. Color Image Dimensions")
print("Width:", width)
print("Height:", height)
print("Channels:", channel)

plt.figure(figsize=(6, 5))
plt.imshow(img_color)
plt.title("Eagle Image - Color")
plt.axis('off')
plt.show()

# 
# 7. CROP THE EAGLE
# 

# Adjust coordinates if needed to capture the eagle
cropped_img = img[100:500, 100:500]

plt.figure(figsize=(6, 5))
plt.imshow(cropped_img, cmap='gray')
plt.title("Cropped Eagle")
plt.axis('off')
plt.show()

# 
# 8. RESIZE IMAGE BY 2X
# 

resized_img = cv2.resize(
    cropped_img, None,
    fx=2, fy=2,
    interpolation=cv2.INTER_CUBIC
)

plt.figure(figsize=(6, 5))
plt.imshow(resized_img, cmap='gray')
plt.title("Resized Eagle - 2X")
plt.axis('off')
plt.show()

# 
# 9. FLIP IMAGE HORIZONTALLY
# 

flipped_img = cv2.flip(resized_img, 1)

plt.figure(figsize=(6, 5))
plt.imshow(flipped_img, cmap='gray')
plt.title("Horizontally Flipped Eagle")
plt.axis('off')
plt.show()

# 
# 10. READ APOLLO 11 LAUNCH IMAGE
# 

apollo_bgr = cv2.imread('Apollo-11-launch.jpg')

if apollo_bgr is None:
    raise FileNotFoundError("Apollo-11-launch.jpg not found")

apollo = cv2.cvtColor(apollo_bgr, cv2.COLOR_BGR2RGB)

# 
# 11. ADD TEXT TO BOTTOM OF APOLLO IMAGE
# 

text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5
thickness = 2

apollo_annotated = apollo.copy()

text_size, baseline = cv2.getTextSize(
    text, font_face, font_scale, thickness
)

text_x = (apollo.shape[1] - text_size[0]) // 2
text_y = apollo.shape[0] - 20

cv2.putText(
    apollo_annotated,
    text,
    (text_x, text_y),
    font_face,
    font_scale,
    (255, 255, 255),
    thickness,
    cv2.LINE_AA
)

# 
# 12. DRAW MAGENTA RECTANGLE AROUND ROCKET AND TOWER
# 

rect_color = (255, 0, 255)  # Magenta in RGB

# Approximate coordinates; adjust to fit your image
x1 = int(apollo.shape[1] * 0.35)
y1 = int(apollo.shape[0] * 0.10)
x2 = int(apollo.shape[1] * 0.70)
y2 = int(apollo.shape[0] * 0.90)

cv2.rectangle(
    apollo_annotated,
    (x1, y1),
    (x2, y2),
    rect_color,
    3
)

# 13. DISPLAY FINAL ANNOTATED APOLLO IMAGE

plt.figure(figsize=(10, 7))
plt.imshow(apollo_annotated)
plt.title("Apollo 11 Launch - Annotated")
plt.axis('off')
plt.show()

cv2.imwrite(
    'Apollo_11_Annotated.png',
    cv2.cvtColor(apollo_annotated, cv2.COLOR_RGB2BGR)
)

# 14. READ BOY IMAGE
# 

boy_bgr = cv2.imread('boy.jpg')

if boy_bgr is None:
    raise FileNotFoundError("boy.jpg not found")

boy = cv2.cvtColor(boy_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 5))
plt.imshow(boy)
plt.title("Original Boy Image")
plt.axis('off')
plt.show()

# 15. CREATE MATRIX OF ONES FOR BRIGHTNESS ADJUSTMENT
# 

matrix_ones = np.ones(boy.shape, dtype=np.float64)
matrix = matrix_ones * 50

boy_float = boy.astype(np.float64)

# 
# 16. CREATE BRIGHTER AND DARKER IMAGES

img_brighter = np.clip(
    boy_float + matrix, 0, 255
).astype(np.uint8)

img_darker = np.clip(
    boy_float - matrix, 0, 255
).astype(np.uint8)

# 
# 17. DISPLAY ORIGINAL, DARKER AND BRIGHTER IMAGES
# 

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(boy)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_darker)
plt.title("Darker Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_brighter)
plt.title("Brighter Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# 18. MODIFY IMAGE CONTRAST
# 

matrix1 = np.ones(boy.shape, dtype=np.float64) * 1.1
matrix2 = np.ones(boy.shape, dtype=np.float64) * 1.2

img_higher1 = np.clip(
    boy_float * matrix1, 0, 255
).astype(np.uint8)

img_higher2 = np.clip(
    boy_float * matrix2, 0, 255
).astype(np.uint8)

# 
# 19. DISPLAY ORIGINAL, LOWER AND HIGHER CONTRAST IMAGES
# 

img_lower = np.clip(
    boy_float * 0.8, 0, 255
).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(boy)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_lower)
plt.title("Lower Contrast")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_higher1)
plt.title("Higher Contrast - 1.1")
plt.axis('off')

plt.tight_layout()
plt.show()

# 
# 20. SPLIT IMAGE INTO B, G, R CHANNELS
# 

boy_bgr = cv2.cvtColor(boy, cv2.COLOR_RGB2BGR)
B, G, R = cv2.split(boy_bgr)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

# 
# 21. MERGE R, G, B CHANNELS
# 

merged_rgb = cv2.merge((R, G, B))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(boy)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# 
# 22. SPLIT IMAGE INTO H, S, V CHANNELS
# 

hsv = cv2.cvtColor(boy, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsv)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(H, cmap='gray')
plt.title("Hue Channel")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(S, cmap='gray')
plt.title("Saturation Channel")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(V, cmap='gray')
plt.title("Value Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

# 23. MERGE H, S, V CHANNELS
# 

merged_hsv = cv2.merge((H, S, V))
merged_hsv_rgb = cv2.cvtColor(
    merged_hsv, cv2.COLOR_HSV2RGB
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(boy)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(merged_hsv_rgb)
plt.title("Merged HSV Image")
plt.axis('off')

plt.tight_layout()
plt.show()

print("All 23 image-processing tasks completed successfully.")
