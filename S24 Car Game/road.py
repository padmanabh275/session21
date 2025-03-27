import cv2
import numpy as np
import os

# Get the absolute path to the images directory
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "citymap.png")

print("Current directory:", current_dir)
print("Image path:", image_path)
print("File exists:", os.path.exists(image_path))

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Failed to load image!")
    print("OpenCV version:", cv2.__version__)
    exit(1)

print("Image loaded successfully. Shape:", image.shape)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection to highlight the roads
edges = cv2.Canny(gray, 50, 150)

# Use morphological operations to enhance the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Optional: Mask the original image to extract roads
masked_image = cv2.bitwise_and(image, image, mask=dilated)

# Save or display the results
output_path = os.path.join(current_dir, "roads_only.png")
cv2.imwrite(output_path, masked_image)
print("Saved output to:", output_path)

cv2.imshow("Roads Only", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
