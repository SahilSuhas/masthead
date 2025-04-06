from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

def extract_actual_bounding_box(image_path):
    """ Detects actual product boundaries inside transparent PNG """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale using alpha channel (transparency)
    alpha_channel = image[:, :, 3]  # Extract alpha channel
    _, binary = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Find contours to detect actual product
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])  # Get bounding box
        aspect_ratio = round(w / h, 2)  # Calculate width-to-height ratio
        size_category = "Tall" if h > w else "Wide" if w > h else "Compact"
        return {"x": x, "y": y, "width": w, "height": h, "aspect_ratio": aspect_ratio, "size_category": size_category}
    
    return {"x": 0, "y": 0, "width": 0, "height": 0, "aspect_ratio": 1, "size_category": "Unknown"}

def extract_colors(image_path, num_colors=5):
    """ Extract dominant colors using KMeans clustering """
    image = Image.open(image_path).convert("RGB").resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(pixels)
    dominant_colors = [tuple(map(int, color)) for color in kmeans.cluster_centers_]

    return dominant_colors

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image_path = "temp_image.png"
    
    with open(image_path, "wb") as f:
        f.write(image_data)

    bounding_box = extract_actual_bounding_box(image_path)
    dominant_colors = extract_colors(image_path)

    return {"bounding_box": bounding_box, "dominant_colors": dominant_colors}