from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response, Request, Form
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import pytesseract
import os
import io
from typing import List, Dict, Any, Tuple
import cairosvg
import tempfile
import base64
import json
from dataclasses import dataclass, field, asdict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
import re

app = FastAPI()

# Enable CORS for all origins (including Figma plugin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a directory for storing temporary files
TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Default color that will be returned if no processing has been done
current_base_color = "#E5BE3D"

# Add these global variables for temporary storage
# Store processed mastheads
processed_mastheads = []

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

@app.get("/get-base-color")
async def get_base_color():
    """
    Returns the current base color determined from the most recent image processing.
    Used by the Figma plugin to retrieve the latest color.
    """
    try:
        with open(os.path.join(TEMP_DIR, "base_color.txt"), "r") as f:
            color = f.read().strip()
        return {"base_color": color}
    except FileNotFoundError:
        return {"base_color": current_base_color}

def has_transparency(image_path):
    """ Detects if an image has transparency """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if img.shape[-1] == 4:  # If image has 4 channels (RGBA)
        return bool(np.any(img[:, :, 3] < 255))  # True if any pixel is not fully opaque
    return False

def extract_actual_bounding_box(image_path):
    """ Detects actual product boundaries inside transparent PNG """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale using alpha channel (transparency)
    if image.shape[-1] == 4:  # If RGBA
        alpha_channel = image[:, :, 3]
        _, binary = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

        # Find contours to detect actual product
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            if w > 1 and h > 1:  # Ensure the bounding box is valid
                aspect_ratio = round(w / h, 2)
                size_category = "Tall" if h > w else "Wide" if w > h else "Compact"
                return {"x": x, "y": y, "width": w, "height": h, "aspect_ratio": aspect_ratio, "size_category": size_category}

    return {"x": 0, "y": 0, "width": 0, "height": 0, "aspect_ratio": 1, "size_category": "Unknown"}

def extract_colors(image_path, num_colors=5):
    """ Extract dominant colors using KMeans clustering with frequencies, filtering out near-black and near-white colors """
    image = Image.open(image_path).convert("RGB").resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(pixels)
    
    # Count occurrences of each cluster
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    
    # Get the colors with their frequencies
    colors_with_freq = []
    for i in range(len(kmeans.cluster_centers_)):
        color = tuple(map(int, kmeans.cluster_centers_[i]))
        frequency = int(label_counts[i])
        
        # Filter out colors close to black and white
        r, g, b = color
        if not (r > 200 and g > 200 and b > 200) and not (r < 50 and g < 50 and b < 50):
            colors_with_freq.append((color, frequency))
    
    # Sort by frequency (highest first)
    colors_with_freq.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the colors for the return value
    filtered_colors = [color for color, _ in colors_with_freq]
    
    return filtered_colors[:num_colors]  # Return only the top num_colors

def merge_color_palettes(colors1, colors2, image1_path=None, image2_path=None):
    """ Merges two color palettes and assigns hierarchy based on frequency """
    # Get the original image to count actual frequencies
    try:
        # Use provided image paths if given
        if image1_path and image2_path and os.path.exists(image1_path) and os.path.exists(image2_path):
            image1 = Image.open(image1_path).convert("RGB").resize((100, 100))
            image2 = Image.open(image2_path).convert("RGB").resize((100, 100))
        else:
            # Fallback to default paths for backward compatibility
            default_path1 = os.path.join(TEMP_DIR, "temp_image1.png")
            default_path2 = os.path.join(TEMP_DIR, "temp_image2.png")
            
            if os.path.exists(default_path1) and os.path.exists(default_path2):
                image1 = Image.open(default_path1).convert("RGB").resize((100, 100))
                image2 = Image.open(default_path2).convert("RGB").resize((100, 100))
            else:
                # If files don't exist, use simplified approach without frequency analysis
                # Just combine the colors from both palettes
                merged = colors1 + colors2
                unique_colors = list(set(merged))
                return {
                    "base_color": colors1[0] if colors1 else (128, 128, 128),
                    "secondary_color": colors2[0] if colors2 else (128, 128, 128),
                    "accent_colors": unique_colors[2:5] if len(unique_colors) > 2 else []
                }
                
        pixels1 = np.array(image1).reshape(-1, 3)
        pixels2 = np.array(image2).reshape(-1, 3)
        all_pixels = np.vstack([pixels1, pixels2])
        
        # Combine both palettes
        merged = colors1 + colors2
        unique_colors = list(set(merged))  # Remove duplicates

        # Count frequency of each unique color across both images
        color_frequencies = {}
        
        for color in unique_colors:
            # Calculate how close each pixel is to this color
            distances = np.sum((all_pixels - np.array(color)) ** 2, axis=1)
            # Count pixels that are close to this color (using a threshold)
            color_frequencies[color] = np.sum(distances < 1000)  # Adjust threshold as needed
        
        # Sort by frequency
        sorted_colors = sorted(unique_colors, key=lambda c: color_frequencies.get(c, 0), reverse=True)

        return {
            "base_color": sorted_colors[0] if sorted_colors else (128, 128, 128),  # The most frequent color
            "secondary_color": sorted_colors[1] if len(sorted_colors) > 1 else sorted_colors[0] if sorted_colors else (128, 128, 128),
            "accent_colors": sorted_colors[2:] if len(sorted_colors) > 2 else []
        }
    except Exception as e:
        print(f"Error in merge_color_palettes: {str(e)}")
        # Provide a fallback in case of any error
        return {
            "base_color": colors1[0] if colors1 else (128, 128, 128),
            "secondary_color": colors2[0] if colors2 else (128, 128, 128),
            "accent_colors": []
        }

@app.post("/process-images/")
async def process_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Save uploaded files temporarily
        temp_file_path = os.path.join(TEMP_DIR, "temp_image1.png")
        temp_image2_path = os.path.join(TEMP_DIR, "temp_image2.png")
        
        with open(temp_file_path, "wb") as f:
            f.write(await file1.read())
        
        with open(temp_image2_path, "wb") as f:
            f.write(await file2.read())
        
        # Extract dominant colors from both images
        colors1 = extract_colors(temp_file_path)
        colors2 = extract_colors(temp_image2_path)
        
        # Merge color palettes and determine hierarchy
        merged_colors = merge_color_palettes(colors1, colors2, temp_file_path, temp_image2_path)
        
        # Get the base color directly from the dictionary
        base_color = merged_colors["base_color"]
        
        # Convert RGB to hex
        hex_color = "#{:02X}{:02X}{:02X}".format(base_color[0], base_color[1], base_color[2])
        
        # Save color globally for the GET endpoint to access
        global current_base_color
        current_base_color = hex_color
        
        # Save the color to a file so it persists across requests on Render
        with open(os.path.join(TEMP_DIR, "base_color.txt"), "w") as f:
            f.write(hex_color)
        
        # Return only the hex code of the base color
        return {"base_color": hex_color}
        
    except Exception as e:
        print(f"Error in process_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_image2_path):
                os.remove(temp_image2_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up temporary files: {cleanup_error}")

# New functions for text detection

def detect_text_in_image(image_path):
    """
    Detect text in a transparent PNG image and return bounding boxes.
    """
    # Load the image with the alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Check if image has alpha channel (is a transparent PNG)
    if img.shape[-1] != 4:
        raise ValueError("Image does not have an alpha channel (not a transparent PNG)")
    
    # Create a copy of the image for preprocessing
    processed_img = img.copy()
    
    # Convert transparent PNG to white background for better OCR
    alpha_channel = processed_img[:, :, 3]
    _, alpha_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    
    # Create a white background image
    white_background = np.ones(processed_img.shape[:3], dtype=np.uint8) * 255
    
    # Blend the original image with white background based on alpha
    for c in range(3):  # RGB channels
        processed_img[:, :, c] = np.where(alpha_channel[:, :] > 0, 
                                          processed_img[:, :, c], 
                                          white_background[:, :, c])
    
    # Remove the alpha channel
    processed_img = processed_img[:, :, :3]
    
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text visibility
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Additional preprocessing to enhance text visibility
    # Edge enhancement to detect text boundaries better
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(thresh, -1, kernel)
    
    # Use Tesseract to detect text
    # Configure tesseract to detect text blocks with confidence score
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(enhanced, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Collect detected text regions with high confidence
    text_regions = []
    for i in range(len(data['text'])):
        # Filter for non-empty text with reasonable confidence
        if data['text'][i].strip() and int(data['conf'][i]) > 40:  # Adjust confidence threshold as needed
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Skip very small regions (likely noise)
            if w < 5 or h < 5:
                continue
                
            text_regions.append({
                "text": data['text'][i],
                "confidence": data['conf'][i],
                "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            })

    return {
        "text_regions": text_regions,
        "image_dimensions": {
            "width": img.shape[1],
            "height": img.shape[0]
        }
    }

def visualize_text_detection(image_path, text_data):
    """
    Create a visualization of detected text regions on the original image.
    """
    # Load the original image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Prepare the image for visualization
    # If alpha channel exists, create a white background
    if img.shape[-1] == 4:
        # Create a white background
        white_bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        
        # Blend the image with white background based on alpha
        alpha_channel = img[:, :, 3] / 255.0
        rgb_channels = img[:, :, :3]
        
        for c in range(3):
            white_bg[:, :, c] = white_bg[:, :, c] * (1 - alpha_channel) + rgb_channels[:, :, c] * alpha_channel
            
        vis_img = white_bg
    else:
        vis_img = img[:, :, :3].copy()
    
    # Draw bounding boxes around detected text regions
    for i, region in enumerate(text_data["text_regions"]):
        # Get the bounding box
        bbox = region["bbox"]
        text = region["text"]
        confidence = region["confidence"]
        
        # Convert bbox to vertices for drawing
        vertices = np.array(bbox, np.int32).reshape((-1, 1, 2))
        
        # Draw filled polygon for the text region (with transparency)
        overlay = vis_img.copy()
        cv2.fillPoly(overlay, [vertices], (0, 200, 255))  # Light orange fill
        
        # Apply overlay with transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0, vis_img)
        
        # Draw bounding box outline
        cv2.polylines(vis_img, [vertices], True, (0, 0, 255), 2)  # Red outline
        
        # Add text label
        text_to_show = f"{text} ({confidence:.1f}%)"
        cv2.putText(vis_img, text_to_show, 
                   (bbox[0][0], bbox[0][1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text with thickness for outline
        cv2.putText(vis_img, text_to_show, 
                   (bbox[0][0], bbox[0][1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
    
    # Add title to the visualization
    title = f"Detected {len(text_data['text_regions'])} Text Regions"
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 1)
    
    # Save the visualization
    output_path = os.path.join(TEMP_DIR, "text_visualization.png")
    cv2.imwrite(output_path, vis_img)
    
    return output_path

@app.post("/detect-text/")
async def detect_text_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to detect text in a transparent PNG image and return 
    bounding boxes and visualization.
    """
    image_path = os.path.join(TEMP_DIR, "temp_text_image.png")
    
    try:
        # Save uploaded file
        with open(image_path, "wb") as f:
            f.write(await file.read())
        
        # Check if the file is a valid image
        img_test = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_test is None:
            raise ValueError("Invalid image file")
        
        # Detect text in the image
        text_data = detect_text_in_image(image_path)
        
        # Create visualization if text was detected
        visualization_path = None
        if text_data["text_regions"]:
            visualization_path = visualize_text_detection(image_path, text_data)
        
        # Return the results
        return {
            "text_detection_results": text_data,
            "visualization_path": visualization_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New feature: Layout products in SVG mask
@dataclass
class ProductPlacement:
    image: str
    position: Dict[str, int]
    size: Dict[str, int]
    rotation: int = 0

@dataclass
class LayoutResult:
    canvas: Dict[str, int]
    safe_zone_polygon: List[List[int]]
    product_placements: List[ProductPlacement] = field(default_factory=list)
    composed_image: str = ""

def convert_svg_to_mask(svg_content: bytes, width: int = 1440, height: int = 1024) -> np.ndarray:
    """Convert SVG content to a binary mask that accurately represents the SVG shape."""
    try:
        # Save SVG content to a temporary file for debugging
        debug_svg_path = os.path.join(TEMP_DIR, "debug_svg.svg")
        with open(debug_svg_path, "wb") as f:
            f.write(svg_content)
        
        # Try to extract viewBox from SVG to get correct scaling
        import re
        svg_text = svg_content.decode('utf-8')
        
        # Look for viewBox or width/height
        viewbox_match = re.search(r'viewBox=["\']([\d\s.]+)["\']', svg_text)
        width_match = re.search(r'width=["\']([\d.]+)["\']', svg_text)
        height_match = re.search(r'height=["\']([\d.]+)["\']', svg_text)
        
        # If viewBox is found, use it for scaling
        svg_width, svg_height = width, height
        if viewbox_match:
            viewbox = viewbox_match.group(1).split()
            if len(viewbox) == 4:
                _, _, svg_width, svg_height = map(float, viewbox)
        elif width_match and height_match:
            svg_width = float(width_match.group(1))
            svg_height = float(height_match.group(1))
        
        print(f"SVG dimensions: {svg_width}x{svg_height}")
            
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Convert SVG to PNG using cairosvg with proper dimensions
            cairosvg.svg2png(
                bytestring=svg_content, 
                write_to=tmp.name, 
                output_width=width, 
                output_height=height,
                scale=max(width/svg_width, height/svg_height) if svg_width and svg_height else 1
            )
            
            # Read the PNG as grayscale
            mask_img = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED)
            
            # Check if mask was created successfully
            if mask_img is None:
                raise ValueError("Failed to convert SVG to image mask")
            
            # If image has alpha channel, use it as mask
            if mask_img.shape[-1] == 4:
                # Use alpha channel as mask
                alpha_channel = mask_img[:, :, 3]
                _, binary_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            else:
                # Convert to binary mask - handle both black and white fills
                gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                # First try with normal thresholding (assuming white background)
                _, binary_mask1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                # Then try inverted thresholding (assuming black background)
                _, binary_mask2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                
                # Choose the mask that has some white pixels (indicating content)
                if np.sum(binary_mask1) > 0:
                    binary_mask = binary_mask1
                else:
                    binary_mask = binary_mask2
            
            # Save mask for debugging
            debug_mask_path = os.path.join(TEMP_DIR, "debug_mask.png")
            cv2.imwrite(debug_mask_path, binary_mask)
            
            # Clean up
            os.unlink(tmp.name)
            
            return binary_mask
    except Exception as e:
        print(f"Error in convert_svg_to_mask: {str(e)}")
        # Create a fallback mask (a simple rectangle) for debugging
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        fallback_mask[50:height-50, 50:width-50] = 255  # Simple rectangle with margin
        fallback_path = os.path.join(TEMP_DIR, "fallback_mask.png")
        cv2.imwrite(fallback_path, fallback_mask)
        return fallback_mask

def extract_svg_polygon(svg_content: bytes, width: int = 1440, height: int = 1024) -> List[List[int]]:
    """Extract polygon points from SVG for visualization and layout."""
    try:
        # Try to directly extract path data from SVG
        import re
        svg_text = svg_content.decode('utf-8')
        
        # Look for path data - this will help with complex SVGs from Figma
        path_match = re.search(r'<path[^>]*d=["\'](.*?)["\']', svg_text)
        
        if path_match:
            print("Found SVG path data - could use this for more precise polygon extraction")
        
        # Convert SVG to mask
        mask = convert_svg_to_mask(svg_content, width, height)
        
        # Ensure the mask has some content
        if np.sum(mask) == 0:
            print("Warning: Mask is empty, using fallback rectangle")
            # Create a fallback mask (a simple rectangle) for debugging
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[50:height-50, 50:width-50] = 255  # Simple rectangle with margin
        
        # Find contours in the mask - use RETR_EXTERNAL for outer contour only
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: print number of contours found
        print(f"Found {len(contours)} contours in the SVG mask")
        
        # Get the largest contour (assumed to be the main shape)
        if not contours:
            # If no contours found, create a simple rectangle
            print("No contours found, creating fallback rectangle")
            polygon = [[50, 50], [width-50, 50], [width-50, height-50], [50, height-50]]
            return polygon
        
        # Create a visualization of all contours for debugging
        contour_debug = np.zeros((height, width, 3), dtype=np.uint8)
        for i, contour in enumerate(contours):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.drawContours(contour_debug, [contour], -1, color, 2)
            # Add area text
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(contour_debug, f"{area:.1f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        contour_debug_path = os.path.join(TEMP_DIR, "all_contours_debug.png")
        cv2.imwrite(contour_debug_path, contour_debug)
        
        # Find the largest contour by area
        largest_contour_idx = 0
        largest_area = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour_idx = i
                
        main_contour = contours[largest_contour_idx]
        
        # Debug: print area of largest contour
        print(f"Largest contour area: {largest_area}")
        
        # Get a more accurate representation of the contour
        epsilon = 0.001 * cv2.arcLength(main_contour, True)  # Use smaller epsilon for better accuracy
        approx_poly = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Create a more precise polygon with more points for complex shapes
        if len(approx_poly) < 10:  # If we have too few points, get more detail
            epsilon = 0.0005 * cv2.arcLength(main_contour, True)
            approx_poly = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Convert to list of [x, y] points
        polygon = [list(map(int, point[0])) for point in approx_poly]
        
        # Debug: save contour visualization
        contour_vis = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.drawContours(contour_vis, [main_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(contour_vis, [approx_poly], -1, (0, 0, 255), 2)
        # Add points
        for i, point in enumerate(polygon):
            cv2.circle(contour_vis, tuple(point), 5, (255, 0, 0), -1)
            cv2.putText(contour_vis, str(i), (point[0]+5, point[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        debug_contour_path = os.path.join(TEMP_DIR, "debug_contour.png")
        cv2.imwrite(debug_contour_path, contour_vis)
        
        return polygon
        
    except Exception as e:
        print(f"Error in extract_svg_polygon: {str(e)}")
        # Return a default rectangle as fallback
        return [[50, 50], [width-50, 50], [width-50, height-50], [50, height-50]]

def get_visible_content_bbox(image_path: str) -> Tuple[int, int, int, int]:
    """Get the bounding box of visible (non-transparent) content in a PNG."""
    # Load image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return 0, 0, 100, 100  # Default fallback values
        
    if img.shape[-1] != 4:
        # If image doesn't have alpha channel, return full image bbox
        return 0, 0, img.shape[1], img.shape[0]
    
    # Create binary mask from alpha channel
    alpha_channel = img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours to detect actual product boundary
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, img.shape[1], img.shape[0]
    
    # Get bounding rectangle of the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Debug: Save a visualization of the mask
    debug_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(debug_mask, [main_contour], 0, (0, 255, 0), 2)
    cv2.rectangle(debug_mask, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(f"debug_mask_{os.path.basename(image_path)}.png", debug_mask)
    
    return x, y, w, h

def crop_to_visible_content(image_path: str) -> str:
    """
    Crop an image to only include visible (non-transparent) content.
    Returns the path to the cropped image.
    """
    # Get the bounding box of visible content
    x, y, w, h = get_visible_content_bbox(image_path)
    
    # Load the original image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return image_path
    
    # Check if the image has an alpha channel
    if img.shape[-1] != 4:
        print(f"Warning: Image {image_path} does not have an alpha channel")
        return image_path
    
    # Crop the image to the visible content
    if w > 0 and h > 0:
        cropped = img[y:y+h, x:x+w]
        
        # Generate output path
        base_name = os.path.basename(image_path)
        name_parts = os.path.splitext(base_name)
        cropped_path = f"{name_parts[0]}_cropped{name_parts[1]}"
        
        # Save the cropped image
        cv2.imwrite(cropped_path, cropped)
        
        print(f"Cropped image from {img.shape[:2]} to {cropped.shape[:2]}, saved as {cropped_path}")
        return cropped_path
    
    # If no visible content was found, return the original image path
    return image_path

def classify_product_size(width: int, height: int) -> str:
    """
    Classify product size based on aspect ratio.
    Tall: Height > Width
    Wide: Width > Height
    Compact: Width ≈ Height
    """
    aspect_ratio = width / height
    
    if aspect_ratio < 0.9:  # Height is significantly greater than width
        return "Tall"
    elif aspect_ratio > 1.1:  # Width is significantly greater than height
        return "Wide"
    else:  # Width and height are approximately equal
        return "Compact"

def create_overlap_vector_shape(
    product1_img: np.ndarray, 
    product2_img: np.ndarray, 
    p1_x_pos: int, 
    p1_y_pos: int, 
    p2_x_pos: int, 
    p2_y_pos: int,
    canvas_width: int,
    canvas_height: int
) -> Dict[str, Any]:
    """
    Create a vector shape of the pixel area overlap between two products.
    Also creates base vectors (5px high strips) at the bottom of each product.
    Returns the path to the SVG file and overlap coordinates.
    """
    print("Creating overlap vector shape...")
    
    # Create binary masks for visible content of both products on the canvas
    canvas_mask1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas_mask2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Extract alpha channels
    alpha1 = product1_img[:, :, 3] if product1_img.shape[-1] == 4 else np.ones(product1_img.shape[:2], dtype=np.uint8) * 255
    alpha2 = product2_img[:, :, 3] if product2_img.shape[-1] == 4 else np.ones(product2_img.shape[:2], dtype=np.uint8) * 255
    
    # Fill the masks based on product positions and transparency
    h1, w1 = product1_img.shape[:2]
    h2, w2 = product2_img.shape[:2]
    
    # Place product 1 visible pixels on canvas mask
    for y in range(h1):
        for x in range(w1):
            if alpha1[y, x] > 127:  # Only visible pixels
                canvas_y = p1_y_pos + y
                canvas_x = p1_x_pos + x
                if 0 <= canvas_y < canvas_height and 0 <= canvas_x < canvas_width:
                    canvas_mask1[canvas_y, canvas_x] = 255
    
    # Place product 2 visible pixels on canvas mask
    for y in range(h2):
        for x in range(w2):
            if alpha2[y, x] > 127:  # Only visible pixels
                canvas_y = p2_y_pos + y
                canvas_x = p2_x_pos + x
                if 0 <= canvas_y < canvas_height and 0 <= canvas_x < canvas_width:
                    canvas_mask2[canvas_y, canvas_x] = 255
    
    # Calculate overlap using bitwise AND
    overlap_mask = cv2.bitwise_and(canvas_mask1, canvas_mask2)
    
    # Save masks for debugging
    cv2.imwrite(os.path.join(TEMP_DIR, "debug_mask1.png"), canvas_mask1)
    cv2.imwrite(os.path.join(TEMP_DIR, "debug_mask2.png"), canvas_mask2)
    cv2.imwrite(os.path.join(TEMP_DIR, "debug_overlap_mask.png"), overlap_mask)
    
    # Find the bottom edge of visible pixels for each product
    # For product 1
    p1_base_segments = []
    p1_bottom_edge = np.zeros((canvas_width,), dtype=bool)
    
    for x in range(w1):
        for y in range(h1-1, -1, -1):  # Start from bottom, move up
            if alpha1[y, x] > 127:  # Found a visible pixel
                canvas_x = p1_x_pos + x
                canvas_y = p1_y_pos + y
                if 0 <= canvas_y < canvas_height and 0 <= canvas_x < canvas_width:
                    p1_bottom_edge[canvas_x] = True
                break
    
    # Form continuous segments for product 1
    segment_start = -1
    for x in range(canvas_width):
        if p1_bottom_edge[x] and segment_start == -1:
            segment_start = x
        elif not p1_bottom_edge[x] and segment_start != -1:
            p1_base_segments.append((segment_start, x-1))
            segment_start = -1
    
    # Don't forget the last segment if it ends at the edge
    if segment_start != -1:
        p1_base_segments.append((segment_start, canvas_width-1))
    
    # For product 2
    p2_base_segments = []
    p2_bottom_edge = np.zeros((canvas_width,), dtype=bool)
    
    for x in range(w2):
        for y in range(h2-1, -1, -1):  # Start from bottom, move up
            if alpha2[y, x] > 127:  # Found a visible pixel
                canvas_x = p2_x_pos + x
                canvas_y = p2_y_pos + y
                if 0 <= canvas_y < canvas_height and 0 <= canvas_x < canvas_width:
                    p2_bottom_edge[canvas_x] = True
                break
    
    # Form continuous segments for product 2
    segment_start = -1
    for x in range(canvas_width):
        if p2_bottom_edge[x] and segment_start == -1:
            segment_start = x
        elif not p2_bottom_edge[x] and segment_start != -1:
            p2_base_segments.append((segment_start, x-1))
            segment_start = -1
    
    # Don't forget the last segment if it ends at the edge
    if segment_start != -1:
        p2_base_segments.append((segment_start, canvas_width-1))
    
    print(f"Found {len(p1_base_segments)} base segments for product 1")
    print(f"Found {len(p2_base_segments)} base segments for product 2")
    
    # Check if there's overlap
    if np.sum(overlap_mask) == 0:
        print("No overlap detected between products.")
        return {"svg_path": None, "coordinates": [], "overlap_percentage": 0}
    
    # Find contours of the overlap
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in overlap.")
        return {"svg_path": None, "coordinates": [], "overlap_percentage": 0}
    
    # Create a visualization image
    visualization = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    # Draw the first product in blue with transparency
    visualization[canvas_mask1 > 0] = [0, 0, 255, 100]
    # Draw the second product in green with transparency
    visualization[canvas_mask2 > 0] = [0, 255, 0, 100]
    # Draw the overlap in red
    visualization[overlap_mask > 0] = [255, 0, 0, 200]
    
    # Draw the base segments for visualization
    base_height = 5  # Height of base vector in pixels
    for start_x, end_x in p1_base_segments:
        for y in range(5):  # 5px height
            for x in range(start_x, end_x + 1):
                # Get y-coordinate of the bottom pixel at this x
                for y_check in range(canvas_height-1, -1, -1):
                    if canvas_mask1[y_check, x] > 0:
                        bottom_y = y_check
                        break
                else:
                    continue
                
                # Draw base vector pixel
                if 0 <= bottom_y + y + 1 < canvas_height:
                    visualization[bottom_y + y + 1, x] = [0, 0, 0, 255]  # Black, fully opaque
    
    for start_x, end_x in p2_base_segments:
        for y in range(5):  # 5px height
            for x in range(start_x, end_x + 1):
                # Get y-coordinate of the bottom pixel at this x
                for y_check in range(canvas_height-1, -1, -1):
                    if canvas_mask2[y_check, x] > 0:
                        bottom_y = y_check
                        break
                else:
                    continue
                
                # Draw base vector pixel
                if 0 <= bottom_y + y + 1 < canvas_height:
                    visualization[bottom_y + y + 1, x] = [0, 0, 0, 255]  # Black, fully opaque
    
    # Save visualization
    cv2.imwrite(os.path.join(TEMP_DIR, "overlap_and_base_visualization.png"), visualization)
    
    # Calculate overlap percentage
    total_pixels1 = np.sum(canvas_mask1 > 0)
    total_pixels2 = np.sum(canvas_mask2 > 0)
    overlap_pixels = np.sum(overlap_mask > 0)
    
    smaller_product_pixels = min(total_pixels1, total_pixels2)
    overlap_percentage = (overlap_pixels / smaller_product_pixels) * 100 if smaller_product_pixels > 0 else 0
    
    print(f"Overlap: {overlap_pixels} pixels, {overlap_percentage:.2f}% of smaller product")
    
    # Find the largest contour for overlap
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour for SVG (approximate polygon)
    epsilon = 0.003 * cv2.arcLength(largest_contour, True)
    approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Extract coordinates for overlap SVG
    coordinates = [{"x": int(point[0][0]), "y": int(point[0][1])} for point in approx_poly]
    
    # Create SVG content
    svg_width = canvas_width
    svg_height = canvas_height
    
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">\n'
    
    # Add description
    svg_content += '  <desc>Product Overlap Area and Base Vectors</desc>\n'
    
    # Create a separate SVG file just for base vectors
    base_svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">\n'
    base_svg_content += '  <desc>Product Base Vectors</desc>\n'
    
    # Add base vectors to SVG content - Product 1
    for start_x, end_x in p1_base_segments:
        # For each segment, find its bottom edge y-coordinate
        for x_mid in range(start_x, end_x + 1):
            # Find the y-coordinate at the bottom of the product for this x
            for y_check in range(canvas_height-1, -1, -1):
                if x_mid < canvas_width and canvas_mask1[y_check, x_mid] > 0:
                    bottom_y = y_check
                    
                    # Create a rectangle for this base segment
                    rect_x = start_x
                    rect_y = bottom_y + 1  # Start just below the bottom edge
                    rect_width = end_x - start_x + 1
                    rect_height = base_height
                    
                    # Add rectangle to SVG
                    svg_content += f'  <rect x="{rect_x}" y="{rect_y}" width="{rect_width}" height="{rect_height}" fill="black" stroke="none" />\n'
                    base_svg_content += f'  <rect x="{rect_x}" y="{rect_y}" width="{rect_width}" height="{rect_height}" fill="black" stroke="none" />\n'
                    break
            # Only need to find one y-coordinate per segment
            break
    
    # Add base vectors to SVG content - Product 2
    for start_x, end_x in p2_base_segments:
        # For each segment, find its bottom edge y-coordinate
        for x_mid in range(start_x, end_x + 1):
            # Find the y-coordinate at the bottom of the product for this x
            for y_check in range(canvas_height-1, -1, -1):
                if x_mid < canvas_width and canvas_mask2[y_check, x_mid] > 0:
                    bottom_y = y_check
                    
                    # Create a rectangle for this base segment
                    rect_x = start_x
                    rect_y = bottom_y + 1  # Start just below the bottom edge
                    rect_width = end_x - start_x + 1
                    rect_height = base_height
                    
                    # Add rectangle to SVG
                    svg_content += f'  <rect x="{rect_x}" y="{rect_y}" width="{rect_width}" height="{rect_height}" fill="black" stroke="none" />\n'
                    base_svg_content += f'  <rect x="{rect_x}" y="{rect_y}" width="{rect_width}" height="{rect_height}" fill="black" stroke="none" />\n'
                    break
            # Only need to find one y-coordinate per segment
            break
    
    # Convert overlap contour to SVG path
    path_data = "M "
    for i, point in enumerate(approx_poly):
        x, y = point[0]
        if i == 0:
            path_data += f"{x},{y} "
        else:
            path_data += f"L {x},{y} "
    path_data += "Z"  # Close the path
    
    # Add the overlap path with black fill, no stroke, and 100% opacity
    svg_content += f'  <path d="{path_data}" fill="black" stroke="none" />\n'
    
    # Add coordinate data as metadata
    svg_content += '  <metadata>\n'
    svg_content += f'    <overlap_percentage>{overlap_percentage:.2f}</overlap_percentage>\n'
    svg_content += '    <coordinates>\n'
    for i, coord in enumerate(coordinates):
        svg_content += f'      <point id="{i}" x="{coord["x"]}" y="{coord["y"]}" />\n'
    svg_content += '    </coordinates>\n'
    svg_content += '  </metadata>\n'
    
    # Close SVG tags
    svg_content += '</svg>'
    base_svg_content += '</svg>'
    
    # Save both SVG files
    svg_path = os.path.join(TEMP_DIR, "overlap_vector_shape.svg")
    base_svg_path = os.path.join(TEMP_DIR, "base_vector_shape.svg")
    
    with open(svg_path, "w") as f:
        f.write(svg_content)
    
    with open(base_svg_path, "w") as f:
        f.write(base_svg_content)
    
    print(f"Saved overlap vector shape with base vectors to {svg_path}")
    print(f"Saved separate base vector shape to {base_svg_path}")
    
    return {
        "svg_path": svg_path,
        "base_svg_path": base_svg_path,
        "coordinates": coordinates,
        "overlap_percentage": overlap_percentage
    }

def calculate_product_placement(
    product1_path: str, 
    product2_path: str, 
    mask_polygon: List[List[int]], 
    canvas_width: int, 
    canvas_height: int,
    padding: int = 5,
    right_offset: int = 100  # New parameter for right alignment offset
) -> Tuple[List[ProductPlacement], np.ndarray, Dict[str, Any]]:
    """
    Calculate placement of two products within SVG mask.
    Applies different scaling based on product categories.
    Maintains aspect ratios and ensures products stay within boundaries.
    
    Parameters:
        right_offset: Offset from the right edge, larger values move products toward center
    """
    # Normalize and convert polygon to numpy array for easier processing
    try:
        # Ensure all points are exactly [x,y] format with integer values
        normalized_points = [[int(point[0]), int(point[1])] for point in mask_polygon]
        polygon_points = np.array(normalized_points, dtype=np.int32)
        print(f"Normalized polygon points shape: {polygon_points.shape}")
    except Exception as e:
        print(f"Error normalizing polygon points: {str(e)}")
        # Create a fallback rectangle as polygon
        polygon_points = np.array([
            [50, 50], 
            [canvas_width-50, 50], 
            [canvas_width-50, canvas_height-50], 
            [50, canvas_height-50]
        ], dtype=np.int32)
        print("Using fallback rectangle for polygon")
    
    # 1. Get the bounding box of the mask polygon
    min_x, min_y = np.min(polygon_points, axis=0)
    max_x, max_y = np.max(polygon_points, axis=0)
    mask_width = max_x - min_x
    mask_height = max_y - min_y
    
    print(f"Mask bounds: ({min_x}, {min_y}) to ({max_x}, {max_y}), size: {mask_width}x{mask_height}")
    print(f"Using right offset: {right_offset}px")
    
    # Create a mask from the polygon for boundary checking
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Save mask for debugging
    mask_debug_path = os.path.join(TEMP_DIR, "mask_debug.png")
    cv2.imwrite(mask_debug_path, mask)
    
    # 2. Load the product images and get original dimensions
    product1_img = cv2.imread(product1_path, cv2.IMREAD_UNCHANGED)
    product2_img = cv2.imread(product2_path, cv2.IMREAD_UNCHANGED)
    
    orig_h1, orig_w1 = product1_img.shape[:2]
    orig_h2, orig_w2 = product2_img.shape[:2]
    
    # Calculate original aspect ratios
    p1_aspect = orig_w1 / orig_h1  # width/height ratio
    p2_aspect = orig_w2 / orig_h2
    
    # Determine which product is taller (smaller aspect ratio means taller)
    p1_is_taller = p1_aspect < p2_aspect
    
    print(f"Product 1 original dimensions: {orig_w1}x{orig_h1}, aspect ratio: {p1_aspect:.3f}")
    print(f"Product 2 original dimensions: {orig_w2}x{orig_h2}, aspect ratio: {p2_aspect:.3f}")
    print(f"Product 1 is taller: {p1_is_taller}")
    
    # 3. Classify products by size
    p1_category = classify_product_size(orig_w1, orig_h1)
    p2_category = classify_product_size(orig_w2, orig_h2)
    
    print(f"Product 1 category: {p1_category}")
    print(f"Product 2 category: {p2_category}")
    
    # 4. Calculate initial scaling based on product categories
    # Available space in the mask
    available_height = mask_height - (padding * 2)
    available_width = mask_width - (padding * 3)  # Extra padding for middle gap
    
    # Initial scaling variables
    p1_target_h = 0
    p1_target_w = 0
    p2_target_h = 0
    p2_target_w = 0
    
    # Special case: When both products are "Wide", stack them vertically
    is_vertical_stack = False
    if p1_category == "Wide" and p2_category == "Wide":
        print("Special case: Both products are Wide - implementing vertical stacking")
        is_vertical_stack = True
        
        # Determine which product is wider in actual pixels
        p1_is_wider = orig_w1 > orig_w2
        
        if p1_is_wider:
            print("Product 1 is wider - placing at bottom")
            bottom_product = product1_img
            top_product = product2_img
            bottom_orig_w, bottom_orig_h = orig_w1, orig_h1
            top_orig_w, top_orig_h = orig_w2, orig_h2
            bottom_aspect = p1_aspect
            top_aspect = p2_aspect
        else:
            print("Product 2 is wider - placing at bottom")
            bottom_product = product2_img
            top_product = product1_img
            bottom_orig_w, bottom_orig_h = orig_w2, orig_h2
            top_orig_w, top_orig_h = orig_w1, orig_h1
            bottom_aspect = p2_aspect
            top_aspect = p1_aspect
        
        # Maximize space utilization - use full mask width with small padding
        # Use more of the available space by reducing padding for vertical stack
        inner_padding = padding  # Smaller padding between products for vertical stack
        
        # First, scale the bottom (wider) product to full mask width
        bottom_target_w = available_width - (padding * 2)  # Use almost full width
        bottom_target_h = int(bottom_target_w / bottom_aspect)
        
        # Check if bottom product's height is too large, scale down if needed
        max_bottom_height = available_height * 0.6  # Allow bottom product to use up to 60% of height
        if bottom_target_h > max_bottom_height:
            bottom_target_h = int(max_bottom_height)
            bottom_target_w = int(bottom_target_h * bottom_aspect)
        
        # Scale the top (narrower) product to same width as bottom product if possible
        # This creates a more uniform appearance
        ideal_top_w = bottom_target_w  # Try to match bottom width for cleaner look
        ideal_top_h = int(ideal_top_w / top_aspect)
        
        # Check if ideal height for top product fits
        remaining_height = available_height - bottom_target_h - inner_padding
        if ideal_top_h <= remaining_height:
            # We can use the ideal width (matching bottom width)
            top_target_w = ideal_top_w
            top_target_h = ideal_top_h
        else:
            # Scale based on available height
            top_target_h = int(remaining_height * 0.95)  # Use 95% of remaining height
            top_target_w = int(top_target_h * top_aspect)
            
            # If too wide, scale down to mask width
            if top_target_w > available_width - (padding * 2):
                top_target_w = available_width - (padding * 2)
                top_target_h = int(top_target_w / top_aspect)
        
        # Make sure both products fit within available height
        total_height = bottom_target_h + top_target_h + inner_padding
        if total_height > available_height:
            # Scale both proportionally to fit
            scale_factor = available_height / total_height
            bottom_target_h = int(bottom_target_h * scale_factor)
            bottom_target_w = int(bottom_target_h * bottom_aspect)
            top_target_h = int(top_target_h * scale_factor)
            top_target_w = int(top_target_h * top_aspect)
        
        # Now set target dimensions for both products
        if p1_is_wider:
            p1_target_w, p1_target_h = bottom_target_w, bottom_target_h
            p2_target_w, p2_target_h = top_target_w, top_target_h
        else:
            p1_target_w, p1_target_h = top_target_w, top_target_h
            p2_target_w, p2_target_h = bottom_target_w, bottom_target_h
            
        print(f"Vertical stacking dimensions:")
        print(f"Bottom product: {bottom_target_w}x{bottom_target_h}")
        print(f"Top product: {top_target_w}x{top_target_h}")
        print(f"Total height: {total_height} / Available: {available_height}")

    # Apply conditional scaling rules based on product categories
    elif p1_category == "Tall" and p2_category == "Tall":
        # Determine which product is taller in relative terms
        p1_is_taller = (orig_h1 / orig_w1) > (orig_h2 / orig_w2)
        
        # Scale the taller product to max height, slightly scaled up
        if p1_is_taller:
            p1_height_scale = 1.0  # Use 100% of available height
            p2_height_scale = 0.9  # Use 90% of available height
        else:
            p1_height_scale = 0.9
            p2_height_scale = 1.0
            
        p1_target_h = int(available_height * p1_height_scale)
        p2_target_h = int(available_height * p2_height_scale)
        
        # Calculate widths based on aspect ratios
        p1_target_w = int(p1_target_h * p1_aspect)
        p2_target_w = int(p2_target_h * p2_aspect)
        
    elif (p1_category == "Wide" and p2_category == "Tall") or (p1_category == "Compact" and p2_category == "Tall"):
        # Scale the tall product (p2) to max height
        p2_target_h = int(available_height * 0.95)
        p2_target_w = int(p2_target_h * p2_aspect)
        
        # Scale wide/compact product to fit remaining space
        remaining_width = available_width - p2_target_w
        p1_target_w = int(remaining_width * 0.95)
        p1_target_h = int(p1_target_w / p1_aspect)
        
        # If p1 height is too large, scale down
        if p1_target_h > available_height:
            p1_target_h = int(available_height * 0.95)
            p1_target_w = int(p1_target_h * p1_aspect)
            
    elif (p1_category == "Tall" and p2_category == "Wide") or (p1_category == "Tall" and p2_category == "Compact"):
        # Scale the tall product (p1) to max height
        p1_target_h = int(available_height * 0.95)
        p1_target_w = int(p1_target_h * p1_aspect)
        
        # Scale wide/compact product to fit remaining space
        remaining_width = available_width - p1_target_w
        p2_target_w = int(remaining_width * 0.95)
        p2_target_h = int(p2_target_w / p2_aspect)
        
        # If p2 height is too large, scale down
        if p2_target_h > available_height:
            p2_target_h = int(available_height * 0.95)
            p2_target_w = int(p2_target_h * p2_aspect)
    
    elif p1_category == "Compact" and p2_category == "Compact":
        # Similar to tall+tall, scale the larger one slightly up
        p1_is_larger = (orig_w1 * orig_h1) > (orig_w2 * orig_h2)
        
        if p1_is_larger:
            p1_height_scale = 1.0
            p2_height_scale = 0.9
        else:
            p1_height_scale = 0.9
            p2_height_scale = 1.0
            
        p1_target_h = int(available_height * p1_height_scale)
        p2_target_h = int(available_height * p2_height_scale)
        
        # Calculate widths based on aspect ratios
        p1_target_w = int(p1_target_h * p1_aspect)
        p2_target_w = int(p2_target_h * p2_aspect)
        
    elif (p1_category == "Wide" and p2_category == "Compact") or (p1_category == "Compact" and p2_category == "Wide"):
        # For wide+compact, distribute available width proportionally
        total_width_ratio = orig_w1 + orig_w2
        p1_width_portion = orig_w1 / total_width_ratio
        p2_width_portion = orig_w2 / total_width_ratio
        
        p1_target_w = int(available_width * p1_width_portion)
        p2_target_w = int(available_width * p2_width_portion)
        
        p1_target_h = int(p1_target_w / p1_aspect)
        p2_target_h = int(p2_target_w / p2_aspect)
        
        # Check if heights exceed available height, scale down if needed
        if p1_target_h > available_height or p2_target_h > available_height:
            max_height = max(p1_target_h, p2_target_h)
            scale_factor = available_height / max_height
            
            p1_target_h = int(p1_target_h * scale_factor)
            p1_target_w = int(p1_target_h * p1_aspect)
            
            p2_target_h = int(p2_target_h * scale_factor)
            p2_target_w = int(p2_target_h * p2_aspect)
    
    else:
        # Default fallback if categories don't match any specific rule
        p1_target_h = int(available_height * 0.9)
        p1_target_w = int(p1_target_h * p1_aspect)
        
        p2_target_h = int(available_height * 0.9)
        p2_target_w = int(p2_target_h * p2_aspect)
    
    # 5. Verify that products will fit within mask width
    total_width = p1_target_w + p2_target_w + padding
    if total_width > available_width:
        # Scale both products down proportionally
        scale_factor = available_width / total_width
        
        p1_target_w = int(p1_target_w * scale_factor)
        p1_target_h = int(p1_target_h * scale_factor)
        
        p2_target_w = int(p2_target_w * scale_factor)
        p2_target_h = int(p2_target_h * scale_factor)
    
    # 6. Resize the products
    p1_resized = cv2.resize(product1_img, (p1_target_w, p1_target_h), interpolation=cv2.INTER_AREA)
    p2_resized = cv2.resize(product2_img, (p2_target_w, p2_target_h), interpolation=cv2.INTER_AREA)
    
    # 7. Calculate positions based on the stacking logic (if applicable) or the regular positioning
    if is_vertical_stack:
        # For vertical stacking (wide + wide case)
        # Center both products horizontally within the mask
        p1_is_wider = orig_w1 > orig_w2
        inner_padding = padding  # Smaller padding between products
        
        if p1_is_wider:
            # Product 1 is wider (bottom), Product 2 is on top
            # Center horizontally within mask bounds
            p1_x_pos = min_x + (mask_width - p1_target_w) // 2  # Center horizontally
            p2_x_pos = min_x + (mask_width - p2_target_w) // 2  # Center horizontally
            
            # Stack vertically from bottom, maximizing space
            p1_y_pos = max_y - p1_target_h - padding  # Bottom product at the bottom
            p2_y_pos = p1_y_pos - p2_target_h - inner_padding  # Top product above it
        else:
            # Product 2 is wider (bottom), Product 1 is on top
            p1_x_pos = min_x + (mask_width - p1_target_w) // 2  # Center horizontally
            p2_x_pos = min_x + (mask_width - p2_target_w) // 2  # Center horizontally
            
            # Stack vertically from bottom, maximizing space
            p2_y_pos = max_y - p2_target_h - padding  # Bottom product at the bottom
            p1_y_pos = p2_y_pos - p1_target_h - inner_padding  # Top product above it
            
        print(f"Vertical stacking: P1 at ({p1_x_pos}, {p1_y_pos}), P2 at ({p2_x_pos}, {p2_y_pos})")
        print(f"Mask width: {mask_width}, Mask height: {mask_height}")
    else:
        # Original horizontal positioning logic
        # Calculate overlap to ensure minimum 30% overlap of visible pixels
        min_overlap_percent = 0.3
        
        # Calculate minimum required overlap in pixels
        min_overlap_pixels = int(min(p1_target_w, p2_target_w) * min_overlap_percent)
        print(f"Enforcing minimum overlap of {min_overlap_pixels} pixels ({min_overlap_percent*100}%)")
        
        # Set initial overlap
        overlap = min_overlap_pixels
        
        # If the products don't fit within the available width with the minimum overlap, 
        # we'll need to scale them differently
        total_width_with_min_overlap = p1_target_w + p2_target_w - min_overlap_pixels + padding
        if total_width_with_min_overlap > available_width:
            print(f"Products with min overlap would exceed available width: {total_width_with_min_overlap} > {available_width}")
            # Recalculate sizes to fit with required overlap
            scale_factor = (available_width + min_overlap_pixels - padding) / (p1_target_w + p2_target_w)
            
            # Scale down both products
            p1_target_w = int(p1_target_w * scale_factor)
            p1_target_h = int(p1_target_h * scale_factor)
            p2_target_w = int(p2_target_w * scale_factor)
            p2_target_h = int(p2_target_h * scale_factor)
            
            # Resize with adjusted dimensions
            p1_resized = cv2.resize(product1_img, (p1_target_w, p1_target_h), interpolation=cv2.INTER_AREA)
            p2_resized = cv2.resize(product2_img, (p2_target_w, p2_target_h), interpolation=cv2.INTER_AREA)
        
        # Determine placement based on which product is taller (smaller aspect ratio)
        # Place the taller product on the right side, offset from edge by right_offset
        if p1_is_taller:
            # Product 1 is taller, place it on right
            p1_x_pos = max_x - p1_target_w - padding - right_offset
            p2_x_pos = p1_x_pos - p2_target_w + overlap
            print("Placing product 1 (taller) on the right side")
        else:
            # Product 2 is taller, place it on right
            p2_x_pos = max_x - p2_target_w - padding - right_offset
            p1_x_pos = p2_x_pos - p1_target_w + overlap
            print("Placing product 2 (taller) on the right side")
        
        # Maximize space usage - if the left product can be scaled up
        left_product_is_wide = False
        if p1_is_taller:
            # Product 2 is on the left
            left_product_is_wide = (p2_category == "Wide" or p2_category == "Compact")
            if left_product_is_wide and p2_x_pos > min_x + padding:
                # Calculate how much extra space we have on the left
                extra_space = p2_x_pos - (min_x + padding)
                if extra_space > 20:  # If we have significant space, scale up left product
                    print(f"Scaling up product 2 to use extra space: {extra_space}px")
                    # Scale up width while maintaining aspect ratio
                    new_p2_target_w = p2_target_w + extra_space
                    new_p2_target_h = int(new_p2_target_w / p2_aspect)
                    
                    # Make sure height doesn't exceed available height
                    if new_p2_target_h <= available_height:
                        p2_target_w = new_p2_target_w
                        p2_target_h = new_p2_target_h
                        p2_resized = cv2.resize(product2_img, (p2_target_w, p2_target_h), interpolation=cv2.INTER_AREA)
                        # Recalculate position with new width but maintain overlap
                        p2_x_pos = p1_x_pos - p2_target_w + overlap
        else:
            # Product 1 is on the left
            left_product_is_wide = (p1_category == "Wide" or p1_category == "Compact")
            if left_product_is_wide and p1_x_pos > min_x + padding:
                # Calculate how much extra space we have on the left
                extra_space = p1_x_pos - (min_x + padding)
                if extra_space > 20:  # If we have significant space, scale up left product
                    print(f"Scaling up product 1 to use extra space: {extra_space}px")
                    # Scale up width while maintaining aspect ratio
                    new_p1_target_w = p1_target_w + extra_space
                    new_p1_target_h = int(new_p1_target_w / p1_aspect)
                    
                    # Make sure height doesn't exceed available height
                    if new_p1_target_h <= available_height:
                        p1_target_w = new_p1_target_w
                        p1_target_h = new_p1_target_h
                        p1_resized = cv2.resize(product1_img, (p1_target_w, p1_target_h), interpolation=cv2.INTER_AREA)
                        # Recalculate position with new width but maintain overlap
                        p1_x_pos = p2_x_pos - p1_target_w + overlap
        
        # Bottom alignment for horizontal layout
        p1_y_pos = max_y - p1_target_h - padding
        p2_y_pos = max_y - p2_target_h - padding
    
    # 8. Check if the placement is valid (products fully within mask)
    # Create test masks for both products - but check ONLY visible pixels
    test_img = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Draw p1 visible content
    if p1_x_pos >= 0 and p1_y_pos >= 0:
        alpha1 = p1_resized[:, :, 3] if p1_resized.shape[-1] == 4 else np.ones(p1_resized.shape[:2], dtype=np.uint8) * 255
        for y in range(min(p1_target_h, canvas_height - p1_y_pos)):
            for x in range(min(p1_target_w, canvas_width - p1_x_pos)):
                if alpha1[y, x] > 127:  # Only consider truly visible pixels
                    y_pos = p1_y_pos + y
                    x_pos = p1_x_pos + x
                    if 0 <= y_pos < canvas_height and 0 <= x_pos < canvas_width:
                        test_img[y_pos, x_pos] = 255
    
    # Draw p2 visible content
    if p2_x_pos >= 0 and p2_y_pos >= 0:
        alpha2 = p2_resized[:, :, 3] if p2_resized.shape[-1] == 4 else np.ones(p2_resized.shape[:2], dtype=np.uint8) * 255
        for y in range(min(p2_target_h, canvas_height - p2_y_pos)):
            for x in range(min(p2_target_w, canvas_width - p2_x_pos)):
                if alpha2[y, x] > 127:  # Only consider truly visible pixels
                    y_pos = p2_y_pos + y
                    x_pos = p2_x_pos + x
                    if 0 <= y_pos < canvas_height and 0 <= x_pos < canvas_width:
                        test_img[y_pos, x_pos] = 255
    
    # Check if any product VISIBLE pixels are outside the mask
    outside_mask = cv2.bitwise_and(test_img, cv2.bitwise_not(mask))
    
    # If products extend outside mask, try different adjustments
    max_iterations = 5
    iteration = 0
    
    while np.any(outside_mask > 0) and iteration < max_iterations:
        iteration += 1
        print(f"Products extend outside mask, adjustment attempt {iteration}")
        
        if iteration == 1:
            # First try shifting products more inward
            shift_x = padding * 2
            p2_x_pos = max_x - p2_target_w - padding - shift_x
            p1_x_pos = p2_x_pos - p1_target_w + overlap
        elif iteration == 2:
            # Try adjusting vertical position
            shift_y = padding * 2
            p1_y_pos = max_y - p1_target_h - padding - shift_y
            p2_y_pos = max_y - p2_target_h - padding - shift_y
        else:
            # Last resort: scale down products
            scale_factor = 0.9
            
            p1_target_w = int(p1_target_w * scale_factor)
            p1_target_h = int(p1_target_h * scale_factor)
            p2_target_w = int(p2_target_w * scale_factor)
            p2_target_h = int(p2_target_h * scale_factor)
            
            # Resize with new dimensions
            p1_resized = cv2.resize(product1_img, (p1_target_w, p1_target_h), interpolation=cv2.INTER_AREA)
            p2_resized = cv2.resize(product2_img, (p2_target_w, p2_target_h), interpolation=cv2.INTER_AREA)
            
            # Recalculate positions - place at right side again
            p2_x_pos = max_x - p2_target_w - padding
            p1_x_pos = p2_x_pos - p1_target_w + overlap
            p1_y_pos = max_y - p1_target_h - padding
            p2_y_pos = max_y - p2_target_h - padding
        
        # Re-test with new positions/sizes
        test_img = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        # Draw p1 visible content
        if p1_x_pos >= 0 and p1_y_pos >= 0:
            alpha1 = p1_resized[:, :, 3] if p1_resized.shape[-1] == 4 else np.ones(p1_resized.shape[:2], dtype=np.uint8) * 255
            for y in range(min(p1_target_h, canvas_height - p1_y_pos)):
                for x in range(min(p1_target_w, canvas_width - p1_x_pos)):
                    if alpha1[y, x] > 127:
                        y_pos = p1_y_pos + y
                        x_pos = p1_x_pos + x
                        if 0 <= y_pos < canvas_height and 0 <= x_pos < canvas_width:
                            test_img[y_pos, x_pos] = 255
        
        # Draw p2 visible content
        if p2_x_pos >= 0 and p2_y_pos >= 0:
            alpha2 = p2_resized[:, :, 3] if p2_resized.shape[-1] == 4 else np.ones(p2_resized.shape[:2], dtype=np.uint8) * 255
            for y in range(min(p2_target_h, canvas_height - p2_y_pos)):
                for x in range(min(p2_target_w, canvas_width - p2_x_pos)):
                    if alpha2[y, x] > 127:
                        y_pos = p2_y_pos + y
                        x_pos = p2_x_pos + x
                        if 0 <= y_pos < canvas_height and 0 <= x_pos < canvas_width:
                            test_img[y_pos, x_pos] = 255
        
        # Check again
        outside_mask = cv2.bitwise_and(test_img, cv2.bitwise_not(mask))
    
    if iteration == max_iterations and np.any(outside_mask > 0):
        print("Warning: Could not fully contain products within mask after maximum adjustments")
    
    # Save mask testing visualization for debugging
    debug_mask_test = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    debug_mask_test[mask > 0] = [150, 150, 150]  # Gray background for mask
    debug_mask_test[test_img > 0] = [0, 255, 0]  # Green for products
    debug_mask_test[outside_mask > 0] = [0, 0, 255]  # Red for out-of-bounds pixels
    mask_test_debug_path = os.path.join(TEMP_DIR, "mask_test_debug.png")
    cv2.imwrite(mask_test_debug_path, debug_mask_test)
    
    # 9. Verify final aspect ratios
    final_p1_aspect = p1_target_w / p1_target_h
    final_p2_aspect = p2_target_w / p2_target_h
    
    print(f"Product 1 final dimensions: {p1_target_w}x{p1_target_h}, aspect ratio: {final_p1_aspect:.3f} (original: {p1_aspect:.3f})")
    print(f"Product 2 final dimensions: {p2_target_w}x{p2_target_h}, aspect ratio: {final_p2_aspect:.3f} (original: {p2_aspect:.3f})")
    
    # 10. Create placement data
    placements = [
        ProductPlacement(
            image="product1_placed.png",
            position={"x": int(p1_x_pos), "y": int(p1_y_pos)},
            size={"width": p1_target_w, "height": p1_target_h},
            rotation=0
        ),
        ProductPlacement(
            image="product2_placed.png",
            position={"x": int(p2_x_pos), "y": int(p2_y_pos)},
            size={"width": p2_target_w, "height": p2_target_h},
            rotation=0
        )
    ]
    
    # 11. Create a composed image
    # Create transparent canvas
    composed = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    
    # Add gray background for the mask area
    mask_vis = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    mask_vis[mask > 0] = [150, 150, 150, 255]  # Gray background
    
    # Create region of interest for the mask
    roi = composed[mask > 0]
    background_roi = mask_vis[mask > 0]
    roi[:] = background_roi[:]
    
    # Create debug visualization
    debug_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # Draw mask area
    debug_img[mask > 0] = [150, 150, 150]
    
    # Draw product bounding boxes
    cv2.rectangle(debug_img, 
                 (int(p1_x_pos), int(p1_y_pos)), 
                 (int(p1_x_pos + p1_target_w), int(p1_y_pos + p1_target_h)), 
                 (0, 255, 0), 2)
    cv2.rectangle(debug_img, 
                 (int(p2_x_pos), int(p2_y_pos)), 
                 (int(p2_x_pos + p2_target_w), int(p2_y_pos + p2_target_h)), 
                 (0, 255, 0), 2)
    
    # Draw SVG polygon
    cv2.polylines(debug_img, [polygon_points], True, (0, 0, 255), 2)
    
    placement_debug_path = os.path.join(TEMP_DIR, "placement_debug.png")
    cv2.imwrite(placement_debug_path, debug_img)
    
    # Place products on the canvas
    for placement, img in zip(placements, [p1_resized, p2_resized]):
        x, y = placement.position["x"], placement.position["y"]
        h, w = img.shape[:2]
        
        # Ensure we don't exceed canvas bounds
        if y < 0 or x < 0 or y + h > canvas_height or x + w > canvas_width:
            print(f"Warning: Product placement exceeds canvas bounds: ({x}, {y}, {w}, {h})")
            # Adjust placement to fit within canvas
            x = max(0, min(x, canvas_width - w))
            y = max(0, min(y, canvas_height - h))
            placement.position["x"] = int(x)
            placement.position["y"] = int(y)
        
        # Create region of interest
        y_end = min(y + h, canvas_height)
        x_end = min(x + w, canvas_width)
        h_roi = y_end - y
        w_roi = x_end - x
        
        if h_roi <= 0 or w_roi <= 0:
            print(f"Warning: Product ROI has invalid dimensions: {w_roi}x{h_roi}")
            continue
            
        roi = composed[y:y_end, x:x_end]
        img_roi = img[:h_roi, :w_roi]
        
        # For each pixel, if it's not transparent, copy it to the canvas
        if img_roi.shape[-1] == 4:  # Has alpha channel
            # Create a mask of visible pixels
            mask = img_roi[:, :, 3] > 0
            
            # Copy RGB channels for visible pixels
            for c in range(3):  # RGB channels
                roi[:, :, c][mask] = img_roi[:, :, c][mask]
                
            # Copy alpha channel
            roi[:, :, 3][mask] = img_roi[:, :, 3][mask]
    
    product1_placed_path = os.path.join(TEMP_DIR, "product1_placed.png")
    cv2.imwrite(product1_placed_path, p1_resized)
    
    product2_placed_path = os.path.join(TEMP_DIR, "product2_placed.png")
    cv2.imwrite(product2_placed_path, p2_resized)
    
    # 12. Create overlap vector shape
    overlap_data = create_overlap_vector_shape(
        product1_img=p1_resized, 
        product2_img=p2_resized, 
        p1_x_pos=p1_x_pos, 
        p1_y_pos=p1_y_pos, 
        p2_x_pos=p2_x_pos, 
        p2_y_pos=p2_y_pos,
        canvas_width=canvas_width,
        canvas_height=canvas_height
    )
    
    return placements, composed, overlap_data

def create_visualization(
    composed_img: np.ndarray, 
    placements: List[ProductPlacement], 
    mask_polygon: List[List[int]],
    canvas_width: int,
    canvas_height: int
) -> str:
    """Create a visualization of the product placement."""
    # Create a copy for visualization
    vis_img = composed_img.copy()
    
    # Draw mask polygon with semi-transparent fill
    polygon_pts = np.array(mask_polygon, np.int32)
    mask_overlay = vis_img.copy()
    
    # Add semi-transparent red fill to show the mask area
    cv2.fillPoly(mask_overlay, [polygon_pts], (0, 0, 255, 80))
    alpha = 0.3
    cv2.addWeighted(mask_overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    # Add solid outline of the mask
    cv2.polylines(
        vis_img, 
        [polygon_pts], 
        True, 
        (0, 0, 255, 255),  # Red
        2
    )
    
    # Draw boxes around product placements
    for i, placement in enumerate(placements):
        x, y = placement.position["x"], placement.position["y"]
        w, h = placement.size["width"], placement.size["height"]
        
        # Draw rectangle
        cv2.rectangle(
            vis_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0, 255),  # Green
            2
        )
        
        # Add label
        label = f"Product {i+1}"
        cv2.putText(
            vis_img,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0, 255),  # Black outline
            2
        )
        cv2.putText(
            vis_img,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255, 255),  # White text
            1
        )
    
    # Add title
    title = "Products Placed Inside SVG Safe Zone"
    cv2.putText(
        vis_img,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0, 255),  # Black outline
        2
    )
    cv2.putText(
        vis_img,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 100, 255, 255),  # Orange text
        1
    )
    
    # Save visualization
    output_path = "product_placement_visualization.png"
    cv2.imwrite(output_path, vis_img)
    
    return output_path

@app.post("/place-products-in-svg/")
async def place_products_in_svg(
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...),
    svg_file: UploadFile = File(...),
    padding: int = Query(10, description="Padding in pixels around products"),
    bottom_offset: int = Query(0, description="Offset from bottom in pixels"),
    right_offset: int = Query(0, description="Controls how far from the right edge the products are placed")
):
    """
    Upload two product images and an SVG mask to place products inside the safe zone.
    Returns placement data and a visualization of the result.
    
    Parameters:
    - file1, file2: The product images to place
    - svg_file: SVG file defining the safe zone
    - padding: Padding in pixels
    - bottom_offset: Offset from bottom in pixels
    - right_offset: Controls how far from the right edge the products are placed
    """
    product1_path = os.path.join(TEMP_DIR, "temp_product1.png")
    product2_path = os.path.join(TEMP_DIR, "temp_product2.png")
    svg_path = os.path.join(TEMP_DIR, "temp_mask.svg")
    
    try:
        # Save uploaded files
        with open(product1_path, "wb") as f:
            f.write(await file1.read())
        
        with open(product2_path, "wb") as f:
            f.write(await file2.read())
        
        with open(svg_path, "wb") as f:
            f.write(await svg_file.read())
        
        # Load images
        product1_img = cv2.imread(product1_path, cv2.IMREAD_UNCHANGED)
        product2_img = cv2.imread(product2_path, cv2.IMREAD_UNCHANGED)
        
        if product1_img is None or product2_img is None:
            raise HTTPException(status_code=400, detail="Could not read product images")
        
        # Create a mask from the SVG
        try:
            mask = convert_svg_to_mask(svg_path)
            if mask is None:
                raise ValueError("Could not create mask from SVG")
        except Exception as e:
            print(f"Error creating mask from SVG: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing SVG file: {str(e)}")
        
        # Extract polygon from SVG for visualizing the safe zone
        try:
            polygon = extract_svg_polygon(svg_path, mask)
        except Exception as e:
            print(f"Error extracting polygon from SVG: {str(e)}")
            polygon = []
        
        # Calculate dimensions for the canvas
        canvas_width = mask.shape[1]
        canvas_height = mask.shape[0]
        
        # Calculate product placement positions and sizes
        placement_data = calculate_product_placement(
            product1_path, product2_path, polygon, canvas_width, canvas_height, padding, right_offset
        )
        
        if not placement_data:
            raise HTTPException(status_code=400, detail="Could not calculate suitable product placement")
        
        # Create a visualization of the result
        visualization = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
        
        # Draw the mask area with a semi-transparent color
        for y in range(canvas_height):
            for x in range(canvas_width):
                if mask[y, x] > 0:
                    visualization[y, x] = [200, 200, 200, 100]  # Light gray with transparency
        
        # Draw the two products on the visualization
        p1_x = placement_data["product1"]["x"]
        p1_y = placement_data["product1"]["y"]
        p1_width = placement_data["product1"]["width"]
        p1_height = placement_data["product1"]["height"]
        
        p2_x = placement_data["product2"]["x"]
        p2_y = placement_data["product2"]["y"]
        p2_width = placement_data["product2"]["width"]
        p2_height = placement_data["product2"]["height"]
        
        # Resize products for visualization
        product1_resized = cv2.resize(product1_img, (p1_width, p1_height))
        product2_resized = cv2.resize(product2_img, (p2_width, p2_height))
        
        # Draw products on visualization
        for y in range(p1_height):
            for x in range(p1_width):
                if y + p1_y < canvas_height and x + p1_x < canvas_width:
                    alpha = product1_resized[y, x, 3] / 255.0 if product1_resized.shape[-1] == 4 else 1.0
                    if alpha > 0:
                        visualization[y + p1_y, x + p1_x] = [
                            product1_resized[y, x, 0],
                            product1_resized[y, x, 1],
                            product1_resized[y, x, 2],
                            int(alpha * 255)
                        ]
        
        for y in range(p2_height):
            for x in range(p2_width):
                if y + p2_y < canvas_height and x + p2_x < canvas_width:
                    alpha = product2_resized[y, x, 3] / 255.0 if product2_resized.shape[-1] == 4 else 1.0
                    if alpha > 0:
                        visualization[y + p2_y, x + p2_x] = [
                            product2_resized[y, x, 0],
                            product2_resized[y, x, 1],
                            product2_resized[y, x, 2],
                            int(alpha * 255)
                        ]
        
        # Draw polygon outline (the safe zone) on visualization
        if polygon:
            for i in range(len(polygon) - 1):
                pt1 = (int(polygon[i][0]), int(polygon[i][1]))
                pt2 = (int(polygon[i+1][0]), int(polygon[i+1][1]))
                cv2.line(visualization, pt1, pt2, (255, 0, 0, 255), 2)
            # Close the polygon
            cv2.line(visualization, 
                    (int(polygon[-1][0]), int(polygon[-1][1])), 
                    (int(polygon[0][0]), int(polygon[0][1])), 
                    (255, 0, 0, 255), 2)
        
        # Draw bounding box of products with different colors
        cv2.rectangle(visualization, (p1_x, p1_y), (p1_x + p1_width, p1_y + p1_height), (0, 255, 0, 255), 2)
        cv2.rectangle(visualization, (p2_x, p2_y), (p2_x + p2_width, p2_y + p2_height), (0, 0, 255, 255), 2)
        
        # Save visualization
        visualization_path = os.path.join(TEMP_DIR, "product_placement_visualization.png")
        cv2.imwrite(visualization_path, visualization)
        
        # Create vector shape of pixel overlap
        vector_result = create_overlap_vector_shape(
            product1_resized,
            product2_resized,
            p1_x,
            p1_y,
            p2_x,
            p2_y,
            canvas_width,
            canvas_height
        )
        
        # Return the placement data and visualization path
        return {
            "placement": placement_data,
            "visualization_path": visualization_path,
            "canvas_dimensions": {
                "width": canvas_width,
                "height": canvas_height
            },
            "overlap_vector": vector_result.get("svg_path"),
            "base_vector": vector_result.get("base_svg_path"),
            "overlap_percentage": vector_result.get("overlap_percentage", 0)
        }
    
    except Exception as e:
        print(f"Error in placing products: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        for path in [product1_path, product2_path, svg_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing temp file {path}: {str(e)}")

@app.post("/extract-logo/")
async def extract_logo(file: UploadFile = File(...)):
    """
    Endpoint to extract a logo by cropping a transparent PNG to its visible pixel dimensions.
    """
    try:
        # Save uploaded file temporarily
        temp_input_path = os.path.join(TEMP_DIR, "temp_logo_input.png")
        temp_output_path = os.path.join(TEMP_DIR, "extracted_logo.png")
        
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())
        
        # Extract actual boundary box and crop the image
        crop_info = crop_to_visible_pixels(temp_input_path, temp_output_path)
        
        # Get the file dimensions and size
        image = Image.open(temp_output_path)
        width, height = image.size
        file_size = os.path.getsize(temp_output_path)
        
        return {
            "success": True,
            "message": "Logo extracted successfully",
            "crop_info": crop_info,
            "dimensions": {
                "width": width,
                "height": height
            },
            "file_size": file_size
        }
        
    except Exception as e:
        print(f"Error in extract_logo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up temporary input file: {cleanup_error}")

@app.get("/get-extracted-logo/")
async def get_extracted_logo():
    """
    Endpoint to retrieve the extracted logo image.
    """
    try:
        logo_path = os.path.join(TEMP_DIR, "extracted_logo.png")
        
        if not os.path.exists(logo_path):
            raise HTTPException(status_code=404, detail="No extracted logo found")
        
        return FileResponse(
            logo_path, 
            media_type="image/png", 
            filename="extracted_logo.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_extracted_logo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add the crop_to_visible_pixels function before the extract_logo endpoint
def crop_to_visible_pixels(input_path, output_path):
    """
    Crops a transparent PNG to the visible pixel dimensions
    by removing excess transparent areas.
    """
    # Read the image with alpha channel
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Ensure image has an alpha channel
    if image.shape[-1] != 4:
        raise ValueError("Image must have an alpha channel (transparent PNG)")
    
    # Extract alpha channel
    alpha_channel = image[:, :, 3]
    
    # Get non-zero points (visible pixels)
    non_zero_points = cv2.findNonZero(alpha_channel)
    
    if non_zero_points is None:
        raise ValueError("No visible pixels found in the image")
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(non_zero_points)
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    
    return {"x": x, "y": y, "width": w, "height": h}

# Add these new functions for the masthead generation feature
async def download_image_from_url(url):
    """
    Download an image from a URL and return it as bytes
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return response.content

async def download_from_google_drive(drive_url):
    """
    Download a file from Google Drive
    """
    try:
        # Extract the file ID from the URL
        file_id = re.search(r'id=([^&]+)', drive_url)
        if not file_id:
            raise ValueError("Invalid Google Drive URL format")
        
        file_id = file_id.group(1)
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        response = requests.get(direct_url, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        raise

async def process_masthead_item(item, index):
    """
    Process a single masthead item by calling other features
    """
    try:
        result = {
            "id": index,
            "brand_name": item.get("Brand Name", ""),
            "copy_line_1": item.get("Copy Line 1", ""),
            "copy_line_2": item.get("Copy Line 2", "")
        }
        
        # 1. Process product images to get base color
        if "product1" in item and "product2" in item:
            product1_url = item["product1"].get("image_url", "")
            product2_url = item["product2"].get("image_url", "")
            
            if product1_url and product2_url:
                # Download product images
                product1_content = await download_image_from_url(product1_url)
                product2_content = await download_image_from_url(product2_url)
                
                # Save them temporarily
                product1_path = os.path.join(TEMP_DIR, f"temp_product1_{index}.png")
                product2_path = os.path.join(TEMP_DIR, f"temp_product2_{index}.png")
                
                with open(product1_path, "wb") as f:
                    f.write(product1_content)
                
                with open(product2_path, "wb") as f:
                    f.write(product2_content)
                
                # Extract colors from both images
                colors1 = extract_colors(product1_path)
                colors2 = extract_colors(product2_path)
                
                # Merge color palettes and determine hierarchy
                merged_colors = merge_color_palettes(colors1, colors2, product1_path, product2_path)
                
                # Get the base color
                base_color = merged_colors["base_color"]
                
                # Convert RGB to hex
                hex_color = "#{:02X}{:02X}{:02X}".format(base_color[0], base_color[1], base_color[2])
                
                result["base_color"] = hex_color
                
                # Clean up temp files
                os.remove(product1_path)
                os.remove(product2_path)
        
        # 2. Process logo to extract and crop
        logo_url = item.get("Logo drive link (Transparent PNG)", "")
        if logo_url:
            try:
                # Download logo from Google Drive
                logo_content = await download_from_google_drive(logo_url)
                
                # Save temporarily
                logo_input_path = os.path.join(TEMP_DIR, f"temp_logo_input_{index}.png")
                logo_output_path = os.path.join(TEMP_DIR, f"extracted_logo_{index}.png")
                
                with open(logo_input_path, "wb") as f:
                    f.write(logo_content)
                
                # Crop logo to visible pixels
                crop_info = crop_to_visible_pixels(logo_input_path, logo_output_path)
                
                # Get logo dimensions
                image = Image.open(logo_output_path)
                width, height = image.size
                
                # Construct logo URL for retrieval
                result["logo"] = {
                    "url": f"/get-masthead-logo/{index}",
                    "dimensions": {
                        "width": width,
                        "height": height
                    },
                    "crop_info": crop_info
                }
                
                # Keep the logo in memory for retrieval
                # We'll store the path instead of the content for efficiency
                result["_logo_path"] = logo_output_path
                
                # Clean up temp input file
                os.remove(logo_input_path)
            except Exception as logo_error:
                print(f"Error processing logo: {str(logo_error)}")
                result["logo_error"] = str(logo_error)
        
        return result
    except Exception as e:
        print(f"Error processing masthead item {index}: {str(e)}")
        return {
            "id": index,
            "error": str(e),
            "brand_name": item.get("Brand Name", ""),
            "copy_line_1": item.get("Copy Line 1", ""),
            "copy_line_2": item.get("Copy Line 2", "")
        }

@app.post("/generate-mastheads/")
async def generate_mastheads(request: Request):
    """
    Master endpoint that processes multiple mastheads at once, integrating
    product image processing, logo extraction, and text handling.
    """
    try:
        global processed_mastheads
        
        # Parse request body
        body = await request.json()
        
        if not isinstance(body, list):
            raise HTTPException(
                status_code=400, 
                detail="Request body must be an array of masthead data"
            )
        
        # Process each masthead item
        results = []
        for i, item in enumerate(body):
            result = await process_masthead_item(item, i)
            results.append(result)
        
        # Store the processed results
        processed_mastheads = results
        
        # Return success response with processed data
        return {
            "success": True,
            "message": f"Successfully processed {len(results)} mastheads",
            "mastheads": [
                {
                    k: v for k, v in masthead.items() 
                    if not k.startswith('_')  # Exclude fields starting with _
                } 
                for masthead in results
            ]
        }
    except Exception as e:
        print(f"Error in generate_mastheads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-mastheads/")
async def get_mastheads():
    """
    Retrieve all processed mastheads
    """
    global processed_mastheads
    
    if not processed_mastheads:
        return {
            "success": False,
            "message": "No mastheads have been processed yet",
            "mastheads": []
        }
    
    # Return all mastheads without internal fields
    return {
        "success": True,
        "message": f"Retrieved {len(processed_mastheads)} mastheads",
        "mastheads": [
            {
                k: v for k, v in masthead.items() 
                if not k.startswith('_')  # Exclude fields starting with _
            } 
            for masthead in processed_mastheads
        ]
    }

@app.get("/get-masthead/{masthead_id}")
async def get_masthead(masthead_id: int):
    """
    Retrieve a specific processed masthead by ID
    """
    global processed_mastheads
    
    if not processed_mastheads:
        raise HTTPException(
            status_code=404, 
            detail="No mastheads have been processed yet"
        )
    
    for masthead in processed_mastheads:
        if masthead.get("id") == masthead_id:
            # Return masthead without internal fields
            return {
                k: v for k, v in masthead.items() 
                if not k.startswith('_')  # Exclude fields starting with _
            }
    
    raise HTTPException(
        status_code=404, 
        detail=f"Masthead with ID {masthead_id} not found"
    )

@app.get("/get-masthead-logo/{masthead_id}")
async def get_masthead_logo(masthead_id: int):
    """
    Retrieve the extracted logo for a specific masthead
    """
    global processed_mastheads
    
    if not processed_mastheads:
        raise HTTPException(
            status_code=404, 
            detail="No mastheads have been processed yet"
        )
    
    for masthead in processed_mastheads:
        if masthead.get("id") == masthead_id and "_logo_path" in masthead:
            logo_path = masthead["_logo_path"]
            
            if os.path.exists(logo_path):
                return FileResponse(
                    logo_path, 
                    media_type="image/png", 
                    filename=f"logo_{masthead_id}.png"
                )
    
    raise HTTPException(
        status_code=404, 
        detail=f"Logo for masthead with ID {masthead_id} not found"
    )

@app.post("/generate-mastheads-binary/")
async def generate_mastheads_binary(
    product1: UploadFile = File(...),
    product2: UploadFile = File(...),
    logo: UploadFile = File(...),
    copy_line_1: str = Form(...),
    copy_line_2: str = Form(...),
    brand_name: str = Form(...)
):
    """
    Master endpoint that processes a masthead with direct binary uploads
    instead of URLs. Accepts multipart form data with files and text fields.
    """
    try:
        global processed_mastheads
        
        # Create a unique ID for this masthead
        masthead_id = len(processed_mastheads) if processed_mastheads else 0
        
        # Initialize result object
        result = {
            "id": masthead_id,
            "brand_name": brand_name,
            "copy_line_1": copy_line_1,
            "copy_line_2": copy_line_2
        }
        
        # 1. Process product images to get base color
        # Save uploaded files temporarily
        product1_path = os.path.join(TEMP_DIR, f"temp_product1_{masthead_id}.png")
        product2_path = os.path.join(TEMP_DIR, f"temp_product2_{masthead_id}.png")
        logo_path = os.path.join(TEMP_DIR, f"temp_logo_input_{masthead_id}.png")
        logo_output_path = os.path.join(TEMP_DIR, f"extracted_logo_{masthead_id}.png")
        
        # Save product1
        with open(product1_path, "wb") as f:
            f.write(await product1.read())
        
        # Save product2
        with open(product2_path, "wb") as f:
            f.write(await product2.read())
        
        # Save logo
        with open(logo_path, "wb") as f:
            f.write(await logo.read())
        
        try:
            # Extract colors from both product images
            colors1 = extract_colors(product1_path)
            colors2 = extract_colors(product2_path)
            
            # Merge color palettes and determine hierarchy
            merged_colors = merge_color_palettes(colors1, colors2, product1_path, product2_path)
            
            # Get the base color from the dictionary
            base_color = merged_colors["base_color"]
            
            # Convert RGB to hex
            hex_color = "#{:02X}{:02X}{:02X}".format(base_color[0], base_color[1], base_color[2])
            
            # Add base color to result
            result["base_color"] = hex_color
            
            # Crop logo to visible pixels
            crop_info = crop_to_visible_pixels(logo_path, logo_output_path)
            
            # Get logo dimensions
            image = Image.open(logo_output_path)
            width, height = image.size
            
            # Add logo info to result
            result["logo"] = {
                "url": f"/get-masthead-logo/{masthead_id}",
                "dimensions": {
                    "width": width,
                    "height": height
                },
                "crop_info": crop_info
            }
            
            # Store the logo path for retrieval
            result["_logo_path"] = logo_output_path
            
            # Add this masthead to the processed list
            processed_mastheads.append(result)
            
            # Return success response
            return {
                "success": True,
                "message": "Successfully processed masthead",
                "masthead": {
                    k: v for k, v in result.items() 
                    if not k.startswith('_')  # Exclude fields starting with _
                }
            }
            
        except Exception as processing_error:
            print(f"Error processing images: {str(processing_error)}")
            raise HTTPException(status_code=500, detail=str(processing_error))
        finally:
            # Clean up product files (but keep logo for retrieval)
            try:
                os.remove(product1_path)
                os.remove(product2_path)
                os.remove(logo_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary files: {cleanup_error}")
                
    except Exception as e:
        print(f"Error in generate_mastheads_binary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))