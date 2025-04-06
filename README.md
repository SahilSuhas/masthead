<<<<<<< HEAD
# FastAPI Image Processing API

This API provides image processing capabilities for use with the Figma plugin.

## Features

- **Color Extraction**: Extract dominant colors from two product images
- **Text Detection**: Detect text in transparent PNG images
- **Product Placement**: Place two product images within an SVG safe zone with visualization

## Endpoints

### /process-images/

Upload two product images to extract the base color in hexadecimal format.

### /detect-text/

Upload an image to detect text regions with visualization.

### /place-products-in-svg/

Upload two product images and an SVG mask to place products inside the safe zone with proper alignment and overlap.

## Deployment

This API is designed to be deployed on Render.

## Development

To run the API locally:

```bash
uvicorn main:app --reload
```

## Requirements

See `requirements.txt` for dependencies. 
=======
# masthead
>>>>>>> f2c1ef8d500930b96c5268f61b1f95ff1842cb71
