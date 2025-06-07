from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import os
import io
import ast
from PIL import Image
from pathlib import Path
from final.main import main
from final.get_report_s3 import face_report
import requests 
import numpy as np
import cv2
import base64 

app = FastAPI(title="FastAPI Server", description="A basic FastAPI server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directory for templates
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Create a directory for storing images if it doesn't exist
IMAGES_DIR = Path("")
IMAGES_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/report/{start_date}/{end_date}", response_class=HTMLResponse)
async def get_report(request: Request, start_date: str, end_date: str):
    """
    Returns a report for the given date range.
    
    Args:
        start_date: The start date for the report in YYYY-MM-DD format
        end_date: The end date for the report in YYYY-MM-DD format
        
    Returns:
        HTML page with the report for the given date range
    """
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Start date and end date are required")
    
    start_date = start_date.strip()
    end_date = end_date.strip()

    DATE_FORMAT = "%Y-%m-%d"
    try:
        # Validate the date format
        datetime.strptime(start_date, DATE_FORMAT)
        datetime.strptime(end_date, DATE_FORMAT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Check if the start date is before the end date
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")
    
    # Call the main function with the provided start and end dates
    report = main(start_date, end_date, cli=False)
    
    # Return HTML template with the report data
    return templates.TemplateResponse(
        "report.html", 
        {"request": request, "report": report, "start_date": start_date, "end_date": end_date}
    )

@app.get("/facereport/{fid}", response_class=HTMLResponse)
async def get_face_report(request: Request, fid: str):
    """
    Returns a report for the given date range.
    
    Args:
        start_date: The start date for the report in YYYY-MM-DD format
        end_date: The end date for the report in YYYY-MM-DD format
        
    Returns:
        HTML page with the report for the given date range
    """
    if not fid:
        raise HTTPException(status_code=400, detail="Start date and end date are required")
    
    # Call the main function with the provided start and end dates
    report = face_report(fid)
    
    # Return HTML template with the report data
    return templates.TemplateResponse(
        "facereport.html", 
        {"request": request, "report": report, "start_date": "", "end_date": ""}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/image")
async def get_image(image_path: str = Query(..., description="The path of the image file to return"), 
                    location: str = Query(..., description="The location of the face in the image file (top, right, bottom, left)"),
                    ):
    """
    Returns a cropped image based on the location coordinates.
    
    Args:
        image_path: The path of the image file to return
        location: The location coordinates (top, right, bottom, left) for cropping
        
    Returns:
        The cropped image file
    """
    print(f"RECEIVED image path first:{image_path}")
    image_path = base64.urlsafe_b64decode(image_path).decode()
    full_image_url = image_path
    location = base64.urlsafe_b64decode(location).decode()
    print(f"RECEIVED image url:{full_image_url}")
    # if not full_image_path.exists():
        # print(f"Image not found: {full_image_path}")
        # raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Parse the location string to get the coordinates
        location = location.strip()
        # convert url encoded string to tuple
        location = location.replace("%2C", ",").replace("%28", "(").replace("%29", ")").replace("%20", " ")

        # Expected format: "(top, right, bottom, left)"
        coords = ast.literal_eval(location)
        top, right, bottom, left = coords
        
        # Open the image and crop it
        response = requests.get(full_image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Image not found.")

        """Detect faces in an image and return locations"""
        image_content = response.content
        image_array = np.frombuffer(image_content, dtype=np.uint8)
        
        # image = cv2.imread(image_path)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(img)

        format_img = image_path.split('.')[-1].split('?')[0].upper()
        if format_img == 'JPG':
            format_img = 'JPEG'
        

        print(f"image format: {format_img},  image_url: {full_image_url}")
        cropped_img = pil_img.crop( (left, top, right, bottom) )
        
        # Save the cropped image to a bytes buffer
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format=format_img)
        img_byte_arr.seek(0)
        
        # Determine content type based on image format
        # content_type = full_image_url#f"image/{format_img.lower()}"
        content_type = f"image/{format_img.lower()}"

        return Response(content=img_byte_arr.getvalue(), media_type=content_type)
        # return Response(content=img_byte_arr.getvalue(), media_type=f"image/{format_img.lower()}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


