import cv2
import pytesseract
import pyttsx3
import numpy as np
import os
import time

# If Tesseract is installed at a non-standard location, set its path here:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_image():
    """Capture a single frame from the default camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return None

    print("Press 'c' to capture an image or 'q' to quit.")
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from camera.")
            break

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            captured_frame = frame.copy()
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame

def preprocess_image(image):
    """
    Preprocess the image for OCR using an alternative approach:
    1. Convert to grayscale.
    2. Apply median blur.
    3. Use adaptive thresholding to binarize the image.
    4. Apply a morphological opening to remove noise.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 3)
    
    # Apply adaptive thresholding (Gaußian method)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological opening to remove small noise (kernel size can be tuned)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return opened

def extract_text(image):
    """Use pytesseract to extract text from the preprocessed image."""
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()

def speak_text(text):
    """Convert text to speech using pyttsx3."""
    if not text:
        print("No text found to speak.")
        return

    try:
        engine = pyttsx3.init()
        print("Speaking out the extracted text...")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error in text-to-speech conversion:", e)

def display_extracted_text(text):
    """Overlay the extracted text on a white background and display it."""
    # Create a white image with fixed dimensions
    img_height, img_width = 600, 800
    blank_image = np.ones((img_height, img_width, 3), np.uint8) * 255

    # Set font parameters for display
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Black text
    thickness = 2

    # Split text into lines
    lines = text.split('\n')
    y0, dy = 30, 30  # Starting y position and line spacing
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(blank_image, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Extracted Text", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Starting the Talking Camera for Low Vision/Elderly application.")
    
    # Step 1: Capture Image
    image = capture_image()
    if image is None:
        print("No image captured. Exiting application.")
        return

    # Optional: Display the captured image
    cv2.imshow("Captured Image", image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # Step 2: Preprocess Image using the new approach
    preprocessed_image = preprocess_image(image)
    # Optional: Display the preprocessed image
    cv2.imshow("Preprocessed Image", preprocessed_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # Step 3: Extract Text using OCR
    extracted_text = extract_text(preprocessed_image)
    print("Extracted Text:\n", extracted_text)

    # Step 4: Display the extracted text on a white background
    display_extracted_text(extracted_text)

    # Step 5: Read Text Aloud using pyttsx3
    speak_text(extracted_text)

if __name__ == "__main__":
    main()
