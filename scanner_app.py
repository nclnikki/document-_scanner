import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import img2pdf

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped

def find_document_contour(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    document_contour = None
    
    # Loop through contours
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If we have found a contour with 4 points, we can assume it's the document
        if len(approx) == 4:
            document_contour = approx
            break
    
    return document_contour

def enhance_document(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7
    )
    
    # Apply some morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def main():
    st.title("Document Scanner App")
    
    # Create sidebar for settings and controls
    st.sidebar.title("Settings")
    
    # Initialize session state for captured images
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'scanned_image' not in st.session_state:
        st.session_state.scanned_image = None
    
    # Camera input
    img_file = st.camera_input("Take a picture of your document")
    
    if img_file is not None:
        # Convert to OpenCV format
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.session_state.current_image = cv_img
        
        # Document scanning
        document_contour = find_document_contour(cv_img)
        
        if document_contour is not None:
            # Draw contour on copy of original image
            display_img = cv_img.copy()
            cv2.drawContours(display_img, [document_contour], -1, (0, 255, 0), 2)
            
            # Transform the image to get the document perspective
            document_pts = document_contour.reshape(4, 2)
            warped = four_point_transform(cv_img, document_pts)
            
            # Image enhancement options
            st.sidebar.subheader("Document Enhancement")
            apply_enhancement = st.sidebar.checkbox("Apply Document Enhancement", value=True)
            
            if apply_enhancement:
                enhanced = enhance_document(warped)
                st.session_state.scanned_image = enhanced
                
                # Show original with contour and processed images
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Detected Document")
                    st.image(display_img, channels="BGR")
                with col2:
                    st.subheader("Scanned Result")
                    st.image(enhanced)
            else:
                st.session_state.scanned_image = warped
                
                # Show original with contour and warped image
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Detected Document")
                    st.image(display_img, channels="BGR")
                with col2:
                    st.subheader("Warped Result")
                    st.image(warped, channels="BGR")

            if st.button("Save Image"):

                if apply_enhancement:
                    # For enhanced document (grayscale)
                    img_to_save = Image.fromarray(st.session_state.scanned_image)
                else:
                    # For warped image (BGR)
                    img_to_save = Image.fromarray(cv2.cvtColor(st.session_state.scanned_image, cv2.COLOR_BGR2RGB))
            
                # Add to session state
                img_byte_arr = io.BytesIO()
                img_to_save.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.session_state.captured_images.append(img_byte_arr)
                st.success(f"Image saved! Total images: {len(st.session_state.captured_images)}")
        
        else:
            st.error("No document detected. Please ensure your document is clearly visible against the background.")
            st.image(cv_img, channels="BGR")

    # Show captured images
    if st.session_state.captured_images:
        st.subheader("Captured Images")
        cols = st.columns(3)
        for idx, img_bytes in enumerate(st.session_state.captured_images):
            with cols[idx % 3]:
                st.image(img_bytes, caption=f"Image {idx + 1}")

        # Generate PDF
        if st.button("Generate PDF"):
            try:
                # Create PDF
                pdf_bytes = io.BytesIO()
                
                # Convert images to PIL format for PDF creation
                pil_images = []
                for img_bytes in st.session_state.captured_images:
                    pil_images.append(Image.open(io.BytesIO(img_bytes)))
                
                # Save as PDF
                pdf_bytes.write(img2pdf.convert([io.BytesIO(img_bytes) for img_bytes in st.session_state.captured_images]))
                
                # Offer download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes.getvalue(),
                    file_name=f"scanned_document_{timestamp}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

        # Clear all images
        if st.button("Clear All Images"):
            st.session_state.captured_images = []
            st.success("All images cleared!")

if __name__ == "__main__":
    main()
