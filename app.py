import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import imutils
import scan
import hmac

def image_to_bytes(image: np.ndarray) -> BytesIO:
    """Convert an image to bytes.
    
    Args:
        image (np.ndarray): The input image.
    
    Returns:
        BytesIO: The image in bytes.
    """
    _, img_encoded = cv2.imencode('.jpg', image)
    return BytesIO(img_encoded.tobytes())

def generate_download_button(image_bytes: BytesIO, filename: str):
    """Generate a download button for the image.
    
    Args:
        image_bytes (BytesIO): The image in bytes.
        filename (str): The name for the downloaded file.
    """
    st.sidebar.download_button(label="Download image", data=image_bytes, file_name=filename, mime="image/jpeg")
    
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False



def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")  # Set the app layout to wide mode
    st.title("Papyrus")
    st.subheader("Document Scanner App")
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.
    # Single file uploader for image upload
    uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False, type=["jpg", "png"])

    # Sidebar for user parameters
    st.sidebar.title("Parameters")
    canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 50)
    canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 150)
    gaussian_blur_size = st.sidebar.slider("Gaussian Blur Size", 1, 15, 5, step=2)
    final_image_format = st.sidebar.radio("Choose Final Image Format", ('Color', 'Black & White'))

    # Additional parameters for B&W conversion if the user selects Black & White
    if final_image_format == 'Black & White':
        method = st.sidebar.selectbox("Thresholding Method", ['adaptive', 'clahe', 'otsu', 'combined'])

        if method == 'adaptive':
            with st.sidebar.expander("Adaptive Threshold Parameters"):
                adaptive_block_size = st.sidebar.slider("Block Size", 3, 51, 35, step=2)
                adaptive_c = st.sidebar.slider("C", 0, 30, 11)
        elif method == 'clahe':
            with st.sidebar.expander("CLAHE Parameters"):
                clahe_clip_limit = st.sidebar.slider("Clip Limit", 1.0, 4.0, 2.0, step=0.1)
        elif method == 'otsu':
            with st.sidebar.expander("Otsu's Method Parameters"):
                otsu_blur_size = st.sidebar.slider("Blur Size", 1, 15, 5, step=2)
        elif method == 'combined':
            with st.sidebar.expander("Combined Parameters"):
                combined_block_size = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
                adaptive_c_comb = st.sidebar.slider("C", 0, 10, 2)
                clahe_clip_limit_comb = st.sidebar.slider("Clip Limit", 1.0, 4.0, 2.0, step=0.1)

    def find_screen_contour(contours: list[np.ndarray]) -> np.ndarray:
        """Find the appropriate screen contour with 4 points.
        
        Args:
            contours (list[np.ndarray]): List of contours found in the image.
        
        Returns:
            np.ndarray: The found contour that matches a quadrilateral, otherwise None.
        """
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
        return None
    
    def draw_contours(image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        """Draw contours on the image.
        
        Args:
            image (np.ndarray): The original image.
            contours (np.ndarray): The contours to draw on the image.
        
        Returns:
            np.ndarray: The image with contours drawn.
        """
        image_with_contours = image.copy()
        if contours is not None and len(contours.shape) == 3 and contours.shape[0] == 4:  # Ensure contours are valid before drawing
            cv2.drawContours(image_with_contours, [contours], -1, (0, 255, 0), 2)
        return image_with_contours

    if uploaded_file:
        # Read image data from the uploaded file
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the image")
            
            st.write("Processing Image")
            
            # Process the image
            ratio = image.shape[0] / 500.0
            orig_image = image.copy()
            resized_image = imutils.resize(image, height=500)
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (gaussian_blur_size, gaussian_blur_size), 0)
            edged_image = cv2.Canny(blurred_image, canny_thresh1, canny_thresh2)

            contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            screen_contour = find_screen_contour(contours)
            if screen_contour is None:
                raise ValueError("Could not find a valid contour")

            # Draw contours on the original image
            original_with_contours = draw_contours(orig_image, screen_contour.reshape(4, 2) * ratio)

            # Apply four-point transform to get the top-down view
            warped_image = scan.four_point_transform(orig_image, screen_contour.reshape(4, 2) * ratio)

            if final_image_format == 'Black & White':
                if method == 'adaptive':
                    final_image = scan.apply_threshold(warped_image, method='adaptive', block_size=adaptive_block_size, c=adaptive_c)
                elif method == 'clahe':
                    final_image = scan.apply_threshold(warped_image, method='clahe', clip_limit=clahe_clip_limit)
                elif method == 'otsu':
                    final_image = scan.apply_threshold(warped_image, method='otsu', blur_size=otsu_blur_size)
                elif method == 'combined':
                    final_image = scan.apply_threshold(warped_image, method='combined', block_size=combined_block_size, c=adaptive_c_comb, clip_limit=clahe_clip_limit_comb)
            else:
                final_image = warped_image

            # Display the original image with contours and the final processed image side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(original_with_contours, cv2.COLOR_BGR2RGB), caption="Original Image with Contours", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="Final Processed Image", use_column_width=True)

            final_image_bytes = image_to_bytes(final_image)
            generate_download_button(final_image_bytes, "Scanned.jpg")
        except Exception as e:
            st.warning(f"Could not process image. Error: {e}")

if __name__ == "__main__":
    main()