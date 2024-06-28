# Import the necessary packages
import streamlit as st
import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
from io import BytesIO
import base64

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def apply_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 11, offset=10, method="gaussian")
    return (gray > T).astype("uint8") * 255

def image_to_bytes(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    return BytesIO(img_encoded.tobytes())

def generate_download_link(image_bytes, filename):
    b64 = base64.b64encode(image_bytes.getvalue()).decode()
    href = f'<a href="data:file/jpeg;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def process_image(image, canny_thresh1, canny_thresh2, gaussian_blur_size):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    resized_image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
    edged = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None, "Could not find contour"

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped, "Success"

def main():
    st.title("Document Scanner App")

    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"])

    st.sidebar.title("Parameters")
    canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 50)
    canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 150)
    gaussian_blur_size = st.sidebar.slider("Gaussian Blur Size", 1, 15, 5, step=2)
    option = st.sidebar.radio("Choose Final Image Format", ('Color', 'Black & White'))

    if st.button("Process Images"):
        if not uploaded_files:
            st.warning("Please upload at least one image.")
        else:
            for index, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                st.write(f"Processing Image {index + 1}")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

                warped, status = process_image(image, canny_thresh1, canny_thresh2, gaussian_blur_size)
                if status == "Success":
                    if option == 'Black & White':
                        final_image = apply_threshold(warped)
                    else:
                        final_image = warped

                    st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="Scanned Image", use_column_width=True)

                    final_image_bytes = image_to_bytes(final_image)
                    download_link = generate_download_link(final_image_bytes, f"Scanned{index+1}.jpg")
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.warning(f"Could not process image {uploaded_file.name}. Skipping...")

if __name__ == "__main__":
    main()
