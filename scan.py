import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
import os
import glob
from tqdm import tqdm
from typing import List, Tuple

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perform a four-point perspective transform on an image.
    
    Args:
        image (np.ndarray): The input image.
        pts (np.ndarray): A NumPy array of shape (4, 2) representing the four points.

    Returns:
        np.ndarray: The warped and transformed image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in a consistent manner: top-left, top-right, bottom-right, and bottom-left.
    
    Args:
        pts (np.ndarray): A NumPy array of shape (4, 2) representing the four points.

    Returns:
        np.ndarray: Ordered coordinates of the points.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Top-left point and bottom-right point
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right point and bottom-left point
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def apply_threshold(image: np.ndarray) -> np.ndarray:
    """Convert image to black and white using a local threshold.
    
    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The thresholded image in binary form.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 11, offset=10, method="gaussian")
    return (gray > T).astype("uint8") * 255

def process_image(image: np.ndarray, canny_thresh1: int = 0, canny_thresh2: int = 100, 
                  gaussian_blur_size: int = 5) -> np.ndarray:
    """
    Process an image to detect document boundary and apply a perspective transform.
    
    Args:
        image (np.ndarray): Input image data.
        canny_thresh1 (int): First threshold for the Canny edge detector. Defaults to 0.
        canny_thresh2 (int): Second threshold for the Canny edge detector. Defaults to 100.
        gaussian_blur_size (int): Kernel size for Gaussian blur. Defaults to 5.
    
    Returns:
        np.ndarray: The warped and transformed image.
    
    Raises:
        ValueError: If no contour is detected.
    """
    if image is None:
        raise ValueError("Image is None.")
    
    # Resize and process the image
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    resized = imutils.resize(image, height=500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
    edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # Find contours in the edged image, keeping only the largest ones
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        raise ValueError("Could not find contour")
    
    # Apply four-point transform to get the top-down view
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped

def process_images_in_directory(input_directory: str, output_directory: str, canny_thresh1: int, 
                                canny_thresh2: int, gaussian_blur_size: int) -> None:
    """
    Process all images in a directory and save the scanned results.
    
    Args:
        input_directory (str): Path to the input image directory.
        output_directory (str): Path to the output directory.
        canny_thresh1 (int): First threshold for the Canny edge detector.
        canny_thresh2 (int): Second threshold for the Canny edge detector.
        gaussian_blur_size (int): Kernel size for Gaussian blur.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    extensions = ('*.jpg', '*.png')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_directory, ext)))

    if not image_paths:
        print("No images found in the specified directory.")
        return

    for index, image_path in enumerate(tqdm(image_paths)):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image at {image_path} could not be read.")
            
            warped = process_image(image, canny_thresh1, canny_thresh2, gaussian_blur_size)
            final_image = apply_threshold(warped)
            output_path = os.path.join(output_directory, f"Scanned{index+1}.jpg")
            cv2.imwrite(output_path, final_image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Scan documents from images.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input image directory")
    parser.add_argument("-o", "--output", type=str, default="Output",
                        help="Path to the output directory")
    parser.add_argument("--canny1", type=int, default=0,
                        help="First threshold for the Canny edge detector")
    parser.add_argument("--canny2", type=int, default=100,
                        help="Second threshold for the Canny edge detector")
    parser.add_argument("--blur", type=int, default=5,
                        help="Gaussian blur kernel size")

    args = parser.parse_args()
    
    process_images_in_directory(args.input, args.output, args.canny1, args.canny2, args.blur)
