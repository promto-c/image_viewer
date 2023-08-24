import cv2
import numpy as np

def to_uint8_gray(image_data):
    # Check if the image is colored (3 channels)
    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # Ensure the image is 8-bit
    if image_data.dtype != np.uint8:
        # Normalize the image to be in the range [0, 255]
        image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
        image_data = image_data.astype(np.uint8)

    return image_data

def fps_to_interval_msc(fps) -> int:
    """Converts frames per second to time interval in milliseconds.

    Args:
        fps (float): Frames per second.

    Returns:
        float: Time interval between frames in milliseconds.

    Raises:
        ValueError: If FPS is not a positive value.
    """
    if fps <= 0:
        raise ValueError("FPS must be a positive value.")
    
    return int(1000 / fps)
