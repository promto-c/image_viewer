import numpy as np
import OpenEXR
import Imath

def read_exr(image_path: str) -> np.ndarray:
    """Read an EXR image from file and return it as a NumPy array.

    Args:
        image_path (str): The path to the EXR image file.

    Returns:
        np.ndarray: The image data as a NumPy array.

    """
    # Open the EXR file for reading
    exr_file = OpenEXR.InputFile(image_path)

    # Get the image header
    header = exr_file.header()

    # Get the data window (bounding box) of the image
    data_window = header['dataWindow']

    # Get the channels present in the image
    channels = header['channels']

    # Calculate the width and height of the image
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Determine the channel keys
    channel_keys = 'RGB' if len(channels.keys()) == 3 else channels.keys()

    # Read all channels at once
    channel_data = exr_file.channels(channel_keys, Imath.PixelType(Imath.PixelType.FLOAT))

    # Create an empty NumPy array to store the image data
    image_data = np.zeros((height, width, len(channel_keys)), dtype=np.float32)

    # Populate the image data array
    for i, data in enumerate(channel_data):
        # Retrieve the pixel values for the channel
        pixels = np.frombuffer(data, dtype=np.float32)
        # Reshape the pixel values to match the image dimensions and store them in the image data array
        image_data[:, :, i] = pixels.reshape((height, width))

    return image_data
