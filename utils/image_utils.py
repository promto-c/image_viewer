import os, math
import numpy as np
from numbers import Number
try:
    import OpenEXR
except ImportError:
    IS_SUPPORT_OPENEXR_LIB = False
else:
    IS_SUPPORT_OPENEXR_LIB = True
import cv2
import Imath
import struct

class ImageIO:

    @classmethod
    def read_image(cls, file_path: str):

        file_type_handlers = {
                'exr': cls.read_exr,
                'dpx': cls.read_dpx,
            }
                
        file_extension = file_path.split('.')[-1].lower()
        
        # Lookup read method for given file extension
        read_method = file_type_handlers.get(file_extension, cls.cv2_read_image)

        # if not read_method:
        #     supported_types = ", ".join(self.file_type_handlers.keys())
        #     raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: {supported_types}")

        return read_method(file_path)

    @staticmethod
    def cv2_read_image(file_path: str):
        # Read file using OpenCV
        image_data = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # Check if the image was successfully loaded
        if image_data is None:
            raise FileNotFoundError(f"Unable to load image at {file_path}")

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        return image_data

    @classmethod
    def read_exr(cls, image_path: str) -> np.ndarray:
        """Read an EXR image from file and return it as a NumPy array.

        Args:
            image_path (str): The path to the EXR image file.

        Returns:
            np.ndarray: The image data as a NumPy array.

        """
        if not os.path.isfile(image_path):
            return None

        if not IS_SUPPORT_OPENEXR_LIB:
            return cls.read_image(image_path)

        # Open the EXR file for reading
        exr_file = OpenEXR.InputFile(image_path)

        # Get the image header
        header = exr_file.header()

        # Get the data window (bounding box) and channels of the image
        data_window = header['dataWindow']
        channels = header['channels']

        # Calculate the width and height of the image
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1

        # Determine the channel keys
        channel_keys = 'RGB' if len(channels.keys()) == 3 else channels.keys()

        # Read all channels at once
        channel_data = exr_file.channels(channel_keys, Imath.PixelType(Imath.PixelType.FLOAT))

        # Using list comprehension to transform the channel data
        channel_data = [
            np.frombuffer(data, dtype=np.float32).reshape(height, width)
            for data in channel_data
        ]

        # Convert to NumPy array
        image_data = np.array(channel_data)

        return image_data.transpose(1, 2, 0)

    @staticmethod
    def read_dpx_header(file):
        headers = {}
        
        generic_file_header_format = ">I I 8s I I I I I 100s 24s 100s 200s 200s I 104s"
        headers['GenericFileHeader'] = struct.unpack(
            generic_file_header_format, file.read(768)
        )

        # Determine endianness based on magic number
        magic_number = headers['GenericFileHeader'][0]
        if magic_number == 0x53445058:
            headers['endianness'] = 'be'  # big-endian
        elif magic_number == 0x58504453:
            headers['endianness'] = 'le'  # little-endian

        generic_image_header_format = ">H H I I"
        headers['GenericImageHeader'] = struct.unpack(
            generic_image_header_format, file.read(12)
        )
    
        return headers

    @classmethod
    def read_dpx(cls, image_path: str) -> np.ndarray:
        with open(image_path, "rb") as file:

            meta = cls.read_dpx_header(file)
            width = meta['GenericImageHeader'][2]
            height = meta['GenericImageHeader'][3]
            offset = meta['GenericFileHeader'][1]

            file.seek(offset)
            raw = np.fromfile(file, dtype=np.int32, count=width*height)

        raw = raw.reshape(height, width)

        if meta['endianness'] == 'be':
            raw.byteswap(True)

        image_data = np.array([raw >> 22, raw >> 12, raw >> 2], dtype=np.uint16)
        image_data &= 0x3FF

        # NOTE: to uint8
        # image_data = (image_data >> 2).astype(np.uint8)

        # Convert to float32 and normalize
        image_data = image_data.astype(np.float32)
        image_data /= 0x3FF

        return image_data.transpose(1, 2, 0)

    @classmethod
    def read_dpx_12bit_packed(cls, image_path: str) -> np.ndarray:
        with open(image_path, "rb") as file:

            meta = cls.read_dpx_header(file)
            width = meta['GenericImageHeader'][2]
            height = meta['GenericImageHeader'][3]
            offset = meta['GenericFileHeader'][1]

            file.seek(offset)

            words_per_line = math.ceil(width * 9/4)
            raw = np.fromfile(file, dtype=np.uint16, count=words_per_line*height)

        word_lines = raw.reshape(height, words_per_line)

        if meta['endianness'] == 'be':
            raw.byteswap(True)

        # TODO: read num channels from metadata
        # Constants for the process
        components_per_pixel = 3  # RGB components

        # Extract 8 components from 6 halfwords (read as 16bit)
        # word1: B5 B6 B7 B8 B9 B10 B11 B12  G1 G2 G3 G4 G5 G6 G7 G8    G9 G10 G11 G12  R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12
        # word2: BBBBGGGGGGGGGGGG RRRRRRRRRRRR B1 B2 B3 B4
        # word3: GGGGGGGGGGGGRRRR RRRRRRRRBBBBBBBB
        image_data = np.array([
            (word_lines[:, 1::6] & 0xFFF),
            ((word_lines[:, 0::6] & 0xFF) << 4) | (word_lines[:, 1::6] >> 12),
            ((word_lines[:, 3::6] & 0xF) << 8) | (word_lines[:, 0::6] >> 8),
            (word_lines[:, 3::6] >> 4),
            (word_lines[:, 2::6] & 0xFFF),
            ((word_lines[:, 5::6] & 0xFF) << 4) | (word_lines[:, 2::6] >> 12),
            ((word_lines[:, 4::6] & 0xF) << 8) | (word_lines[:, 5::6] >> 8),
            (word_lines[:, 4::6] >> 4 & 0xFFF)
        ], dtype=np.uint16).transpose(1, 2, 0).reshape(height, width, components_per_pixel)  # Reshape to (height, width, components_per_pixel)

        # Convert to float32 and normalize
        image_data = image_data.astype(np.float32)
        image_data /= 0x0FFF

        return image_data
