from enum import Enum


class FileExtension(Enum):
    JPG = 'jpg'
    JPEG = 'jpeg'
    PNG = 'png'
    NPY = 'npy'
    XML = 'xml'
    JXL = 'jxl'
    BMP = 'bmp'
    TIF = 'tif'
    TIFF = 'tiff'
    WEBP = 'webp'


class OverlapStrategy(str, Enum):
    """
    Holds strategies for keeping bounding boxes based on overlap
    """
    DISABLED = "disabled"
    BIGGEST = "biggest"
    SMALLEST = "smallest"


all_image_extensions = [FileExtension.JXL, FileExtension.JPG, FileExtension.PNG, FileExtension.JPEG, FileExtension.BMP, FileExtension.TIF, FileExtension.TIFF, FileExtension.WEBP]
