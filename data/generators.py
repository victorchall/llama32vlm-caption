"""
Copyright [2022-2024] Victor C Hall

Licensed under the Open Software License 3.0

   https://opensource.org/license/osl-3-0-php
"""

import os
from typing import Generator
from PIL import Image, ImageOps
import tarfile
import logging

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def image_path_generator(image_dir: str, do_recurse: bool = True) -> Generator[str, None, None]:
    if do_recurse:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                    yield os.path.join(root, file)
    else:
        for file in os.listdir(image_dir):
            if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                yield os.path.join(image_dir, file)