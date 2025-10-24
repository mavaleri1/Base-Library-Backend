"""
Utilities for working with files and images in core.
"""

import shutil
import logging
import hashlib
from pathlib import Path
from typing import List
from PIL import Image

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class ImageFileManager:
    """Manager for working with image files"""

    def __init__(self):
        self.settings = get_settings()

    def create_temp_directory(self, thread_id: str) -> Path:
        """
        Creates temporary directory for thread_id.

        Args:
            thread_id: Thread identifier

        Returns:
            Path: Path to created directory
        """
        temp_dir = Path(self.settings.temp_storage_path) / thread_id / "images"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        return temp_dir

    def validate_image_file(self, file_path: Path) -> bool:
        """
        Validates image file.

        Args:
            file_path: Path to file

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            # Check extension
            if file_path.suffix.lower() not in self.settings.supported_image_formats:
                logger.warning(f"Unsupported image format: {file_path.suffix}")
                return False

            # Check file size
            if file_path.stat().st_size > self.settings.max_image_size:
                logger.warning(f"Image too large: {file_path.stat().st_size} bytes")
                return False

            # Check that file is actually an image
            with Image.open(file_path) as img:
                img.verify()

            logger.info(f"Image validated successfully: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Image validation failed for {file_path}: {e}")
            return False

    def save_uploaded_images(
        self, thread_id: str, image_data_list: List[bytes]
    ) -> List[str]:
        """
        Saves uploaded images to temporary directory.

        Args:
            thread_id: Thread identifier
            image_data_list: List of image byte data

        Returns:
            List[str]: List of paths to saved files
        """
        if len(image_data_list) > self.settings.max_images_per_request:
            raise ValueError(
                f"Too many images: {len(image_data_list)} > {self.settings.max_images_per_request}"
            )

        temp_dir = self.create_temp_directory(thread_id)
        saved_paths = []

        for i, image_data in enumerate(image_data_list):
            # Create hash for filename
            image_hash = hashlib.md5(image_data).hexdigest()[:10]
            file_path = temp_dir / f"image_{i:02d}_{image_hash}.png"

            # Save file
            with open(file_path, "wb") as f:
                f.write(image_data)

            # Validate saved file
            if self.validate_image_file(file_path):
                saved_paths.append(str(file_path))
                logger.info(f"Saved image: {file_path}")
            else:
                # Remove invalid file
                file_path.unlink(missing_ok=True)
                logger.warning(f"Removed invalid image: {file_path}")

        return saved_paths

    def cleanup_temp_directory(self, thread_id: str) -> None:
        """
        Cleans up temporary directory for thread_id.

        Args:
            thread_id: Thread identifier
        """
        temp_dir = Path(self.settings.temp_storage_path) / thread_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")

    def get_image_paths_for_thread(self, thread_id: str) -> List[str]:
        """
        Gets list of image paths for thread_id.

        Args:
            thread_id: Thread identifier

        Returns:
            List[str]: List of image paths
        """
        temp_dir = Path(self.settings.temp_storage_path) / thread_id / "images"
        if not temp_dir.exists():
            return []

        image_paths: list[str] = []
        for ext in self.settings.supported_image_formats:
            image_paths.extend(str(p) for p in temp_dir.glob(f"*{ext}"))
            image_paths.extend(str(p) for p in temp_dir.glob(f"*{ext.upper()}"))

        return sorted(image_paths)


def ensure_temp_storage() -> None:
    """Creates base directory for temporary storage"""
    settings = get_settings()
    temp_path = Path(settings.temp_storage_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured temp storage directory: {temp_path}")
