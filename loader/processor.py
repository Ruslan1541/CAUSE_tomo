import os
import cv2
import torch
import logging
import numpy as np
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from patchify import patchify
from typing import Tuple, List
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageDataProcessor:
    """Handles both image and dataset processing operations."""
    
    def __init__(self, config):
        """
        Initialize the processor with configuration.
        
        Args:
            config: ProcessingConfig object containing processing parameters
        """
        self.step = None
        self.config = config
        self.padding_size = None
        
        self.output_dir = Path(self.config.dataset.data_dir) / self.config.dataset.dst
        self.stats_dir = self.output_dir / f'{self.config.dataset.dst}_std_mean'
        self.patches_dir = self.output_dir / f'{self.config.dataset.dst}_patches'
        
        if os.path.exists(self.patches_dir):
            logger.info(f"{self.config.dataset.dst} dataset with extracted patches already exists, please check its content")
            
        self.stats_dir.mkdir(exist_ok=True)
        self.patches_dir.mkdir(exist_ok=True)
        
        #update config file
        self.config.std_mean_path = str(self.stats_dir)
        self.config.dataset.data_dir = str(self.output_dir)
        self.config.dataset.dst = f'{self.config.dataset.dst}_patches'
        
    def process_dataset(self) -> Tuple[int, int]:
        """
        Process the entire dataset by extracting and saving patches.
        
        Returns:
            Tuple[int, int]: Shape of patches (height, width)
        """
        try:
            img_files = sorted(glob(f'{self.output_dir}/*.tif'))[:self.config.max_images]
            if not img_files:
                raise FileNotFoundError(f"No image files found in {self.output_dir}")

            if self.config.dataset.patchify:
                self._process_all_images(img_files)
                
            return self._get_patch_dimensions(img_files[0])
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def _process_all_images(self, 
                            img_files: List[str]) -> None:
        """
        Process all images in the dataset.
        
        Args:
            img_files: List of image file paths
        """
        images = [] if self.config.dataset.standardization else None
        idx = 0
        
        logger.info(f"Processing {len(img_files)} images...")
        
        for img_path in tqdm(img_files):
            try:
                patches, input_image = self._process_single_image(img_path)
                
                if self.config.dataset.standardization:
                    images.append(input_image)
                    
                #Save extracted patches to files.
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch_path = self.patches_dir / f'{idx}.tif'
                        cv2.imwrite(str(patch_path), patches[i, j, 0])
                        idx += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
                
        if self.config.dataset.standardization and images:
            self._calculate_and_save_statistics(images)

    def _process_single_image(self, 
                              image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single image and extract patches.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing patches array and original image
        """
        input_image = cv2.imread(image_path, 1)
        if self.padding_size is None:
            self.padding_size = self._calculate_padding(input_image)
        
        cropped_image = self._crop_image(input_image)
        patches = self._create_patches(cropped_image)
        
        return patches, input_image

    def _create_patches(self, 
                        image: np.ndarray) -> np.ndarray:
        """
        Create patches from image.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Array of patches
        """
        return patchify(
            image,
            (self.config.patch_size, self.config.patch_size, 3),
            step=self.step
        )

    def _crop_image(self, 
                    image: np.ndarray) -> np.ndarray:
        """
        Crop image based on padding size.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Cropped image
        """
        if self.padding_size > 0:
            return image[self.padding_size:image.shape[0]-self.padding_size,
                         self.padding_size:image.shape[0]-self.padding_size]
        return image

    def _calculate_padding(self, 
                           image: np.ndarray) -> int:
        """
        Calculate the padding size needed for consistent patching.
        
        Args:
            image: Input image array
            
        Returns:
            int: Padding size
        """
        self.step = int(self.config.patch_size * (1 - self.config.overlap_fraction))
        patches = self._create_patches(image)
        condition = (patches.shape[0] - 1) * self.step + self.config.patch_size
        
        if condition == image.shape[0]:
            return 0
            
        size = (image.shape[0] - condition) // 2
        return size - (size % 2)

    def _calculate_and_save_statistics(self,
                                       images: List[np.ndarray]) -> None:
        """
        Calculate and save standardization statistics.
        
        Args:
            images: List of image arrays
        """
        try:
            std = 0
            mean = 0
            logger.info(f"Std_mean estimation for {len(images)} images...")
            for i in tqdm(range(len(images))):
                transform = transforms.Compose([transforms.ToTensor()])
                img_tr = transform(images[i])
                m, s = img_tr.mean([1,2]), img_tr.std([1,2])
                mean += m
                std += s
            std /= len(images)
            mean /= len(images)
            
            torch.save(std, self.stats_dir / "std.pt")
            torch.save(mean, self.stats_dir / "mean.pt")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def _get_patch_dimensions(self, 
                              sample_image_path: str) -> Tuple[int, int]:
        """
        Get the dimensions of patches for a sample image.
        
        Args:
            sample_image_path: Path to sample image
            
        Returns:
            Tuple[int, int]: Patch dimensions
        """
        patches, _ = self._process_single_image(sample_image_path)
        return patches.shape[0], patches.shape[1]
