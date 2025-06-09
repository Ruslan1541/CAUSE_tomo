
import cv2
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode

#custom functionality 
from utils.utils import set_seeds, ToTargetTensor

logger = logging.getLogger(__name__)

def create_transform(res: Optional[int] = None, 
                     target: bool = False) -> transforms.Compose:
    """
    Create appropriate transform composition.
    
    Args:
        res: Resolution for resize transform
        target: Whether to create target transform
        
    Returns:
        Composed transformation
    """
    if res is not None:
        return transforms.Compose([
            transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(res),
            transforms.ToTensor()
        ])
    elif target:
        return transforms.Compose([ToTargetTensor()])
    return transforms.Compose([transforms.ToTensor()])

def create_dataset(args: Any, 
                   label: bool = False) -> Dataset:
    """
    Create appropriate dataset based on configuration.
    
    Args:
        args: Configuration arguments
        label: Whether to create labeled dataset
        
    Returns:
        Dataset instance
    """
    try:
        # Determine transform based on resolution
        transform = create_transform(
            res=args.train_resolution if args.train_resolution % 14 == 0 else None
        )
        
        # Load normalization parameters
        std = torch.load(Path(args.std_mean_path) / "std.pt")
        mean = torch.load(Path(args.std_mean_path) / "mean.pt")
        normalize = transforms.Normalize(mean=mean, std=std)
        
        # Create appropriate dataset
        if label:
            target_transform = create_transform(target=True)
            return CroppedLabelDataset(
                img_dir=args.dataset.data_dir,
                dataset_name=args.dataset.dst,
                transform=transform,
                target_transform=target_transform,
                normalize=normalize
            )
            
        return CroppedDataset(
            img_dir=args.dataset.data_dir,
            dataset_name=args.dataset.dst,
            transform=transform,
            normalize=normalize
        )
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def dataloader(args: Any,
               no_ddp_train_shuffle: bool = False,
               label: bool = False
               ) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create dataloader with appropriate configuration.
    
    Args:
        args: Configuration arguments
        no_ddp_train_shuffle: Whether to shuffle in non-distributed mode
        label: Whether to create labeled dataset
        
    Returns:
        Tuple of (DataLoader, Optional[DistributedSampler])
    """
    try:
        # Create dataset
        dataset = create_dataset(args, label)
        
        # Configure sampler
        sampler = None
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            
        # Create and return loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=not args.distributed and no_ddp_train_shuffle,
            pin_memory=True,
            sampler=sampler,
            num_workers=args.num_workers
        )
        
        return loader, sampler
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        raise

class CroppedDataset(Dataset):
    """
    Dataset for cropped images.
    """
    def __init__(self, 
                 img_dir: str, 
                 dataset_name: str, 
                 transform: Optional[transforms.Compose] = None,
                 normalize: Optional[transforms.Normalize] = None):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.transform = transform
        self.normalize = normalize
        self.img_dir = Path(img_dir) / self.dataset_name
        self._validate_directory()
        
    def _validate_directory(self) -> None:
        """
        Validate image directory exists and contains images.
        """
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")
        
        self.num_images = len(list(self.img_dir.glob("*.tif")))
        if self.num_images == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
    def _load_image(self, 
                    path: Path) -> Image.Image:
        """
        Load image from path.
        """
        return Image.fromarray(cv2.imread(str(path), cv2.IMREAD_COLOR))

    def __getitem__(self, 
                    index: int) -> Dict[str, torch.Tensor]:
        """
        Get transformed and normalized image.
        """
        image = self._load_image(self.img_dir / f"{index}.tif")
        set_seeds()
        transformed_img = self.transform(image).unsqueeze(0)
        
        return {
            "ind": index,
            "img": self.normalize(transformed_img[0])
        }

    def __len__(self) -> int:
        """
        Get number of images.
        """
        return self.num_images
    
class CroppedLabelDataset(CroppedDataset):
    """
    Dataset for cropped images with labels to estimate difference between CAUSE predictions and ground truth
    """
    def __init__(self, 
                 img_dir: str,
                 dataset_name: str, 
                 transform: transforms.Compose,
                 target_transform: transforms.Compose,
                 normalize: transforms.Normalize):
        
        base_dir = Path(img_dir) / dataset_name
        super().__init__(str(base_dir / "img"), dataset_name, transform, normalize)
        self.label_dir = base_dir / "label"
        self.target_transform = target_transform
        self._validate_label_directory()
        
    def _validate_label_directory(self) -> None:
        """
        Validate label directory exists and matches images.
        """
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        num_labels = len(list(self.label_dir.glob("*.tif")))
        if num_labels != self.num_images:
            raise ValueError(f"Mismatch between images ({self.num_images}) and labels ({num_labels})")

    def __getitem__(self, 
                    index: int) -> Dict[str, torch.Tensor]:
        """
        Get transformed image and label pair.
        """
        image = self._load_image(self.img_dir / f"{index}.tif")
        target = Image.fromarray(cv2.imread(
            str(self.label_dir / f"{index}.tif"),
            cv2.IMREAD_GRAYSCALE
        ))
        
        set_seeds()
        transformed_image = self.transform(image)
        set_seeds()
        transformed_target = self.target_transform(target)
        
        return {
            "ind": index,
            "img": self.normalize(transformed_image.unsqueeze(0)),
            "label": transformed_target.squeeze(0)
        }
