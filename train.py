import logging
import os
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig, OmegaConf

from train_mediator import mediator_train
from train_front_door import front_train
from fine_tuning import fine_tune
from inference import inference
from loader.processor import ImageDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Manages the complete training pipeline execution."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the training pipeline.
        
        Args:
            cfg: Configuration object from Hydra
        """
        self.cfg = cfg
        self.gpu_list = list(map(int, cfg.gpu.split(',')))
        self.ngpus_per_node = len(self.gpu_list)
        
    def run_training_stage(self, 
                           stage_name: str, 
                           training_func, 
                           rank: int) -> None:
        """
        Execute a single training stage.
        
        Args:
            # stage_name: singe training stage run
            rank: GPU rank for distributed training
        """
        if rank == 0:
            logger.info(f'-------------{stage_name}-------------')
        try:
            training_func(rank, self.cfg, self.ngpus_per_node)
        except Exception as e:
            logger.error(f"Error in {stage_name}: {str(e)}")
            logger.debug("Detailed traceback:", exc_info=True)

    def run_complete_pipeline(self, rank: int) -> None:
        """
        Execute all training stages in sequence.
        
        Args:
            rank: GPU rank for distributed training
        """
        self.cfg.load_segment = False
        self.cfg.load_cluster = False
        # Mediator training
        self.run_training_stage("Train Mediator", mediator_train, rank)
        
        # Front door training
        self.run_training_stage("Train Front", front_train, rank)
        
        # Fine tuning
        self.cfg.load_segment = True
        self.run_training_stage("Fine Tune", fine_tune, rank)
        
        # Inference
        self.cfg.load_cluster = True
        self.run_training_stage("Inference", inference, rank)

    def process_dataset(self) -> None:
        """
        Process the dataset and update configuration.
        """
        try:
            processor = ImageDataProcessor(self.cfg)
            patches_shape = processor.process_dataset()
            
            # Configure model parameters
            self._configure_model_parameters(patches_shape)
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            raise
        
    def run(self, rank: int) -> None:
        """
        Main execution method for the pipeline.
        
        Args:
            rank: GPU rank for distributed training
        """
        try:
            if rank == 0:
                logger.info("Starting training pipeline...")
                logger.info("\nConfiguration:")
                logger.info(OmegaConf.to_yaml(self.cfg))
                
            self.run_complete_pipeline(rank)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.debug("Detailed traceback:", exc_info=True)
            raise
            
    def _configure_model_parameters(self, patches_shape) -> None:
        """
        Configure model-specific parameters based on checkpoint.
        """
        self.cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if 'dinov2' in self.cfg.model.ckpt:
            self.cfg.train_resolution = 322
            
        if 'small' in self.cfg.model.ckpt:
            self.cfg.dim = 384
        elif 'base' in self.cfg.model.ckpt:
            self.cfg.dim = 768
        
        #set patches_shape for inference stage
        self.cfg.patches_shape = patches_shape
        
        # Calculate number of queries based on patch size
        patch_size = int(self.cfg.model.ckpt.split('_')[-1].split('.')[0])
        self.cfg.num_queries = (self.cfg.train_resolution ** 2 // 
                                    patch_size ** 2)
        
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the training pipeline.
    
    Args:
        cfg: Configuration object from Hydra
    """
    try:
        pipeline = TrainingPipeline(cfg)
        pipeline.process_dataset()
        
        if cfg.distributed:
            # Set up distributed training
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.hardware.gpu
            mp.set_start_method('spawn')
            mp.set_sharing_strategy('file_system')
            mp.spawn(pipeline.run, nprocs=pipeline.ngpus_per_node, join=True)
        else:
            # Single GPU training
            # param_grid = {
            #     'batch_size': [8, 16, 32],
            #     'num_codebook': [512, 1024, 2048, 4096],
            #     'temp': [0.1, 0.2, 0.3],
            #     'thresh_neg': [0.1, 0.2, 0.3]
            # }
            for bs in [8]:
                for num_codebook in [1024]:
                    for temp in [0.1]:
                        for thresh_neg in [0.1]:
                            
                            print(f'codebook_{num_codebook}-batch_size_{bs}---temp---{temp}---thresh_neg_{thresh_neg}')
                        
                            cfg.num_codebook = num_codebook
                            cfg.temp = temp
                            cfg.thresh_neg = thresh_neg
                            cfg.batch_size = bs
                            
                            cfg.model.run_name = f"codebook_{num_codebook}_batch_size_{bs}_temp_{temp}_thresh_neg_{thresh_neg}"
                            pipeline.run(rank=pipeline.gpu_list[0])
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.debug("Detailed traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    #parse the name of config 
    main()
