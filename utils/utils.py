import os
import cv2
import torch
import random
import numpy as np
import torchvision
from pathlib import Path
import torch.distributed as dist
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_clean():
    dist.destroy_process_group()
    
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def get_norm_transforms(args):
    
    std = torch.load(Path(args.std_mean_path) / "std.pt")
    mean = torch.load(Path(args.std_mean_path) / "mean.pt")
    
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / std[0], 1 / std[1], 1 / std[2]]),
                                   transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]],
                                                        std=[1., 1., 1.]),
                                  ])
    Trans = transforms.Normalize(mean=[mean[0], mean[1], mean[2]], std=[std[0], std[1], std[2]])
    
    return Trans, invTrans

class ToTargetTensor:
    """
    Transform for converting targets to tensor format.
    """
    def __call__(self, target: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

class Writer(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
                
class Wrapper(object):
    @staticmethod
    def EpochPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------TRAIN EPOCH: {epoch+1}-------------', rank)
            func(*args, **kwards)
        return wrapper

def pickle_path_and_exist(args):
    
    baseline = args.model.ckpt.split('/')[-1].split('.')[0]

    modular_dir = os.path.join("CAUSE", args.dataset.dst, "modularity", baseline, args.model.run_name)
    check_dir(modular_dir) 
    modular_path = os.path.join(modular_dir, "modular.npy")
    
    weights_dir = os.path.join("CAUSE", args.dataset.dst, baseline, args.model.run_name)
    check_dir(weights_dir)
    segment_path = os.path.join(weights_dir, "segment_tr.pth")
    cluster_path = os.path.join(weights_dir, "cluster_tr.pth")
    
    return {
        "modular_path":  modular_path,
        "modular_exists": os.path.exists(modular_path),
        "segment_path":  segment_path,
        "cluster_path":  cluster_path
    }

def freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

def no_freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = True

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
def path(args, add=None):
    baseline = args.model.ckpt.split('/')[-1].split('.')[0]
    y = f'{args.num_codebook}'
    new_path = f'{args.dataset.data_dir}/results/{args.dataset.n_classes}_classes'
    check_dir(new_path)
    
    root = os.path.join(new_path, args.dataset.dst, baseline, y, args.model.run_name, add)

    check_dir(f'{root}/imgs')
    check_dir(f'{root}/kmeans')
    check_dir(f'{root}/crfs')
    
    return root
        
def save_all(args, img, ind, cluster_preds, crf_preds, cmap, add=None):
    root = path(args, add)
    
    _, invTrans = get_norm_transforms(args)

    # save image
    for id, i in [(id, x.item()) for id, x in enumerate(list(ind))]:
        torchvision.utils.save_image(invTrans(img)[id].cpu(), f'{root}/imgs/imgs_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[cluster_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/kmeans/kmeans_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[crf_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/crfs/crfs_{i}.png')
    return root 

def rprint(msg, rank=0):
    if rank==0: print(msg)

def num_param(f):
    out = 0
    for param in f.head.parameters():
        out += param.numel()
    return out

def merge_patches(args, patches, patch_size, step):
    N, M = patches.shape[0], patches.shape[1]
    
    out_h = (N - 1) * step + patch_size
    out_w = (M - 1) * step + patch_size
    
    classes = np.unique(patches)
    if classes.shape[0] != args.dataset.n_classes:
        full_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        for i in range(N):
            for j in range(M):
                y0, x0 = i * step, j * step
                full_mask[y0:y0+patch_size, x0:x0+patch_size] = patches[i, j]
        return full_mask
    else:  
        full_mask = np.zeros((out_h, out_w, classes.shape[0]), dtype=np.uint8)
        for i in range(N):
            for j in range(M):
                patch = patches[i, j]
                y0 = i * step
                x0 = j * step
                for c in range(classes.shape[0]):
                    full_mask[y0 : y0 + patch_size, x0 : x0 + patch_size, c] += (patch == classes[c])
                    
        merged = full_mask.argmax(axis=2) 

        return (merged*(255//(classes.shape[0]-1))).astype(np.uint8)


def handle_writer_logs(args, 
                       writer: Writer, 
                       mIoU: torch.tensor,
                       rank: int) -> None:
    """
    Handle tensorboard logging for training metrics and hyperparameters.

    Args:
        args: Configuration object containing training parameters
        writer (Writer): Tensorboard writer instance for logging
        mIoU (torch.tensor): Mean Intersection over Union metric
        rank (int): Process rank in distributed training setup
    """
    if args.distributed and rank == 0:
        check = args.model.ckpt.split('/')[-1].split('.')[0]
    elif not args.distributed:
        check = int(args.model.ckpt.split('/')[-1].split('.')[0].split('_')[-1])
        
    print('smt1efe')
    writer.add_hparams({
        'check': check,
        'classes': args.dataset.n_classes,
        'resoltuions': args.train_resolution
    }, {"train_mIoU": mIoU})
    

def process_image_files(file_path: str, 
                       train_resolution: int) -> np.ndarray:
    """
    Process and resize image files to the specified training resolution.

    Args:
        file_path (str): Path to the image file
        train_resolution (int): Target resolution for the image
    """
    if cv2.imread(file_path, 0).shape[0] == train_resolution:
        img = cv2.imread(file_path, 0)
    else:
        img = cv2.resize(cv2.imread(file_path, 1), 
                        (train_resolution, train_resolution))
    return img

def save_reconstructed_images(args, 
                            reconstructed_images: dict, 
                            count: int, 
                            writer: Writer, 
                            rank: int) -> None:
    """
    Save reconstructed images to disk and log them to tensorboard.

    Args:
        args: Configuration object containing path information
        reconstructed_images (dict): Dictionary containing reconstructed images
            with keys as image types and values as image data
        count (int): Current iteration count
        writer (Writer): Tensorboard writer instance
        rank (int): Process rank in distributed training setup
    """
    root = path(args, add='full')
    
    for img_type, img_data in reconstructed_images.items():
        cv2.imwrite(f'{root}/{img_type}/{img_type}_{count}.jpeg', img_data)
    
    if (args.distributed and rank == 0) or (not args.distributed):
        for img_name, img_data in [('init image', reconstructed_images['imgs']),
                                   ('prediction', reconstructed_images['kmeans'])]:
            rgb_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            grid = torchvision.utils.make_grid(
                [torch.from_numpy(rgb_img).permute(2, 0, 1)])
            writer.add_image(f'Inference ({img_name})', grid)

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    stem, _ = os.path.splitext(filename)
    
    prefix, num_str = stem.split('_', maxsplit=1)
    
    number = int(num_str)
    return prefix, number

def ckpt_to_name(ckpt):
    name = ckpt.split('/')[-1].split('_')[0]
    return name

def ckpt_to_arch(ckpt):
    arch = ckpt.split('/')[-1].split('.')[0]
    return arch

def print_argparse(args, rank=0):
    dict = vars(args)
    print(dict.keys())
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys(): print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')

from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

class NiceTool(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.histogram = torch.zeros((self.n_classes, self.n_classes)).cuda()

    def scores(self, label_trues, label_preds):
        mask = (label_trues >= 0) & (label_trues < self.n_classes) & (label_preds >= 0) & (label_preds < self.n_classes)  # Exclude unlabelled data.
        hist = torch.bincount(self.n_classes * label_trues[mask] + label_preds[mask], \
                              minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes).t().cuda()
        return hist

    def eval(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        self.histogram += self.scores(label, pred)

        self.assignments = linear_sum_assignment(self.histogram.cpu(), maximize=True)
        hist = self.histogram[np.argsort(self.assignments[1]), :]

        tp = torch.diag(hist)
        fp = torch.sum(hist, dim=0) - tp
        fn = torch.sum(hist, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(hist)

        # metric
        metric_dict = OrderedDict({"mIoU": iou[~torch.isnan(iou)].mean().item() * 100,
                       # "Precision per Class (%)": prc * 100,
                       "mAP": prc[~torch.isnan(prc)].mean().item() * 100,
                       "Acc": opc.item() * 100})


        self.metric_dict_by_class = OrderedDict({"mIoU": iou * 100,
                       # "Precision per Class (%)": prc * 100,
                       "mAP": prc * 100,
                       "Acc": (torch.diag(hist) / hist.sum(dim=1)) * 100})


        # generate desc
        sentence = ''
        for key, value in metric_dict.items():
            if type(value) == torch.Tensor: continue
            sentence += f'[{key}]: {value:.1f}, '
        return metric_dict, sentence, iou[~torch.isnan(iou)].mean().item() * 100

    def reset(self):
        self.histogram = torch.zeros((self.n_classes, self.n_classes)).cuda()

    def do_hungarian(self, clusters):
        return torch.tensor(self.assignments[1])[clusters.cpu()]

import torch
import torch.nn.functional as F
import pydensecrf.utils as utils
import pydensecrf.densecrf as dcrf
import torchvision.transforms.functional as VF

def dense_crf(args, image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor, max_iter: int):
    MAX_ITER = max_iter
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3
    
    _, invTrans = get_norm_transforms(args)

    image = np.array(VF.to_pil_image(invTrans(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def _apply_crf(args, tup, max_iter):
    return dense_crf(args, tup[0], tup[1], max_iter=max_iter)

def do_crf(args, img_tensor, prob_tensor, max_iter=10):
    # from functools import partial
    # outputs =  pool.map(partial(_apply_crf, args, max_iter=max_iter), zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    outputs = [_apply_crf(args, (img_tensor.detach().cpu()[i], prob_tensor.detach().cpu()[i]), max_iter=max_iter) for i in range(len(img_tensor.detach().cpu()))]
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

def create_colormap():
    def bit_get(val, idx):
        return (val >> idx) & 1
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap / 255
