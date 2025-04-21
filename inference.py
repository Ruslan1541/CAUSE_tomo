import torch
import shutil
from glob import glob
from tqdm.auto import tqdm
from torch.cuda.amp import autocast

#custom functionality 
from utils.utils import *
from loader.dataloader import dataloader
from modules.segment_module import transform, untransform
from loader.netloader import network_loader, segment_tr_loader, cluster_tr_loader

def test(args, net, segment, cluster, train_loader, cmap, nice, writer, rank):
    segment.eval()
    
    if os.path.exists(path(args, add='temp')):
        shutil.rmtree(path(args, add='temp'))

    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    count = 0
    for _, batch in prog_bar:
        ind = batch["ind"].to(args.device)
        img = batch["img"].to(args.device)
        
        with autocast():

            # intermediate feature
            feat = net(img)[:, 1:, :]
            feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            
            if 'label' in batch.keys():
                label = batch["label"].to(args.device)
                seg_feat_ema = segment.head_ema(feat)
                linear_logits = segment.linear(seg_feat_ema)
                linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
                
                interp_seg_feat = F.interpolate(transform(seg_feat_ema), img.shape[-2:], mode='bilinear', align_corners=False)

                cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), inference=True)

                _, _, mIoU = nice.eval(cluster_preds, label)
                handle_writer_logs(args, writer, mIoU, rank)
        
        seg_feat = transform(segment.head_ema(feat))
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

        # interp feat
        interp_seg_feat = F.interpolate(transform(seg_feat), img.shape[-2:], mode='bilinear', align_corners=False)

        # cluster preds
        cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), crf=True)
            
        # crf
        crf_preds = do_crf(args, img, cluster_preds).argmax(1)
        
        temp_root = save_all(args, img, ind, cluster_preds.argmax(dim=1), crf_preds, cmap, add='temp')
        
        # Process image patches
        masks_files = sorted(glob(f'{temp_root}/kmeans//*.png'), key=lambda p: parse_filename(p))
        if len(masks_files)>= args.patches_shape[0]*args.patches_shape[1]:
            file_types = ['imgs', 'kmeans', 'crfs']
            patches = {t: [] for t in file_types}
            
            for file_type in file_types:
                files = sorted(glob(f'{temp_root}/{file_type}//*.png'), 
                                    key=lambda p: parse_filename(p))
                for j in range(args.patches_shape[0] * args.patches_shape[1]):
                    img = process_image_files(files[j], args.train_resolution)
                    patches[file_type].append(img)
                    os.remove(files[j])
                
            reconstructed = {}
            for file_type, patch_list in patches.items():
                patches_array = np.array(patch_list)
                reshaped = np.reshape(patches_array, 
                                     (args.patches_shape[0], 
                                      args.patches_shape[1], 
                                      args.train_resolution, 
                                      args.train_resolution))
                step = int(args.train_resolution * (1 - args.overlap_fraction))
                reconstructed[file_type] = merge_patches(reshaped, 
                                                         patch_size=args.train_resolution, 
                                                         step=step)
            
            save_reconstructed_images(args, reconstructed, count, writer, rank)
            count += 1
        
    shutil.rmtree(temp_root)

def inference(rank, args, _):

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # dataset loader
    train_loader, _ = dataloader(args, no_ddp_train_shuffle=False)

    # network loader
    net = network_loader(args, rank)
    segment = segment_tr_loader(args, rank)
    cluster = cluster_tr_loader(args, rank)
    
    # evaluation
    nice = NiceTool(args.n_classes)

    # color map
    cmap = create_colormap()

    paths = pickle_path_and_exist(args)
    
    from datetime import datetime
    log_dir = os.path.join('logs_train',
                           datetime.today().strftime("%m-%d_%H-%M"),
                           os.path.dirname(paths['segment_path']),
                           'inference')
    check_dir(log_dir)
    writer = Writer(log_dir=log_dir)

    # early save for time
    if paths['modular_exists']:
        codebook = np.load(paths['modular_path'])
        cb = torch.from_numpy(codebook).to(args.device)
        cluster.codebook.data = cb
        cluster.codebook.requires_grad = False
        segment.head.codebook = cb
        segment.head_ema.codebook = cb

        # print successful loading modularity
        rprint(f'Modularity {paths["modular_exists"]} loaded', rank)

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return

    # param size
    print(f'# of Parameters: {num_param(segment)/10**6:.2f}(M)') 

    test(
        args,
        net,
        segment,
        cluster,
        train_loader,
        cmap, 
        nice,
        writer,
        rank
        )
    
    if writer is not None:
        writer.close()
    
