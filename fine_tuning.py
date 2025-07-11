from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

#custom functionality 
from utils.utils import *
from loader.dataloader import dataloader
from modules.segment_module import compute_modularity_based_codebook
from loader.netloader import network_loader, segment_tr_loader, cluster_tr_loader

cudnn.benchmark = False
scaler = GradScaler()

set_seeds()

@Wrapper.EpochPrint
def train(args, net, segment, cluster, train_loader, optimizer_segment, optimizer_cluster):
    segment.train()
    cluster.train()

    total_loss = 0

    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, batch in prog_bar:

        # optimizer
        with autocast():

            # image and label and self supervised feature
            img = batch["img"].to(args.device)

            # intermediate features
            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat, segment.dropout)

            # computing modularity based codebook
            loss = compute_modularity_based_codebook(cluster.cluster_probe, seg_feat_ema, temp=args.temp, k=args.degree_of_sampling, grid=args.grid)

        optimizer_segment.zero_grad()
        optimizer_cluster.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_segment)
        torch.nn.utils.clip_grad_norm_(segment.parameters(), 1)
        optimizer_segment.step()
        scaler.step(optimizer_cluster)
        scaler.update()

        # loss check
        total_loss += loss.item()

        # real-time print
        desc = f'[Train] Loss: {total_loss / (idx + 1):.2f}'
        prog_bar.set_description(desc, refresh=True)
        
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

def fine_tune(rank, args, ngpus_per_node):
    
    # setup ddp process
    if args.distributed and not torch.distributed.is_initialized(): ddp_setup(args, rank, ngpus_per_node)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # dataset loader
    train_loader, sampler = dataloader(args)

    # network loader
    net = network_loader(args, rank)
    segment = segment_tr_loader(args, rank)
    cluster = cluster_tr_loader(args, rank)
    
    # distributed parsing
    if args.distributed: net = net.module; segment = segment.module; cluster = cluster.module

    # optimizer
    optimizer_segment = torch.optim.Adam(segment.parameters(), args.learning_rate * ngpus_per_node, weight_decay=1e-4)
    optimizer_cluster = torch.optim.Adam(cluster.parameters(), args.learning_rate * ngpus_per_node)
    
    # scheduler
    scheduler_segment = torch.optim.lr_scheduler.StepLR(optimizer_segment, step_size=2, gamma=0.5)
    scheduler_cluster = torch.optim.lr_scheduler.StepLR(optimizer_cluster, step_size=2, gamma=0.5)
    
    paths = pickle_path_and_exist(args)

    # early save for time
    if paths['modular_path']:
        codebook = np.load(paths['modular_path'])
        cluster.codebook.data = torch.from_numpy(codebook).to(args.device)
        cluster.codebook.requires_grad = False
        segment.head.codebook = torch.from_numpy(codebook).to(args.device)
        segment.head_ema.codebook = torch.from_numpy(codebook).to(args.device)

        # print successful loading modularity
        rprint(f'Modularity {paths["modular_path"]} loaded', rank)
        
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return


    for epoch in range(args.fine_tune_epoch):
        
        # for shuffle
        if args.distributed: sampler.set_epoch(epoch)

        train(
            epoch,
            rank,
            args,
            net,
            segment,
            cluster,
            train_loader,
            optimizer_segment,
            optimizer_cluster,
            )

        scheduler_segment.step()
        scheduler_cluster.step()
        if (rank == 0):
            
            torch.save(segment.state_dict(), paths['segment_path'])
            torch.save(cluster.state_dict(), paths['cluster_path'])
            
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()
        
    # Closing DDP
    if args.distributed: dist.barrier(); ddp_clean()
