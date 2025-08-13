from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

#custom functionality 
from utils.utils import *
from loader.dataloader import dataloader
from modules.segment_module import stochastic_sampling, ema_init, ema_update
from loader.netloader import network_loader, segment_tr_loader, cluster_tr_loader

cudnn.benchmark = False
scaler = GradScaler()

# set_seeds()

# tensorboard
counter = 0

@Wrapper.EpochPrint
def train(args, net, segment, cluster, train_loader, optimizer_segment):
    segment.train()

    total_loss = 0

    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, batch in prog_bar:

        # optimizer
        with autocast():

            img = batch["img"].to(args.device)
            
            # intermediate features
            feat = net(img)[:, 1:, :]

            # teacher
            seg_feat_ema = segment.head_ema(feat, drop=segment.dropout)
            proj_feat_ema = segment.projection_head_ema(seg_feat_ema)

            # student
            seg_feat = segment.head(feat, drop=segment.dropout)
            proj_feat = segment.projection_head(seg_feat)

            # grid
            if args.grid:
                feat, order = stochastic_sampling(feat, k=args.degree_of_sampling)
                proj_feat, _ = stochastic_sampling(proj_feat, order=order, k=args.degree_of_sampling)
                proj_feat_ema, _ = stochastic_sampling(proj_feat_ema, order=order, k=args.degree_of_sampling)

            # bank compute and contrastive loss
            cluster.bank_compute()
            loss = cluster.contrastive_ema_with_codebook_bank(feat, proj_feat, proj_feat_ema)

        # optimizer
        optimizer_segment.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_segment)
        torch.nn.utils.clip_grad_norm_(segment.parameters(), 1)
        scaler.step(optimizer_segment)
        scaler.update()

        # ema update
        ema_update(segment.head, segment.head_ema)
        ema_update(segment.projection_head, segment.projection_head_ema)

        # bank update
        cluster.bank_update(feat, proj_feat_ema)
        
        # loss check
        total_loss += loss.item()

        # real-time print
        desc = f'[Train] Loss: {total_loss / (idx + 1):.2f}'
        prog_bar.set_description(desc, refresh=True)
        
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

def front_train(rank, args, ngpus_per_node):
    
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
    
    # Bank and EMA
    cluster.bank_init()
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)

    paths = pickle_path_and_exist(args)
    
    if paths['modular_exists']:
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

    optimizer_segment = torch.optim.Adam(segment.parameters(), args.learning_rate* ngpus_per_node, weight_decay=1e-4)

    for epoch in range(args.front_epoch):
        
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
            )
        
        if (rank == 0):
            torch.save(segment.state_dict(), paths['segment_path'])
            print(f'----------------- Epoch {epoch}: SAVING CHECKPOINT IN {paths["segment_path"]}-----------------')
            
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()
        
        # Closing DDP
    if args.distributed: dist.barrier(); ddp_clean()