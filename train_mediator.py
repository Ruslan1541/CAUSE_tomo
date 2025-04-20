from tqdm.auto import tqdm
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

#custom functionality 
from utils.utils import *
from loader.dataloader import dataloader
from loader.netloader import network_loader, cluster_tr_loader
from modules.segment_module import compute_modularity_based_codebook

cudnn.benchmark = False
scaler = GradScaler()

@Wrapper.EpochPrint
def train(args, net, cluster, train_loader, optimizer):
    cluster.train()
    
    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    prog_bar.set_description("[Train]")
    
    for idx, batch in prog_bar:
        img = batch["img"].to(args.device)

        # intermediate feature
        with autocast():
            feat = net(img)[:, 1:, :]

            # computing modularity based codebook
            loss_mod = compute_modularity_based_codebook(cluster.codebook, feat, temp=args.temp, k=args.degree_of_sampling, grid=args.grid)

        # optimization
        optimizer.zero_grad()
        scaler.scale(loss_mod).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # real-time print
        desc = f'[Train] Modulatiry_loss: {loss_mod / (idx + 1):.6f}'
        prog_bar.set_description(desc, refresh=True)
            
        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

def mediator_train(rank, args, ngpus_per_node):

    # setup ddp process
    if args.distributed and not torch.distributed.is_initialized(): ddp_setup(args, rank, ngpus_per_node)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # dataset loader
    train_loader, sampler = dataloader(args)

    # network loader
    net = network_loader(args, rank)
    cluster = cluster_tr_loader(args, rank)
    
    # distributed parsing
    if args.distributed: net = net.module; cluster = cluster.module

    # optimizer and scheduler
    optimizer = torch.optim.Adam(cluster.parameters(), args.learning_rate * ngpus_per_node)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    paths = pickle_path_and_exist(args)

    # early save for time
    if not paths['modular_exists']:
        rprint("No File Exists!!", rank)
        for epoch in range(args.mediator_epoch):
            
            # for shuffle
            if args.distributed: sampler.set_epoch(epoch)
            
            train(
                epoch,
                rank,
                args,
                net,
                cluster,
                train_loader,
                optimizer,
                )
                
            # scheduler step
            scheduler.step()

            # save
            if rank == 0:
                np.save(paths['modular_path'], cluster.codebook.detach().cpu().numpy()
                if args.distributed else cluster.codebook.detach().cpu().numpy())
                
            # Interrupt for sync GPU Process
            if args.distributed: dist.barrier()

    else:
        rprint("Already Exists!!", rank)
    
        # clean ddp process
    if args.distributed: ddp_clean()