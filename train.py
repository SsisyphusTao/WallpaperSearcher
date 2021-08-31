from torch.optim import optimizer
from resnet18 import get_model
from data import get_dataloader
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time

def train(local_rank):

    model = get_model().cuda()
    model.train()
    loss_fn = torch.nn.SmoothL1Loss().cuda()
    if not local_rank == -1:
        model =DDP(model, [local_rank])
    compute_loss = lambda x, y : loss_fn(model(x)[0], y)

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9,
                weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_epoch = 10
    adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, 0)
    loader = get_dataloader(64, local_rank)
    length = len(loader)
    # create batch iterator
    for e in range(1, total_epoch+1):
        loss_amount = 0
        for iteration, (images, targets) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda().permute(0,2,1)
            # forward
            optimizer.zero_grad()
            loss = compute_loss(images.permute(0, 3, 1, 2).contiguous()/255. , targets) * torch.distributed.get_world_size()
            loss_amount = (loss_amount * iteration + loss.item()) / (iteration + 1)
            if iteration % 10 == 0 and local_rank==0:
                s = ('[%s], %s' + '  Loss:%.4f, iter:%04d/%04d') % (
                        time.asctime(time.localtime(time.time())), 'Epoch:[%g/%g]' % (e, total_epoch), loss_amount, iteration, length)
                print(s)
            loss.backward()

            optimizer.step()
            adjust_learning_rate.step()
        if local_rank == 0:
            torch.save(model.module.state_dict(), 'checkpoints/v3_%depochs_%03d_%g.pth'%(total_epoch, e, loss_amount))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    train(args.local_rank)