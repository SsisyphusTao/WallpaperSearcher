from resnet18 import get_model
from data import get_dataloader
import torch
import torch.optim as optim

import time

def train():
    torch.backends.cudnn.benchmark = True
    model = get_model().cuda()
    model.train()
    compute_loss = torch.nn.SmoothL1Loss(reduction='none').cuda()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                weight_decay=5e-4)
    totalrounds = 20
    adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, totalrounds, 0)
    loader = get_dataloader(64, 8)
    length = len(loader)
    # create batch iterator
    for e in range(totalrounds):
        loss_amount = 0
        for iteration, (images, targets, masks) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda().squeeze(-2)
            masks = masks.cuda().unsqueeze(-1)
            # forward
            optimizer.zero_grad()
            loss = (compute_loss(model(images.permute(0, 3, 1, 2)/255.)[0].sigmoid(), targets.sigmoid()) * masks).sum() / masks.sum()
            loss_amount = (loss_amount * iteration + loss.item()) / (iteration + 1)
            if iteration % 10 == 0:
                s = ('[%s], %s' + '  Loss:%.4f, iter:%04d/%04d') % (
                        time.asctime(time.localtime(time.time())), 'Epoch:[%g/%g]' % (e, totalrounds), loss_amount, iteration, length)
                print(s)
            loss.backward()

            optimizer.step()
            adjust_learning_rate.step()
        torch.save(model.state_dict(), 'checkpoints/ebd_%03d_%g.pth'%(e, loss_amount))
        

if __name__ == '__main__':
    train()