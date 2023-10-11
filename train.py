import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.my_net import LeFeD_Net
from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/userdisk1/qjzeng/semi_seg/LeFeD/code/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='LA', help='dataset_name')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu') 
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='learning rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labelnum', type=int,  default=16, help='number of labeled data')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='6', help='GPU to use')
parser.add_argument('--temperature', type=float, default=0.05, help='temperature of sharpening')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + \
    "_{}labels/".format(args.labelnum)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen
    

@torch.no_grad()
def test(net, val_loader, maxdice=0, stride_xy=16, stride_z=4):
    metrics = test_calculate_metric(net, val_loader.dataset, stride_xy=stride_xy, stride_z=stride_z)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag    



if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    testset = Pancreas(train_data_path, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    def create_model(ema=False):
        # Network definition
        net = LeFeD_Net(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    #ce_loss = BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = MSELoss()


    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    maxdice1 = 0.
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch = label_batch == 1
            
        
            for num in range(3):
                if num == 0:
                    outputs1, outputs2, masks, stage_out1, _ = model(volume_batch, [])
                else:
                    outputs1, outputs2, masks, stage_out1, _ = model(volume_batch, en)

                consistency_weight = get_current_consistency_weight(iter_num//150)
                
                en = []
                for idx in range(len(masks[0])):
                    mask1 = masks[0][idx].detach()
                    mask2 = masks[1][idx].detach()
                    #mask1 = sharpening(mask1)
                    #mask2 = sharpening(mask2)
                    en.append(1e-3*(mask1-mask2))    # 1e-3
                    
                out5, out4, out3, out2, out1 = stage_out1[0], stage_out1[1], stage_out1[2], stage_out1[3], stage_out1[4]
                out1_soft = F.softmax(out1, dim=1)
                out2_soft = F.softmax(out2, dim=1)
                out3_soft = F.softmax(out3, dim=1)
                out4_soft = F.softmax(out4, dim=1)
                out5_soft = F.softmax(out5, dim=1)
    
                outputs_soft1 = F.softmax(outputs1, dim=1)
                outputs_soft2 = F.softmax(outputs2, dim=1)
                
                # calculate the loss
                # supervised loss
                loss_sup1 = losses.dice_loss(outputs_soft1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                loss_sup2 = losses.dice_loss(outputs_soft2[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                #loss_sup2 = torch.mean(ce_loss(outputs2[:labeled_bs], label_batch[:labeled_bs]))
                #loss_sup2 = -torch.sum(label_batch[:labeled_bs]*(outputs_soft2[:labeled_bs, 1, :, :, :]+1e-6).log()) / (torch.sum(label_batch[:labeled_bs])+ 1e-6) \
                           #-torch.sum((1.-label_batch[:labeled_bs])*(1.-outputs_soft2[:labeled_bs, 1, :, :, :]+1e-6).log()) / (torch.sum(1.-label_batch[:labeled_bs]) + 1e-6)
                loss_sup = loss_sup1 + loss_sup2
                
                # deep supervision
                los1 = losses.dice_loss(out1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los2 = losses.dice_loss(out2_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los3 = losses.dice_loss(out3_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los4 = losses.dice_loss(out4_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los5 = losses.dice_loss(out5_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los = 0.8*los1 + 0.6*los2 + 0.4*los3 + 0.2*los4 + 0.1*los5
                
                loss_ds = los
                
                # MSE loss
                loss_cons = losses.mse_loss(outputs_soft1, outputs_soft2)
                
                # total loss
                loss = loss_sup + loss_ds + loss_cons 
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

            iter_num = iter_num + 1

            logging.info(
                'iteration %d : loss : %f, loss_dice: %f, loss_ds: %f, loss_cons: %f' %
                (iter_num, loss.item(), loss_sup.item(), loss_ds.item(), loss_cons.item()))
            
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


            if iter_num % 200 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    
    
    
    
    
    
