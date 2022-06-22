from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from dataset.MultiNormal.multi_normal_generate import *
from models.GANModels import *
from utils.functions import train, LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.visualizationMetrics import visualization

import torch
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
from torchinfo import summary

# synthesis data
import warnings
import wandb

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True



def main():
    args = cfg.parse_args()

    # set seed for experiment, to reproduce our results
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False # the accelerator for CNN computation
        torch.backends.cudnn.deterministic = True

    # gpu setting
    if args.gpu is not None:
        main_worker(args.gpu, args)
        
def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    #-------------------------------------------------------------------#
    #------------------------   dataset loading ------------------------#
    # -------------------------------------------------------------------#
    # [batch_size, channles, seq-len]
    train_set = MultiNormaldataset(size=10000,  mode='train',
                                       channels = args.simu_channels,  simu_dim=args.simu_dim,
                                       transform=args.transform, truncate=args.truncate)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_set = MultiNormaldataset(size=args.eval_num, mode='test',
                                      channels = args.simu_channels, transform=args.transform,
                                      truncate=args.truncate, simu_dim=args.simu_dim)
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print('------------------------------------------------------------')
    print('How many iterations in a training epoch: ', len(train_loader))
    print('------------------------------------------------------------')
    assert len(train_set) >= args.eval_num, 'The number of evaluation should be less than test_set'

    #-------------------------------------------------------------------#
    #------------------------   Model & optimizer init ------------------#
    # -------------------------------------------------------------------#
    channels, seq_len = train_set[1].shape
    gen_net = Generator(seq_len=seq_len, channels=channels,
                        num_heads= args.heads, latent_dim=args.latent_dim,
                        depth=args.g_depth, patch_size=args.patch_size)
    dis_net = Discriminator(seq_len=seq_len, channels=channels,
                            num_heads=args.heads, depth=args.d_depth,
                            patch_size=args.patch_size)
    # Distributed computing is not used in our setting
    # If you would like to use Gaussian GANs on larger datasets
    # or larger model, please contact me: qilong.pan@kaust.edu.sa,
    # we would consider to add the module of distributed computing.
    torch.cuda.set_device(args.gpu)
    gen_net.cuda(args.gpu)
    dis_net.cuda(args.gpu)
    # model size for generator
    summary(gen_net, (args.batch_size, args.latent_dim))

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2), weight_decay=args.wd)
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2), weight_decay=args.wd)
    else:
        # TO DO: add other optimizer
        raise NotImplementedError
    # decay for learning rate
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_gen)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_dis)

    # global setting for later training process
    start_epoch = 0
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_num, args.latent_dim)))
    dis_best, p_dis_best, cor_dis_best, moment_dis_best = 99, 99, 99, 99

    #-------------------------------------------------------------------#
    #------------------------ Writer Configuration  --------------------#
    # -------------------------------------------------------------------#

    # set writer
    assert args.exp_name
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    writer = SummaryWriter(args.path_helper['log_path'])
    logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch,
    }
    print('------------------------------------------------------------')
    print(f"Log file path: {args.path_helper['prefix']}")
    print('------------------------------------------------------------')

    # wandb ai monitoring
    # project_name = 'loss: ' + args.loss + ', n_gen: ' + str(args.n_gen) + ', n_dis: '+ str(args.n_dis)
    project_name = 'n_gen: ' + str(args.n_gen) + ', n_dis: ' + str(args.n_dis) + ', ' + \
                       str(args.simu_channels) + '*' + str(args.simu_dim)
    wandb.init(project= args.exp_name, entity="qilong77", name = project_name)
    wandb.config = {
        "epochs": int(args.epochs) - int(start_epoch),
        "batch_size": args.batch_size
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.epochs)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
              train_loader, epoch, writer_dict, lr_schedulers)

        # Monitor for traing process
        # save the generated time series after using PCA and t-SNE
        if (epoch) % args.eval_epochs == 0:
            with torch.no_grad():
                # sample_imgs is on GPU device
                sample_imgs = gen_net(fixed_z).detach().to('cpu').numpy()

            # visualization for generated data & wandb for generated data
            visualization(ori_data=train_set[:args.eval_num],
                          generated_data=sample_imgs[:args.eval_num], analysis='pca',
                          save_name=args.exp_name, epoch=epoch, args=args)
            visu_pca = plt.imread(
                os.path.join(args.path_helper['log_path_img_pca'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
            img_visu_pca = wandb.Image(visu_pca, caption="Epoch: " + str(epoch))
            wandb.log({'PCA Visualization': img_visu_pca})

            # add the GRF plot
            visu_rline = plt.imread(
                os.path.join(args.path_helper['log_path_img_pca'], f'{args.exp_name}_epoch_{epoch + 1}_rline.png'))
            img_visu_rline = wandb.Image(visu_rline, caption="Epoch: " + str(epoch))
            wandb.log({'Real sampless': img_visu_rline})
            visu_gline = plt.imread(
                os.path.join(args.path_helper['log_path_img_pca'], f'{args.exp_name}_epoch_{epoch + 1}_gline.png'))
            img_visu_gline = wandb.Image(visu_gline, caption="Epoch: " + str(epoch))
            wandb.log({'Generated sampless': img_visu_gline})
            # x = np.arange(1, args.simu_dim+1)
            # data_g = [[x, y] for y in sample_imgs[:args.eval_num//50, 0, :]]
            # table_g = wandb.Table(data=data_g, columns=["x", "y"])
            # data_real = [[x, y] for y in test_set[:args.eval_num //50, 0, :]]
            # table_real = wandb.Table(data=data_real, columns=["x", "y"])
            # wandb.log({'Generated Samples': wandb.plot.line(table_g, "x", "y", title = 'Generated Samples')})
            # wandb.log({'Real Samples': wandb.plot.line(table_real, "x", "y", title = 'Real Samples')})
            # sigma estimated
            # sample_imgs [batch_size, channels = 1, length]
            sigma_true = anal_solution(train_set[:args.eval_num].squeeze(1), train_set.mean)
            sigma_gen = anal_solution(sample_imgs.squeeze(1), train_set.mean)
            wandb.log({'Generated sigma': sigma_gen,
                       'True sigma': sigma_true})

            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'path_helper': args.path_helper,
            }, args.path_helper['ckpt_path'], filename="checkpoint_best_dis")

    print('===============================================')
    print('Training Finished & Model Saved, the path is: ', args.path_helper['ckpt_path']+ '/'+ 'checkpoint')

if __name__ == '__main__':
    main()
