from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import sys
# sys.path.extend(['/home/track/pycharm_project_315/MCMOT/src/home/track/pycharm_project_315/MCMOT/src'])
import _init_paths

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

my_visible_devs = '0'  # '0, 3'  # 设置可运行GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task, opt.multi_scale)  # if opt.task==mot -> JointDataset
    Test_dataset=get_dataset('jde','mot',False)
    f = open(opt.data_cfg)  # choose which dataset to train '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    trainset_paths = data_config['train']  # 训练集路径
    testset_paths=data_config['test']
    dataset_root = data_config['root']  # 数据集所在目录
    print("Dataset root: %s" % dataset_root)
    f.close()

    # Image data transformations
    transforms = T.Compose([T.ToTensor()])

    # Dataset
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=trainset_paths,
                      img_size=opt.input_wh,
                      augment=True,
                      transforms=transforms)

    test_dataset=Dataset(opt=opt,
                      root=dataset_root,
                      paths=testset_paths,
                      img_size=opt.input_wh,
                      augment=False,
                      transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)

    logger = Logger(opt)

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 多GPU训练
    # print("opt.gpus_str: ", opt.gpus_str)

    # opt.device = torch.device('cuda:0' if opt.gpus[0] >= 0 else 'cpu')  # 设置GPU

    opt.device = device
    opt.gpus = my_visible_devs

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    if opt.is_debug:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
    else:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=False,
                                            num_workers=opt.num_workers,#opt.num_workers,5
                                            pin_memory=True,
                                            drop_last=False
                                            )
    print('Starting training...')

    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    trainer.set_device(opt.gpus, opt.chunk_sizes, device)
    min_loss=1000
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
        log_dict_test, _ = trainer.val(epoch,test_loader)


        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        for k, v in log_dict_test.items():
            logger.scalar_summary('test_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        trainer.logger.write('test',log_dict_test,epoch,1,0,flag=False)
        trainer.logger.write('train', log_dict_train, epoch, 1, 0, flag=False)
        if log_dict_test['loss']<min_loss:
            min_loss=log_dict_test['loss']
            save_model(os.path.join(opt.save_dir, f'model_'+ opt.arch +f'_best_{log_dict_test}.pth'),
                       epoch, model, optimizer)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:  # mcmot_last_track or mcmot_last_det
            if opt.id_weight > 0:  # do tracking(detection and re-id)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_track_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
            else:  # only do detection
                # save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                #        epoch, model, optimizer)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 10 == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '0, 1'
    opt = opts().parse()
    # print("opt.gpus: ", opt.gpus)
    run(opt)
