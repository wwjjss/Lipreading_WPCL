# 导入相关包
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines


# 定义超参数
def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    # 数据集设置
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    # 类别数目
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    # 模态
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio'], help='choose the modality')
    # -- directory
    # 数据路径设置
    parser.add_argument('--data-dir', default='./datasets/visual_data', help='Loaded data directory')
    # 标签路径
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt',
                        help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default='./lrw/lipread_mp4', help='Loaded data directory')
    # -- model config
    # 主干网络
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'],
                        help='Architecture used for backbone')
    # 激活函数
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu', 'prelu'], help='what relu to use')
    # mobilenet和shufflenet的宽度倍数
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    # TCN模块层数
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    # 随机失活率
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    # 是否使用深度可分离卷积
    parser.add_argument('--tcn-dwpw', default=False, action='store_true',
                        help='If True, use the depthwise seperable convolution in TCN architecture')
    # tcn宽度倍数
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    # 批量大小32
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size')
    # 优化器 Adamw = Adam + weight decate
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    # 学习率
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    # 训练轮次
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    # 是否是测试
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    # 测试使用的权重
    parser.add_argument('--model-path', type=str, default="./models/lrw_resnet18_mstcn.pth.tar",
                        help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor 特征提取
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None,
                        help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None,
                        help='Save mouth embeddings to a specificed path')
    # -- json pathname配置文件
    parser.add_argument('--config-path', type=str, default="./configs/lrw_resnet18_mstcn.json",
                        help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='KL Loss parameters')

    args = parser.parse_args()
    return args


args = load_args()

# 随机种子固定随机序列
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
# 启动 cudNN 自动调整器，对cudNN中计算卷积的多种不同方法进行基准测试，以获得最佳的性能指标。
torch.backends.cudnn.benchmark = True


# 提取特征 裁剪、灰度化，标准化
def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


# 验证
# eq()判断相等，view_as()将调用函数的变量，转变为同参数tensor同样的形状，.sum()累加,.item()将只有一个元素的tensor转换为标量
def evaluate(model, dset_loader, criterion):
    model.eval()

    running_loss = 0.
    running_corrects_main = 0.
    running_corrects_branch = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            # logits [B,500]
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            logits_fusion = (logits[0] + logits[1]) / 2

            _, preds_main = torch.max(F.softmax(logits[0], dim=1).data, dim=1)
            _, preds_branch = torch.max(F.softmax(logits[1], dim=1).data, dim=1)
            _, predicted_fusion = torch.max(F.softmax(logits_fusion, dim=1).data, dim=1)
            # preds = (preds_main + preds_branch) / 2

            running_corrects_main += preds_main.eq(labels.cuda().view_as(preds_main)).sum().item()
            running_corrects_branch += preds_branch.eq(labels.cuda().view_as(preds_branch)).sum().item()
            running_corrects += predicted_fusion.eq(labels.cuda().view_as(predicted_fusion)).sum().item()

            loss_main = criterion(logits[0], labels.cuda()) + args.beta * F.kl_div(
                F.log_softmax(logits[1], dim=-1), F.softmax(logits[0], dim=-1), reduction='batchmean')
            loss_branch = criterion(logits[1], labels.cuda()) + args.beta * F.kl_div(F.log_softmax(logits[0], dim=-1),
                                                                                     F.softmax(logits[1], dim=-1),
                                                                                     reduction='batchmean')
            loss = (loss_main + loss_branch) / 2
            running_loss += loss.item() * input.size(0)

    print('{} in total\tCR: {}'.format(len(dset_loader.dataset), running_corrects / len(dset_loader.dataset)))
    return running_corrects_main / len(dset_loader.dataset), running_corrects_branch / len(
        dset_loader.dataset), running_corrects / len(dset_loader.dataset), running_loss / len(dset_loader.dataset)


# 训练
def train(model, dset_loader, criterion, epoch, optimizer, logger):
    # 时间信息
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # 学习率
    lr = showLR(optimizer)

    # 日志信息
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current learning rate: {}'.format(lr))

    # 模型训练
    model.train()
    running_loss = 0.
    running_corrects_main = 0.
    running_corrects_branch = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, (input, lengths, labels) in enumerate(dset_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # mixup混合数据增强
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()  # 梯度清0

        logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
        # print("logits:", logits)
        # KL损失

        # mixup混合数据增强
        loss_func = mixup_criterion(labels_a, labels_b, lam)
        # loss = loss_func(criterion, logits)
        # 总
        loss_main = loss_func(criterion, logits[0]) + args.beta * F.kl_div(
            F.log_softmax(logits[1], dim=-1), F.softmax(logits[0], dim=-1), reduction='batchmean')
        # 分支
        loss_branch = loss_func(criterion, logits[1]) + args.beta * F.kl_div(F.log_softmax(logits[0], dim=-1),
                                                                             F.softmax(logits[1], dim=-1),
                                                                             reduction='batchmean')
        loss_total = loss_main + loss_branch
        # print("loss_main,loss_branch:", loss_main, loss_branch)
        # print("loss_total:", loss_total)

        # loss.backward()
        loss_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        logits_fusion = (logits[0] + logits[1]) / 2
        _, predicted_main = torch.max(F.softmax(logits[0], dim=1).data, dim=1)
        _, predicted_branch = torch.max(F.softmax(logits[1], dim=1).data, dim=1)
        _, predicted_fusion = torch.max(F.softmax(logits_fusion, dim=1).data, dim=1)
        # predicted = (predicted_main + predicted_branch) / 2

        # 损失
        running_loss += loss_total.item() * input.size(0)
        # 准确率
        # 总
        running_corrects_main += lam * predicted_main.eq(labels_a.view_as(predicted_main)).sum().item() + (
                1 - lam) * predicted_main.eq(
            labels_b.view_as(predicted_main)).sum().item()
        # 分支
        running_corrects_branch += lam * predicted_branch.eq(labels_a.view_as(predicted_branch)).sum().item() + (
                1 - lam) * predicted_branch.eq(
            labels_b.view_as(predicted_branch)).sum().item()
        # 合并
        running_corrects += lam * predicted_fusion.eq(labels_a.view_as(predicted_fusion)).sum().item() + (
                1 - lam) * predicted_fusion.eq(
            labels_b.view_as(predicted_fusion)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader) - 1):
            update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects_main,
                                running_corrects_branch,
                                running_corrects, running_all,
                                batch_time, data_time)

    return model


# 获取模型参数
def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)
    args_loaded = load_json(args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = {'num_layers': args_loaded['tcn_num_layers'],
                   'kernel_size': args_loaded['tcn_kernel_size'],
                   'dropout': args_loaded['tcn_dropout'],
                   'dwpw': args_loaded['tcn_dwpw'],
                   'width_mult': args_loaded['tcn_width_mult'],
                   }
    # 调用模型
    model = Lipreading(modality=args.modality,
                       num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()
    calculateNorm2(model)
    return model


def main():
    # -- logging
    save_path = get_save_folder(args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    model = get_model_from_json()  # 获取模型

    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)  # 加载数据集
    # -- get loss function
    criterion = nn.CrossEntropyLoss()  # 定义损失函数(使用交叉熵损失函数)
    # -- get optimizer Adamw = Adam + weight decate
    optimizer = get_optimizer(args, optim_policies=model.parameters())  # 定义优化器(使用Adamw优化器)
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)  # 动态调整学习率(CosineScheduler)

    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint 继续上次训练
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info('Model and states have been successfully loaded from {}'.format(args.model_path))
        # init from trained model 从头训练
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info('Model has been successfully loaded from {}'.format(args.model_path))
        # feature extraction
        if args.mouth_patch_path:
            save2npz(args.mouth_embedding_out_path, data=extract_feats(model).cpu().detach().numpy())
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        # 是否为测试模式
        if args.test:
            acc_main_avg_test, acc_branch_avg_test, acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'],
                                                                                           criterion)
            logger.info('Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}'.format('test', loss_avg_test,
                                                                                                 acc_avg_test))
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch - 1)

    epoch = args.init_epoch

    # 每一轮训练打印信息
    while epoch < args.epochs:
        # 模型训练
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        # 模型验证
        acc_main_avg_val, acc_branch_avg_val, acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'],
                                                                                   criterion)
        # 打印信息
        logger.info(
            '{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc_main val:{:.4f},\tAcc_branch val:{:.4f},\tAcc val:{:.4f}, LR: {}'.format(
                'val', epoch, loss_avg_val, acc_main_avg_val, acc_branch_avg_val, acc_avg_val,
                showLR(optimizer)))
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    # 保存最好结果
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_main_avg_test, acc_branch_avg_test, acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'],
                                                                                   criterion)
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))


if __name__ == '__main__':
    main()
