# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Modified for Visual Query Task
# --------------------------------------------------------

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_vis_query_dataset
from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.vis_query_model
from models.labram import labram_base_patch200_200  # 直接导入我们的模型

def get_args():
    parser = argparse.ArgumentParser('Visual Query fine-tuning script', add_help=False)
    
    # 数据参数
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size)')
    
    # 模型参数
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--window_size', type=int, default=200,
                        help='EEG signal window size')
    parser.add_argument('--in_chans', type=int, default=62,
                        help='Number of EEG channels')
    parser.add_argument('--patch_size', type=int, default=200,
                        help='Patch size for EEG signal')
    
    # 训练参数
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enable distributed evaluation')
    
    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', type=float, default=1e-8,
                        help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', type=float, nargs='+', default=[0.9, 0.999],
                        help='Optimizer Betas')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='steps to warmup LR (override warmup_epochs if > 0)')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help='Final weight decay (default: same as weight_decay)')
    
    # 路径参数
    parser.add_argument('--data_path', default='/home/shiyao/EEG/eegprocessed/', type=str,
                        help='path to EEG data directory')
    parser.add_argument('--output_dir', default='./checkpoints/vis_query/', type=str)
    parser.add_argument('--log_dir', default='./log/vis_query', type=str)
    
    # 分布式训练参数
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    
    # 其他参数
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--nb_classes', type=int, default=70,
                        help='number of the classification types')
    parser.add_argument('--num_sql_keywords', type=int, default=50,
                        help='Number of SQL keywords')
    parser.add_argument('--max_table_columns', type=int, default=20,
                        help='Maximum number of table columns')
    
    # 设备参数
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    
    # 添加start_epoch参数
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.use_cuda and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # 计算有效batch size
    args.eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()
    
    # 设置warmup_steps的默认值
    if args.warmup_steps < 0:  # 使用warmup_epochs
        # 使用一个估计值，实际值会在main()中更新
        estimated_total_steps = 3000  # 估计的每轮迭代次数
        args.warmup_steps = args.warmup_epochs * estimated_total_steps
    
    # 设置weight_decay_end的默认值
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    
    return args

def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate position embeddings from checkpoint to current model
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 通常是1 (cls token)
        
        # 只对patch embeddings进行插值
        if num_extra_tokens > 0:
            pos_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_embed_checkpoint = pos_embed_checkpoint[:, num_extra_tokens:]

        # 添加位置编码
        if pos_embed_checkpoint.shape[1] != num_patches:
            print(f"Interpolating position embeddings from {pos_embed_checkpoint.shape[1]} to {num_patches} patches")
            pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)
            pos_embed_checkpoint = torch.nn.functional.interpolate(
                pos_embed_checkpoint, size=(num_patches,), mode='linear')
            pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)

        if num_extra_tokens > 0:
            pos_embed_checkpoint = torch.cat((pos_tokens, pos_embed_checkpoint), dim=1)
        
        checkpoint_model['pos_embed'] = pos_embed_checkpoint

def print_model_structure(model):
    print("\nDetailed Model Structure:")
    print("=" * 50)
    print(f"Model Type: {type(model).__name__}")
    
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"\nLayer: {name}")
        print(f"Shape: {list(param.shape)}")
        print(f"Parameters: {param_count:,}")
    
    print("\n" + "=" * 50)
    print(f"Total Parameters: {total_params:,}")
    print("=" * 50)

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_vis_query_dataset('train', args)
    dataset_val = build_vis_query_dataset('val', args)
    
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.model == 'labram_base_patch200_200':
        model = labram_base_patch200_200(
            num_classes=args.nb_classes,
            window_size=args.window_size,
            in_chans=args.in_chans
        )
        print_model_structure(model)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # 加载预训练模型
    pretrained_path = '/home/shiyao/EEG/LaBraM/checkpoints/labram-base.pth'
    print(f"Loading pre-trained checkpoint from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 获取模型状态字典
    checkpoint_model = checkpoint['model']
    
    # 创建新的状态字典，移除"student."前缀
    new_state_dict = {}
    for k, v in checkpoint_model.items():
        if k.startswith('student.'):
            # 移除"student."前缀
            new_k = k[8:]  # len('student.') == 8
            new_state_dict[new_k] = v
        elif not k.startswith(('logit_scale', 'projection_head')):  # 跳过不需要的参数
            new_state_dict[k] = v
    
    # 插值位置编码
    interpolate_pos_embed(model, new_state_dict)
    
    # 加载处理后的参数
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pre-trained model with msg: {msg}")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader_train))
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * len(data_loader_train))
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema=model_ema,
            log_writer=log_writer,
            start_steps=epoch * len(data_loader_train),
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=len(data_loader_train),
            update_freq=args.update_freq,
            ch_names=args.ch_names,
            is_binary=True
        )
        
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, header='Val:', 
                            ch_names=args.ch_names, metrics=['accuracy', 'precision', 'recall', 'f1'])
        print(f"Accuracy of the network on the validation set: {test_stats['accuracy']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["accuracy"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.update(test_acc=test_stats['accuracy'], head="perf", step=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 