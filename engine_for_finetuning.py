import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
from timm.utils import accuracy
from trainer.vis_query_loss import VisQueryLoss

import utils

def train_one_epoch(model: torch.nn.Module,
                   data_loader: Iterable,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   epoch: int,
                   loss_scaler,
                   max_norm: float = 0,
                   model_ema: Optional[torch.nn.Module] = None,
                   log_writer=None,
                   start_steps=None,
                   lr_schedule_values=None,
                   wd_schedule_values=None,
                   num_training_steps_per_epoch=None,
                   update_freq=None,
                   ch_names=None,
                   is_binary=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # 初始化损失函数
    criterion = VisQueryLoss().to(device)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        # 更新学习率
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 将数据移到设备上
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(batch['table_eeg'], batch['query_eeg'], batch['table'])
            loss, loss_dict = criterion(outputs, batch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 反向传播
        loss /= update_freq
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                   parameters=model.parameters(), create_graph=False,
                   update_grad=(data_iter_step + 1) % update_freq == 0)
        
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        # 记录损失
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 记录各个组件的损失
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=optimizer.param_groups[0]["lr"], head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=None):
    criterion = VisQueryLoss().to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # 切换到评估模式
    model.eval()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        # 将数据移到设备上
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(batch['table_eeg'], batch['query_eeg'], batch['table'])
            loss, loss_dict = criterion(outputs, batch)
            
        # 计算评估指标
        if metrics:
            for metric in metrics:
                if metric == 'accuracy':
                    acc = compute_accuracy(outputs, batch)
                    metric_logger.update(accuracy=acc)
                elif metric == 'precision':
                    prec = compute_precision(outputs, batch)
                    metric_logger.update(precision=prec)
                elif metric == 'recall':
                    rec = compute_recall(outputs, batch)
                    metric_logger.update(recall=rec)
                elif metric == 'f1':
                    f1 = compute_f1(outputs, batch)
                    metric_logger.update(f1=f1)
        
        metric_logger.update(loss=loss.item())
        
        # 记录各个组件的损失
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_accuracy(outputs, targets):
    """计算预测准确率
    
    分别计算:
    1. 图表类型预测准确率
    2. SQL关键字预测准确率
    3. 表格列选择准确率
    """
    accuracies = {}
    
    # 1. 图表类型准确率
    chart_pred = outputs['chart_type'].argmax(dim=1)
    chart_true = targets['chart_type']
    chart_acc = (chart_pred == chart_true).float().mean()
    accuracies['chart_acc'] = chart_acc.item()
    
    # 2. SQL关键字准确率
    keyword_pred = (outputs['keyword_logits'] > 0).float()
    keyword_true = targets['keyword_labels']
    keyword_acc = (keyword_pred == keyword_true).float().mean()
    accuracies['keyword_acc'] = keyword_acc.item()
    
    # 3. 表格列选择准确率
    column_pred = (outputs['column_logits'] > 0).float()
    column_true = targets['value_labels']
    column_acc = (column_pred == column_true).float().mean()
    accuracies['column_acc'] = column_acc.item()
    
    # 综合准确率
    accuracies['total_acc'] = (chart_acc + keyword_acc + column_acc) / 3
    
    return accuracies

def compute_precision(outputs, targets):
    """计算精确率
    
    分别计算:
    1. SQL关键字预测精确率
    2. 表格列选择精确率
    """
    precisions = {}
    
    # 1. SQL关键字精确率
    keyword_pred = (outputs['keyword_logits'] > 0).float()
    keyword_true = targets['keyword_labels']
    keyword_tp = (keyword_pred * keyword_true).sum()
    keyword_fp = (keyword_pred * (1 - keyword_true)).sum()
    keyword_precision = keyword_tp / (keyword_tp + keyword_fp + 1e-8)
    precisions['keyword_precision'] = keyword_precision.item()
    
    # 2. 表格列选择精确率
    column_pred = (outputs['column_logits'] > 0).float()
    column_true = targets['value_labels']
    column_tp = (column_pred * column_true).sum()
    column_fp = (column_pred * (1 - column_true)).sum()
    column_precision = column_tp / (column_tp + column_fp + 1e-8)
    precisions['column_precision'] = column_precision.item()
    
    return precisions

def compute_recall(outputs, targets):
    """计算召回率
    
    分别计算:
    1. SQL关键字预测召回率
    2. 表格列选择召回率
    """
    recalls = {}
    
    # 1. SQL关键字召回率
    keyword_pred = (outputs['keyword_logits'] > 0).float()
    keyword_true = targets['keyword_labels']
    keyword_tp = (keyword_pred * keyword_true).sum()
    keyword_fn = ((1 - keyword_pred) * keyword_true).sum()
    keyword_recall = keyword_tp / (keyword_tp + keyword_fn + 1e-8)
    recalls['keyword_recall'] = keyword_recall.item()
    
    # 2. 表格列选择召回率
    column_pred = (outputs['column_logits'] > 0).float()
    column_true = targets['value_labels']
    column_tp = (column_pred * column_true).sum()
    column_fn = ((1 - column_pred) * column_true).sum()
    column_recall = column_tp / (column_tp + column_fn + 1e-8)
    recalls['column_recall'] = column_recall.item()
    
    return recalls

def compute_f1(outputs, targets):
    """计算F1分数"""
    precisions = compute_precision(outputs, targets)
    recalls = compute_recall(outputs, targets)
    
    f1_scores = {}
    
    # 计算SQL关键字的F1
    keyword_p = precisions['keyword_precision']
    keyword_r = recalls['keyword_recall']
    keyword_f1 = 2 * keyword_p * keyword_r / (keyword_p + keyword_r + 1e-8)
    f1_scores['keyword_f1'] = keyword_f1
    
    # 计算表格列选择的F1
    column_p = precisions['column_precision']
    column_r = recalls['column_recall']
    column_f1 = 2 * column_p * column_r / (column_p + column_r + 1e-8)
    f1_scores['column_f1'] = column_f1
    
    return f1_scores

def compute_query_similarity(pred_query, target_query):
    """计算查询语义相似度
    
    使用预训练语言模型计算查询之间的相似度
    """
    from transformers import AutoModel, AutoTokenizer
    
    # 加载预训练模型
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 编码查询
    pred_tokens = tokenizer(pred_query, return_tensors='pt', padding=True)
    target_tokens = tokenizer(target_query, return_tensors='pt', padding=True)
    
    # 获取查询表示
    with torch.no_grad():
        pred_emb = model(**pred_tokens).pooler_output
        target_emb = model(**target_tokens).pooler_output
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(pred_emb, target_emb)
    
    return similarity.item()