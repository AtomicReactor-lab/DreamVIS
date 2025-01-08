import torch
import torch.nn as nn
import torch.nn.functional as F

class VisQueryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def keyword_selection_loss(self, pred_logits, target_labels):
        """SQL关键字选择的损失"""
        return F.binary_cross_entropy_with_logits(pred_logits, target_labels)
    
    def value_selection_loss(self, pred_logits, target_labels):
        """表格值选择的损失"""
        return F.binary_cross_entropy_with_logits(pred_logits, target_labels)
    
    def chart_type_loss(self, pred_logits, target_labels):
        """图表类型选择的损失"""
        return F.cross_entropy(pred_logits, target_labels)
    
    def structure_matching_loss(self, pred_structure, target_structure):
        """SQL结构匹配损失"""
        # 比较SQL组件的选择是否匹配
        return F.binary_cross_entropy(
            pred_structure, 
            target_structure
        )
    
    def forward(self, predictions, targets):
        """计算总损失"""
        losses = {}
        
        # 1. 图表类型损失
        if 'chart_type' in predictions and 'chart_type' in targets:
            losses['chart'] = self.chart_type_loss(
                predictions['chart_type'],
                targets['chart_type']
            )
        
        # 2. SQL关键字选择损失
        if 'keyword_logits' in predictions and 'keyword_labels' in targets:
            losses['keyword'] = self.keyword_selection_loss(
                predictions['keyword_logits'],
                targets['keyword_labels']
            )
        
        # 3. 表格值选择损失
        if 'column_logits' in predictions and 'value_labels' in targets:
            losses['value'] = self.value_selection_loss(
                predictions['column_logits'],
                targets['value_labels']
            )
        
        # 4. SQL结构匹配损失
        if 'sql_structure' in predictions and 'sql_structure' in targets:
            losses['structure'] = self.structure_matching_loss(
                predictions['sql_structure'],
                targets['sql_structure']
            )
        
        # 计算总损失
        total_loss = sum(losses.values())
        
        return total_loss, losses

    def get_metrics(self, predictions, targets):
        """计算评估指标"""
        metrics = {}
        
        # 1. 图表类型准确率
        if 'chart_type' in predictions and 'chart_type' in targets:
            chart_pred = predictions['chart_type'].argmax(dim=1)
            chart_acc = (chart_pred == targets['chart_type']).float().mean()
            metrics['chart_acc'] = chart_acc.item()
        
        # 2. 关键字选择准确率
        if 'keyword_logits' in predictions and 'keyword_labels' in targets:
            keyword_pred = (predictions['keyword_logits'] > 0).float()
            keyword_acc = (keyword_pred == targets['keyword_labels']).float().mean()
            metrics['keyword_acc'] = keyword_acc.item()
        
        # 3. 值选择准确率
        if 'column_logits' in predictions and 'value_labels' in targets:
            column_pred = (predictions['column_logits'] > 0).float()
            column_acc = (column_pred == targets['value_labels']).float().mean()
            metrics['column_acc'] = column_acc.item()
        
        return metrics 