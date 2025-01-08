import torch
import torch.nn as nn
import torch.nn.functional as F
from models.labram import labram_base_patch200_200  # 导入LaBraM基础模型

class VisQueryModel(nn.Module):
    def __init__(self, 
                 num_sql_keywords: int,
                 max_table_columns: int,
                 base_model=None,
                 hidden_dim: int = 768):
        super().__init__()
        
        # LaBraM基础模型，用于处理EEG信号
        self.base_model = base_model if base_model else labram_base_patch200_200()
        
        # Table内容编码器
        self.table_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 多头注意力，用于关联EEG特征和Table内容
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # SQL关键字预测器
        self.keyword_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_sql_keywords)
        )
        
        # Table列选择器
        self.column_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_table_columns)
        )
        
    def forward(self, table_eeg, query_eeg, table_content):
        """
        Args:
            table_eeg: Table对应的EEG信号 [batch_size, channels, time_points]
            query_eeg: Query对应的EEG信号 [batch_size, channels, time_points]
            table_content: Table内容的编码 [batch_size, num_columns, hidden_dim]
        """
        # 1. 使用LaBraM处理EEG信号
        eeg_features = self.base_model(
            torch.cat([table_eeg, query_eeg], dim=0)
        )  # [2*batch_size, hidden_dim]
        
        table_eeg_feat, query_eeg_feat = torch.chunk(eeg_features, 2, dim=0)
        
        # 2. 编码Table内容
        table_encoded = self.table_encoder(table_content)
        
        # 3. 通过注意力机制关联EEG特征和Table内容
        attended_table, _ = self.cross_attention(
            query=table_eeg_feat.unsqueeze(0),
            key=table_encoded.transpose(0, 1),
            value=table_encoded.transpose(0, 1)
        )
        attended_table = attended_table.squeeze(0)
        
        # 4. 合并特征
        combined_features = torch.cat([
            attended_table,
            query_eeg_feat
        ], dim=-1)
        
        # 5. 预测SQL关键字和Table列选择
        keyword_logits = self.keyword_predictor(combined_features)
        column_logits = self.column_selector(combined_features)
        
        return {
            'keyword_logits': keyword_logits,
            'column_logits': column_logits
        } 