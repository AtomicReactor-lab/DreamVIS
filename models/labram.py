import torch
import torch.nn as nn
from functools import partial

class PatchEmbed(nn.Module):
    """ EEG signal to Patch Embedding
    """
    def __init__(self, window_size=200, patch_size=200, in_chans=62, embed_dim=20):
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_patches = window_size // patch_size
        
        # 匹配预训练模型的卷积层维度
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 15))  # [8,1,1,15]
        self.norm1 = nn.LayerNorm(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(1, 3))   # [8,8,1,3]
        self.norm2 = nn.LayerNorm(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(1, 3))   # [8,8,1,3]
        self.norm3 = nn.LayerNorm(8)
        
        # 最终投影到20维度
        self.proj = nn.Linear(8, embed_dim)

    def forward(self, x):
        # [B, C, L] -> [B, 1, C, L]
        x = x.unsqueeze(1)
        
        # 应用卷积层
        x = self.conv1(x)
        x = x.transpose(2, 3)
        x = self.norm1(x)
        x = x.transpose(2, 3)
        
        x = self.conv2(x)
        x = x.transpose(2, 3)
        x = self.norm2(x)
        x = x.transpose(2, 3)
        
        x = self.conv3(x)
        x = x.transpose(2, 3)
        x = self.norm3(x)
        
        # 投影到目标维度
        x = self.proj(x)  # [B, 1, L, embed_dim]
        x = x.squeeze(1)  # [B, L, embed_dim]
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 添加单独的q,k,v投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 添加q,k的归一化层
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # 应用归一化并分别计算q,k,v
        q = self.q_norm(self.q(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # 添加gamma参数
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LaBraM(nn.Module):
    """LaBraM基础模型"""
    def __init__(self, window_size=200, in_chans=62, patch_size=200, embed_dim=20,
                 depth=12, num_heads=4, mlp_ratio=4., qkv_bias=False, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_classes=1000, **kwargs):
        super().__init__()
        
        # patch embedding
        self.patch_embed = PatchEmbed(
            window_size=window_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim)
        
        # 保存num_patches作为类属性
        self.num_patches = self.patch_embed.num_patches
        
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def labram_base_patch200_200(pretrained=False, **kwargs):
    model = LaBraM(
        patch_size=200,
        embed_dim=200,  # 保持200维度不变
        depth=12,
        num_heads=10,   # 保持10个头不变
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model 