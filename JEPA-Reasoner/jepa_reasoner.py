import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class HybridNorm(nn.Module):
    # RMS + L2 Hybrid Normalization
    def __init__(self, dim):
        super().__init__()
        self.rms_norm = RMSNorm(dim)

    def forward(self, x):
        # Normalization
        x = self.rms_norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class ModifiedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # QK-Norm
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x, causal_mask=False):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled Dot-Product Attention
        attn_weights = torch.einsum('b l h d, b s h d -> b h l s', q, k) / (self.head_dim ** 0.5)
        
        if causal_mask:
            mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.einsum('b h l s, b s h d -> b l h d', attn_probs, v)
        out = out.reshape(B, L, D)
        return self.out_proj(out)

class ModifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ModifiedAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x, causal_mask=False):
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class JEPAReasoner(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=16, ffn_dim=1536, num_layers=18):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.predictor = nn.ModuleList([
            ModifiedTransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        self.hybrid_norm = HybridNorm(embed_dim)

    def forward(self, input_tokens, reasoning_steps=1):
        x = self.embedding(input_tokens)
        latent_trajectory = []
        
        for _ in range(reasoning_steps):
            for block in self.predictor:
                x = block(x)
            
            x = self.hybrid_norm(x)
            latent_trajectory.append(x)
            
        return torch.stack(latent_trajectory, dim=1) # Batch, Reasoning_Steps, Seq_Len, Dim

class MonoTalker(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, ffn_dim=1536, num_layers=6):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            ModifiedTransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(embed_dim, vocab_size) # LM head only

    def forward(self, latents):
        x = latents
        for block in self.decoder_blocks:
            x = block(x, causal_mask=True)
            
        logits = self.lm_head(x)
        return logits
    

class DualTalker(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, ffn_dim=1536, num_enc_layers=4, num_dec_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_dec_layers)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, latents, target_tokens):
        
        memory = self.encoder(latents)
        
        tgt_emb = self.embedding(target_tokens)
        
        seq_len = target_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(latents.device)
        
        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.lm_head(out)
        
        return logits