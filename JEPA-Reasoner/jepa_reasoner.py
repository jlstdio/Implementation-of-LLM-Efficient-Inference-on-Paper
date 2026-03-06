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

        # Query, Key에 비학습형 LayerNorm 적용
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
        
        # Predictor
        self.blocks = nn.ModuleList([
            ModifiedTransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        # 잠재 공간 출력을 단위 구(Unit hypersphere)로 강제하는 정규화 레이어
        self.hybrid_norm = HybridNorm(embed_dim)

    def forward(self, input_tokens, reasoning_steps=1):
        """토큰 확률(LM Head) 투사 없이 잠재 공간 안에서 자가회귀적으로 생성"""
        x = self.embedding(input_tokens)
        latent_trajectory = []
        
        # 잠재 공간 내의 Autoregressive 생성 프로세스
        for _ in range(reasoning_steps):
            for block in self.blocks:
                x = block(x)
            
            # 다음 입력을 위한 하이브리드 정규화
            x = self.hybrid_norm(x)
            latent_trajectory.append(x)
            
        # (Batch, Reasoning_Steps, Seq_Len, Dim) 반환
        return torch.stack(latent_trajectory, dim=1)

class MonoTalker(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, ffn_dim=1536, num_layers=6):
        super().__init__()
        # 논문 Table 1에 따라 임베딩(Embedding) 레이어와 인코더(Encoder)가 없습니다.
        # 디코더 블록만 존재하며, 입력을 잠재 벡터(Latent Vector)로 직접 받습니다.
        self.decoder_blocks = nn.ModuleList([
            ModifiedTransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        # 토큰 출력을 위한 LM Head 존재
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, latents):
        """JEPA-Reasoner에서 생성된 잠재 궤적을 토큰으로 One-pass 재구성"""
        x = latents
        # 단일 순전파(One forward pass)로 전체 시퀀스 생성
        for block in self.decoder_blocks:
            x = block(x, causal_mask=True)
            
        logits = self.lm_head(x)
        return logits
    

class DualTalker(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, ffn_dim=1536, num_enc_layers=4, num_dec_layers=4):
        super().__init__()
        # 1. 임베딩 레이어 (문맥 토큰 인코딩용)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. 표준 Transformer 인코더 및 디코더
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_dec_layers)
        
        # 3. LM Head (토큰 확률 출력)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, latents, target_tokens):
        
        # 인코더는 JEPA-Reasoner의 잠재 궤적을 입력으로 받음 (Continuous Latent Guidance)
        memory = self.encoder(latents)
        
        # 디코더는 이전 토큰의 임베딩을 받고 인코더 출력(memory)과 Cross-Attention 수행
        tgt_emb = self.embedding(target_tokens)
        
        # Causal Mask 생성 (미래 토큰 참조 방지)
        seq_len = target_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(latents.device)
        
        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.lm_head(out)
        
        return logits