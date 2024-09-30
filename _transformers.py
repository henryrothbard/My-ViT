import torch
from torch import nn
import torch.nn.functional as F

# B = batch size
# L = sequence length
# N or blocks = number of transformer blocks
# h or heads = number of heads
# d_model = dimensionality of each `token` vector
# d_k = key/query vector dimensionality
# d_v = value vector dimensionality
# d_ff = feed forward dimensionality
# a_ff = activation function
# pos_encoding = position encoding module
# m_L = max length
# e_p_drop = encoding probability of dropout
# p_drop = probability of dropout


class SinusoidalPositionEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.

    Attributes:
        d_model (int): Dimensionality of the model.
        m_L (int): Maximum sequence length.

    Methods:
        forward(x): Adds positional encodings to the input tensor.
    """
    def __init__(self, d_model, m_L):
        super(SinusoidalPositionEncoding, self).__init__()
        pe = torch.zeros(m_L, d_model)
        pos = torch.arange(0, m_L, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class LearnedPositionEncoding(nn.Module):
    """
    Implements learned positional encoding.

    Attributes:
        d_model (int): Dimensionality of the model.
        m_L (int): Maximum sequence length.

    Methods:
        forward(x): Adds learned positional encodings to the input tensor.
    """
    def __init__(self, d_model, m_L):
        super(LearnedPositionEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(m_L, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention mechanism.

    Attributes:
        h (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_k (int): Dimensionality of key/query vectors.
        d_v (int): Dimensionality of value vectors. Defaults to d_k if not specified.
        p_drop (float): Dropout probability.
        causal (bool): If True, applies causal masking.
        wk_bias (bool): If True, applies a bias to the key weights.
        wv_bias (bool): If True, applies a bias to the value weights.
        wo_bias (bool): If True, applies a bias to the output weights.

    Methods:
        forward(x): Processes the input tensor with attention mechanism.
    """
    def __init__(self, h, d_model, d_k, d_v=None, p_drop=0, causal=False, 
                 wk_bias=True, wv_bias=True, wo_bias=True):
        super(MultiHeadSelfAttention, self).__init__()
        
        d_v = d_v if d_v is not None else d_k
        
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop
        self.causal = causal
        
        self.Wqk = nn.Linear(d_model, 2 * d_k * h, bias=wk_bias)
        self.Wv = nn.Linear(d_model, d_v * h, bias=wv_bias)
        self.Wo = nn.Linear(d_v * h, d_model, bias=wo_bias)

    def forward(self, x): # (B, L, d_model)
        B, L, _ = x.shape
        qk = self.Wqk(x) # (B, L, 2 * d_k * h)
        q, k = torch.split(qk, self.d_k * self.h, dim=-1) # (B, L, d_k * h), (B, L, d_k * h)

        v = self.Wv(x) # (B, L, d_v * h)

        # splitting heads
        q = q.view(B, L, self.h, self.d_k).transpose(1, 2) # (B, h, L, d_k)
        k = k.view(B, L, self.h, self.d_k).transpose(1, 2) # (B, h, d_k, L)
        v = v.view(B, L, self.h, self.d_v).transpose(1, 2) # (B, h, L, d_v)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_drop, is_causal=self.causal) # (B, h, L, d_v)

        x = x.transpose(1, 2) # (B, L, h, d_v)
        x = x.contiguous().view(B, L, self.h * self.d_v) # (B, L, h * d_v)
        return self.Wo(x) # (B, L, d_model)


class TransformerBlock(nn.Module):
    """
    Defines a single Transformer block layer.

    Attributes:
        h (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_k (int): Dimensionality of key/query vectors.
        d_v (int): Dimensionality of value vectors. Defaults to d_k if not specified.
        d_ff (int): Dimensionality of the feed-forward network. Defaults to 4*d_model if not specified.
        a_ff (activation function): Activation function used in feed-forward network.
        p_drop (float): Dropout probability.
        causal (bool): If True, applies causal masking.
        wk_bias (bool): If True, applies a bias to the key weights.
        wv_bias (bool): If True, applies a bias to the value weights.
        wo_bias (bool): If True, applies a bias to the output weights.
        ff_bias (bool): If True, applies a bias to the feed-forward network weights.

    Methods:
        forward(x): Processes the input through one transformer block.
    """
    def __init__(self, h, d_model, d_k, d_v=None, d_ff=None, a_ff=nn.ReLU, p_drop=0, causal=False, 
                 wk_bias=True, wv_bias=True, wo_bias=True, ff_bias=True):
        super(TransformerBlock, self).__init__()

        d_v = d_v if d_v is not None else d_k
        d_ff = d_ff if d_ff is not None else 4 * d_model
        
        self.mhsa = MultiHeadSelfAttention(h, d_model, d_k, d_v, p_drop, causal, wk_bias, wv_bias, wo_bias)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=ff_bias),
            a_ff(),
            nn.Linear(d_ff, d_model, bias=ff_bias)
        )
        self.drop = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x): # (B, L, d_model)
        x = x + (self.mhsa(x)) # (B, L, d_model) -- dropout included inside mhsa
        x = self.norm1(x) # (B, L, d_model)
        x = x + self.drop(self.ffn(x)) # (B, L, d_model)
        return self.norm2(x) # (B, L, d_model)


class Transformer(nn.Module):
    """
    A Torch (nn.Module) Transformer model using an encoder-encoder architecture with configurable parameters for flexibility.
    
    Attributes:
        blocks (int): The number of transformer blocks in the model.
        heads (int): The number of attention heads in each MultiHeadSelfAttention module.
        d_model (int): The dimensionality of the input token vectors.
        d_k (int): The dimensionality of the key and query vectors in the attention mechanism.
        d_v (int): The dimensionality of the value vectors in the attention mechanism. Defaults to d_k if not specified.
        d_ff (int): The dimensionality of the feedforward network model. Defaults to 4*d_model if not specified.
        a_ff (nn.Module): The activation function to be used in the feedforward network. Defaults to nn.ReLU.
        p_drop (float): The dropout probability used in the dropout layers to prevent overfitting.
        pos_encoder (str or nn.Module): Type of positional encoder to use. Options are 'sinusoidal', 'learned', or an instance of a positional encoding class.
        max_length (int): The maximum length of the input sequences for positional encoding.
        e_p_drop (float): Dropout probability specifically for the positional encoding. Defaults to p_drop if not specified.
        causal (bool): If True, the model will apply causal masking to ensure that predictions for a position can depend only on known outputs at positions before it.
        wk_bias (bool): If True, adds bias to key and query weight matrices.
        wv_bias (bool): If True, adds bias to value weight matrices.
        wo_bias (bool): If True, adds bias to output weight matrices.
        ff_bias (bool): If True, adds bias to feedforward network layers.

    Methods:
        forward(x): Defines the computation performed at every call.
        
        - x (Tensor): The input tensor to the transformer model. Shape (B, L, d_model), where B is batch size, L is sequence length, and d_model is the feature dimension of each token.
        
        Returns:
        - Tensor: The output of the transformer model. Shape (B, L, d_model).

    Example:
        >>> transformer = Transformer(blocks=6, heads=8, d_model=512, d_k=64, d_ff=2048, a_ff=nn.ReLU, p_drop=0.1, pos_encoder='sinusoidal', max_length=5000, causal=False)
        >>> inputs = torch.randn(10, 20, 512)
        >>> outputs = transformer(inputs)
        >>> print(outputs.shape)
        <<< (10, 20, 512)
    """
    def __init__(self, blocks, heads, d_model, d_k, 
                 d_v=None, d_ff=None, a_ff=nn.ReLU, p_drop=0,
                 pos_encoder=None, max_length=5000, e_p_drop=None, causal=False, 
                 wk_bias=True, wv_bias=True, wo_bias=True, ff_bias=True):
        super(Transformer, self).__init__()

        d_v = d_v if d_v is not None else d_k
        d_ff = d_ff if d_ff is not None else 4 * d_model

        encoding = []
        self.pos_encoder = None
        if pos_encoder is not None:
            if not isinstance(pos_encoder, str):
                self.pos_encoder = pos_encoder(d_model=d_model, m_L=max_length)
            elif pos_encoder.lower() == 'sinusoidal':
                self.pos_encoder = SinusoidalPositionEncoding(d_model, max_length)
            elif pos_encoder.lower() == 'learned':
                self.pos_encoder = LearnedPositionEncoding(d_model, max_length)
            else:
                raise ValueError('Invalid `pos_encoder` value.')
            
            if e_p_drop is None:
                e_p_drop = p_drop
        
        if e_p_drop is not None and e_p_drop > 0:
            encoding.append(nn.Dropout(e_p_drop))

        self.blocks = blocks
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.a_ff = a_ff
        self.p_drop = p_drop
        self.e_p_drop = e_p_drop
        self.causal = causal
        self.wk_bias = wk_bias 
        self.wv_bias = wv_bias 
        self.wo_bias = wo_bias
        self.ff_bias = ff_bias

        params = [heads, d_model, d_k, d_v, d_ff, a_ff, p_drop, causal, wk_bias, wv_bias, wo_bias, ff_bias]
        
        self.seq = nn.Sequential(*encoding, *[TransformerBlock(*params) for i in range(blocks)])

    def forward(self, x): # (B, L, d_model)
        return self.seq(x) # (B, L, d_model)


class LearningRateScheduler:
    """
    Implements a learning rate scheduler based on the transformer model's attention mechanism.

    Attributes:
        d_model (int): The model dimensionality which influences the learning rate.
        warmup_steps (int): The number of steps during which the learning rate will increase.

    Methods:
        __call__(): Updates and returns the current learning rate.
        do_step(step, set_step=False): Calculates the learning rate at `step` and optionally saves it.
    """
    def __init__(self, d_model, warmup_steps):
        self.dm = torch.tensor(d_model ** -0.5)
        self.ws = torch.tensor(warmup_steps ** -1.5)
        self.step = torch.tensor(0)

    def __call__(self):
        self.step += 1
        return self.dm * torch.min(self.step ** -0.5, self.step * self.ws)

    def do_step(self, step, set_step=False):
        step = min(step, 1)
        if set_step:
            self.step = step
        
        return self.dm * torch.min(step ** -0.5, step * self.ws)