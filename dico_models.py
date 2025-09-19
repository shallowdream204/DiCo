import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size,bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class LabelEmbedder_3(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size_0, hidden_size_1, hidden_size_2, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table_0 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size_0)
        self.embedding_table_1 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size_1)
        self.embedding_table_2 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size_2)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings_0 = self.embedding_table_0(labels)
        embeddings_1 = self.embedding_table_1(labels)
        embeddings_2 = self.embedding_table_2(labels)

        return embeddings_0, embeddings_1, embeddings_2


#################################################################################
#                                 Core DiCo Model                                #
#################################################################################

class DiCoBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=1, groups=hidden_size,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size , kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        ffn_channel = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(in_channels=hidden_size, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(hidden_size,affine=False, eps=1e-6)
        self.norm2 = LayerNorm2d(hidden_size,affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, inp, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = self.norm1(inp)

        x = modulate(x, shift_msa, scale_msa)

        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)

        x = inp + gate_msa.unsqueeze(-1).unsqueeze(-1) * x 

        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.conv5(F.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp))))
        
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiCo.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.out_proj(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class DiCo(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=[2,5,8,5,2],
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size, bias=True)

        self.t_embedder_1 = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder_3(num_classes, hidden_size, hidden_size*2, hidden_size*4, class_dropout_prob)

        self.t_embedder_2 = TimestepEmbedder(hidden_size*2)

        self.t_embedder_3 = TimestepEmbedder(hidden_size*4)


        # encoder-1
        self.encoder_level_1 = nn.ModuleList([
            DiCoBlock(hidden_size, mlp_ratio, **kwargs) for _ in range(depth[0])
        ])
        self.down1_2 = Downsample(hidden_size) 

        # encoder-2
        self.encoder_level_2 = nn.ModuleList([
            DiCoBlock(hidden_size*2, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth[1])
        ])
        self.down2_3 = Downsample(hidden_size*2) 

        # latent
        self.latent = nn.ModuleList([
            DiCoBlock(hidden_size*4,mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth[2])
        ])

        # decoder-2
        self.up3_2 = Upsample(int(hidden_size*4))  ## From Level 4 to Level 3
        self.reduce_chan_level2 = nn.Conv2d(int(hidden_size*4), int(hidden_size*2), kernel_size=1, bias=True)
        self.decoder_level_2 = nn.ModuleList([
            DiCoBlock(hidden_size*2, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth[3])
        ])

        # decoder-1
        self.up2_1 = Upsample(int(hidden_size*2))  ## From Level 4 to Level 3
        self.reduce_chan_level1 = nn.Conv2d(int(hidden_size*2), int(hidden_size*2), kernel_size=1, bias=True)
        self.decoder_level_1 = nn.ModuleList([
            DiCoBlock(hidden_size*2, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth[4])
        ])

        self.final_layer = FinalLayer(hidden_size*2, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table_0.weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table_1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table_2.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder_1.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_1.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder_2.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_2.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder_3.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_3.mlp[2].weight, std=0.02)

        blocks = self.encoder_level_1 + self.encoder_level_2 + self.latent + self.decoder_level_2 + self.decoder_level_1
        for block in blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiCo.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)                   # (N, C, H, W)
        t1 = self.t_embedder_1(t)    # (N, C, 1, 1)
        y1, y2, y3 = self.y_embedder(y, self.training)    # (N, C, 1, 1)
        c1 = t1 + y1                                # (N, D, 1, 1)

        t2 = self.t_embedder_2(t)    # (N, C, 1, 1)
        c2 = t2 + y2                                # (N, D, 1, 1)

        t3 = self.t_embedder_3(t)    # (N, C, 1, 1)
        c3 = t3 + y3                                # (N, D, 1, 1)

        # encoder_1
        out_enc_level1 = x
        for block in self.encoder_level_1:
            out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # encoder_2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # latent
        latent = inp_enc_level3
        for block in self.latent:
            latent = block(latent, c3)

        # decoder_2
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level_2:
            out_dec_level2 = block(out_dec_level2, c2)

        # decoder_1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level_1:
            out_dec_level1 = block(out_dec_level1, c2)

        # output
        output = self.final_layer(out_dec_level1, c2)                # (N, T, patch_size ** 2 * out_channels)

        return output

    def forward_with_cfg(self, x, t, y, cfg_scale):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                                   Configs                                  #
#################################################################################

def DiCo_S(**kwargs):
    return DiCo(hidden_size=128, depth=[5,4,4,4,4], mlp_ratio=2)

def DiCo_B(**kwargs):
    return DiCo(hidden_size=256, depth=[5,4,4,4,4], mlp_ratio=2)

def DiCo_L(**kwargs):
    return DiCo(hidden_size=352, depth=[9,8,9,8,9], mlp_ratio=2)

def DiCo_XL(**kwargs):
    return DiCo(hidden_size=416, depth=[9,9,10,9,9], mlp_ratio=2)

def DiCo_H(**kwargs):
    return DiCo(hidden_size=416, depth=[14,12,10,12,14], mlp_ratio=4)

DiT_models = {
    'DiCo-S': DiCo_S,
    'DiCo-B': DiCo_B,
    'DiCo-L': DiCo_L,
    'DiCo-XL': DiCo_XL,
    'DiCo-H': DiCo_H,
}

def test_model_throughout(model, model_name='', bs=None):
    from tqdm import tqdm
    import time
    torch.cuda.empty_cache()
    model.cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    torch.cuda.empty_cache()
    inputs = torch.rand(bs, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    # warm up
    print(f"warm up")
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in tqdm(range(20)):
            model(inputs, t, y)
    iters = 20
    print(f"start test throughout")
    torch.cuda.empty_cache()
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(iters)):
            model(inputs, t, y)
    end_time = time.time()
    throughout = (iters*bs)/(end_time-start_time)
    print(f"{model_name}, throughout = {throughout:.2f} samples/second")


if __name__=="__main__":
    from torchprofile import profile_macs

    model = DiCo_S()
    # model = DiCo_B()
    # model = DiCo_L()
    # model = DiCo_XL()
    # model = DiCo_H()

    model.cuda()
    model.eval()

    inputs = torch.rand(1, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()

    test_model_throughout(model,'DiCo-S',64)

    flops = profile_macs(model, (inputs, t, y))
    print(f'FLOPS: {flops/1e6:.2f}')

    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print("Number of parameter: %.4fM" % (total / 1e6))