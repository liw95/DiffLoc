import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from models.model_utils import adapt_input_conv, resize_pos_embed, init_weights, padding, unpadding, GeMPooling
from models.stems import PatchEmbedding, ConvStem
from models.decoders import DecoderLinear
# DINOv2
from models.layers import Mlp, NestedTensorBlock as Block

# Modified from https://github.com/valeoai/rangevit/blob/main/models/rangevit.py

logger = logging.getLogger(__name__)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            n_layers,
            d_model,
            d_ff,
            n_heads,
            dropout=0.1,
            drop_path_rate=0.0,
            channels=3,
            ls_init_values=None,
            patch_stride=None,
            conv_stem='none',
            stem_base_channels=32,
            stem_hidden_dim=None,
            n_cls=1
    ):
        super().__init__()

        self.conv_stem = conv_stem

        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                patch_stride,
                d_model,
                channels, )
        else:  # in this case self.conv_stem = 'ConvStem'
            assert patch_stride == patch_size  # patch_size = patch_stride if a convolutional stem is used

            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        mlp_ratio = 4
        qkv_bias = True
        proj_bias = True
        ffn_bias = True

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        ffn_layer = Mlp
        init_values = 1.0

        blocks_list = [
            Block(
                dim=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(n_layers)
        ]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))


        self.blocks = nn.ModuleList(blocks_list)
        self.chunked_blocks = False

        self.norm = norm_layer(d_model)
        self.head = nn.Identity()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def prepare_tokens(self, x):
        B, _, W, H = x.shape
        x, skip = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.pos_embed
        num_extra_tokens = 1

        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (grid_H, grid_W),
                num_extra_tokens,
            )

        x = x + pos_embed
        x = self.dropout(x)

        return x, skip

    def forward(self, im, return_features=False):
        x, skip = self.prepare_tokens(im)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)

        return x, skip


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size')
    new_patch_stride = model_cfg.pop('new_patch_stride')

    if (new_patch_size is not None):
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    model = VisionTransformer(**model_cfg)

    return model


def create_decoder(decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
    decoder = DecoderLinear(**decoder_cfg)

    return decoder, name


def create_rangevit(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')

    encoder = create_vit(model_cfg)

    decoder, name = create_decoder(decoder_cfg)

    model = RangeViT(encoder, decoder, n_cls=model_cfg['n_cls'])

    return model


class RangeViT(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder
        # self.pool = GeMPooling()

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
             append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params


    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)  # [B*N, 32, 720, 5]
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, skip = self.encoder(im, return_features=True)

        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]

        pred_mask, feats = self.decoder(x, (H, W), skip)
        # sigmoid
        pred_mask = torch.sigmoid(pred_mask)
        #
        feats = F.interpolate(feats, size=(H, W), mode='bilinear')
        feats = unpadding(feats, (H_ori, W_ori))
        x = (x + x * pred_mask).mean(1)

        return x, feats


class ImageFeatureExtractor(nn.Module):
    def __init__(
            self,
            backbone: str = "vit_base_patch16_384",
            freeze=False,
            in_channels=5,
            new_patch_size=(4, 16),
            new_patch_stride=(4, 16),
            conv_stem='ConvStem',  # 'none' or 'ConvStem'
            stem_base_channels=32,
            D_h=256,  # hidden dimension of the stem
            image_size=(32, 512),
            decoder='up_conv',
            pretrained_path="dino_vitbase16_pretrain.pth",
            reuse_pos_emb=True,
            reuse_patch_emb=False,
            n_cls=1
    ):
        super().__init__()

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif backbone == 'vit_base_patch16_384':
            n_heads = 12
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 768
        elif backbone == 'vit_large_patch16_384':
            n_heads = 16
            n_layers = 24
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 1024
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        if decoder == 'linear':
            decoder_cfg = {'name': 'linear',
                           'patch_size': new_patch_size,
                            'patch_stride': new_patch_stride,
                            'd_encoder': d_model,
                            'n_cls': n_cls}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'name': 'up_conv',
                'patch_size': new_patch_size,
                'patch_stride': new_patch_stride,
                'd_encoder': d_model,
                'n_cls': n_cls,
                'd_decoder': 64,  # hidden dim of the decoder
                'scale_factor': new_patch_size,  # scaling factor in the PixelShuffle layer
                'skip_filters': 256 }  # channel dim of the skip connection (between the convolutional stem and the up_conv decoder)

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model,  # dim of features
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels,  # nb of channels for the 3D point projections
            'image_size': image_size,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size,  # old patch size for the ViT encoder
            'new_patch_size': new_patch_size,  # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride,  # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': D_h,
            'n_cls': n_cls  # moving objects / static objects
        }

        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs)
        old_state_dict = self.rangevit.state_dict()

        # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            print(f'Loading pretrained parameters from {pretrained_path}')
            if pretrained_path == 'timmImageNet21k':
                vit_imagenet = timm.create_model(backbone, pretrained=True)  # .cuda()
                pretrained_state_dict = vit_imagenet.state_dict()  # nb keys: 152
                all_keys = list(pretrained_state_dict.keys())
                for key in all_keys:
                    pretrained_state_dict['encoder.' + key] = pretrained_state_dict.pop(key)
            else:
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'pos_embed' in pretrained_state_dict.keys():
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.' + key] = pretrained_state_dict.pop(key)

            # Reuse pre-trained positional embeddings
            if reuse_pos_emb:
                # Resize the existing position embeddings to the desired size
                print('Reusing positional embeddings.')
                gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
                gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
                num_extra_tokens = 1
                resized_pos_emb = resize_pos_embed(pretrained_state_dict['encoder.pos_embed'],
                                                   grid_old_shape=None,
                                                   grid_new_shape=(gs_new_h, gs_new_w),
                                                   num_extra_tokens=num_extra_tokens)
                pretrained_state_dict['encoder.pos_embed'] = resized_pos_emb
            else:
                del pretrained_state_dict['encoder.pos_embed']  # remove positional embeddings

            # Reuse pre-trained patch embeddings
            if reuse_patch_emb:
                assert conv_stem == 'none'  # no patch embedding if a convolutional stem is used
                print('Reusing patch embeddings.')

                assert old_state_dict['encoder.patch_embed.proj.bias'].shape == pretrained_state_dict[
                    'encoder.patch_embed.proj.bias'].shape
                old_state_dict['encoder.patch_embed.proj.bias'] = pretrained_state_dict['encoder.patch_embed.proj.bias']

                _, _, gs_new_h, gs_new_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
                reshaped_weight = adapt_input_conv(in_channels,
                                                   pretrained_state_dict['encoder.patch_embed.proj.weight'])
                reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                pretrained_state_dict['encoder.patch_embed.proj.weight'] = reshaped_weight
            else:
                del pretrained_state_dict['encoder.patch_embed.proj.weight']  # remove patch embedding layers
                del pretrained_state_dict['encoder.patch_embed.proj.bias']  # remove patch embedding layers

            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            print(f'{msg}')

        if freeze:
            print('==> Freeze the ViT encoder (without the pos_embed and stem)')
            for param in self.feature_extractor.blocks.parameters():
                param.requires_grad = False

            self.feature_extractor.norm.weight.requires_grad = False
            self.feature_extractor.norm.bias.requires_grad = False


    def get_output_dim(self):
        return self._output_dim

    def forward(self, *args):
        return self.rangevit(*args)