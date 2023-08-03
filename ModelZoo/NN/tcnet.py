import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import flow_warp, make_layer, ResidualBlockNoBN
from basicsr.archs.spynet_arch import SpyNet


class BasicAttention_VSR(nn.Module):
    def __init__(self,
                 image_ch=3,
                 num_feat=64,
                 feat_size=64,
                 num_frame=7,
                 num_extract_block=5,
                 depth=2,
                 heads=1,
                 patch_size=8,
                 num_block=15,
                 spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # Attention
        self.center_frame_idx = num_frame // 2
        self.num_frame = num_frame

        # Feature extractor
        self.conv_first = nn.Conv2d(image_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Transformer
        self.transformer = Transformer(num_feat, feat_size, depth, patch_size, heads)

        # DualAttention
        self.DualAttention1 = DANetHead(num_feat, num_feat)
        self.DualAttention2 = DANetHead(num_feat, num_feat)
        self.DualAttention3 = DANetHead(num_feat, num_feat)
        self.DualAttention4 = DANetHead(num_feat, num_feat)
        self.DualAttention5 = DANetHead(num_feat, num_feat)
        self.DualAttention6 = DANetHead(num_feat, num_feat)
        self.DualAttention7 = DANetHead(num_feat, num_feat)
        self.DualAttention8 = DANetHead(num_feat, num_feat)

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.Pyramidfusion = PyramidFusion(ConvBlock, num_feat)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.tensor_fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        # n = 7
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def get_attention(self, x):
        b, n, c, h, w = x.size()
        feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat = self.feature_extraction(feat).view(b, n, -1, h, w)
        attention_feat = self.transformer(feat)
        return attention_feat

    def forward(self, x):
        flows_forward, flows_backward = self.get_flow(x)
        b, n, c, h, w = x.size()
        attention_feat = self.get_attention(x)
        # branch
        out_l = []
        out_l_attn = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            # feature extraction
            feat_prop = self.backward_trunk(feat_prop)
            feat_prop_attn = self.DualAttention1(feat_prop)
            feat_prop_attn = self.DualAttention2(feat_prop_attn)
            feat_prop_attn = self.DualAttention3(feat_prop_attn)
            feat_prop_attn = self.DualAttention4(feat_prop_attn)
            feat_prop_attn = self.DualAttention5(feat_prop_attn)
            feat_prop_attn = self.DualAttention6(feat_prop_attn)
            feat_prop_attn = self.DualAttention7(feat_prop_attn)
            feat_prop_attn = self.DualAttention8(feat_prop_attn)
            out_l_attn.insert(0, feat_prop_attn)
            out_l.insert(0, feat_prop)

        out_o_attn = torch.stack(out_l_attn, dim=1)  # DualAttention feature tensor
        out_o = torch.stack(out_l, dim=1)  # [b, 14, 64, 64, 64] Residual block feature tensor
        attention_feat = self.get_attention(out_o)  # TemporalAttention feature tensor
        # print(out_o.shape)

        # refine
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            # feature extraction
            feat_prop = self.forward_trunk(feat_prop)
            # fusion and upsample
            attention_feat_f = torch.cat([out_o_attn[:, i, ...], attention_feat[:, i, ...]], dim=1)
            attention_feat_f = self.tensor_fusion(attention_feat_f)
            attention_feat_f = self.Pyramidfusion(attention_feat_f)
            out = torch.cat([out_o[:, i, ...], attention_feat_f], dim=1)
            out = self.tensor_fusion(out)
            out = self.Pyramidfusion(out)
            out = torch.cat([out, feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
        # outo = torch.stack(out_l, dim=1)
        # print(outo.shape)

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


# Attention
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, num_feat, feat_size, fn):
        super().__init__()
        self.norm = nn.LayerNorm([num_feat, feat_size, feat_size])
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MatmulNet(nn.Module):
    def __init__(self) -> None:
        super(MatmulNet, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, y)
        return x


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2
        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)
        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(64, 64), kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim
        q = self.to_q(x.view(-1, c, h, w))
        k = self.to_k(x.view(-1, c, h, w))
        v = self.to_v(x.view(-1, c, h, w))
        unfold_q = self.feat2patch(q)
        unfold_k = self.feat2patch(k)
        unfold_v = self.feat2patch(v)
        unfold_q = unfold_q.view(b, t, H, d, n)
        unfold_k = unfold_k.view(b, t, H, d, n)
        unfold_v = unfold_v.view(b, t, H, d, n)
        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()
        unfold_q = unfold_q.view(b, H, d, t * n)
        unfold_k = unfold_k.view(b, H, d, t * n)
        unfold_v = unfold_v.view(b, H, d, t * n)
        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)
        attn = attn * (d ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))
        attn_x = attn_x.view(b, H, t, n, d)
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()
        attn_x = attn_x.view(b * t, D, n)
        feat = self.patch2feat(attn_x)
        out = self.conv(feat).view(x.shape)
        out += x
        return out


class Transformer(nn.Module):
    def __init__(self, num_feat, feat_size, depth, patch_size, heads):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(num_feat, feat_size, globalAttention(num_feat, patch_size, heads))),
                Residual(PreNorm(num_feat, feat_size, globalAttention(num_feat, patch_size, heads)))
            ]))

    def forward(self, x):
        for attn, attn1 in self.layers:
            x = attn(x)
            x = attn1(x)
        return x


# Dual Attention
class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        mid_channels = in_channels
        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
        #                             nn.ReLU())

        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
        #                             nn.ReLU())

        self.sa = PAM_Module(mid_channels)
        self.sc = CAM_Module(mid_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                                    nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(mid_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(mid_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(2 * mid_channels, out_channels, 1))

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        sc_feat = self.sc(x)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = torch.cat([sa_conv, sc_conv], 1)

        sasc_output = self.conv8(feat_sum)
        return sasc_output


# for Fusion
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class PyramidFusion(nn.Module):
    """ Fusion Module """

    def __init__(self, block=ConvBlock, dim=64):
        super(PyramidFusion, self).__init__()
        self.dim = dim
        self.ConvBlock1 = ConvBlock(dim, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.ConvBlock3 = block(dim, dim, strides=1)
        self.upv5 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock5 = block(dim * 2, dim, strides=1)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)
        conv3 = self.ConvBlock3(pool1)
        up5 = self.upv5(conv3)
        up5 = torch.cat([up5, conv1], dim=1)
        conv5 = self.ConvBlock5(up5)
        conv6 = self.conv6(conv5)
        out = x + conv6
        return out


if __name__ == '__main__':
    a = torch.rand(1, 7, 3, 180, 270)
    print(a.shape)
    model = BasicAttention_VSR()
    b = model(a)
    # model.train()
    print(b.shape)
