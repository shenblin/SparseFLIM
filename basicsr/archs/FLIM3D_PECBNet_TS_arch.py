import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from .edvr_arch import PCDAlignment, TSAFusion



class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class PECBNet_TS(nn.Module):
    """photon enrichment with coupled bidirectional propagation network. Temporal sparsity (TS)

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframes (list): Keyframe lists
        temporal_padding (int): Temporal padding. Default: 2.
        snum_in_ch (int): Channel number of inputs.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_block=8,
                 num_in_ch=1,
                 upscale=1,
                 upscale_time=1,
                 keyframes=10,
                 temporal_padding=2,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.num_in_ch = num_in_ch
        self.upscale = upscale
        self.upscale_time = upscale_time
        self.temporal_padding = temporal_padding
        self.keyframes = keyframes

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, self.num_in_ch, edvr_path)
        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + self.num_in_ch, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + self.num_in_ch, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        if self.upscale == 3:
            self.conv_hr = nn.Conv2d(32, 64, 3, 1, 1)
        else:
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, self.num_in_ch, 3, 1, 1)

        if self.upscale == 3:
            self.pixel_shuffle = nn.PixelShuffle(3)
        else:
            self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, t, chn, h_input, w_input = x.size()

        s = self.upscale_time
        X = x.new_zeros(b, t * s, chn, h_input, w_input)
        for i in range(t):
            for j in range(s):
                if i < t-1:
                    X[:, i*s+j, :, :, :] = x[:, i, :, :, :] + (x[:, i+1, :, :, :] - x[:, i, :, :, :])*j/s
                else:
                    X[:, i * s + j, :, :, :] = x[:, i, :, :, :] + (x[:, -1, :, :, :] - x[:, i, :, :, :])*j/s

        x = X
        b, t, chn, h_input, w_input = x.size()

        h, w = x.shape[3:]

        keyframe_idx = self.keyframes

        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            # if i < n - 1:
            #     flow = flows_backward[:, i, :, :, :]
            #     feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            x_i = x[:, i, :, :, :]
            # if i > 0:
            #     flow = flows_forward[:, i - 1, :, :, :]
            #     feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            # out = torch.cat([out_l[i], feat_prop], dim=1)
            # out = self.lrelu(self.fusion(out))
            if self.upscale == 4:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            elif self.upscale in [2, 3]:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            else:
                out = feat_prop

            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)

class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor used in IconVSR.

    Args:
        num_input_frame (int): Number of input frames.
        num_feat (int): Number of feature channels
        num_in_ch (int): Channel number of inputs.
        load_path (str): Path to the pretrained weights of EDVR. Default: None.
    """

    def __init__(self, num_input_frame, num_feat, num_in_ch, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)


