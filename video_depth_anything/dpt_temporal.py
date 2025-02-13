# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dpt import DPTHead
from .motion_module.motion_module import TemporalModule
from easydict import EasyDict


class DPTHeadTemporal(DPTHead):
    def __init__(self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken)

        assert num_frames > 0
        motion_module_kwargs = EasyDict(num_attention_heads                = 8,
                                        num_transformer_block              = 1,
                                        num_attention_blocks               = 2,
                                        temporal_max_len                   = num_frames,
                                        zero_initialize                    = True,
                                        pos_embedding_type                 = pe)

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], 
                           **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3],
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs)
        ])

    def forward(self, out_features, patch_h, patch_w, frame_length, micro_batch_size=4):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length

        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(
                out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
            )
            ori_type = out.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.scratch.output_conv2(out.float())
            return out.to(ori_type)
        else:
            ret = []
            for i in range(0, batch_size, micro_batch_size):
                path_2 = self.scratch.refinenet2(path_3[i:i + micro_batch_size], layer_2_rn[i:i + micro_batch_size], size=layer_1_rn[i:i + micro_batch_size].shape[2:])
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn[i:i + micro_batch_size])
                out = self.scratch.output_conv1(path_1)
                out = F.interpolate(
                    out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
                )
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(out.to(ori_type))
            return torch.cat(ret, dim=0)
