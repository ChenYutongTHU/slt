import os, sys
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from models_3d.I3D.pytorch_i3d import InceptionI3d
from models_3d.CoCLR.backbone.s3dg import S3D
from models_3d.S3D.model import S3Dsup
from models_3d.S3D_HowTo100M.s3dg import S3D as S3Dtup

class my_resnet50(torch.nn.Module):
    def __init__(self, pretrained_ckpt):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.resnet.load_state_dict(torch.load(pretrained_ckpt),strict=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)  # return logits
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x.squeeze(dim=-1).squeeze(dim=-1)

class I3D(InceptionI3d):
    def __init__(self, input_channel=3, final_endpoint='Mixed_5c'):
        super(I3D, self).__init__(in_channels=input_channel, final_endpoint=final_endpoint)
        self.build()
        #self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return x

class S3Ds(S3Dsup):
    def __init__(self):
        super(S3Ds, self).__init__(num_class=400)

    def forward(self, x):
        x = self.base(x)
        return x

class S3Dt(S3Dtup):
    def __init__(self, init_path):
        super(S3Dt, self).__init__(dict_path=init_path, num_classes=512)

    def forward(self, x):
        if self.space_to_depth:
            x = self._space_to_depth(x)
        net = self.conv1(x)
        if self.space_to_depth:
            # we need to replicate 'SAME' tensorflow padding
            net = net[:, :, 1:, 1:, 1:]
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        feat = self.mixed_5c(net)
        return feat


def select_backbone(network, ckpt_dir=None, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dt':
        init_path = os.path.join(ckpt_dir, 's3dt_milnce_ckpt', 's3d_dict.npy')
        model = S3Dt(init_path)
    elif network == 's3ds':
        model = S3Ds()
    elif network == 'i3d':
        model = I3D(input_channel=first_channel)
    else: 
        raise NotImplementedError

    return model, param

class backbone_3D(torch.nn.Module):
    def __init__(self, ckpt_dir, network='i3d'):
        super(backbone_3D, self).__init__()
        self.network = network
        self.backbone, self.param = select_backbone(network, ckpt_dir)
    
    def forward(self, block):
        (B, C, T, H, W) = block.shape
        feat3d = self.backbone(block)
        return feat3d