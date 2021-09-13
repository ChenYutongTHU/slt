import os, sys
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from models_3d.I3D.pytorch_i3d import InceptionI3d
from models_3d.CoCLR.backbone.s3dg import S3D
from models_3d.S3D.model import S3Dsup
from models_3d.S3D_HowTo100M.s3dg import S3D as S3Dtup
BLOCK2SIZE = {1:64, 2:192, 3:480, 4:832, 5:1024}

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
    def __init__(self, input_channel=3, freeze_block=0, use_block=5, stride=2):
        self.END_POINT2BLOCK = {
            'Conv3d_1a_7x7': 'block1',
            'Conv3d_2c_3x3': 'block2',  # block2
            'Mixed_3c': 'block3',  # block3
            'Mixed_4f': 'block4',  # block4
            'Mixed_5c': 'block5',  # block5
        }
        self.BLOCK2END_POINT = {blk: ep for ep,
                                blk in self.END_POINT2BLOCK.items()}
        self.final_endpoint = self.BLOCK2END_POINT['block{}'.format(use_block)]
        super(I3D, self).__init__(in_channels=input_channel, final_endpoint=self.final_endpoint, stride=stride)
        self.build()

        #self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.frozen_modules = []
        if freeze_block>0:
            self.final_freeze_endpoint = self.BLOCK2END_POINT['block{}'.format(
                freeze_block)]
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    self.frozen_modules.append(self._modules[end_point])
                if end_point == self.final_freeze_endpoint:
                    break

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        assert end_point==self.final_endpoint
        return x

class S3Ds(S3Dsup):
    def __init__(self, use_block=5, freeze_block=0, stride=2):
        self.use_block = use_block
        super(S3Ds, self).__init__(num_class=400, use_block=use_block, stride=stride)
        self.freeze_block = freeze_block
        self.END_POINT2BLOCK = {
            0: 'block1',
            3: 'block2',
            6: 'block3',
            12: 'block4',
            15: 'block5',
        }
        self.BLOCK2END_POINT = {blk:ep for ep, blk in self.END_POINT2BLOCK.items()}

        self.frozen_modules = []
        self.use_block = use_block

        if freeze_block>0:
            for i in range(0, self.base_num_layers): #base  0,1,2,...,self.BLOCK2END_POINT[blk]
                module_name = 'base.{}'.format(i)
                submodule = self.base[i]
                assert submodule != None, module_name
                if i <= self.BLOCK2END_POINT['block{}'.format(freeze_block)]:
                    self.frozen_modules.append(submodule)


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


def select_backbone(network, ckpt_dir=None, first_channel=3, use_block=5, freeze_block=0, stride=2):
    param = {'feature_size': BLOCK2SIZE[use_block]}
    assert network in ['s3ds'], 'i3d s3d haven\'t been debug'
    assert (use_block<=5 and freeze_block<=use_block), (use_block, freeze_block)
    if network == 's3d':
        assert use_block<=5, use_block
        model = S3D(input_channel=first_channel, freeze_block=freeze_block, use_block=use_block, stride=stride)
    elif network == 's3dt':
        init_path = os.path.join(ckpt_dir, 's3dt_milnce_ckpt', 's3d_dict.npy')
        model = S3Dt(init_path)
    elif network == 's3ds':
        model = S3Ds(use_block=use_block, freeze_block=freeze_block, stride=stride)
    elif network == 'i3d':
        model = I3D(input_channel=first_channel, use_block=use_block,
                    freeze_block=freeze_block, stride=stride)
    else: 
        raise NotImplementedError

    return model, param

class backbone_3D(torch.nn.Module):
    def __init__(self, network, ckpt_dir,  use_block=5, freeze_block=0, stride=2):
        super(backbone_3D, self).__init__()
        self.network = network
        self.backbone, self.param = select_backbone(
            network=network,
            ckpt_dir=ckpt_dir,
            use_block=use_block,
            freeze_block=freeze_block,
            stride=stride)       
        self.set_frozen_layers()
    
    def set_train(self):
        self.train()
        for m in self.backbone.frozen_modules:
            m.eval()

    def set_frozen_layers(self):
        for m in self.backbone.frozen_modules:
            for param in m.parameters():
                #print(param)
                param.requires_grad = False
            m.eval()

    def forward(self, block):
        (B, C, T, H, W) = block.shape
        feat3d = self.backbone(block)
        return feat3d
