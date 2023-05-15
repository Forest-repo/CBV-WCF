import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class Fusion_4_residual_learn(nn.Module):
    """
    in_channel:  input channel
    out_channel: output channel
    p: λ1
    q: λ2
    """
    def __init__(self, in_channel, out_channel, kernelsize, p, q):
        super(Fusion_4_residual_learn,self).__init__()
        if kernelsize == 1:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1)
        if kernelsize == 3:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if kernelsize == 5:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        if kernelsize == 7:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        if kernelsize == 9:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

        self.conv_compute = conv_compute
        self.pp =torch.nn.Parameter(torch.tensor([p],requires_grad=True))
        self.qq =torch.nn.Parameter(torch.tensor([q],requires_grad=True))
        print('Learnable λa init value is:',self.pp.data)
        print('Learnable λg init value is:',self.qq.data)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,a,g):
        B,C,H,W = a.size()
        cat_input = torch.cat([a,g],dim=1)

        output_conv = self.conv_compute(cat_input)
        output_attention = a + g + output_conv

        result_a = a + self.pp * output_attention
        result_g = g + self.qq * output_attention

        return result_a, result_g

class Fusion_4_residual_learn_linear(nn.Module):
    """
    in_channel:  input channel
    out_channel: output channel
    p: λ1
    q: λ2
    residual_learn_linear: Xa+Xg+conv(concat(Xa+Xg)) is used, and λa and λg,
    namely p and q, are learnable and participate in gradient propagation (nn.Parameter).
    The initial value is set to 0.3, and two linears are set at the same time Adaptive selection as a channel layer.
    """
    def __init__(self, in_channel, out_channel, kernelsize, p, q):
        super(Fusion_4_residual_learn_linear,self).__init__()
        if kernelsize == 1:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1)
        if kernelsize == 3:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if kernelsize == 5:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        if kernelsize == 7:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        if kernelsize == 9:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

        self.conv_compute = conv_compute
        self.pp =torch.nn.Parameter(torch.tensor([p],requires_grad=True))
        self.qq =torch.nn.Parameter(torch.tensor([q],requires_grad=True))
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.channel_select_a = nn.Sequential(nn.Linear(in_channel,out_channel), nn.Sigmoid())
        self.channel_select_g = nn.Sequential(nn.Linear(in_channel,out_channel), nn.Sigmoid())
        print('λa init value is:',self.pp.data)
        print('λg init value is:',self.qq.data)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,a,g):
        B,C,H,W = a.size()
        cat_input = torch.cat([a,g],dim=1)

        output_conv = self.conv_compute(cat_input)
        output_attention = a + g + output_conv
        gap_output = self.GAP(output_attention).reshape([B,C])

        a_attention=output_attention * self.channel_select_a(gap_output).reshape([B,C,1,1])
        g_attention=output_attention * self.channel_select_g(gap_output).reshape([B,C,1,1])

        result_a = a + self.pp * a_attention
        result_g = g + self.qq * g_attention


        return result_a, result_g

class Fusion_4_residual_learn_conv1d(nn.Module):
    """
    in_channel:  input channel
    out_channel: output channel
    p: λ1
    q: λ2
    residual_learn_conv1d: Xa+Xg+conv(concat(Xa+Xg)) is used, and λa and λg,
    that is, p and q, are learnable and participate in gradient propagation (nn.Parameter).
    The initial value is set to 0.3, and two conv1d are set at the same time Adaptive selection as a channel layer.
    """
    def __init__(self, in_channel, out_channel, kernelsize, p, q):
        super(Fusion_4_residual_learn_conv1d,self).__init__()

        if kernelsize == 1:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1)
        if kernelsize == 3:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if kernelsize == 5:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        if kernelsize == 7:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        if kernelsize == 9:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

        self.conv_compute = conv_compute
        self.pp =torch.nn.Parameter(torch.tensor([p],requires_grad=True))
        self.qq =torch.nn.Parameter(torch.tensor([q],requires_grad=True))
        self.GAP = nn.AdaptiveAvgPool2d(1)

        k_size = int(abs((math.log(in_channel, 2) + 1) / 2))
        k_size = k_size if k_size % 2 else k_size + 1
        print(k_size)
        self.channel_select_a = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid())
        self.channel_select_g = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid())

        print('λa init value is:',self.pp.data)
        print('λg init value is:',self.qq.data)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,a,g):
        B,C,H,W = a.size()
        cat_input = torch.cat([a,g],dim=1)

        output_conv = self.conv_compute(cat_input)
        output_attention = a + g + output_conv
        gap_output = self.GAP(output_attention).reshape([B,1,C])

        a_attention = output_attention * self.channel_select_a(gap_output).reshape([B,C,1,1])
        g_attention = output_attention * self.channel_select_g(gap_output).reshape([B,C,1,1])

        result_a = a + self.pp * a_attention
        result_g = g + self.qq * g_attention

        return result_a, result_g

class Fusion_4_residual_learn_linear_spatial(nn.Module):
    """
    in_channel:  input channel
    out_channel: output channel
    p: λ1
    q: λ2
    residual_learn_linear_spatial: Xa+Xg+conv(concat(Xa+Xg)) is used, and λa and λg, that is,
    p and q, are learnable and participate in gradient propagation (nn.Parameter).
    The initial value is set to 0.3, and two linears are set at the same time Adaptive selection as a channel layer.
    """
    def __init__(self, in_channel, out_channel, kernelsize, p, q):
        super(Fusion_4_residual_learn_linear_spatial,self).__init__()
        if kernelsize == 1:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1)
        if kernelsize == 3:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if kernelsize == 5:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        if kernelsize == 7:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        if kernelsize == 9:
            conv_compute = torch.nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

        self.conv_compute = conv_compute
        self.pp =torch.nn.Parameter(torch.tensor([p],requires_grad=True))
        self.qq =torch.nn.Parameter(torch.tensor([q],requires_grad=True))
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.channel_select_a = nn.Sequential(nn.Linear(in_channel,out_channel), nn.Sigmoid())
        self.spatial_select_a = nn.Sequential(nn.Sigmoid())
        self.channel_select_g = nn.Sequential(nn.Linear(in_channel,out_channel), nn.Sigmoid())
        self.spatial_select_g = nn.Sequential(nn.Sigmoid())
        print('λa init value is:',self.pp.data)
        print('λg init value is:',self.qq.data)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,a,g):
        B,C,H,W = a.size()
        cat_input = torch.cat([a,g],dim=1)

        output_conv = self.conv_compute(cat_input)
        output_attention = a + g + output_conv
        gap_output = self.GAP(output_attention).reshape([B,C])
        mean_pool_out = torch.mean(output_attention, dim=1, keepdim=True)

        a_attention=output_attention * self.channel_select_a(gap_output).reshape([B,C,1,1])*self.spatial_select_a(mean_pool_out)
        g_attention=output_attention * self.channel_select_g(gap_output).reshape([B,C,1,1])*self.spatial_select_g(mean_pool_out)

        result_a = a + self.pp * a_attention
        result_g = g + self.qq * g_attention


        return result_a, result_g 

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return  x


class ResNet_CBV(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 hidlayer_dim=512):
        super(ResNet_CBV, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.ProheadMLP = nn.Sequential(
            nn.Linear(512 * block.expansion, hidlayer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidlayer_dim, 512),
            nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            Prohead_x = self.ProheadMLP(x)
            Prohead_contrast = F.normalize(Prohead_x, dim=1)
            x = self.fc(Prohead_x)

        return Prohead_contrast, x


class ResNet_MLP(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 hidlayer_dim=512):
        super(ResNet_MLP, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.MLP = nn.Sequential(
            nn.Linear(512 * block.expansion, hidlayer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidlayer_dim, 512),
            nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.MLP(x)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet18_CBV(num_classes=1000,hidlayer_dim=512, include_top=True):

    return ResNet_CBV(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, hidlayer_dim=hidlayer_dim)

def resnet18_MLP(num_classes=1000,hidlayer_dim=512, include_top=True):

    return ResNet_MLP(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, hidlayer_dim=hidlayer_dim)

def resnet34(num_classes=1000, include_top=True):

    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):

    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):

    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

class resnet18_Siamese(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18(num_classes=num_classes) 
            net2 = resnet18(num_classes=num_classes)
            self.view1_net = net1
            self.view2_net = net2
            print("train from scaratch")

        if flag_share == 'True':

            print('flag_share == True')
            net3 = resnet18(num_classes=num_classes)
            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1 = self.view1_net(x1)

        x2 = self.view2_net(x2)

        return x1, x2

class resnet18_Siamese_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_PT, self).__init__()
        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)

            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=num_classes)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1 = self.view1_net(x1)

        x2 = self.view2_net(x2)

        return x1, x2

class resnet18_Siamese_MLP(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_MLP, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_MLP(num_classes=num_classes,hidlayer_dim=512)
            net2 = resnet18_MLP(num_classes=num_classes,hidlayer_dim=512)
            print("train from scratch")
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_MLP(num_classes=num_classes,hidlayer_dim=512)
            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1 = self.view1_net(x1)
        x2 = self.view2_net(x2)

        return x1, x2

class resnet18_Siamese_MLP_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_MLP_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_MLP(num_classes=1000,hidlayer_dim=512)
            net2 = resnet18_MLP(num_classes=1000,hidlayer_dim=512)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)

            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_MLP(num_classes=num_classes)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1 = self.view1_net(x1)

        x2 = self.view2_net(x2)

        return x1, x2

class resnet18_Siamese_CBV_hid_512(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_512, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=512)
            net2 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=512)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True,权重共享模式打开')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=512)
            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)
        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_CBV_hid_128_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_128_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=128)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=128)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=128)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)
        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_CBV_hid_256_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_256_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=256)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=256)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=256)
            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)

        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_CBV_hid_512_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_512_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=512)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=512)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False) #strict=False
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False) #strict=False
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=512)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)

        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_CBV_hid_768_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_768_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=768)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=768)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=512)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)

        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_CBV_hid_1024_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_hid_1024_PT, self).__init__()

        if flag_share == 'False':

            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=1024)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=1024)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)

            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=num_classes,hidlayer_dim=1024)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2):

        x1_proj, x1 = self.view1_net(x1)

        x2_proj, x2 = self.view2_net(x2)

        return x1, x2, x1_proj, x2_proj

class resnet18_Siamese_WCF_residual_learn_layer4_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_WCF_residual_learn_layer4_PT, self).__init__()

        if flag_share == 'False':

            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer4 = Fusion_4_residual_learn(in_channel=512, out_channel=512, kernelsize=1, p=0.3, q=0.3)

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)
            x1_layer3 = self.view1_net.layer3(x1_layer2)
            x1_layer4_0 = self.view1_net.layer4[0](x1_layer3)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)
            x2_layer3 = self.view2_net.layer3(x2_layer2)
            x2_layer4_0 = self.view2_net.layer4[0](x2_layer3)

            x1_layer4_0,x2_layer4_0 = self.fusion_layer4(x1_layer4_0,x2_layer4_0)

            x1_layer4_1 = self.view1_net.layer4[1](x1_layer4_0)
            x2_layer4_1 = self.view2_net.layer4[1](x2_layer4_0)

            x1_avgpool=self.view1_net.avgpool(x1_layer4_1)
            x2_avgpool=self.view2_net.avgpool(x2_layer4_1)

            x1_flatten = torch.flatten(x1_avgpool, 1)
            x2_flatten = torch.flatten(x2_avgpool, 1)

            x1_all = self.view1_net.fc(x1_flatten)
            x2_all = self.view2_net.fc(x2_flatten)


            return x1_all, x2_all
        else:

            x1_all = self.view1_net(x1)
            x2_all = self.view2_net(x2)
            return x1_all, x2_all

class resnet18_Siamese_add_layer4_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_add_layer4_PT, self).__init__()

        if flag_share == 'False':

            print('flag_share == False')
            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':

            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)
            x1_layer3 = self.view1_net.layer3(x1_layer2)
            x1_layer4_0 = self.view1_net.layer4[0](x1_layer3)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)
            x2_layer3 = self.view2_net.layer3(x2_layer2)
            x2_layer4_0 = self.view2_net.layer4[0](x2_layer3)

            x1_layer4_0 = x1_layer4_0 + x2_layer4_0
            x2_layer4_0 = x2_layer4_0 + x1_layer4_0

            x1_layer4_1 = self.view1_net.layer4[1](x1_layer4_0)
            x2_layer4_1 = self.view2_net.layer4[1](x2_layer4_0)

            x1_avgpool=self.view1_net.avgpool(x1_layer4_1)
            x2_avgpool=self.view2_net.avgpool(x2_layer4_1)

            x1_flatten = torch.flatten(x1_avgpool, 1)
            x2_flatten = torch.flatten(x2_avgpool, 1)

            x1_all = self.view1_net.fc(x1_flatten)
            x2_all = self.view2_net.fc(x2_flatten)


            return x1_all, x2_all
        else:
            x1_all = self.view1_net(x1)
            x2_all = self.view2_net(x2)
            return x1_all, x2_all

class resnet18_Siamese_CBV_WCF_residual_learn_layer24_PT(nn.Module):
    """
    CBV
    WCF
    weight adaptive learning
    """
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet18_Siamese_CBV_WCF_residual_learn_layer24_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18_CBV(num_classes=1000,hidlayer_dim=512)
            net2 = resnet18_CBV(num_classes=1000,hidlayer_dim=512)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint,strict=False)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)
            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer2 = Fusion_4_residual_learn(in_channel=128, out_channel=128, kernelsize=1, p=0.3, q=0.3)
            self.fusion_layer4 = Fusion_4_residual_learn(in_channel=512, out_channel=512, kernelsize=1, p=0.3, q=0.3)

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18_CBV(num_classes=1000,hidlayer_dim=512)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)

            x1_layer2,x2_layer2 = self.fusion_layer2(x1_layer2,x2_layer2)

            x1_layer3 = self.view1_net.layer3(x1_layer2)
            x1_layer4_0 = self.view1_net.layer4[0](x1_layer3)

            x2_layer3 = self.view2_net.layer3(x2_layer2)
            x2_layer4_0 = self.view2_net.layer4[0](x2_layer3)

            x1_layer4_0,x2_layer4_0 = self.fusion_layer4(x1_layer4_0,x2_layer4_0)

            x1_layer4_1 = self.view1_net.layer4[1](x1_layer4_0)
            x2_layer4_1 = self.view2_net.layer4[1](x2_layer4_0)

            x1_avgpool=self.view1_net.avgpool(x1_layer4_1)
            x2_avgpool=self.view2_net.avgpool(x2_layer4_1)

            x1_flatten = torch.flatten(x1_avgpool, 1)
            x2_flatten = torch.flatten(x2_avgpool, 1)

            Prohead_x1=self.view1_net.ProheadMLP(x1_flatten)
            Prohead_x2=self.view2_net.ProheadMLP(x2_flatten)
            x1_proj_norm = F.normalize(Prohead_x1, dim=1)
            x2_proj_norm = F.normalize(Prohead_x2, dim=1)

            x1_all = self.view1_net.fc(Prohead_x1)
            x2_all = self.view2_net.fc(Prohead_x2)


            return x1_all, x2_all, x1_proj_norm, x2_proj_norm
            #return x1_all, x2_all, Prohead_x1, Prohead_x2
        else:
            x1_proj, x1_all = self.view1_net(x1)
            x2_proj, x2_all = self.view2_net(x2)

            return x1_all, x2_all, x1_proj, x2_proj

class resnet_without_WCF_layer2fusion(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_without_WCF_layer2fusion, self).__init__()
        if flag_share == 'False':
            print('flag_share == False')
            net1 = resnet18(num_classes=num_classes)
            net2 = resnet18(num_classes=num_classes)

            fusion_layer2 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer2 = fusion_layer2

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':

            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)

            concat_features = torch.cat([x1_layer2,x2_layer2],dim=1)
            fusion_features = self.fusion_layer2(concat_features)

            fusion_layer3 = self.view1_net.layer3(fusion_features)
            fusion_layer4 = self.view1_net.layer4(fusion_layer3)

            fusion_avgpool=self.view1_net.avgpool(fusion_layer4)

            fusino_flatten = torch.flatten(fusion_avgpool, 1)

            fusion_all = self.view1_net.fc(fusino_flatten)

            return fusion_all

class resnet_without_WCF_layer2fusion_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_without_WCF_layer2fusion_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)

            with torch.no_grad():
                pretrained_conv1 = net1.layer2[1].conv2.weight.detach()
                fusion_layer2 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                fusion_layer2.weight.detach()
                fusion_layer2.weight[:, :128, :, :]=pretrained_conv1
                fusion_layer2.weight[:, 128:, :, :]=pretrained_conv1
                fusion_layer2.weight.requires_grad_()
            fusion_layer2.weight.requires_grad_()

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer2 = fusion_layer2

        if flag_share == 'True':
            print('flag_share == True')

            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)

            concat_features = torch.cat([x1_layer2,x2_layer2],dim=1)
            fusion_features = self.fusion_layer2(concat_features)

            fusion_layer3 = self.view1_net.layer3(fusion_features)
            fusion_layer4 = self.view1_net.layer4(fusion_layer3)

            fusion_avgpool=self.view1_net.avgpool(fusion_layer4)

            fusino_flatten = torch.flatten(fusion_avgpool, 1)

            fusion_all = self.view1_net.fc(fusino_flatten)

            return fusion_all

class resnet_six_channel_fusion(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_six_channel_fusion, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=num_classes)
            new_conv1 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            net1.conv1 = new_conv1

            self.view1_net = net1

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            concat_features = torch.cat([x1, x2],dim=1)
            fusion_all = self.view1_net(concat_features)

            return fusion_all

class resnet_six_channel_fusion_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_six_channel_fusion_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
            with torch.no_grad():
                pretrained_conv1 = net1.conv1.weight.detach()
                new_conv1 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                new_conv1.weight.detach()
                new_conv1.weight[:, :3, :, :]=pretrained_conv1
                new_conv1.weight[:, 3:, :, :]=pretrained_conv1
                new_conv1.weight.requires_grad_()
            new_conv1.weight.requires_grad_()
            net1.conv1 = new_conv1

            self.view1_net = net1

        if flag_share == 'True':
            print('flag_share == True')

            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            concat_features = torch.cat([x1, x2],dim=1)
            fusion_all = self.view1_net(concat_features)

            return fusion_all

class resnet_Late_conv_Fusion(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Late_conv_Fusion, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=num_classes)
            net2 = resnet18(num_classes=num_classes)

            fusion_layer_late = torch.nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer_late = fusion_layer_late

        if flag_share == 'True':
            print('flag_share == True')

            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)
            x1_layer3 = self.view1_net.layer3(x1_layer2)
            x1_layer4 = self.view1_net.layer4(x1_layer3)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)
            x2_layer3 = self.view2_net.layer3(x2_layer2)
            x2_layer4 = self.view2_net.layer4(x2_layer3)

            concat_features = torch.cat([x1_layer4,x2_layer4],dim=1)
            fusion_features = self.fusion_layer_late(concat_features)

            fusion_avgpool=self.view1_net.avgpool(fusion_features)

            fusino_flatten = torch.flatten(fusion_avgpool, 1)

            fusion_all = self.view1_net.fc(fusino_flatten)

            return fusion_all

class resnet_Late_conv_Fusion_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Late_conv_Fusion_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)

            fusion_layer_late = torch.nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer_late = fusion_layer_late

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)
            x1_layer2 = self.view1_net.layer2(x1_layer1)
            x1_layer3 = self.view1_net.layer3(x1_layer2)
            x1_layer4 = self.view1_net.layer4(x1_layer3)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)
            x2_layer2 = self.view2_net.layer2(x2_layer1)
            x2_layer3 = self.view2_net.layer3(x2_layer2)
            x2_layer4 = self.view2_net.layer4(x2_layer3)

            concat_features = torch.cat([x1_layer4,x2_layer4],dim=1)
            fusion_features = self.fusion_layer_late(concat_features)

            fusion_avgpool=self.view1_net.avgpool(fusion_features)

            fusino_flatten = torch.flatten(fusion_avgpool, 1)

            fusion_all = self.view1_net.fc(fusino_flatten)

            return fusion_all

class resnet_Late_fc_concat(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Late_fc_concat, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, 512)
                net2.fc = nn.Linear(in_channel, 512)

            late_fusion_fc = nn.Linear(1024, num_classes)

            self.view1_net = net1
            self.view2_net = net2
            self.late_fusion_fc = late_fusion_fc

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1_fc = self.view1_net(x1)
            x2_fc = self.view2_net(x2)
            concat_features = torch.cat([x1_fc,x2_fc],dim=1)
            fusion_all = self.late_fusion_fc(concat_features)

            return fusion_all


class resnet_Late_fc_concat_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Late_fc_concat_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, 512)
                net2.fc = nn.Linear(in_channel, 512)

            late_fusion_fc = nn.Linear(1024, num_classes)

            self.view1_net = net1
            self.view2_net = net2
            self.late_fusion_fc = late_fusion_fc

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1_fc = self.view1_net(x1)
            x2_fc = self.view2_net(x2)
            concat_features = torch.cat([x1_fc,x2_fc],dim=1)
            fusion_all = self.late_fusion_fc(concat_features)

            return fusion_all


class resnet_Early_Fusion(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Early_Fusion, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=num_classes)
            net2 = resnet18(num_classes=num_classes)
            fusion_layer1 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer1 = fusion_layer1

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)

            concat_features = torch.cat([x1_layer1,x2_layer1],dim=1)
            fusion_features = self.fusion_layer1(concat_features)
            fusion_layer2 = self.view1_net.layer2(fusion_features)
            fusion_layer3 = self.view1_net.layer3(fusion_layer2)
            fusion_layer4 = self.view1_net.layer4(fusion_layer3)

            fusion_avgpool=self.view1_net.avgpool(fusion_layer4)

            fusion_flatten = torch.flatten(fusion_avgpool, 1)

            fusion_all = self.view1_net.fc(fusion_flatten)

            return fusion_all

class resnet_Early_Fusion_PT(nn.Module):
    def __init__(self, num_classes=11, flag_share='False'):
        super(resnet_Early_Fusion_PT, self).__init__()

        if flag_share == 'False':
            print('flag_share == False')

            net1 = resnet18(num_classes=1000)
            net2 = resnet18(num_classes=1000)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pre_path=f"./hub/checkpoints/resnet18-f37072fd.pth"
            checkpoint = torch.load(pre_path, map_location=device)
            missing_keys,unexpected_keys = net1.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)
            missing_keys,unexpected_keys = net2.load_state_dict(checkpoint)
            print(missing_keys,unexpected_keys)

            if num_classes!=1000:
                in_channel = net1.fc.in_features
                net1.fc = nn.Linear(in_channel, num_classes)
                net2.fc = nn.Linear(in_channel, num_classes)

            with torch.no_grad():
                pretrained_conv1 = net1.layer1[1].conv2.weight.detach()
                fusion_layer1 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                fusion_layer1.weight.detach()
                fusion_layer1.weight[:, :64, :, :]=pretrained_conv1
                fusion_layer1.weight[:, 64:, :, :]=pretrained_conv1
                fusion_layer1.weight.requires_grad_()
            fusion_layer1.weight.requires_grad_()

            self.view1_net = net1
            self.view2_net = net2
            self.fusion_layer1 = fusion_layer1

        if flag_share == 'True':
            print('flag_share == True')
            net3 = resnet18(num_classes=1000)

            self.view1_net = net3
            self.view2_net = net3

    def forward(self, x1, x2, double_flag='True'):

        if double_flag=='True':
            x1 = self.view1_net.conv1(x1)
            x1 = self.view1_net.bn1(x1)
            x1 = self.view1_net.relu(x1)
            x1 = self.view1_net.maxpool(x1)
            x1_layer1 = self.view1_net.layer1(x1)

            x2 = self.view2_net.conv1(x2)
            x2 = self.view2_net.bn1(x2)
            x2 = self.view2_net.relu(x2)
            x2 = self.view2_net.maxpool(x2)
            x2_layer1 = self.view2_net.layer1(x2)

            concat_features = torch.cat([x1_layer1,x2_layer1],dim=1)
            fusion_features = self.fusion_layer1(concat_features)
            fusion_layer2 = self.view1_net.layer2(fusion_features)
            fusion_layer3 = self.view1_net.layer3(fusion_layer2)
            fusion_layer4 = self.view1_net.layer4(fusion_layer3)
            fusion_avgpool=self.view1_net.avgpool(fusion_layer4)
            fusion_flatten = torch.flatten(fusion_avgpool, 1)
            fusion_all = self.view1_net.fc(fusion_flatten)

            return fusion_all