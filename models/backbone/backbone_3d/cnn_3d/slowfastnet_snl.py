import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from torch.hub import load_state_dict_from_url
from .NonLocalBlock import NonLocalBlock3D
from .triplet_attention_3d import *
from .nls.basic import Stage 
__all__ = ['slowfast50', 'slowfast101','slowfast152', 'slowfast200']



class Bottleneck(nn.Module):
    # 這邊 expansion 是 4, 這代表每個 block 輸出是輸入通道數的 4倍
    expansion = 4 

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1, use_triplet_attention=False):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_triplet_attention:
            self.triplet_attention = TripletAttention(planes*4, 16)
        else:
            self.triplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if not self.triplet_attention is None:
            out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out
    
# Slowfast2*16
# 這邊建立 Slowfast 每一個 block 都是 Bottleneck形式 , layers 分別是 [3,4,6,3] 每個都會做三次 (3+4+6+3)*3 = 48 層 conv 在加上最前面一層 conv 與最後的 FC layer 形成 slowfast
class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5, nl_type=None, nl_nums=None, stage_num=5, div=2, is_sys=True, is_norm=True):
        super(SlowFast, self).__init__()
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) # 改成 4*16修改這邊 stride=(1, 4, 4)
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)
        
        self.lateral_p1 = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32,32*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64,64*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128,128*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))

        self.slow_inplanes = 64+64//8*2 #80 64+64//alpha*t2s_mul (取樣64偵間隔8 每次取2)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        '''
        新增 NonLocalBlock3D
        在每一層 resnet 後面加上 Non-local block 去強化空間特徵 (Slow Path)
        '''

        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.nonlocal_block_slow2 =  NonLocalBlock3D(256,use_cbam=True)
        
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.nonlocal_block_slow3 =  NonLocalBlock3D(512,use_cbam=True)

        if not nl_nums:
            self.slow_res4 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        else:
            self.slow_res4 = self._make_layer_slow(block, 256, layers[2], stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div, is_sys=is_sys, is_norm=is_norm)    
        self.nonlocal_block_slow4 =  NonLocalBlock3D(1024,use_cbam=True)

        if nl_nums != 5:
            self.slow_res5 = self._make_layer_slow(block, 512, layers[3], stride=2, head_conv=3)
        else:
            self.slow_res5 = self._make_layer_slow(block, 512, layers[3], stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div, is_sys=is_sys, is_norm=is_norm)

        
        #self.dp = nn.Dropout(dropout)
        #self.fc = nn.Linear(self.fast_inplanes+2048, class_num, bias=False)

    def forward(self, input):
        fast, lateral = self.FastPath(input[:, :, ::4, :, :])
        slow = self.SlowPath(input[:, :, ::32, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        #x = self.dp(x)
        #x = self.fc(x)
        #x.shape (batch_size,2304,7,7)
        return x



    def SlowPath(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]],dim=1)
        
        x = self.slow_res2(x)
        x = self.nonlocal_block_slow2(x)

        x = torch.cat([x, lateral[1]],dim=1)
        
        x = self.slow_res3(x)
        x = self.nonlocal_block_slow3(x)  # 將 NonLocalBlock3D 放在這裡

        x = torch.cat([x, lateral[2]],dim=1)
        
        x = self.slow_res4(x)
        x = self.nonlocal_block_slow4(x)  # 將 NonLocalBlock3D 放在這裡

        x = torch.cat([x, lateral[3]],dim=1)
        x = self.slow_res5(x)

    
        #x = self.nonlocal_block_slow5(x)
        
        # ===============================================
        # 在第三個維度上取平均值, 代表將 4 個 frame 的特徵取平均 (Slow,進來是每一次吃 4 frames) K
        # ===============================================

        if x.size(2) > 1:
            x = torch.mean(x, dim=2, keepdim=True) 

        '''
        這裡有修改過
        下面的 x.squeeze(2) 會把 x.size()從 [4,2048,1,7,7] -> [4,2048,7,7] 
        x = nn.AdaptiveAvgPool3d(1)(x)
        print('slow x size1:',x.size())
        x = x.view(-1, x.size(1)) 這邊原本是要做 fc (flatten) 我這邊不接 fc layer 直接拿slowfast5 layer output 來用
        print('slow x size:',x.size())
        '''  
        return x.squeeze(2)

    # 有修改過
    def FastPath(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)
        
        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        if res5.size(2) > 1:
            x = torch.mean(res5, dim=2, keepdim=True)
        
        '''
        x = nn.AdaptiveAvgPool3d(1)(res5)
        print('x',x.size())
        x = nn.AdaptiveAvgPool3d((7, 7, 7))(res5)  # 修改此行
        x = x.view(-1, x.size(1))
        print(x.size())
        '''
        
        return x.squeeze(2), lateral
    
    def _addNonlocal(self, in_planes, sub_planes, nl_type='snl', stage_num=None, is_sys=True, is_norm=True):
        if nl_type == 'snl':
            return Stage(
                in_planes, sub_planes, stage_num=stage_num, nl_type=nl_type, is_sys=is_sys, is_norm=is_norm)

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1, nl_nums=None, nl_type='snl', stage_num=None, is_sys=True, is_norm=True, div=2):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion

        sub_planes = int(self.slow_inplanes / div)
        for i in range(1, blocks):
            #######Add Nonlocal Block#######
            if nl_nums == 1 and (i == 5 and blocks == 6) or (i == 22 and blocks == 23) or (i == 35 and blocks == 36):
                layers.append(self._addNonlocal(self.slow_inplanes,sub_planes, nl_type=nl_type, stage_num=stage_num, is_sys=is_sys, is_norm=is_norm))
            if nl_nums == 5 and (stride == 2 and ((i == 1 and blocks == 6) or (i == 3 and blocks == 6) or (i == 5 and blocks == 6) or (i == 1 and blocks ==3))):
                layers.append(self._addNonlocal(self.slow_inplanes, sub_planes, nl_type=nl_type, stage_num=stage_num, is_sys=is_sys, is_norm=is_norm))
            
            #######Add Res Block#######
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

            if nl_nums == 5 and stride == 2 and blocks==3:
                layers.append(self._addNonlocal(self.slow_inplanes, sub_planes, nl_type, stage_num))
  
        self.slow_inplanes = planes * block.expansion + planes * block.expansion//8*2
        return nn.Sequential(*layers)



def slowfast50(pretrained=False,**kwargs):
    """Constructs a slowfast-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def slowfast101(pretrained=False,**kwargs):
    """Constructs a slowfast-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], nl_nums=1, nl_type='snl',is_norm=True,is_sys=True, **kwargs)
    return model


def slowfast152(pretrained=False,**kwargs):
    """Constructs a slowfast-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def slowfast200(pretrained=False,**kwargs):
    """Constructs a slowfast-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model





# 建立 slowfast 模型根據模型名稱選擇不同大小的 model
def build_slowfast_3d(model_name='slowfast50', pretrained=False, **kwargs):
    if model_name == 'slowfastnet':
        model = slowfast50(pretrained=pretrained, **kwargs)
        feats = 2304
    elif model_name == 'slowfast101':
        model = slowfast101(pretrained=pretrained, **kwargs)
        feats = 2304
    elif model_name == 'slowfast152':
        model = slowfast152(pretrained=pretrained, **kwargs)
        feats = 2304
    elif model_name == 'slowfast200':
        model = slowfast200(pretrained=pretrained, **kwargs)
        feats = 2304
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    return model, feats


if __name__ == "__main__":
    import time

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #測試資料
    num_classes = 6 # slowfast 80 類, 這邊自定義只有 6 類
    model, feats = build_slowfast_3d(model_name='slowfastnet',pretrained=False,class_num = num_classes)
    model.to(device)
    summary(model=model,input_size=(4,3,16,224,224))
    x = torch.autograd.Variable(torch.rand(4, 3, 64, 224, 224).to(device)) #影片輸入 (bacth_size,channels,T frames,Height,Width) (train.py 中是 (4,3,16,224,224))
    
    #start time
    t0 = time.time()
    # inference
    output = model(x)
    for y in output:
        pass
    #end time
    print('Inference time: {}'.format(time.time() - t0))
    

    # FLOPs & Params
    #print('==============================')
    #flops, params = profile(model, inputs=(x, ), verbose=False)
    #print('==============================')
    #print('GFLOPs : {:.2f}'.format(flops / 1e9))
    #print('Params : {:.2f} M'.format(params / 1e6))
