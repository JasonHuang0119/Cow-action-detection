import torch
from .yowo import YOWO
from .loss import build_criterion
from torchsummary import summary

# build YOWO detector
def build_yowo(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=6, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # build YOWO
    model = YOWO(
        cfg = m_cfg,
        device = device,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        topk = args.topk,
        trainable = trainable,
        multi_hot = d_cfg['multi_hot'],
        )
    
    #print(model)
  # 指定文本檔案的路徑
    txt_file_path = "model_summary.txt"

# 使用模型的 __str__ 方法獲取模型的字符串表示形式
    model_summary_str = str(model)

# 將模型字符串表示形式寫入文本文件
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(model_summary_str)

    print(f"Model summary has been written to {txt_file_path}")


    if trainable:
        # Freeze backbone
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        if resume is not None:
            print('keep training: ', resume)
            checkpoint = torch.load(resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # 加載預訓練權重, 我後來有加入 GAT , 不在預訓練權重當中
            model.load_state_dict(checkpoint_state_dict,strict=False)

        # build criterion
        criterion = build_criterion(
            args, d_cfg['train_size'], num_classes, d_cfg['multi_hot'])
    
    else:
        criterion = None
                        
    return model, criterion

