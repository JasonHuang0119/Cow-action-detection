import torch
from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            eight_decay=cfg['weight_decay'])
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=cfg['weight_decay'])
          
    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        #print(checkpoint)
        # checkpoint state dict
        # checkpoint_state_dict = checkpoint.pop("optimizer")
        if 'optimizer' in checkpoint:
            checkpoint_state_dict = checkpoint.pop("optimizer")
            optimizer.load_state_dict(checkpoint_state_dict)
        else:
            print(cfg['optimizer'],optimizer)
            print("Warning: 'optimizer' key not found in checkpoint. Optimizer not loaded.")

        #print(checkpoint_state_dict)
        #optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
                        
                                
    return optimizer, start_epoch
