import torch
from torch import optim as optim

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def create_optimizer(args, model):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    
    if hasattr(args, 'filter_bias_and_bn'):
        skip = {}
        if args.filter_bias_and_bn:
            skip = {'bias', 'bn'}
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr,
                            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=weight_decay,
                             eps=args.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=weight_decay,
                              eps=args.opt_eps)
    else:
        raise RuntimeError(f'Unknown optimizer: {args.opt}')
        
    return optimizer 