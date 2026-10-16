import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def set_network_grad_by_keyword(model, keyword, grad=True, printInfo = False):
    print("keyword:", keyword, " Grad:", grad)
    for name, p in model.named_parameters():
        if keyword in name:
            if printInfo:
                print ("Freezing layer : ", name)
            p.requires_grad = grad
        elif printInfo:
            print("Non freezing layer : ", name)

    if grad == False:
        print("Batchnorm will be freezed. keyword:", keyword, " Grad:", grad)
        for name, p in model.named_modules():
            if keyword in name:
                if isinstance(p, nn.BatchNorm2d):
                    p.eval()

class ESCondition(Enum):
    VALID = 0
    INVALID = 1
    STOP = 2


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            print("Loss is improved. Best: %.6f" % (self.best))
            return ESCondition.VALID

        if self.is_better(metrics, self.best):
            print("Loss is improved. Best: %.6f" % (self.best))
            self.num_bad_epochs = 0
            self.best = metrics
            return ESCondition.VALID
        else:
            self.num_bad_epochs += 1
            print("Loss is not improved. Best: %.6f Current: %.6f" %
                  (self.best, metrics))
            print("Patience: %d/%d" % (self.num_bad_epochs, self.patience))

            if self.num_bad_epochs >= self.patience:
                print("Procedure is ealy stopped.")
                return ESCondition.STOP

            return ESCondition.INVALID

        return 0

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta /
                                                             100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta /
                                                             100)

def change_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_lr_dict_list(model, base_lr, keywords, mult, use_lr_mult=False):
    dict_list = []
    name_list = []

    if use_lr_mult is False:
        named_dict = {}
        named_dict["params"] = model.parameters()
        named_dict["lr"] = base_lr
        dict_list.append(named_dict)
        print("- lr multiply is not applied")
        return dict_list

    print("- lr multiply is applied")
    print("-- base lr: ", base_lr)
    for name, p in model.named_parameters():
        if p.requires_grad:
            tmp_name = name.split('.')[0]
            if tmp_name not in name_list:
                name_list.append(tmp_name)

    for i in range(len(name_list)):
        named_dict = {}
        idx = -1
        new_lr = base_lr

        for j in range(len(keywords)):
            if keywords[j] in name_list[i]:
                idx = j
                new_lr = base_lr * mult[idx]

        print("-- group name: ", name_list[i], " new lr: ", new_lr)
        named_dict["params"] = getattr(model, name_list[i]).parameters()
        named_dict["lr"] = new_lr
        dict_list.append(named_dict)

    print(dict_list)
    return dict_list

def get_optimizer(target_model, **kwargs):
    opt_name = kwargs.pop('OPT_NAME', None)
    base_lr = kwargs.pop('LR', None)
    momentum = kwargs.pop('MOMENTUM', 0.9)
    use_lr_mult = kwargs.pop('USE_LR_MULT', None)
    lr_mult_target = kwargs.pop('LR_MULT_TARGET', None)
    lr_mult_val = kwargs.pop('LR_MULT_VAL', None)
    amsgrad = kwargs.pop('AMS_GRAD', False)
    wd = kwargs.pop('WEIGHT_DECAY', 0)
    kwargs.pop('LR_SCHEDULER', None)
    others = kwargs

    params_dict = make_lr_dict_list(model=target_model,
                                    base_lr=base_lr,
                                    keywords=lr_mult_target,
                                    mult=lr_mult_val,
                                    use_lr_mult=use_lr_mult)
    if opt_name == "adam":
        opt = torch.optim.Adam(params_dict, base_lr, amsgrad=amsgrad,
                               weight_decay=wd, **others)
    elif opt_name == 'adamw':
        opt = torch.optim.AdamW(params_dict, lr=base_lr, amsgrad=amsgrad,
                                      weight_decay=wd, **others)
    elif opt_name == "rmsprop":
        opt = torch.optim.RMSprop(params_dict, lr=base_lr,
                                  weight_decay=wd, **others)
    elif opt_name == "adadelta":
        opt = torch.optim.Adadelta(params_dict, lr=base_lr,
                                   weight_decay=wd, **others)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(params_dict, base_lr, momentum=momentum,
                              weight_decay=wd, **others)
    else:
        print("ERROR: Unknown optimizer.")
        print("Rquested optimizer is ", opt_name)
        exit(1)

    return opt

def get_scheduler(opt, conf):

    sch_name = conf['OPTIMIZER']['LR_SCHEDULER']['SCHEDULER_NAME']

    if sch_name == "steplr":
        step_size = conf['OPTIMIZER']['LR_SCHEDULER']['step_size']
        gamma = conf['OPTIMIZER']['LR_SCHEDULER']['gamma']
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma)
    elif sch_name == "multisteplr":
        milestones = conf['OPTIMIZER']['LR_SCHEDULER']['milestones']
        gamma = conf['OPTIMIZER']['LR_SCHEDULER']['gamma']
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma)
    # elif sch_name == "plateau":
    #     sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **others)
    elif sch_name == 'poly':
        from utils.custom_lr_scheduler import PolynomialLRDecay
        lr = conf['OPTIMIZER']['LR']
        power = conf['OPTIMIZER']['LR_SCHEDULER']['POWER']
        max_steps = int(conf['NUM_TRAIN_STEPS'] * conf['TRAIN']['END_EPOCH'])
        sch = PolynomialLRDecay(opt, max_steps, power)
    else:
        print("Unknown scheduler.")
        print("Rquested scheduler is ", sch_name)
        print("Return None")
        return None

    return sch

def set_backend_random_seed(conf):
    train_conf = conf['TRAIN']
    torch.manual_seed(train_conf["SEED_NUMBER"])
    np.random.seed(train_conf["SEED_NUMBER"])
    torch.backends.cudnn.deterministic = train_conf["CUDNN_DETER"]
    torch.backends.cudnn.benchmark = train_conf["CUDNN_BENCH"]
