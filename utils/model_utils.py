
import torch
from thop import profile
from utils.train_utils import set_network_grad_by_keyword
import os
import errno
from urllib.parse import urlparse 
import sys

def print_flops(model, input_shape=(1, 3, 256, 256)):

    input_sample = torch.randn(input_shape)
    total_ops, total_params = profile(model, (input_sample,), verbose=False)
    print("%s | %s | %s" % ("Params(M)", "FLOPs(G)", "FLOPs(M)"))
    print("%.2fM | %.2fG | %.2fM" % (total_params / (1000 ** 2), total_ops / (1000 ** 3), total_ops / (1000 ** 2)))


def print_model_summary(model, input_shape=(1, 3, 256, 256)):

    try:
        from torchsummaryX import summary
        input_sample = torch.randn(input_shape)
        summary(model, input_sample)
    except:
        from torchsummary import summary
        summary(model, input_shape[1:], device='cpu')

def get_model(conf, print_model_info=False):

    net_name = conf['MODEL']['MODEL_NAME']
    exec('from net.' + net_name + ' import Network')
    model = eval('Network(conf=conf)')

    if print_model_info:
        print('Net name: ', net_name, '-' * 50)
        input_h, input_w = conf['MODEL']['INPUT_SIZE']
        print_flops(model, input_shape=(1, 3, input_h, input_w))
        print_model_summary(model, input_shape=(1, 3, input_h, input_w))

    device = torch.device("cuda:0" if conf['DEVICE'] == 'GPU' else "cpu")

    model = model.to(device)

    return model

def freeze_model(conf, model, printInfo=False):

    config_train = conf['TRAIN']
    if config_train["FREEZE"]:
        for freeze_target in config_train["FREEZE_TARGET"]:
            set_network_grad_by_keyword(model, freeze_target,
                                        grad=False, printInfo=printInfo)

    return model

def load_check_point(conf, model):

    train_conf = conf['TRAIN']
    USE_CKPT = train_conf["PREWEIGHT_INFO"]["USE_CKPT"]

    if USE_CKPT and train_conf["PREWEIGHT_INFO"]["LOCAL_CKPT"] != '':
        path_checkpoint = train_conf["PREWEIGHT_INFO"]["LOCAL_CKPT"]
        state = torch.load(path_checkpoint, map_location=conf['DEVICE'])
        load_result = model.load_state_dict(state['model'], strict=False)
        print('- load model from: %s' % path_checkpoint)
        print('-- missing keys')
        for miss in load_result[0]:
            print(miss)
        print('-- unexpected keys')
        for unexpected in load_result[1]:
            if 'total_' not in unexpected:
                print(unexpected)
        print('model loaded!')

    return model

def load_model_from_path(path_torch_weight, torch_model, is_cpu=True, strict=True):
    map_location = torch.device('cpu') if is_cpu else torch.device('cuda')
    state = torch.load(path_torch_weight, map_location=map_location)
    load_result = torch_model.load_state_dict(state['model'], strict=strict)
    print('- load model form %s' % path_torch_weight)
    print('-- missing keys')
    for miss in load_result[0]:
        print(miss)
    print('-- unexpected keys')
    for unexpected in load_result[1]:
        if 'total_' not in unexpected:
            print(unexpected)
    print('model loaded!')

    return torch_model

MODELS_BACKBONE_WEIGHT = ['mnasnet_050', 'mnasnet_075', 'mnasnet_small_025', 'mnasnet_small_050', 'mnasnet_small_100', 
'mobilenet_v3_small_010_relu', 'mobilenet_v3_small_025_relu', 'mobilenet_v3_small_050_relu', 'semnasnet_050', 
'mobilenet_v3_small_050_rt_cocoseg']

IGNORE_LAYERS = {
    "mobilenet_v3_small_050_rt_cocoseg" : ["classifier.low_classifier.weight", "classifier.low_classifier.bias", 
    "classifier.high_classifier.weight", "classifier.high_classifier.bias"]
}

def get_ratio(name: str) -> float:
    ratio = int(name[-3:]) / 100.0
    return ratio


def load_backbone(model, name: str, pretrained: bool):
    if name in MODELS_BACKBONE_WEIGHT and pretrained:
        print ("Loading backbone weights ....")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sd = load_state_dict_from_url('https://pages.oss.navercorp.com/Team3DAvatar/LightweightBackbone/' + name + '.pth.tar', map_location=device)
        key = "state_dict"
        if key not in sd:
            key = "model"
        dict = sd[key]
        if name in IGNORE_LAYERS:
            for layer_to_ignore in IGNORE_LAYERS[name]:
                print ("Ignore layer ", layer_to_ignore)
                dict.pop(layer_to_ignore)
        
        model.load_state_dict(dict, strict=False)

def load_state_dict_from_url(url, map_location=None, progress=True):
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)
    
    return torch.load(cached_file, map_location=map_location)
