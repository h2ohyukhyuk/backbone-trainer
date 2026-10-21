"""ImageNet pretraining backbone matching FaceDetection/net/apag_fpn_net2.py.

Backbone state_dict names match the detector directly. Load with strict=False
in the detector; only classifier and detector-specific FPN/head keys differ.
"""
import torch
import torch.nn as nn
from model.layers import DSConv, RConv1 as ResidualConv1, SumClassificationHead


class Network(nn.Module):
    def __init__(self, num_classes=1000):
        super(Network, self).__init__()

        # 256 -> 64
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2),
                                    nn.Tanh(),
                                    DSConv(16, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        # 64 -> 32
        self.layer2 = nn.Sequential(DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer2_extra = nn.Sequential(ResidualConv1(64,64,1, False),
                                          ResidualConv1(64,64,1, False),
                                          DSConv(64, 128),
                                          nn.ReLU(inplace=True))


        # 32 -> 16
        self.layer3 = nn.Sequential(DSConv(64, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer3_extra = nn.Sequential(ResidualConv1(128,128,1, False),
                                          ResidualConv1(128,128,1, False),
                                          DSConv(128, 128),
                                          nn.ReLU(inplace=True))

        # 16 -> 8
        self.layer4 = nn.Sequential(DSConv(128, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer4_extra = nn.Sequential(ResidualConv1(256, 128, 1, False),
                                          ResidualConv1(128, 128, 1, False),
                                          DSConv(128, 128),
                                          nn.ReLU(inplace=True))
        # 8 -> 4
        self.layer5 = nn.Sequential(DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True))

        # Train every detector backbone branch with ImageNet classification.
        self.classifier = SumClassificationHead(
            in_channels=(256, 128, 128, 128),
            out_channels=num_classes,
        )

    def forward_features(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        c1 = self.layer2_extra(x2)
        x3 = self.layer3(x2)
        c2 = self.layer3_extra(x3)
        x4 = self.layer4(x3)
        c3 = self.layer4_extra(x4)
        c4 = self.layer5(x4)
        return c4, c1, c2, c3

    def forward(self, x):
        return self.classifier(self.forward_features(x))


if __name__ == '__main__':

    from pathlib import Path
    import yaml
    from utils.model_utils import print_flops, print_model_summary
    import time

    # param_path = '../experiments/ShiftNet.yml'
    # with open(param_path) as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cpu'
    model = Network().to(device).eval()

    #input_shape = (1, 3, 224, 224)  # output 12
    input_shape = (1, 3, 256, 384)  # output 12

    input = torch.zeros(size=input_shape).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input)
    end_time = time.perf_counter()

    print((end_time - start_time)*1000, 'ms')


    def print_output(output):
        if isinstance(output, torch.Tensor):
            print('output.shape')
            print(output.shape)
        elif isinstance(output, list) or isinstance(output, tuple):
            for i, tensor in enumerate(output):
                print('output', i, ': ', tensor.shape)


    print_output(output)

    # from rf_calc import receptive_field
    # RF = receptive_field(model, 640)

    print_flops(model, input_shape=input_shape)
    # printmodel_summary(model, input_shape=(1, 3, 256, 256))

    # Requires onnx, onnxscript and onnxruntime.
    onnx_dir = Path(__file__).resolve().parents[1] / 'onnx_'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f'{Path(__file__).stem}.onnx'

    onnx_program = torch.onnx.export(
        model,
        (input,),
        dynamo=True,
        input_names=['image'],
        output_names=['refine_out'],
        optimize=True,
        verify=True,
    )
    onnx_program.save(onnx_path, external_data=False)
    print(f'ONNX model saved to: {onnx_path}')
