import torch
import torch.nn as nn
from model.layers import SumClassificationHead
from model.APAG import APAG


class Network(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.apag = APAG()
        self.classifier = SumClassificationHead(
            in_channels=(256, 128, 128),
            out_channels=num_classes,
        )

    def forward(self, x):
        x4, e2, e3 = self.apag(x)
        return self.classifier([x4, e2, e3])




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
