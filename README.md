# Classification

원하는 딥러닝 네트워크 구조를 직접 작성하고, 분류 태스크의 모델을 학습할 수 있는 PyTorch 기반 학습 프레임워크입니다.

`net/`에 모델을 작성하고 YAML에 데이터셋과 학습 조건을 지정하면 학습, 검증, 체크포인트 저장을 실행할 수 있습니다. ImageNet-1K와 CIFAR-10을 지원하며 CPU, 단일 GPU, 다중 GPU의 DataParallel 및 DistributedDataParallel(DDP) 환경에서 사용할 수 있습니다.

## 설치

학습에 필요한 패키지를 Python 환경에 설치합니다. GPU 학습에는 CUDA를 사용할 수 있는 PyTorch 환경이 필요합니다.

```bash
python -m pip install torch torchvision PyYAML numpy
```

이후 명령은 프로젝트 루트에서 실행합니다.

## 빠른 시작: APAGNet으로 CIFAR-10 학습

제공된 [experiments/APAGNet_cifar10.yml](experiments/APAGNet_cifar10.yml)을 사용합니다.

```bash
python main.py --config_path experiments/APAGNet_cifar10.yml
```

이 설정은 CIFAR-10을 `./data/cifar10`에 다운로드하고, 32×32 이미지로 APAGNet을 학습합니다. 클래스 수는 10개로 자동 설정되며 결과는 `./checkpoints/APAGNet-cifar10/`에 저장됩니다.

학습 조건은 실행 전에 YAML에서 변경합니다.

```yaml
dataset: cifar10
data: ./data/cifar10
input_size: 32
download: true
arch: APAGNet
batch_size: 128
epochs: 90
lr: 0.1
save: ./checkpoints
```

## 프로젝트 구조

```text
classification/
├── main.py                         # 설정 로드, 실행 환경 초기화, 학습 실행
├── experiments/
│   ├── config_default.yaml         # 전체 설정의 기본값
│   ├── APAGNet.yml                 # APAGNet + ImageNet-1K
│   └── APAGNet_cifar10.yml          # APAGNet + CIFAR-10
├── net/
│   └── APAGNet.py                  # 전체 네트워크 구조
├── model/
│   └── layers.py                   # 네트워크에서 사용하는 공통 레이어
├── dataset/
│   ├── __init__.py                 # 데이터셋 선택
│   ├── common.py                   # DDP 분할, 배치, worker 설정
│   ├── image_net_1K.py             # ImageNet-1K 로더
│   └── cifar10.py                  # CIFAR-10 로더
├── train_val/
│   └── classification.py           # Trainer, Validationer
└── utils/                          # 학습 및 모델 관련 유틸리티
```

## 원하는 네트워크 작성하기

`net/`에 Python 파일을 만들고, `torch.nn.Module`을 상속하는 `Network` 클래스를 작성합니다. 예를 들어 `net/MyNet.py`를 다음과 같이 구성할 수 있습니다.

```python
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_classes=1000, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)
```

입력은 `[batch_size, 3, height, width]` 형태의 이미지 텐서입니다. 출력은 `[batch_size, num_classes]` 형태의 logits이며, `CrossEntropyLoss`에 전달하므로 모델의 마지막에 Softmax를 적용하지 않습니다.

생성자는 `num_classes`를 받아야 합니다. 프레임워크가 데이터셋에 따라 ImageNet-1K는 1,000개, CIFAR-10은 10개를 전달합니다. 추가 생성자 인자는 YAML의 `model_kwargs`로 전달합니다.

`experiments/MyNet_cifar10.yml`을 만들고 다음 내용을 저장합니다.

```yaml
dataset: cifar10
data: ./data/cifar10
input_size: 32
download: true
arch: MyNet
model_kwargs:
  width: 64
batch_size: 128
epochs: 90
lr: 0.1
save: ./checkpoints
seed: 1024
dist_url: env://
```

```bash
python main.py --config_path experiments/MyNet_cifar10.yml
```

`arch: MyNet`은 자동으로 `net/MyNet.py`의 `Network` 클래스를 불러옵니다. torchvision 모델을 사용하려면 `arch: resnet18`처럼 모델 이름을 지정합니다. 선택한 모델이 설정한 입력 크기를 처리할 수 있도록 구성해야 합니다.

## YAML 설정

기본 설정은 [experiments/config_default.yaml](experiments/config_default.yaml)에서 읽습니다. `--config_path`로 지정한 실험 YAML은 같은 이름의 항목만 덮어쓰므로, 실험 파일에는 바꿀 값만 작성하면 됩니다.

```bash
# 기본 설정으로 실행
python main.py

# 실험 설정을 적용하여 실행
python main.py --config_path experiments/APAGNet.yml
```

기본 설정의 `data`는 ImageNet 경로입니다. 실행 전에 사용하는 컴퓨터의 데이터 경로로 변경합니다. `data`, `save`, `resume`의 상대 경로는 명령을 실행한 작업 디렉터리를 기준으로 해석합니다.

| 설정 | 의미 |
| --- | --- |
| `dataset` | `image_net_1K` 또는 `cifar10` |
| `data` | 데이터셋 저장 위치 |
| `input_size` | 모델에 입력할 정사각형 이미지의 한 변 길이 |
| `download` | CIFAR-10 파일이 없을 때 다운로드할지 여부 |
| `arch` | `net/`의 모델 파일 이름 또는 torchvision 모델 이름 |
| `model_kwargs` | 모델 생성자에 전달할 추가 인자 |
| `epochs` | 학습을 종료할 전체 epoch 수 |
| `batch_size` | 노드당 배치 크기. DDP에서는 로컬 프로세스 수로 나눔 |
| `workers` | DataLoader worker 수. DDP에서는 로컬 프로세스에 분배 |
| `lr`, `momentum`, `weight_decay` | SGD 설정 |
| `lr_step_size`, `lr_gamma` | StepLR의 갱신 주기와 학습률 배율 |
| `seed` | 난수 시드. `null`이면 시드 고정을 적용하지 않음 |
| `save` | 체크포인트 및 로그를 저장할 상위 디렉터리 |
| `resume` | 불러올 체크포인트 경로 |
| `evaluate` | `true`이면 학습 없이 검증만 실행 |
| `gpu` | 사용할 GPU 번호. `null`이면 자동 선택 |
| `no_accel` | `true`이면 CPU 사용 |
| `dummy` | 선택한 데이터셋의 크기와 클래스 수에 맞는 FakeData 사용 |

현재 학습 경로는 `CrossEntropyLoss`, SGD, StepLR을 사용합니다. StepLR은 epoch마다 한 번 갱신합니다. `pretrained: true`는 현재 ImageNet용 torchvision 모델에 적용하며, 직접 작성한 모델의 가중치는 `resume`으로 불러옵니다.

학습 설정은 YAML에서 관리합니다. 명령행 옵션은 설정 파일을 선택하는 `--config_path`이며, `--lr` 같은 개별 학습 옵션은 사용하지 않습니다.

## 데이터셋

### ImageNet-1K

데이터는 다음과 같은 클래스별 폴더 구조로 준비합니다. 학습과 검증의 클래스 폴더 이름은 같아야 합니다.

```text
ImageNet1K/
├── train/
│   ├── n01440764/
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   └── ...
    └── ...
```

`experiments/APAGNet.yml`의 `data`에 데이터 루트를 지정합니다. ImageNet 파일은 직접 준비하며 `download` 설정은 CIFAR-10에 사용합니다.

```yaml
dataset: image_net_1K
data: /path/to/ImageNet1K
input_size: 224
```

학습에는 RandomResizedCrop과 수평 뒤집기를 적용하고, 검증에는 Resize와 CenterCrop을 적용합니다.

### CIFAR-10

공식 학습 분할을 학습용으로, 공식 테스트 분할을 검증용으로 사용합니다. `download: true`이면 torchvision의 CIFAR-10 다운로드 기능을 사용합니다.

학습에는 32×32 RandomCrop과 수평 뒤집기를 적용한 뒤 `input_size`에 맞춰 변환합니다. 두 데이터셋 모두 평균 `[0.485, 0.456, 0.406]`, 표준편차 `[0.229, 0.224, 0.225]`로 정규화합니다.

## 다중 GPU 학습

두 GPU에서 DDP 학습을 실행하는 예시입니다.

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --config_path experiments/APAGNet_cifar10.yml
```

`--nproc_per_node`를 사용할 GPU 수에 맞춥니다. `torchrun`이 프로세스별 GPU, rank, world size를 설정하므로 YAML의 `gpu`는 `null`, `multiprocessing_distributed`는 `false`로 둡니다.

`batch_size`는 로컬 프로세스 수로 나누어떨어져야 합니다. 예를 들어 `batch_size: 128`로 두 GPU를 사용하면 각 프로세스의 배치 크기는 64입니다. 검증 샘플은 프로세스에 중복 없이 분배하고, 결과를 합산하여 전체 검증 정확도를 계산합니다.

`python main.py`로 실행하면 GPU 한 개에서는 단일 GPU를 사용합니다. 여러 GPU가 보이고 `gpu: null`이면 DataParallel을 사용합니다. CPU로 실행하려면 `no_accel: true`를 설정합니다. CPU DDP에는 `dist_backend: gloo`를 사용합니다.

## 학습 재개와 검증

사용 중인 실험 YAML에 체크포인트 경로와 목표 epoch를 설정합니다.

```yaml
resume: ./checkpoints/APAGNet-cifar10/090.pt
epochs: 200
evaluate: false
```

```bash
python main.py --config_path experiments/APAGNet_cifar10.yml
```

모델 가중치, optimizer, scheduler, 마지막 epoch와 최고 정확도를 복원하고 다음 epoch부터 학습합니다. 위 예시는 91번째 epoch부터 200번째 epoch까지 실행합니다. 모델 가중치만 담긴 기존 `model` 형식의 체크포인트를 불러오면 optimizer, scheduler와 epoch는 새로 시작합니다.

학습 없이 평가하려면 같은 실험 YAML에서 다음 값을 설정하고 동일한 명령을 실행합니다.

```yaml
resume: ./checkpoints/APAGNet-cifar10/model_best.pth.tar
evaluate: true
```

## 결과 저장과 학습 흐름

CIFAR-10 예제의 저장 구조는 다음과 같습니다.

```text
checkpoints/APAGNet-cifar10/
├── 001.pt
├── 002.pt
├── ...
├── model_best.pth.tar
├── config.resolved.yml
└── log-APAGNet-cifar10.txt
```

체크포인트는 매 epoch 저장하며, 검증 Top-1 정확도가 최고 기록을 갱신하면 `model_best.pth.tar`도 저장합니다. `config.resolved.yml`에는 실행 시 적용된 설정을 기록합니다. ImageNet은 `APAGNet-imgn/`에 저장하고, FakeData를 사용하면 폴더 이름에 `-dummy`를 붙입니다.

데이터셋 모듈은 `get_train_loader(conf)`, `get_valid_loader(conf)`를 제공하고, 학습과 검증은 다음 인터페이스로 분리되어 있습니다.

```python
trainer = Trainer(conf, train_loader, optimizer, scheduler)
validationer = Validationer(conf, optimizer=optimizer, scheduler=scheduler)

for epoch in range(start_epoch + 1, epochs + 1):
    trainer(model, epoch)
    validationer(model, epoch, valid_loader)
```

`Trainer`는 학습과 학습률 갱신을, `Validationer`는 loss·Top-1·Top-5 집계와 체크포인트 저장을 담당합니다. 검증 통계는 매 epoch 새로 계산합니다.
