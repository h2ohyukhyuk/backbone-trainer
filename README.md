# Backbone Trainer

직접 작성한 이미지 분류 모델을 ImageNet-1K 또는 CIFAR-10으로 학습하는 PyTorch 프레임워크입니다.
모델은 `net/`, 학습 설정은 `experiments/*.yml`에서 관리합니다.
단일 GPU, 멀티 GPU, 멀티 노드 DDP, AMP와 ClearML 기록을 지원합니다.

## 목차

- [설치와 빠른 시작](#설치와-빠른-시작)
- [설정과 데이터 준비](#설정과-데이터-준비)
- [GPU 및 노드 구성](#gpu-및-노드-구성)
- [ClearML 원격 실행](#clearml-원격-실행)
- [AMP와 배치 크기](#amp와-배치-크기)
- [옵티마이저](#옵티마이저)
- [학습률 스케줄러](#학습률-스케줄러)
- [기록과 체크포인트](#기록과-체크포인트)
- [새 모델 추가](#새-모델-추가)
- [프로젝트 구조와 테스트](#프로젝트-구조와-테스트)

## 설치와 빠른 시작

모든 명령은 프로젝트 루트에서 실행합니다.
CUDA 12.8 환경의 설치 예시입니다. 다른 환경에서는 GPU 드라이버에 맞는 PyTorch 빌드를 사용하세요.

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### CIFAR-10으로 시작

```bash
python main.py --config_path experiments/APAGNet_cifar10.yml
```

CIFAR-10을 `./data/cifar10`에 다운로드하고 32×32 이미지로 학습합니다.
결과는 `/home/j/models_cla/APAGNet-cifar10/`에 저장됩니다. 일반 Python 실행은 ClearML을 사용하지 않습니다.

### ImageNet-1K + 멀티 GPU + ClearML

```bash
python main.py --config_path experiments/config.yml
```

이 설정은 **현재 컴퓨터의 모든 visible GPU에 DDP 프로세스를 하나씩 실행**합니다.
실행 전에 데이터 경로를 준비하세요. 일반 `python main.py` 실행은 로컬 로그, `.pt` 체크포인트와 사용한 config를
저장하며 ClearML 서버에 연결하지 않습니다. ClearML은 `clearml-task`로 제출한 원격 실행에서만 사용합니다.

## 설정과 데이터 준비

### YAML 적용 규칙

[experiments/config_default.yaml](experiments/config_default.yaml)을 먼저 읽고,
선택한 실험 YAML의 같은 항목으로 덮어씁니다.
**다른 실험 YAML을 상속하지는 않습니다.**

```bash
python main.py --config_path experiments/config.yml
```

중첩 딕셔너리는 구역 전체가 교체됩니다. 예를 들어 `LAMBDALR_KWARGS`를 수정할 때는
필요한 키를 포함한 구역 전체를 작성하세요.
`data`, `save`, `resume`의 상대 경로는 현재 작업 디렉터리를 기준으로 합니다.

| 설정 | 의미 |
|---|---|
| `dataset` | `image_net_1K` 또는 `cifar10` |
| `data` / `input_size` | 데이터 루트 / 정사각형 입력 크기 |
| `arch` / `model_kwargs` | 모델 이름 / 추가 생성자 인자 |
| `epochs` | 학습을 종료할 전체 epoch |
| `batch_size` | **노드당** 배치 크기 |
| `workers` | 노드당 DataLoader worker 수; DDP에서 로컬 프로세스에 분배 |
| `OPT_NAME` / `LR` | 옵티마이저 / 기준 학습률 |
| `lr_scheduler` | 스케줄러 선택 |
| `amp` / `amp_dtype` | CUDA 혼합 정밀도 설정 |
| `save` / `resume` | 결과 루트 / 재개할 체크포인트 |
| `seed` | 정수 또는 `null` |
| `dummy` | `true`이면 FakeData 사용 |

학습 조건은 YAML에서 바꿉니다. `-a`, `--lr`, `--world-size` 같은 개별 학습 CLI 옵션은 지원하지 않습니다.
CLI는 `--config_path`, `--plot-lr`을 사용합니다.

### ImageNet-1K

```text
ImageNet1K/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

train/val의 클래스 폴더 이름은 같아야 합니다.

```yaml
dataset: image_net_1K
data: /home/j/database/ImageNet1K
input_size: 224
```

학습에는 RandomResizedCrop, 수평 뒤집기, RandAugment, RandomErasing을 적용합니다.
실제 ImageNet 학습에서는 10% 확률로 MixUp 또는 CutMix를 적용합니다.
검증에는 Resize와 CenterCrop을 사용합니다.

### CIFAR-10

```yaml
dataset: cifar10
data: ./data/cifar10
input_size: 32
download: true
```

공식 train 분할로 학습하고 test 분할로 검증합니다.
학습에는 32×32 RandomCrop과 수평 뒤집기를 사용한 뒤 `input_size`로 변환합니다.
두 데이터셋 모두 평균 `[0.485, 0.456, 0.406]`, 표준편차 `[0.229, 0.224, 0.225]`로 정규화합니다.
ImageNet 데이터는 직접 준비하며 `download`는 CIFAR-10에만 적용합니다.

### Augmentation 조절

`config_default.yaml`의 `AUGMENTATION`에서 조절합니다. 실제 학습에 적용된 값은
ClearML `Training/AUGMENTATION/...`과 `.pt`의 `config.AUGMENTATION`에 함께 기록됩니다.

| 구역 | 주요 설정 |
|---|---|
| `image_net_1K` | crop 면적·종횡비, flip 확률, RandAugment, RandomErasing |
| `cifar10` | native 32x32 crop 사용·padding, flip 확률, RandAugment, RandomErasing |
| `batch` | 적용 데이터셋, 전체 적용 확률, MixUp/CutMix alpha와 선택 비율 |

- `horizontal_flip_p: 0`, `random_erasing_p: 0`으로 해당 변환을 끕니다.
- `randaugment_enabled: false`로 RandAugment를 끕니다.
- `batch.p: 0`으로 MixUp/CutMix를 끕니다. `mixup_weight: 1`은 MixUp만, `0`은 CutMix만 사용합니다.
- 기본 batch 확률은 0.1, 선택 비율은 0.5이므로 MixUp 5%, CutMix 5%, 혼합 없음 90%입니다.
- validation에는 augmentation을 적용하지 않으며 `dummy` 데이터에서도 augmentation을 생략합니다.
- 실험 YAML에서 덮어쓸 때는 `AUGMENTATION` 구역 전체를 복사해 수정하세요.

## GPU 및 노드 구성

### 실행 방식 선택

| 환경 | 실행 방법 | 핵심 설정 |
|---|---|---|
| CPU / 단일 GPU | `python main.py` | `multiprocessing_distributed: false` |
| 단일 노드 · 멀티 GPU DDP | `python main.py` 자동 spawn | `multiprocessing_distributed: true`, `num_nodes: 1`, `node_rank: 0` |
| 멀티 노드 DDP | 노드마다 `python main.py` | 공통 master 주소, 노드 수, 서로 다른 node rank |
| torchrun | 노드마다 `torchrun ... main.py` | `multiprocessing_distributed: false`; 실행기가 rank 설정 |

`gpu: null`은 자동 선택입니다. `CUDA_VISIBLE_DEVICES`로 노출할 GPU를 제한할 수 있습니다.
일반 `python` 실행에서 spawn을 끄고 여러 GPU를 노출하면 DataParallel을 사용합니다.
CPU는 `no_accel: true`, CPU DDP는 `dist_backend: gloo`를 사용합니다.

### 단일 노드 · 멀티 GPU: 자동 spawn

현재 `experiments/APAGNet.yml`에 적용된 방식입니다.

```yaml
multiprocessing_distributed: true
num_nodes: 1
node_rank: 0
gpu: null
dist_backend: nccl
dist_url: tcp://127.0.0.1:29500
batch_size: 256
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path experiments/config.yml
```

두 GPU에 프로세스를 하나씩 띄우고 GPU당 128개 샘플로 학습합니다.
`29500`은 예시 포트이며 같은 컴퓨터에서 다른 학습이 사용 중이면 변경하세요.

### 멀티 노드: 노드마다 자동 spawn

예시: **컴퓨터 2대 × GPU 2장 = 총 4개 프로세스**.
두 노드에서 같은 코드, 모델·데이터·학습 설정을 사용하고 visible GPU 수를 맞춥니다.

다음 파일들을 `experiments/`에 작성합니다. 각 파일은 기본 YAML을 덮어씁니다.
`192.168.0.10`은 Node 0의 실제 접근 가능한 IP로 바꾸세요.

**Node 0 — `experiments/APAGNet_node0.yml`**

```yaml
arch: APAGNet
data: /home/j/database/ImageNet1K
save: /home/j/models_cla/
multiprocessing_distributed: true
num_nodes: 2
node_rank: 0
gpu: null
dist_backend: nccl
dist_url: tcp://192.168.0.10:29500
batch_size: 256
amp: true
```

**Node 1 — `experiments/APAGNet_node1.yml`**

같은 내용을 복사하고 `node_rank`만 `1`로 바꿉니다.
데이터 저장 위치가 다르면 각 노드의 `data`도 맞추되 데이터 내용과 클래스는 동일해야 합니다.

```bash
# Node 0에서 실행
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path experiments/APAGNet_node0.yml

# Node 1에서 실행
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path experiments/APAGNet_node1.yml
```

두 명령이 모두 실행되어야 초기화가 완료됩니다.
master 주소에 `127.0.0.1`을 쓰면 다른 컴퓨터가 연결할 수 없습니다.
노드 간 master 포트 및 NCCL 통신이 가능해야 하며 컨테이너도 서로 접근할 수 있는 네트워크가 필요합니다.
모델 체크포인트와 집계 지표는 global rank 0에서 저장·기록합니다.

### rank와 프로세스 수: 명칭 구분

| 이름 | 의미 | 설정 위치 |
|---|---|---|
| `num_nodes` | 전체 서버 수 | YAML (torchrun은 자동 계산) |
| `node_rank` | 서버 번호 | YAML (torchrun은 자동 계산) |
| `processes_per_node` | 서버당 GPU 프로세스 수 | visible GPU 수 또는 torchrun |
| `local_rank` | 서버 안의 프로세스 번호 | 런타임 자동 배정 |
| `total_processes` | 전체 프로세스 수 | `num_nodes × processes_per_node` |
| `global_rank` | 전체 프로세스 번호 | `node_rank × processes_per_node + local_rank` |

YAML의 기존 `world_size`/`rank` 키는 `num_nodes`/`node_rank`로 변경했습니다.
코드에서 `node_rank`를 `global_rank`로 덮어쓰지 않습니다.
출력·로그·파일 저장·ClearML 기록 기준은 언제나 `global_rank == 0`입니다.
PyTorch API 인자 `world_size`, `rank`에는 각각 `total_processes`, `global_rank`를 전달합니다.

### 배치 크기와 rank의 의미

spawn 방식에서 YAML의 `num_nodes`는 **노드 수**, `node_rank`는 **노드 번호**입니다.
내부에서 이를 전체 GPU 프로세스 수와 global rank로 변환합니다.

| 구성 | `batch_size` | GPU당 배치 | 전체 배치 |
|---|---:|---:|---:|
| 1노드 × 2GPU | 256 | 128 | 256 |
| 2노드 × 2GPU | 256 | 128 | 512 |
| 2노드 × 2GPU | 128 | 64 | 256 |

`batch_size`는 노드 내 GPU 프로세스 수로 나누어떨어져야 합니다.
노드 수를 늘리면서 전체 배치를 유지하려면 노드당 배치를 줄이세요.
검증 데이터는 전체 프로세스에 중복 없이 분배하고 결과를 합산합니다.

### torchrun으로 실행

실험 YAML에서 다음 값을 설정합니다. **자동 spawn과 동시에 사용하면 안 됩니다.**

```yaml
multiprocessing_distributed: false
gpu: null
batch_size: 256
```

```bash
# 단일 노드, GPU 2장
torchrun --standalone --nproc_per_node=2 main.py \
  --config_path experiments/APAGNet_cifar10.yml
```

멀티 노드에서는 위에서 만든 node별 YAML의 `multiprocessing_distributed`를 `false`로 바꾸고 실행합니다.

```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
  --master_addr=192.168.0.10 --master_port=29500 main.py \
  --config_path experiments/APAGNet_node0.yml

# Node 1
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
  --master_addr=192.168.0.10 --master_port=29500 main.py \
  --config_path experiments/APAGNet_node1.yml
```

이 방식은 `torchrun`이 `WORLD_SIZE`, `RANK`, `LOCAL_RANK`를 설정합니다.
코드는 YAML의 num_nodes/node_rank보다 실행기가 제공하는 값을 우선합니다.

## ClearML 원격 실행

### 단일 노드 · 멀티 GPU

`APAGNet.yml`은 자동 spawn을 사용하는 학습 설정입니다.
ClearML 연결 여부는 YAML이 아니라 원격 실행 환경의 `CLEARML_TASK_ID`로 판단합니다.
project와 task 이름은 `clearml-task --project ... --name ...`에서 지정합니다.
다음 명령은 작업 하나를 queue에 제출하며, 해당 agent가 실행하는 컴퓨터의 visible GPU를 사용합니다.

```bash
clearml-task \
  --project APAG \
  --name apag-baseline \
  --repo git@github.com:h2ohyukhyuk/backbone-trainer.git \
  --branch main \
  --script main.py \
  --docker clearml-local/torch:2.9.1-py3.12-cu128 \
  --queue defalult \
  --args config_path=experiments/config.yml
```

실행 준비 사항:

- 변경 코드와 YAML이 원격 `main` branch에 있어야 합니다.
- agent가 해당 queue를 수신해야 합니다. `defalult`는 사용하는 실제 queue 이름과 맞춥니다.
- agent에 사용할 GPU가 모두 노출되어 있어야 합니다.
- ClearML 서버 인증과 Git SSH 접근을 agent 환경에 설정합니다.
- 데이터 경로를 Docker 내부에 마운트하고 DataLoader용 shared memory를 확보합니다.

단일 노드의 Docker 옵션 예시:

```text
--ipc=host -v /home/j/database:/home/j/database:ro
```

`requirements.txt`는 Docker에 맞춰 torch 2.9.1 / torchvision 0.24.1을 사용합니다.

### HPO 실행에서만 원격 hyperparameter 적용

일반 원격 실행과 로컬 실행은 YAML을 기준으로 합니다. 명시적으로 `--hpo`를 켠 원격 실행에서만
ClearML의 `Training/...` 값을 내부 config 복사본에 반영합니다. 원본 YAML 파일은 수정하지 않습니다.
예를 들어 HPO 기준 task를 제출할 때 기존 명령의 인자에 다음을 사용합니다.

```text
--args config_path=experiments/config.yml hpo=true
```

HPO가 복제하는 task에는 `Args/hpo: true`가 유지되어야 합니다.
탐색 변수는 예를 들어 `Training/LAMBDALR_KWARGS/points/15`,
`Training/LAMBDALR_KWARGS/points/45`로 지정합니다. 숫자 epoch 키도 보존합니다.
학습·optimizer·스케줄러·augmentation 설정을 탐색할 수 있으며 데이터/저장 경로와 rank·통신 설정은
YAML/실행기가 계속 관리합니다. 일반 task에서는 UI에서 값을 바꿔도 학습에 반영되지 않습니다.

적용된 최종 config를 다시 검증하고 ClearML hyperparameters, `config.resolved.yml`, `.pt`의 `config`와
Registry 설정에 기록합니다. 원본 config 복사본은 수정 전 YAML이므로 HPO 실행의 실제 값은
`config.resolved.yml`을 확인하세요. 로그에도 변경 전후 값을 기록합니다.
DDP spawn 이전에 적용하고 torchrun의 다른 rank는 task를 읽기만 합니다.
HPO 값 수신은 학습 코드가 담당하며 `hpo-lr.py`가 단계별 Grid Search를 실행합니다.

### 단계별 LR Grid Search

`hpo-lr.py`는 단계별 Grid Search이며 이전 실험과 중복되는 설정은 재사용합니다.

| 단계 | 탐색 값 | 새 학습 수 |
|---|---|---|
| 1 | scale `[0.1, 0.325, 0.55, 0.775, 1.0]` | 최대 5 |
| 2 | 15·20 epoch LR 각각 `[0.008, 0.007, 0.006]` | 최대 8 |
| 3 | 43·48 epoch LR 각각 `[0.001, 0.002, 0.003]` | 최대 8 |

2단계와 3단계의 기준값이 후보에 포함되면 최대 21회입니다.
각 단계의 최고 validation Top-1 설정을 다음 단계에 고정합니다.
각 trial은 같은 seed로 처음부터 학습하며 이전 가중치는 이어받지 않습니다.

```bash
# 계획 확인
python hpo-grid-lr.py --config-path experiments/config.yml
# 완료된 기준 task를 복제하여 실행
python hpo-grid-lr.py --execute --base-task-id TASK_ID --queue defalult
```

`--scale-values`로 scale 후보를 변경할 수 있습니다.
진행 상태는 `outputs/hpo/lr_search.json`에 저장하며 같은 명령으로 재개합니다.
다른 기준 task나 탐색 조건을 사용할 때는 별도 `--state`를 지정하세요.
실제 실행은 기준 task의 코드·설정·Docker를 복제하며 현재 controller는 단일 노드용입니다.
기준 task는 `lambdalr`, 명시적인 seed, 빈 resume, start_epoch 0, evaluate false여야 합니다.
15/20/43/48 point가 있어야 하며 controller는 `Args/hpo=true`를 자동 지정합니다.
controller는 실행 상태를 유지해야 하고 완료 결과는 `hpo_results` artifact로 기록합니다.

ClearML 기본 BO는 별도 파일
`/home/j/projects_mldl/backbone-trainer/hpo-bo-lr.py`를 사용합니다.

### Best model과 Model Registry

ClearML 원격 실행의 global rank 0에서 validation Top-1 최고 기록을 갱신할 때만
`model_best.pt`를 업로드합니다. epoch별 `.pt`는 `save` 경로에만 저장합니다.
모델 설명에는 해당 checkpoint의 validation Top-1, Top-5와 epoch를 기록합니다.
모델 이름은 `<프로젝트>/<task 이름>/best`이며 해당 task의 output model로 연결됩니다.
같은 task에서는 하나의 Registry 모델을 갱신하고, 다른 task는 별도 모델 ID를 가집니다.
따라서 같은 이름으로 실행한 실험끼리도 task/model ID로 구분됩니다.

업로드 대상은 task의 output URI를 사용하며, 없으면 SDK 기본 저장소 또는 ClearML 파일 서버를 사용합니다.
업로드가 끝날 때까지 기다린 후 다음 epoch로 진행하며 업로드 실패는 오류로 처리합니다.
성공적으로 업로드한 best model은 컨테이너 삭제 후에도 저장소에서 접근할 수 있습니다.
ClearML의 Models 또는 task의 output model에서 확인합니다.
로컬 Python 실행과 검증 전용 실행은 Registry에 등록하지 않습니다.

### ClearML과 멀티 노드 함께 사용

현재 코드는 **노드별 작업을 따로 제출하는 방식**을 지원합니다.
`clearml-task` 한 번으로 다른 노드까지 자동 기동하지는 않습니다.

1. 위 멀티 노드 예시의 node별 YAML 두 개를 repo에 저장합니다.
2. 각 물리 노드의 agent를 서로 다른 queue에 연결합니다.
3. 작업을 하나씩 제출합니다. YAML에 별도 ClearML 구역은 필요하지 않습니다.

```bash
# Node 0 전용 agent가 수신하는 queue
clearml-task --project APAG --name apag-node0 \
  --repo git@github.com:h2ohyukhyuk/backbone-trainer.git --branch main \
  --script main.py --docker clearml-local/torch:2.9.1-py3.12-cu128 \
  --queue apag-node0 --args config_path=experiments/APAGNet_node0.yml

# Node 1 전용 agent가 수신하는 queue
clearml-task --project APAG --name apag-node1 \
  --repo git@github.com:h2ohyukhyuk/backbone-trainer.git --branch main \
  --script main.py --docker clearml-local/torch:2.9.1-py3.12-cu128 \
  --queue apag-node1 --args config_path=experiments/APAGNet_node1.yml
```

각 queue에는 해당 노드의 작업을 동시에 실행할 agent가 필요합니다.
같은 순차 실행 agent 하나에 두 작업을 넣으면 첫 작업이 다른 노드를 기다리게 됩니다.
두 작업은 ClearML에서 별도 task로 보이며, 집계 loss/accuracy는 Node 0의 task에 기록됩니다.
멀티 노드 Docker 네트워크는 예를 들어 Linux의 `--network=host` 등으로 구성하고
master 주소와 NCCL 통신이 실제로 연결되는지 확인하세요. `--ipc=host`만으로 노드 간 통신이 설정되지는 않습니다.

한 번의 제출로 노드를 자동 배치하려면 ClearML의
[`Task.launch_multi_node()`](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/) 연동이 추가로 필요합니다.
현재 코드에는 이 자동 배치 호출이 없습니다.

## AMP와 배치 크기

```yaml
amp: true
amp_dtype: float16          # float16 또는 지원 GPU의 bfloat16
```

CUDA에서는 PyTorch native AMP(autocast + GradScaler), CPU에서는 float32를 사용합니다.
FP16 overflow로 optimizer 업데이트를 건너뛰면 step 기반 LR도 진행하지 않습니다.
scaler 상태는 체크포인트에서 함께 복원합니다.

12GB GPU 두 장이라면 **총 batch 256 = GPU당 128**을 초기 시험값으로 사용하세요.
256 → 384 → 512 순서로 올리며 메모리 사용량, epoch 소요 시간과 validation 정확도를 비교합니다.
두 GPU의 메모리가 하나로 합쳐지지는 않습니다. 최대 배치는 GPU 모델·입력 크기 등에 따라 달라지며
배치를 키운다고 항상 빨라지지는 않습니다. 배치 변경에 따른 LR 자동 배율 조정은 없습니다.

## 옵티마이저

공통 설정과 선택한 옵티마이저의 `*_KWARGS` 구역만 변경합니다.

```yaml
OPT_NAME: adamw
LR: 0.001
WEIGHT_DECAY: 0.01
AMS_GRAD: false
ADAMW_KWARGS:
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

| 선택값 | 전용 설정 | 추가 공통 설정 |
|---|---|---|
| `sgd` | `SGD_KWARGS`: `dampening`, `nesterov` | `MOMENTUM` |
| `adam` | `ADAM_KWARGS`: `betas`, `eps` | `AMS_GRAD` |
| `adamw` | `ADAMW_KWARGS`: `betas`, `eps` | `AMS_GRAD` |
| `rmsprop` | `RMSPROP_KWARGS`: `alpha`, `eps`, `centered` | `MOMENTUM` |
| `adadelta` | `ADADELTA_KWARGS`: `rho`, `eps` | — |

`LR`와 `WEIGHT_DECAY`는 모든 옵티마이저에 적용합니다.
소문자 `lr`, `momentum`, `weight_decay` 대신 `LR`, `MOMENTUM`, `WEIGHT_DECAY`를 사용합니다.
모듈별 학습률 배율도 설정할 수 있습니다.

```yaml
USE_LR_MULT: true
LR_MULT_TARGET: [classifier]
LR_MULT_VAL: [10.0]
```

키워드는 최상위 모듈 이름에 적용하며 두 목록의 길이는 같아야 합니다.

## 학습률 스케줄러

`lr_scheduler`로 선택하고 해당 구역을 수정합니다.

| 선택값 | 설정 구역 / 키 | 갱신 단위 |
|---|---|---|
| `steplr` | `STEPLR_KWARGS`: `step_size`, `gamma` | epoch |
| `lambdalr` | `LAMBDALR_KWARGS`: `points`, `scale`, `interpolation` | optimizer update |
| `multisteplr` | `MULTISTEPLR_KWARGS`: `milestones`, `gamma` | epoch |
| `polylr` | `POLYLR_KWARGS`: `power` | epoch |

MultiStepLR은 **milestone epoch가 완료될 때만** `gamma`를 곱합니다.
`[30, 60, 80]`이면 30/60/80 경계에서만 감소합니다.
PolyLR은 `LR × (1 - 완료 epoch / epochs)^power`로 0까지 감소합니다.

### LambdaLR: epoch별 LR과 보간

```yaml
lr_scheduler: lambdalr
LAMBDALR_KWARGS:
  scale: 1.0
  points:
    0: 0.0001
    1: 0.01
    5: 0.01
    30: 0.005
    60: 0.001
    90: 0.0001
  interpolation: hermite    # Hermite만 지원
```

- `scale`은 전체 LR 곡선 배율입니다. `0.5`이면 모든 point와 구간 LR이 절반, `2.0`이면 두 배입니다.
- 배율은 유한한 양수이며 생략하면 1입니다. 원본 `points`는 바꾸지 않고 계산 시 적용합니다.
- HPO 탐색 변수는 `Training/LAMBDALR_KWARGS/scale`입니다. config에는 기준 points와 실제 scale을 기록합니다.
- `points`의 숫자 키는 완료한 epoch, 값은 절대 LR입니다. 소수 epoch도 지원하며 내부에서 정렬합니다.
- 매 optimizer 업데이트에서 epoch 위치를 계산해 보간합니다. 앞부분 점으로 warmup을 표현합니다.
- 첫 점 이전과 마지막 점 이후에는 해당 끝점 LR을 유지합니다.
- 각 point의 기울기는 항상 0으로 고정합니다. 사용자 기울기 설정은 없습니다.
- 구간은 양 끝 LR 사이에서 단조롭게 보간하므로 overshoot가 생기지 않습니다.
- 모듈별 LR 배율은 기준 곡선에 곱해집니다.

### 학습 없이 LR 그래프 저장

```bash
# 네 스케줄러 모두 비교
python main.py --config_path experiments/config.yml --plot-lr

# 선택한 스케줄러만 비교
python main.py --config_path experiments/config.yml --plot-lr multisteplr lambdalr
```

결과는 **`experiments/APAGNet_lr_curve.png`**입니다.
선택한 config 이름에 `_lr_curve.png`를 붙이고, 같은 파일이 있으면 덮어씁니다.
가로축은 epoch이며 1 epoch 간격으로 세로 그리드를 표시합니다.

데이터셋 접근이나 optimizer step 반복 없이 수식·보간 함수를 직접 계산합니다.
`--steps-per-epoch`는 필요하지 않습니다. 재개 상태와 `start_epoch`는 반영하지 않고 전체 곡선을 그립니다.
LambdaLR은 `points` 곡선, 다른 스케줄러는 공통 `LR` 기준 곡선을 표시합니다.

## 기록과 체크포인트

### 측정 지표

| 구간 | 지표 | ClearML 표시 |
|---|---|---|
| Train | loss | `loss/train` |
| Validation | loss | `loss/valid` |
| Validation | Top-1 / Top-5 accuracy | `accuracy/valid_top1`, `accuracy/valid_top5` |
| Train | 다음 업데이트에 사용할 LR | `learning rate/train` |

Train accuracy는 계산하지 않습니다. 지표는 epoch별로 기록하며, DDP loss/accuracy는
전체 rank의 샘플 수로 가중 평균합니다. 검증 통계는 매 epoch 초기화합니다.

### 저장 위치

기본 저장 루트는 `save: /home/j/models_cla/`입니다. 일반 로그와 print, config 복사,
체크포인트 저장 및 ClearML scalar 기록은 global rank 0(Node 0의 첫 GPU)에서만 실행합니다.

```text
/home/j/models_cla/APAGNet-cifar10/
├── 001.pt
├── 002.pt
├── ...
├── model_best.pt
├── APAGNet_cifar10.yml          # 실행에 사용한 원본 config
├── config.resolved.yml          # 기본값을 합친 최종 config
└── log-APAGNet-cifar10.txt
```

로그, `.pt`, 선택한 원본 config 복사본과 `config.resolved.yml`을 함께 저장합니다.
같은 결과 폴더가 이미 있으면 한국 시간 기준 `-YYMMDD`를 붙여 새 폴더를 만듭니다.
예: `APAGNet-cifar10` → `APAGNet-cifar10-261017` → `APAGNet-cifar10-261017-2`.
DDP에서도 모든 rank가 같은 실행 폴더를 사용하고 rank 0만 파일을 저장합니다.
재개 실행도 새 결과 폴더를 사용하며 기존 파일을 덮어쓰지 않습니다.
매 epoch 체크포인트를 저장하고 validation Top-1 최고 기록은 `model_best.pt`로 복사합니다.
체크포인트에는 모델, optimizer, scheduler, AMP scaler, epoch와 최고 정확도가 포함됩니다.
`config`에는 기본값을 합친 학습 설정, `metrics`에는 해당 epoch의 `valid_loss`, `top1`, `top5`를 저장합니다.
`model_best.pt`의 지표도 그 파일에 저장된 가중치의 검증 결과입니다.
`best_acc1`은 해당 epoch까지의 역대 최고 Top-1이며 현재 epoch의 `metrics.top1`과 다를 수 있습니다.
ImageNet의 폴더 접미사는 `-imgn`, FakeData는 추가로 `-dummy`입니다.

### 재개와 검증 전용 실행

기존 실험 YAML에서 다음 값을 변경한 뒤 같은 `--config_path` 명령으로 실행합니다.

```yaml
resume: /home/j/models_cla/APAGNet-cifar10/090.pt
epochs: 200
evaluate: false
```

90 epoch 체크포인트라면 91 epoch부터 시작합니다. `epochs`는 추가 학습량이 아니라 최종 목표입니다.
스케줄러 종류와 보간점, 배치 설정을 유지하세요. PolyLR은 `epochs` 변경 시 감소 곡선 자체가 바뀝니다.
기존 `model` 형식의 가중치 전용 체크포인트는 optimizer·scheduler·epoch를 새로 시작합니다.
멀티 노드 재개에서는 모든 노드가 동일한 체크포인트에 접근할 수 있어야 합니다.

검증만 하려면:

```yaml
resume: /home/j/models_cla/APAGNet-cifar10/model_best.pt
evaluate: true
```

## 새 모델 추가

`net/MyNet.py`에 `num_classes`를 받는 `Network` 클래스를 작성합니다.
입력은 `[batch, 3, height, width]`, 출력은 `[batch, num_classes]` logits입니다.
CrossEntropyLoss를 사용하므로 마지막에 Softmax를 추가하지 않습니다.

```python
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_classes=1000, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

실험 YAML에 다음을 지정합니다.

```yaml
arch: MyNet
model_kwargs:
  width: 64
```

클래스 수는 ImageNet 1,000개 / CIFAR-10 10개로 자동 지정합니다.
`arch: resnet18`처럼 torchvision 모델도 사용할 수 있습니다.
`pretrained: true`는 ImageNet용 torchvision 모델에 적용하며, 직접 작성한 모델의 가중치는 `resume`으로 불러옵니다.

## 프로젝트 구조와 테스트

```text
backbone-trainer/
├── main.py                       # 설정, 실행 환경, 학습·LR 그래프
├── requirements.txt
├── experiments/                  # 기본값과 실험별 YAML
├── net/                          # Network 클래스
├── model/                        # 백본 구성과 공통 레이어
├── dataset/                      # ImageNet / CIFAR-10, DDP 로더
├── train_val/classification.py   # Trainer / Validationer, AMP, 지표, 저장
├── utils/
│   ├── train_utils.py            # 옵티마이저 / 스케줄러
│   ├── lr_interpolation.py       # epoch/LR 보간
│   └── clearml_utils.py          # ClearML 초기화와 rank 0 기록
└── tests/
```

```bash
python -m unittest discover -s tests -v
```

GPU 최대 배치와 실제 CUDA AMP, ClearML 서버 연결은 대상 GPU·agent 환경에서 확인해야 합니다.
