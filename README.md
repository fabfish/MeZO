示例环境：ubuntu22.04，conda24.06, cuda 12.5

从零起步配置该仓库，以下为主要要安装的包。建议用新的env装，装好之后不要动 python 版本。

python 3.9.7

示例环境的cuda太新了，用pip装了torch

```
pip3 install torch torchvision torchaudio
```

torch 2.4.0

datasets

transformers==4.28.1

(building wheels 如果在 tokenizer 中 rust 部分失败，可能是 python 版本过高。尽量保持一致。另外python3.10可能测出结果下降，原因不明）

accelerate==0.33.0

wandb==0.17.7

实验代码

### lisa+hizoo

MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=3e-6 EPS=1e-3 SPARSITY=1.00 STEPS=20000 HESSIAN_SMOOTH_TYPE='constant1e-5' USE_HIZOO="--use_hizoo True" USE_LISA="--use_lisa True" bash mezo.sh

{'accuracy': 0.9185779816513762, 'dev_accuracy': 0.902}

### mezo

MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=3e-7 EPS=1e-3 bash mezo.sh

{'accuracy': 0.9174311926605505, 'dev_accuracy': 0.92}

### hizoo

MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=3e-7 EPS=1e-3 SPARSITY=1.00 STEPS=20000 HESSIAN_SMOOTH_TYPE='constant1e-9' USE_HIZOO="--use_hizoo True" USE_LISA="--use_lisa False" bash mezo.sh

{'accuracy': 0.9174311926605505, 'dev_accuracy': 0.92}
