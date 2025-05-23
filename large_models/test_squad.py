import os

# 参数设置
MODEL = "facebook/opt-1.3b"
TASK = "SQuAD"
MODE = "ft"
EPS = "1e-3"
SPARSITY = "1.00"
STEPS = "20000"

# 不同配置的学习率和HESSIAN_SMOOTH_TYPE
# configurations = [
#     {"LR": 1e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
#     {"LR": 3e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
#     {"LR": 5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
#     {"LR": 1e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
#     {"LR": 3e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
#     {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
#     {"LR": 1e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
#     {"LR": 3e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
#     {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
# ]

# configurations = [
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 4e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # # {"LR": 5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 4e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # # {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 2e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # {"LR": 4e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # # {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
# ]

configurations = [
    # {"LR": 7e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 8e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 9e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 11e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 12e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},

    {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.1e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},


    # {"LR": 3e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 4e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 2e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 3e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 4e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa False"},
    # {"LR": 1e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # {"LR": 2e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # {"LR": 3e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # {"LR": 4e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},
    # {"LR": 5e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": ""},    
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-4", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-6", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-7", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-8", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-9", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.1e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.3e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.4e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.6e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.7e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.8e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.9e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.1e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.2e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.3e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.4e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1.5e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1.6e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1.7e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1.8e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 1.9e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
    # {"LR": 2.0e-6, "HESSIAN_SMOOTH_TYPE": "constant1e-5", "USE_HIZOO": "--use_hizoo True", "USE_LISA": "--use_lisa True"},
]

# 执行grid search
for config in configurations:
    LR = config["LR"]
    HESSIAN_SMOOTH_TYPE = config["HESSIAN_SMOOTH_TYPE"]
    USE_HIZOO = config["USE_HIZOO"]
    USE_LISA = config["USE_LISA"]
    
    command = f"CUDA_VISIBLE_DEVICES=0 MODEL={MODEL} TASK={TASK} MODE={MODE} LR={LR} EPS={EPS} SPARSITY={SPARSITY} STEPS={STEPS} "
    
    if HESSIAN_SMOOTH_TYPE:
        command += f" HESSIAN_SMOOTH_TYPE='{HESSIAN_SMOOTH_TYPE}'"
    
    if USE_HIZOO:
        command += f" USE_HIZOO='{USE_HIZOO}'"
    
    if USE_LISA:
        command += f" USE_LISA='{USE_LISA}'"

    command += f" bash mezo.sh"
    command += f" >> result_squad_newnew.txt 2>&1"
    print(f"Executing: {command}")
    os.system(command)