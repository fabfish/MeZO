import os

# 参数设置
MODEL = "facebook/opt-1.3b"
MODE = "ft"
EPS = "1e-3"
SPARSITY = "1.00"
STEPS = "20000"

# 不同配置的学习率和HESSIAN_SMOOTH_TYPE

configurations = [
    # {"TASK":"RTE", "MODE":"ft", "LR": 1e-5, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"RTE", "MODE":"ft", "LR": 6e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"RTE", "MODE":"ft", "LR": 7e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"RTE", "MODE":"ft", "LR": 8e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"RTE", "MODE":"ft", "LR": 9e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"SQuAD", "MODE":"ft", "LR": 1e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},

    # {"TASK":"CB", "MODE":"ft", "LR": 5e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"BoolQ", "MODE":"ft", "LR": 5e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"WSC", "MODE":"ft", "LR": 5e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"WIC", "MODE":"ft", "LR": 5e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},

    # {"TASK":"BoolQ", "MODE":"ft", "LR": 3e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"BoolQ", "MODE":"ft", "LR": 1e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    # {"TASK":"BoolQ", "MODE":"ft", "LR": 2e-6, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
    {"TASK":"BoolQ", "MODE":"ft", "LR": 6e-7, "HESSIAN_SMOOTH_TYPE": None, "USE_HIZOO": "", "USE_LISA": "", "BS": 1},
]

# 执行grid search
for config in configurations:
    LR = config["LR"]
    HESSIAN_SMOOTH_TYPE = config["HESSIAN_SMOOTH_TYPE"]
    USE_HIZOO = config["USE_HIZOO"]
    USE_LISA = config["USE_LISA"]
    TASK = config["TASK"]
    MODE = config["MODE"]
    BS = config["BS"]
    
    command = f"CUDA_VISIBLE_DEVICES=0 MODEL={MODEL} TASK={TASK} MODE={MODE} LR={LR} EPS={EPS} SPARSITY={SPARSITY} STEPS={STEPS} BS={BS} "
    
    if HESSIAN_SMOOTH_TYPE:
        command += f" HESSIAN_SMOOTH_TYPE='{HESSIAN_SMOOTH_TYPE}'"
    
    if USE_HIZOO:
        command += f" USE_HIZOO='{USE_HIZOO}'"
    
    if USE_LISA:
        command += f" USE_LISA='{USE_LISA}'"

    # if TASK == "SQuAD" or TASK == "BoolQ":
    #     command += f" "

    command += f" bash finetune.sh"
    command += f" >> result_bcd_3.txt 2>&1"
    print(f"Executing: {command}")
    os.system(command)