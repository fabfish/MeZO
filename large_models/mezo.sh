MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MASK_ONLY=${MASK_ONLY:-"--mask_only_mode False"}
MODEL_NAME="${MODEL_NAME[-1]}"


BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SPARSITY=${SPARSITY:-0.95}

WARMUP_STEP=${WARMUP_STEP:-0}
DECAY_STEP=${DECAY_STEP:-0}
ZO_LR_SCHEDULER_TYPE=${ZO_LR_SCHEDULER_TYPE:-'constant'}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
HESSIAN_SMOOTH_TYPE=${HESSIAN_SMOOTH_TYPE:-'constant1e-8'}
# HESSIAN_SMOOTH_TYPE=${HESSIAN_SMOOTH_TYPE:-'constant1e-1'}
# HESSIAN_SMOOTH_TYPE=${HESSIAN_SMOOTH_TYPE:-'constant1e-8'} & 5e-7
# best lisa-hizoo: 1e-1 (0.15)
USE_HIZOO=${USE_HIZOO:-'--use_hizoo False'}
USE_LISA=${USE_LISA:-'--use_lisa False'}


MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir /data/yzy/mezo/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo  \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --sparsity $SPARSITY \
    --warmup_step $WARMUP_STEP --decay_step $DECAY_STEP  --zo_lr_scheduler_type $ZO_LR_SCHEDULER_TYPE \
    --weight_decay $WEIGHT_DECAY --hessian_smooth_type $HESSIAN_SMOOTH_TYPE \
    --overwrite_output_dir \
    $EXTRA_ARGS \
    $TASK_ARGS \
    $MASK_ONLY \
    $USE_HIZOO \
    $USE_LISA \
    "$@"
