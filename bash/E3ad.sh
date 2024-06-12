MODEL=E3AD_plus
VERSION=E3AD_plus

OUTPUT_DIR=0/work_dirs/${VERSION}

python ./train_talk2car.py \
    --output_dir ${OUTPUT_DIR} \
    --device 'cuda:0' --lr 1e-6 \
    --model_name ${MODEL} \
    --batch_size 3 \
    --model_config ./configs/CAVG_V2_plus.py


