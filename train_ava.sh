# Train YOWOv2 on AVA dataset
python train.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_large \
        --root /home/jason/YOWOv2_cow/dataset/ \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
        --max_epoch 20 \
        --lr_epoch 3 4 5 6 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs  16 \
        -accu 16 \
        -K 16 \
        -tbs 16 \
