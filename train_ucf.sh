# Train YOWOv2 on UCF24 dataset
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_large \
        --root /home/jason/YOWOv2_cow/dataset/ \
        --num_workers 16 \
        --eval_epoch 1 \
        --max_epoch 20 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 8 \
        -accu 16 \
        -K 16 \
        -tbs 8 \
        --eval \
