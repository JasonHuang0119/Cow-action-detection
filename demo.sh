# Test YOWOv2 on AVA dataset
python demo.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_slowfast \
        --weight ./weights/yowo_v2_slowfast_epoch_58.pth \
        --video ./video/7.mp4\
        -K 16 \
        --show
        
        

