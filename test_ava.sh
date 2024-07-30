# Test YOWOv2 on AVA dataset
python demo.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_slowfast \
        --weight /home/jason/YOWOv2_cow/weights/ava_v2.2/yowo_v2_slowfast/2024_6_2_Eiou_for_loss_yowov2/yowo_v2_slowfast_epoch_100.pth \
        --video  /home/jason/YOWOv2_cow/video/45.mp4\
        -K 16 \
        --show
        
        

