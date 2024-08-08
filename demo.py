import argparse
import cv2
import os
import time
import numpy as np
import torch
import imageio
from PIL import Image
from datetime import datetime
from dataset.transforms import BaseTransform
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection
from config import build_dataset_config, build_model_config
from models import build_model
from models.boxmot import OCSORT,BYTETracker
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.55, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='./video/10.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # class label config
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False, 
                        help='show 14 action pose of AVA.')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_slowfast', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default='./weights/yowo_v2_slowfast_epoch_58.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.3, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.55, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    return parser.parse_args()
                    
def renormalize_cam_in_bounding_boxes(boxes, image_float_np, orig_w, orig_h, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    return show_cam_on_image(image_float_np, renormalized_cam, use_rgb=False)


def multi_hot_vis(args, frame, out_bboxes, orig_w, orig_h, class_names, act_pose=False, tracker=None):
    # visualize detection results
    for bbox in out_bboxes:
        x1, y1, x2, y2 = bbox[:4]
        if act_pose:
            # only show 14 poses of AVA.
            cls_conf = bbox[5:5+14]
        else:
            # show all actions of AVA.
            cls_conf = bbox[5:]
            
    
        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        # score = obj * cls
        det_conf = float(bbox[4])
        #print (det_conf)
        cls_scores = np.sqrt(det_conf * cls_conf)


        indices = np.where(cls_scores > args.vis_thresh)
        scores = cls_scores[indices]
        indices = list(indices[0]) # (哪一個行為機率最大 然後把它print出來 假如 eat 機率最大 那輸出會是3 因為 1:stand 2: sit 3:eat)
        scores = list(scores) # (行為的機率 也就是會顯示在 demo 畫面上的動作機率數字)
        #print(scores)

        if len(scores) > 0:
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1+3, y1+14+20*_))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)
    
    return frame


@torch.no_grad()
def detect(args, model, device, transform, class_names, class_colors,tracker):
    cam = EigenCAM(model=model, target_layers=target_layers)
    # path to save 
    save_path = os.path.join(args.save_folder, 'demo', 'videos')
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    save_size = (960, 720)
    
    # Get current time
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    formatted_time = f"{current_time[:4]}_{current_time[4:8]}_{current_time[8:]}"
    save_name = os.path.join(save_path, f'detection_{formatted_time}_{args.version}.mp4')
    fps = 16.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # run
    video_clip = []
    image_list = []
    while(True):
        ret, frame = video.read()
        
        if ret:
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]
            print(frame_rgb.shape)

            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(args.len_clip):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

            t0 = time.time()
            # inference
            outputs = model(x)
            print("inference time ", time.time() - t0, "s")

            # Apply EigenCAM
            # grayscale_cam = cam(input_tensor=x)[0]
            # grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))

            outputs_box = outputs[0]  # 假設我們只關心第一個輸出

            # 將 Numpy 陣列轉換為 Tensor
            outputs_tensor = torch.from_numpy(outputs_box).to(device)

            # 使用 outputs_tensor 獲取最可能的類別
            target_categories = torch.argmax(outputs_tensor[:, 4:], dim=1)  # 假設類別分數從第五列開始
            targets = [ClassifierOutputTarget(category.item()) for category in target_categories]
                 
            # Apply EigenCAM
            grayscale_cam = cam(input_tensor=x, targets=targets, eigen_smooth=True)
            grayscale_cam = np.transpose(grayscale_cam, (1, 2, 0)) # (1,224,224) to (224,224,1)
            # grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))
            grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))
            if len(grayscale_cam_resized.shape) == 2:
                grayscale_cam_resized = np.expand_dims(grayscale_cam_resized, axis=2)
                grayscale_cam_resized = np.repeat(grayscale_cam_resized, 3, axis=2)
            # print(grayscale_cam_resized.shape)


            # vis detection results
            if args.dataset in ['ava_v2.2']:
                batch_bboxes = outputs
                bboxes = batch_bboxes[0]
                bboxes_np = np.array(bboxes)

                 # Generate the normalized CAM only inside bounding boxes
                cam_image = renormalize_cam_in_bounding_boxes(bboxes,image_float_np=np.float32(frame_rgb) / 255, orig_h=orig_h,orig_w=orig_w, grayscale_cam=grayscale_cam_resized)
                
                # cam_image = show_cam_on_image(np.float32(frame_rgb) / 255, grayscale_cam_resized , use_rgb=False)
            

                # multi hot
                frame = multi_hot_vis(
                    args=args,
                    frame=cam_image,
                    out_bboxes=bboxes,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    class_names=class_names,
                    act_pose=args.pose,                                                                            
                    tracker=tracker
                    )
                # Update and plot tracker results
                # print(cam_image.shape[1])
                # tracker.update(bboxes_np,frame)
                # print(cam_image.shape[1])
                # frame = tracker.plot_results(frame, show_trajectories=True, orig_w=orig_w, orig_h=orig_h, class_names=class_names)


            elif args.dataset in ['ucf24']:
                batch_scores, batch_labels, batch_bboxes = outputs
                # batch size = 1
                scores = batch_scores[0]
                labels = batch_labels[0]
                bboxes = batch_bboxes[0]
                # rescale
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h])
                # one hot
                frame = vis_detection(
                    frame=frame,
                    scores=scores,
                    labels=labels,
                    bboxes=bboxes,
                    vis_thresh=args.vis_thresh,
                    class_names=class_names,
                    class_colors=class_colors
                    )
            # save
            frame_resized = cv2.resize(frame, save_size)
            out.write(frame_resized)

            if args.gif:
                gif_resized = cv2.resize(frame, (200, 150))
                gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                image_list.append(gif_resized_rgb)

            if args.show:
                # show
                cv2.imshow('key-frame detection', frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    # generate GIF
    if args.gif:
        save_name = os.path.join(save_path, 'detect.gif')
        print('generating GIF ...')
        imageio.mimsave(save_name, image_list, fps=fps)
        print('GIF done: {}'.format(save_name))


if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    target_layers = [model.reg_preds[-1]]
    # target_layers = [model.backbone_2d.backbone.non_shared_heads[-1].cls_feats[0]]

    tracker = BYTETracker()
    # run
    detect(args=args,
            model=model,
            device=device,
            transform=basetransform,
            class_names=class_names,
            class_colors=class_colors,
            tracker=tracker)