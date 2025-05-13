import os
import cv2
import time
import argparse
import numpy as np

import torch

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for RetinaFace")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/retinaface_mv2.pth',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'mobilenetv2_0.25', 'mobilenetv2_0.125', 
            'mobilenetv3', 'mobilenetv3_0.25', 'mobilenetv3_0.125', 
            'mobilenetv4', 'mobilenetv4_0.25', 'mobilenetv4_0.125', 
            'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save the detection results as images'
    )
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )

    # Image input
    parser.add_argument(
        '--image-path',
        type=str,
        default='./assets/test.jpg',
        help='Path to the input image'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    
    # Start inference timing - no warmup
    num_tests = 100  # Number of inference runs to average
    inference_times = []
    
    for _ in range(num_tests):
        start_time = time.time()
        loc, conf, landmarks = model(image)
        inference_times.append((time.time() - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate average inference time
    inference_times = inference_times[50:]
    inference_time = sum(inference_times) / len(inference_times)
    std_dev = np.std(inference_times)
    print(f"\nInference Statistics over {num_tests} runs:")
    print(f"Average Time: {inference_time:.2f} ms")
    print(f"Standard Deviation: {std_dev:.2f} ms")
    
    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks, inference_time


def main(params):
    # Start total time measurement
    total_start_time = time.time()
    
    # load configuration and device setup
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)
    model.eval()

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # loading state_dict
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    print("Model loaded successfully!")

    # read image
    original_image = cv2.imread(params.image_path, cv2.IMREAD_COLOR)
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    # normalize image
    image -= rgb_mean
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
    image = image.to(device)

    # forward pass
    loc, conf, landmarks, inference_time = inference(model, image)

    # generate anchor boxes
    priorbox = PriorBox(cfg, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors().to(device)

    # decode boxes and landmarks
    boxes = decode(loc, priors, cfg['variance'])
    landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

    # scale adjustments
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # filter by confidence threshold
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # sort by scores
    order = scores.argsort()[::-1][:params.pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # apply NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(detections, params.nms_threshold)

    detections = detections[keep]
    landmarks = landmarks[keep]

    # keep top-k detections and landmarks
    detections = detections[:params.post_nms_topk]
    landmarks = landmarks[:params.post_nms_topk]

    # concatenate detections and landmarks
    detections = np.concatenate((detections, landmarks), axis=1)

    # Calculate total processing time
    total_time = (time.time() - total_start_time) * 1000  # Convert to milliseconds

    # Print timing information
    print(f"\nTiming Information:")
    print(f"Model Inference Time: {inference_time:.2f} ms")
    print(f"Total Processing Time: {total_time:.2f} ms")
    print(f"Number of Detections: {len(detections)}")

    # show image
    if params.save_image:
        draw_detections(original_image, detections, params.vis_threshold)
        # save image
        im_name = os.path.splitext(os.path.basename(params.image_path))[0]
        save_name = f"{im_name}_{params.network}_out.jpg"
        cv2.imwrite(save_name, original_image)
        print(f"Image saved at '{save_name}'")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
