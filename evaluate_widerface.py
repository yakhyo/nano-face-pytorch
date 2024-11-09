import os
import cv2
import time
import argparse
import numpy as np

import torch

from layers import PriorBox
from config import get_config
from models import RetinaFace, Slim
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description='RetinaFace Model WiderFace Dataset Evaluation Script')

    # Model settings
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/mobilenetv1_final.pth',
        help='Path to the trained state_dict file'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='mobilenetv1_0.25',
        choices=['mobilenetv1_0.25', 'slim', 'rfb'],
        help='Select a model architecture for face detection'
    )
    # Evaluation settings
    parser.add_argument(
        '--origin-size',
        action='store_true',
        help='Evaluate using the original image size'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold for detection'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )

    # File paths
    parser.add_argument(
        '--save-folder',
        type=str,
        default='./widerface_evaluation/widerface_txt/',
        help='Directory to save the result text files'
    )
    parser.add_argument(
        '--dataset-folder',
        type=str,
        default='./data/widerface/val/images/',
        help='Path to the dataset folder'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks


def resize_image(image, target_size=1600, max_size=2150):
    """Resize the image while maintaining the aspect ratio."""
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)

    return resize


def main(params):
    # load configuration and device setup
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    if params.network == "mobilenetv1_0.25":
        model = RetinaFace(cfg=cfg)
    elif params.network == "slim":
        model = Slim(cfg=cfg)
    elif params.network == "rfb":
        model = RFB(cfg=cfg)
    else:
        raise NameError("Please choose existing face detection method!")
    model.to(device)

    # loading state_dict
    state_dict = torch.load(params.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    model.eval()

    # testing dataset
    testset_folder = params.dataset_folder
    testset_list = params.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing begin
    for idx, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = np.float32(img_raw)

        # Determine resize factor
        resize_factor = 1 if params.origin_size else resize_image(image)

        if resize_factor != 1:
            image = cv2.resize(image, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

        img_height, img_width, _ = image.shape

        # normalize image
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
        image = image.to(device)

        # forward pass
        st = time.time()
        loc, conf, landmarks = inference(model, image)  # forward pass
        forward_pass = time.time() - st

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
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, params.nms_threshold)

        detections = detections[keep]
        landmarks = landmarks[keep]

        # Save results
        save_name = params.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(detections)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in detections:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                fd.write(f"{x} {y} {w} {h} {confidence}\n")

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s'.format(idx + 1, num_images, forward_pass))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
