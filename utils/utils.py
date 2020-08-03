import argparse
import logging
import math
import os
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
from PIL import Image

#from align_faces import get_reference_facial_points, warp_and_crop_face
#from config import image_h, image_w
#from mtcnn.detector import detect_faces


def _clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, acc, is_best=False):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'metric_fc': metric_fc,
             'optimizer': optimizer}
    filename = str(epoch) + '-' + 'facenet.pth'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


'''
def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)  # BGR
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            landmarks = [int(round(x)) for x in landmarks[0]]
            return True, landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[1] + b[3]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def get_central_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


def get_all_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    return bounding_boxes, landmarks


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img
'''


def extract_feature(model, tsfm, img_path, device):
    img = Image.open(img_path).convert('RGB')
    img = tsfm(img)
    feature = model(img.unsqueeze(0).to(device))
    feature = feature.detach().cpu().numpy()
    return feature


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cosine_similarity(x1, x2):
    X = np.vstack([x1, x2])
    d2 = 1 - pdist(X, 'cosine')
    return d2


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def full_log(epoch):
    full_log_dir = 'data/full_log'
    if not os.path.isdir(full_log_dir):
        os.mkdir(full_log_dir)
    filename = 'angles_{}.txt'.format(epoch)
    dst_file = os.path.join(full_log_dir, filename)
    src_file = 'data/angles.txt'
    copyfile(src_file, dst_file)
