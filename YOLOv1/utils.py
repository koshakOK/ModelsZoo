import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torchvision

EPS = 1e-6


def intersection_over_union(boxes_predictions, boxes_labels, box_format):
    # (x1, y1, x2, y2)
    if box_format == "corners":
        bbox1_x1 = boxes_predictions[..., 0]
        bbox1_y1 = boxes_predictions[..., 1]
        bbox1_x2 = boxes_predictions[..., 2]
        bbox1_y2 = boxes_predictions[..., 3]

        bbox2_x1 = boxes_labels[..., 0]
        bbox2_y1 = boxes_labels[..., 1]
        bbox2_x2 = boxes_labels[..., 2]
        bbox2_y2 = boxes_labels[..., 3]

    # (x_c, y_c, w, h)
    if box_format == "midpoint":
        bbox1_x1 = boxes_predictions[..., 0] - boxes_predictions[..., 2] / 2
        bbox1_y1 = boxes_predictions[..., 1] - boxes_predictions[..., 3] / 2
        bbox1_x2 = boxes_predictions[..., 0] + boxes_predictions[..., 2] / 2
        bbox1_y2 = boxes_predictions[..., 1] + boxes_predictions[..., 3] / 2

        bbox2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        bbox2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        bbox2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        bbox2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)

    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = torch.abs((bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1))
    area2 = torch.abs((bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1))
    union = area1 + area2 - intersection

    return intersection / (union + EPS)


def non_maximum_suppression(bboxes, iou_threshold, prob_threshold, box_format):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(chosen_box[2:]),
                                       torch.tensor(box[2:])) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms
