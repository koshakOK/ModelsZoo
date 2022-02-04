import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as patches
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
                                       torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def mean_average_precision(pred_boxes,
                           true_boxes,
                           iou_threshold=0.5,
                           box_format="midpoint",
                           num_classes=20):
    average_precisions = []

    for c in range(num_classes):
        detections = [
            detection for detection in pred_boxes if detection[1] == c
        ]
        gt = [
            true_box for true_box in true_boxes if true_box[1] == c
        ]
        amount_bboxes = Counter([i[0] for i in gt])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True, inplace=True)

        TP = torch.zeros_like(detections)
        FP = torch.zeros_like(detections)

        total_true_bboxes = len(gt)

        if not total_true_bboxes:
            continue

        for detection_idx, detection in enumerate(detections):
            gt_img = [bbox for bbox in gt if bbox[0] == detection[0]]

            best_iou = 0.0

            for idx, groud_true in enumerate(gt_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(groud_true[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if not amount_bboxes[detection[0]][best_gt_idx]:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + EPS)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + EPS))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return torch.mean(average_precisions)


def plot_image(image, boxes, box_fomat="midpoint"):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    if box_fomat == "midpoint":

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        # Create a Rectangle potch
        for box in boxes:
            box = box[2:]
            assert len(
                box) == 4, "Got more values than in x, y, w, h, in a box!"
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()
