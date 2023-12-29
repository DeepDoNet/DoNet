import json
import cv2
import numpy as np
from typing import Any, Iterator, List, Union
import pycocotools.mask as mask_util
from skimage import measure
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, required=True, help="path to the original json file")
parser.add_argument('--output_json_path', type=str, required=True, help="path to the output json file")
args = parser.parse_args()

'''
to run:
python preprocess.py --json_path <path to the original json file> --output_json_path <path to the output json file>
'''

json_path = args.json_path
output_json_path = args.output_json_path

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(np.bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool)


def polygon_area(x, y):
    # Using the shoelace formula
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

with open(json_path, 'r', encoding='utf-8') as f:
    jsonobj = json.load(f)

json_new = jsonobj.copy()
json_new["annotations"] = []

img_id_list = []
for img in jsonobj["images"]:
    img_id_list.append(img['id'])

count = 0.0
count_invisible = 0
count_ann = 0
for img_id in img_id_list:
    count += 1
    print("img_id", img_id, 100*count/len(img_id_list))
    bitmask_list = []
    seg_list = []
    cta_list = []
    ann_list = []

    for ann in jsonobj["annotations"]:
        if ann['image_id'] == img_id:
            ann_list.append(ann)
            seg_list.append(ann['segmentation'])
            cta_list.append(ann['category_id'])

    count_ann += len(ann_list)
    if len(ann_list) > 0:
        for index1, now_ann in enumerate(ann_list):
            if now_ann["category_id"] == 1:
                now_ann["invisible_mask"] = []
                now_ann["visible_mask"] = now_ann["segmentation"]
                json_new["annotations"].append(now_ann)
                continue
            invalid1 = False
            for sub_seg1 in now_ann['segmentation']:
                if len(sub_seg1) < 6:
                    invalid1 = True
            now_ann['invisible_mask'] = []
            if invalid1:
                json_new["annotations"].append(now_ann)
                continue
            ann_bitmask1 = polygons_to_bitmask(
                now_ann['segmentation'], 512, 512)
            now_ann['invisible_mask'] = []
            now_ann['visible_mask'] = []
            overlapping_mask_list = []
            overlapping_polygon_list = []

            for index2, other_ann in enumerate(ann_list):
                if index2 == index1:
                    continue
                if other_ann["category_id"] == 1:
                    continue
                invalid2 = False
                for sub_seg2 in now_ann['segmentation']:
                    if len(sub_seg2) < 6:
                        invalid2 = True
                if invalid2:
                    continue
                ann_bitmask2 = polygons_to_bitmask(
                    other_ann['segmentation'], 512, 512)
                overlapping_mask = np.multiply(ann_bitmask1, ann_bitmask2)
                overlapping_mask_list.append(overlapping_mask)

            whole_overlapping_mask = np.zeros((512, 512), int)
            for overlapping_mask in overlapping_mask_list:
                whole_overlapping_mask = np.logical_or(
                    whole_overlapping_mask, overlapping_mask)
            whole_nonoverlapping_mask = np.logical_xor(
                ann_bitmask1, whole_overlapping_mask)

            if np.count_nonzero(whole_overlapping_mask) < 2:
                now_ann["invisible_mask"] = []
            else:
                overlapping_polygon = binary_mask_to_polygon(
                    whole_overlapping_mask, tolerance=2)
                if len(overlapping_polygon) == 0:
                    now_ann["invisible_mask"] = []
                elif len(overlapping_polygon[0]) < 2 or len(overlapping_polygon[0]) % 2 != 0:
                    now_ann["invisible_mask"] = []
                else:
                    now_ann["invisible_mask"] = overlapping_polygon

            if np.count_nonzero(whole_nonoverlapping_mask) < 2:
                now_ann["visible_mask"] = []
            else:
                nonoverlapping_polygon = binary_mask_to_polygon(
                    whole_nonoverlapping_mask, tolerance=2)
                if len(nonoverlapping_polygon) == 0:
                    now_ann["visible_mask"] = []
                elif len(nonoverlapping_polygon[0]) < 2 or len(nonoverlapping_polygon[0]) % 2 != 0:
                    now_ann["visible_mask"] = []
                else:
                    now_ann["visible_mask"] = nonoverlapping_polygon
            json_new["annotations"].append(now_ann)
            if now_ann["visible_mask"] == []:
                count_invisible += 1
                print(count_invisible, count_invisible*100.0/count_ann)
            if now_ann["invisible_mask"] != []:
                print("overlapping!")

with open(output_json_path, 'w') as f:
    json.dump(json_new, f)