import contextlib
import copy
import io
import itertools
from xxlimited import Null
import numpy as np
import os
import datetime
import time
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO


class AMODALCOCOeval (COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt              # ground truth COCO API
        self.cocoDt = cocoDt              # detections COCO API
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list)
        self.eval = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = AmodalParams(iouType=iouType)  # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        self.ious_for_AJI = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def computeIoU_for_AJI(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return [], [], []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [0] * len(g)
        if len(g) != 0:
            gt_area = maskUtils.area(g)
        else:
            gt_area = None
            # ious = np.zeros((1,len(gt)), dtype=np.double)
            # intersection = np.zeros((1,len(gt)), dtype=np.double)
            # union = np.asarray(map(maskUtils.area, g),dtype= np.double)
            # union = union[np.newaxis,:]
        # pdb.set_trace()
        iouIntUni = maskUtils.iouIntUni(d, g, iscrowd)

        if len(d) == 0 or len(g) == 0:

            ious = []
            dsc = []
            tpp = []
            intersection = []
            if len(d) > 0:
                merge_area = copy.deepcopy(d)
            if len(g) > 0:
                merge_area = copy.deepcopy(g)
            merge_area = maskUtils.merge(merge_area, intersect=False)
            union = [maskUtils.area(merge_area)]
        else:
            ious, intersection, union = iouIntUni[0], iouIntUni[1], iouIntUni[2]
            intersection[ious <= 0] = 0

            # [2 * i/(u + i) for i,u in zip(intersection, union)]
            dsc = 2 * intersection/(union + intersection + 1e-10)
            if dsc.max() > 1:
                pdb.set_trace()
            # if (intersection/gt_area).max

        return ious, intersection, union, gt_area, dsc

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        def visibleToRLE(ann, coco):
            """
            Convert annotation which can be polygons, uncompressed RLE to RLE.
            :return: binary mask (numpy 2D array)
            """
            t = coco.imgs[ann['image_id']]
            h, w = t['height'], t['width']
            segm = ann['visible_mask']
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)
            else:
                # rle
                rle = ann['visible_mask']
            return rle

        def _toVisibleMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = visibleToRLE(ann, coco)
                ann['visible_mask'] = rle

        def invisibleToRLE(ann, coco):
            """
            Convert annotation which can be polygons, uncompressed RLE to RLE.
            :return: binary mask (numpy 2D array)
            """
            t = coco.imgs[ann['image_id']]
            h, w = t['height'], t['width']
            segm = ann.get("invisible_mask", None)
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)
            else:
                # rle
                rle = ann['invisible_mask']
            return rle

        def _toInvisibleMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = invisibleToRLE(ann, coco)
                ann['invisible_mask'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        if p.iouType == 'visible':
            _toVisibleMask(gts, self.cocoGt)
            _toVisibleMask(dts, self.cocoDt)
        if p.iouType == 'invisible':
            # remove segm does not have invisible mask in gts
            my_occulued_gts = [
                gt for gt in gts if gt.get("invisible_mask", None)]
            my_large_gts = [gt for gt in my_occulued_gts if gt['area'] > 5000]
            gts = my_large_gts
            _toInvisibleMask(gts, self.cocoGt)
            _toInvisibleMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        # per-image per-category evaluation results
        self.evalImgs = defaultdict(list)
        self.eval = {}                  # accumulated evaluation results

    def compute_F1(self, gt_area, iou, intersection, UseIOU=True):
        TP = 0
        FP = 0
        FN = 0

        PR_thread = [i for i in np.linspace(0.2, 0.9, 28)]
        TPLIST = [0 for i in range(28)]
        FPLIST = [0 for i in range(28)]
        # PLIST = [0 for i in range(28)]
        # RLIST =[0 for i in range(28)]
        F1LIST = [0 for i in range(28)]
        iou_copy = copy.deepcopy(iou)
        gt_num = iou.shape[1]
        # gt_map_seg = np.zeros((gt_num,2)) # 0 for map idx, 1 for iou value
        # pdb.set_trace()
        iou_list = iou_copy.T.tolist()
        inter_index_list = list(map(lambda x: x.index(
            max(x)) if max(x) > 0 else -1, iou_list))
        inter_value_list = list(map(lambda x: max(x), iou_list))
        # gt_map_seg[:,0] = np.asarray(inter_index_list,)
        # gt_map_seg[:,1] = np.asarray(inter_value_list)
        inter_index_set = set(inter_index_list)
        inter_index_set.discard(-1)

        while (len(inter_index_list) - inter_value_list.count(0)) != len(inter_index_set):
            # find the duplicate index and set another segmented result to ground truth base on criterion

            duplicate_indices = []

            for v in inter_index_set:
                if inter_index_list.count(v) > 1:
                    duplicate_indices = [i for i, x in enumerate(
                        inter_index_list) if x == v]
                    break
            # then get the max iou index in duplicate indices
            if len(duplicate_indices) == 0:
                # pdb.set_trace()
                print('bug')
            iou_for_duplicate = list(
                map(inter_value_list.__getitem__, duplicate_indices))
            # delete the index with max iou
            del duplicate_indices[(
                iou_for_duplicate.index(max(iou_for_duplicate)))]
            # search for best iou match again
            for i in duplicate_indices:
                iou_list[i][v] = 0
                inter_index_list[i] = iou_list[i].index(
                    max(iou_list[i])) if max(iou_list[i]) > 0 else -1
                inter_value_list[i] = max(iou_list[i])
            inter_index_set = set(inter_index_list)
            inter_index_set.discard(-1)
        # so far, for each gt, we map a seg result.
        # Then computer ratio = intersect/union
        for gtidx, segidx in enumerate(inter_index_list):
            if segidx != -1:

                if UseIOU:
                    value = iou_list[gtidx][segidx]
                else:
                    _intersection = intersection[gtidx, segidx]
                    value = _intersection/gt_area[gtidx]

                if value > 0.5:
                    TP += 1

                # LIST
                for k, thread in enumerate(PR_thread):
                    if value > thread:
                        TPLIST[k] += 1

        # add unmatched segmented result to FP
        seg_num = iou.shape[0]
        FNLIST = [len(gt_area) - f for f in TPLIST]

        FPLIST = [iou.shape[0] - t for t in TPLIST]
        # FPLIST = [f + (iou.shape[0] - t ) for t,f in zip(TPLIST,FPLIST)]
        # pdb.set_trace()
        PLIST = [t/(t+f) for t, f in zip(TPLIST, FPLIST)]
        RLIST = [t / (t+f) for t, f in zip(TPLIST, FNLIST)]
        itm = 0
        for p, r in zip(PLIST, RLIST):
            if (p+r) == 0:
                F1LIST[itm] = 0
            else:
                F1LIST[itm] = 2 * p * r/(p + r)
            itm += 1

        FN = len(gt_area) - TP
        FP = (iou.shape[0] - TP)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        fnr_1=FN/(TP + FN)
        if (recall + precision) == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall/(precision + recall)
        return PLIST, RLIST, F1, precision, recall, fnr_1,

    def caclulateMetrics(self, ious, ints, areas, dsc, gt):

        dc_thread = 0.7
        # p = self.params
        try:
            D, G = ious.shape
        except:
            G = len(gt)
            D = 0
        if D == 0:
            gtdsc = np.zeros((G))
            gttpr = np.zeros((G))
            # mdsc = 0
            # mtpr = 0
            FNR = 0
            FDR = 0
            allfpr = 0
            alldsc = gtdsc[gtdsc > dc_thread]
            alltpr = gttpr[gtdsc > dc_thread]
        else:

            allTPR = ints/areas
            if allTPR.max() > 1:

                pdb.set_trace()
            gtmid = - np.ones((G))
            gtdsc = np.zeros((G))
            gttpr = np.zeros((G))
            # AJI
            # DSC = np.zeros((1, 1))
            dsc_shape = dsc.shape

            while dsc.max() > dc_thread:
                maxind = np.argmax(dsc)
                # [detect, gt]
                ind = np.unravel_index(maxind, dsc_shape)
                maxdsc = dsc[ind]
                gtmid[ind[1]] = ind[0]
                gtdsc[ind[1]] = maxdsc
                gttpr[ind[1]] = allTPR[ind]
                dsc[ind[0]] = 0
                dsc[:, ind[1]] = 0
            # pdb.set_trace()
            alldsc = gtdsc[gtdsc > dc_thread]
            # mdsc = np.mean(alldsc)
            alltpr = gttpr[gtdsc > dc_thread]
            # mtpr = np.mean(alltpr)
            allfpr = gttpr[gtdsc < dc_thread]
            FNR = (G - np.count_nonzero(gtdsc))
            # mFNR = FNR/G
            FDR = (D - np.count_nonzero(gtdsc))
            # mFDR = FDR/D
            allfpr=0
        return alldsc, alltpr, allfpr, FNR, FDR

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        AJI = np.zeros((T, 1))
        DSC = np.zeros((G, 1))
        F1 = 0
        FNR = 0
        FDR = 0
        fnr_1=0
        mdsc = np.zeros(G)
        mtpr = np.zeros(G)
        mfpr = np.zeros(G)
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
            # compute AJI, DSC
            if p.iouType == 'segm':
                # if len(self.ious[imgId, catId][0]) > 0 else self.ious[imgId, catId][0]
                ious_for_AJI = self.ious_for_AJI[imgId, catId][0]
                # if len(self.ious[imgId, catId][1]) > 0 else self.ious[imgId, catId][1]
                intersection_for_AJI = self.ious_for_AJI[imgId, catId][1]
                # if len(self.ious[imgId, catId][2]) > 0 else self.ious[imgId, catId][2]
                union_for_AJI = self.ious_for_AJI[imgId, catId][2]
                area_for_AJI = self.ious_for_AJI[imgId, catId][3]
                dsc_for_AJI = self.ious_for_AJI[imgId, catId][4]

                if len(gt) != 0 and len(dt) != 0:
                    PLIST, RLIST, F1, precision, recall,fnr_1 = self.compute_F1(
                        area_for_AJI, ious_for_AJI, intersection_for_AJI, UseIOU=True)
                elif len(gt) == 0 and len(dt) > 0:
                    F1, precision, recall = 1, 0, 1
                    PLIST, RLIST = [0 for i in range(28)], [
                        1 for i in range(28)]
                elif len(gt) > 0 and len(dt) == 0:
                    F1, precision, recall = 0, 1, 0
                    PLIST, RLIST = [1 for i in range(28)], [
                        0 for i in range(28)]
                else:
                    F1, precision, recall = 1, 1, 1
                    PLIST, RLIST = [1 for i in range(28)], [
                        1 for i in range(28)]
                mdsc, mtpr, mtfp, FNR, FDR = self.caclulateMetrics(ious_for_AJI,
                                                             intersection_for_AJI,
                                                             area_for_AJI,
                                                             dsc_for_AJI, gt)

                # calculate AJI
                dc_thread = 0.6
                iouThrsAJI = [0.5]
                T = len(iouThrsAJI)
                G = len(gt)
                D = len(dt)
                gtm_for_AJI = - np.ones((T, G))
                dtm_for_AJI = - np.ones((T, D))
                gtIg_for_AJI = np.array([g['_ignore'] for g in gt])
                dtIg_for_AJI = np.zeros((T, D))
                # AJI
                AJI = np.zeros((T, 1))
                # IOU = np.zeros((T,1))
                INTERSECTION = np.zeros((T, 1))
                UNION = np.zeros((T, 1))
         

                DSC = np.zeros((G, 1))
                if not len(ious_for_AJI) == 0:
                    for tind, t in enumerate(iouThrsAJI):
                        for gind, g in enumerate(gt):
                            iou = min([t, 1 - 1e-10])
                            _intersection = 0
                            _union = 0
                            m = -1
                            _dsc = 0
                            for dind, d in enumerate(dt):
                                # if the dt already matched, continue
                                if dtm_for_AJI[tind, dind] > 0:
                                    continue
                                # continue to next dt unless better match made
                                if ious_for_AJI[dind, gind] < iou:
                                    continue
                                # if match successful and best so far, store it
                                iou = ious_for_AJI[dind, gind]
                                _union = union_for_AJI[dind, gind]
                                _intersection = intersection_for_AJI[dind, gind]
                                # _dsc = dsc[dind, gind]
                                m = dind
                            if m == -1:
                                continue

                            dtm_for_AJI[tind, m] = g['id']
                            gtm_for_AJI[tind, gind] = dt[m]['id']
                            INTERSECTION[tind, 0] = INTERSECTION[tind,
                                                                 0] + _intersection
                            UNION[tind, 0] = UNION[tind, 0] + _union
                            DSC[gind, 0] = _dsc
                        # add missing gt and dt

                        miss_gt = np.argwhere(gtm_for_AJI == -1)
                        miss_dt = np.argwhere(dtm_for_AJI == -1)
                        miss_gt = [gt[gt_index[1]]['segmentation']
                                   for gt_index in miss_gt]
                        miss_dt = [dt[dt_index[1]]['segmentation']
                                   for dt_index in miss_dt]
                        miss_gt = [maskUtils.area(f) for f in miss_gt]
                        miss_dt = [maskUtils.area(f) for f in miss_dt]
                        UNION[tind, 0] = UNION[tind, 0] + \
                            sum(miss_dt) + sum(miss_gt)

                    AJI = np.divide(INTERSECTION, UNION)
                    gtIg_for_AJI = np.array([0 for g in gt])
                    # DSC > 0.7
                    # good_ins = DSC[np.where(DSC > dc_thread)]
                    # if len(np.nonzero(good_ins)[0]) == 0:
                    #     DSC_GOOD = 0
                    # else:
                    #     DSC_GOOD = np.mean(good_ins)
                    # DSC_GOOD = np.asarray(DSC_GOOD).reshape(1,1)
                    # if catId =='nuclei':
                    #     pdb.set_trace()
                    # fno
                    # DSC[DSC> dc_thread] = 1
                    # DSC[DSC<=dc_thread] = 0
                    # FNO =  (G - np.sum(DSC))/G

                else:
                    AJI = np.zeros((T, 1))
                    # DSC_GOOD = np.zeros((T,1))
                    # FNO = 1

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                     for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(
            dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'AJI':          AJI,
            'F1':           F1,
            'DSC':          mdsc,
            'TPRp':          mtpr,
            'FPRp':          mfpr,
            'FNRo':         FNR,
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
            'num_G':        G,
            'num_D': D,
            'fnr_1':         fnr_1,
            'FDR': FDR,
        }

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox' or p.iouType == 'visible' or p.iouType == 'invisible':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}
        if p.iouType == 'segm':
            computeIoU_for_AJI = self.computeIoU_for_AJI
            self.ious_for_AJI = {(imgId, catId): computeIoU_for_AJI(imgId, catId)
                                 for imgId in p.imgIds
                                 for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'visible':
            g = [g['visible_mask'] for g in gt]
            d = [d['visible_mask'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        elif p.iouType == 'invisible':
            g = [g['invisible_mask'] for g in gt]
            d = [d['invisible_mask'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(
                p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(
                1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[11] = _summarize(
                0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        def _summarizeSegm():
            AJI = {}
            DSC = {}
            TPRp = {}
            FPRp = {}
            FNR = {}
            FNR_1 = {}
            FDR={}
            
            F1_score = {}
            summarize_metric = np.zeros((2, 7))
            from tabulate import tabulate

            for catId, cat in enumerate(self._paramsEval.catIds):
                _count = 0
                aji = np.zeros((len(self._paramsEval.iouThrs), 1))
                F1 = 0
                fnr_1=0
                dsc = []
                tprp = []
                fnr = []
                fdr = []
                num_G = 0
                num_D = 0
                for result in self.evalImgs:
                    if result is None:
                        # skip len(gt)=0 & len(dt)
                        continue
                    if result['category_id'] == cat:
                        aji = aji + result['AJI']
                        F1 = F1 + result['F1']
                        fnr_1= fnr_1 + result['fnr_1']
                        dsc.extend(list(result['DSC']))
                        tprp.extend(list(result['TPRp']))
                        fnr.append(result['FNRo'])
                        fdr.append(result['FDR'])
                        num_G = num_G +result["num_G"]
                        num_D = num_D +result["num_G"]
                        _count += 1
                aji = np.divide(aji, _count)
                F1 = F1/_count
                fnr_1 = fnr_1/_count
                dsc = sum(dsc)/(len(dsc)+1e-10)
                tprp = sum(tprp)/(len(tprp)+1e-10)
                fnr = sum(fnr)/num_G
                fdr = sum(fdr)/num_D

                AJI[cat] = aji[0]
                F1_score[cat] = F1
                DSC[cat] = dsc
                TPRp[cat] = tprp
                FNR[cat] = fnr
                FNR_1[cat] = fnr_1
                FDR[cat] = fdr


                if cat == 0:
                    cat_name = 'Cell'
                    summarize_metric[0] = [
                        AJI[cat][0], F1_score[cat], DSC[cat], TPRp[cat],FNR[cat],FNR_1[cat],FDR[cat]]
                    table = [["AJI", "F1", "DSC", "TPRp","FNRp","FNR_o","FDR"], summarize_metric[0]]
                else:
                    cat_name = 'Nuclei'
                    summarize_metric[1] = [
                        AJI[cat][0], F1_score[cat], DSC[cat], TPRp[cat],FNR[cat],FNR_1[cat],FDR[cat]]
                    table = [["AJI", "F1", "DSC", "TPRp","FNRp","FNR_o","FDR"], summarize_metric[1]]
                # table = [["AJI", "F1", "DSC", "TPRp"], [AJI[cat][0], F1_score[cat], DSC[cat]]]
                print("===========================", cat_name, "===========================")
                print(tabulate(table, headers='firstrow', tablefmt='github'))
            print("========================== Average ===========================")
            avg = np.mean(summarize_metric, axis=0)
            table = [["AJI", "F1", "DSC", "TPRp", "FNR","FNR_1","FDR"], [avg[0], avg[1], avg[2], avg[3],avg[4],avg[5],avg[6]]]
            print(tabulate(table, headers='firstrow', tablefmt='github'))

            return AJI
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox' or iouType == 'visible' or iouType == 'invisible':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        if iouType == 'segm':
            _summarizeSegm()

    def __str__(self):
        self.summarize()


class AmodalParams:
    '''
    Params for coco evaluation api
    '''

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(
            np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(
            np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2],
                        [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(
            np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(
            np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2],
                        [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox' or iouType == 'visible' or iouType == 'invisible':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
