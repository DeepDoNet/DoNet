# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn
import copy

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers.mask_ops import _do_paste_mask

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
import matplotlib.pyplot as plt

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.unlabeled = False
        self.semi_supervised = False
        if hasattr(cfg.MODEL, 'SEMI'):
            self.semi_supervised = cfg.MODEL.SEMI

        self.rpn_attention = False
        if hasattr(cfg.MODEL,'RPN_ATTENTION'):
            self.rpn_attention = cfg.MODEL.RPN_ATTENTION

        if self.rpn_attention:  # this is only supported for DBRCNN or TBRCNN
            assert (cfg.MODEL.ROI_HEADS.NAME == "DoubleBranchROIHeads" or cfg.MODEL.ROI_HEADS.NAME == "TripleBranchROIHeads")
            self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

            cfg_for_attention = copy.deepcopy(cfg)
            # cfg_for_attention.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
            cfg_for_attention.MODEL.ROI_HEADS.FOR_NUCLEI = True
            self.nuclei_roi_heads = build_roi_heads(cfg_for_attention, self.backbone.output_shape())

            self.proposal_generator_with_attention = build_proposal_generator(cfg, self.backbone.output_shape())
        else:
            self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())


        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # storage = get_event_storage()
        # current_iter = storage.iter
        if self.semi_supervised and self.training:
            self.unlabeled = batched_inputs[0]['file_name'].__contains__('unlabeled')#batched_inputs
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # if not torch.isfinite(images.tensor).all():
        #     print("error: NaN detected!")
        # imgplot = plt.imshow(torch.Tensor.cpu(images.tensor[0][0]).detach().numpy())
        # plt.colorbar()
        # plt.axis('off')
        # plt.xticks([])
        # plt.savefig("/home/jh/zrs_ORCNN/paper/fig3/orig_image_"+batched_inputs[0]['file_name'][-10:], bbox_inches='tight', pad_inches = -0.1)
        # plt.close()
        features = self.backbone(images.tensor)
        # if not (torch.isfinite(features['p2']).all() and torch.isfinite(features['p3']).all() and torch.isfinite(features['p4']).all() and torch.isfinite(features['p5']).all() and torch.isfinite(features['p6']).all()):
        #     print("error: NaN detected!")

        if self.rpn_attention == False:
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, unlabeled=self.unlabeled)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)

            losses = {}
            if self.unlabeled:
                del proposal_losses
            else:
                losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        else: # rpn attention is on
            ###################### 1. Generate Segmentation For cell ######################
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, unlabeled=self.unlabeled)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, cell_detector_losses, whole_mask_logits, selected_proposals = self.roi_heads(images, features, proposals, gt_instances)


            ###################### 2. Attain RPN attention map ######################
            rpn_attention_map=[]
            proposal_counter = 0
            for proposals_per_image in selected_proposals:
                image_size = proposals_per_image.image_size[0]
                # Getting the cell mask of the specified picture
                cell_whole_mask_list = _do_paste_mask(whole_mask_logits[proposal_counter:proposal_counter + proposals_per_image.proposal_boxes.tensor.size()[0],:1,:,:], 
                                proposals_per_image.proposal_boxes.tensor, image_size, image_size, skip_empty = False)[0]
                
                rpn_attention_map_per_image = torch.zeros(size = (image_size, image_size), device = cell_whole_mask_list.device)
                
                for cell_whole_mask in cell_whole_mask_list:
                    rpn_attention_map_per_image= rpn_attention_map_per_image+torch.clamp(cell_whole_mask,min = 0)

                batch_image_size = images.tensor.size()[2]
                rpn_attention_map_per_image=_do_paste_mask(rpn_attention_map_per_image.resize(1, 1, image_size, image_size), torch.tensor([[0,0,image_size, image_size]], device=images.tensor.device),batch_image_size, batch_image_size, skip_empty = False)[0][0]

                rpn_attention_map_per_image = rpn_attention_map_per_image.sigmoid().resize(1, 1, batch_image_size, batch_image_size)
                proposal_counter += proposals_per_image.proposal_boxes.tensor.size()[0]
                rpn_attention_map.append(rpn_attention_map_per_image)
            
            features_with_attention = {}
            for key,feature in features.items():
                features_with_attention[key] = feature.detach().clone()
            for index, rpn_attention_map_per_image in enumerate(rpn_attention_map):
                for key,feature in features.items():
                    features_with_attention[key][index] = feature[index] * torch.nn.functional.interpolate(rpn_attention_map_per_image, feature.size()[2:])[0]
                    # if key == 'p2' and index == 0 and current_iter == 10000:
                    #     pass
                    #     plt_image = torch.Tensor.cpu(feature[0][26]).detach().squeeze().numpy()
                    #     imgplot = plt.imshow(plt_image)              
                    #     plt.axis('off')
                    #     plt.xticks([])
                    #     plt.savefig("/home/jh/zrs_ORCNN/paper/fig3/feature_map_" + str(current_iter) + ".svg", bbox_inches='tight', pad_inches = -0.1)
                    #     plt.close()

                    #     imgplot = plt.imshow(torch.Tensor.cpu(rpn_attention_map_per_image[0][0]).detach().numpy())
                    #     plt.axis('off')
                    #     plt.xticks([])
                    #     plt.savefig("/home/jh/zrs_ORCNN/paper/fig3/rpn_attention_map_per_image_" + str(current_iter) + ".svg", bbox_inches='tight', pad_inches = -0.1)
                    #     plt.close()

                    #     imgplot = plt.imshow(torch.Tensor.cpu(features_with_attention['p2'][0][26]).detach().numpy())
                    #     plt.axis('off')
                    #     plt.xticks([])
                    #     plt.savefig("/home/jh/zrs_ORCNN/paper/fig3/features_with_attention_" + str(current_iter) + ".svg", bbox_inches='tight', pad_inches = -0.1)
                    #     plt.close()

                    #     imgplot = plt.imshow(torch.Tensor.cpu(images.tensor[0][0]).detach().numpy())
                    #     plt.axis('off')
                    #     plt.xticks([])
                    #     plt.savefig("/home/jh/zrs_ORCNN/paper/fig3/ori_image_" + str(current_iter) + ".svg", bbox_inches='tight', pad_inches = -0.1)
                    #     plt.close()
                        # imgplot = plt.imshow(torch.Tensor.cpu(features_with_attention['p2'][0][26]).detach().numpy())
                        # plt.colorbar()
                        # plt.savefig("/home/jh/zrs_ORCNN/features_with_attention.svg")
                        # plt.close()
                        # imgplot = plt.imshow(torch.Tensor.cpu(rpn_attention_map_per_image[0][0]).detach().numpy())
                        # plt.colorbar()
                        # plt.savefig("/home/jh/zrs_ORCNN/rpn_attention_map_per_image.svg")
                        # plt.close()
            ###################### 3. going trough an independent path for nuclei ######################
            proposals_with_attention, proposal_losses_with_attention = self.proposal_generator_with_attention(images, features_with_attention, gt_instances, with_attention=True, unlabeled=self.unlabeled)
            _, nuclei_detector_losses, _, _ = self.nuclei_roi_heads(images, features_with_attention, proposals_with_attention, gt_instances)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)
            losses = {}


            losses.update({key+"_cell":value for key, value in cell_detector_losses.items()})
            losses.update({key+"_nuclei":value for key, value in nuclei_detector_losses.items()})

            if self.unlabeled:
                del proposal_losses
                del proposal_losses_with_attention
            else:
                losses.update(proposal_losses)
                losses.update(proposal_losses_with_attention)
            return losses



    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # imgplot = plt.imshow(torch.Tensor.cpu(images.tensor[0][0]).detach().numpy())
        # plt.colorbar()
        # plt.savefig("/home/jh/zrs_ORCNN/image.svg")
        # plt.close()

        if detected_instances is None:
            if not self.rpn_attention:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else: # rpn_attention is on
                ###################### 1. Generate Segmentation For cell ######################
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _, whole_mask_logits = self.roi_heads(images, features, proposals, None)
                
                ###################### 2. Attain RPN attention map ######################
                rpn_attention_map=[]
                proposal_counter = 0
                for proposals_per_image in results:
                    image_size = proposals_per_image.image_size[0]
                    # Getting the cell mask of the specified picture
                    cell_whole_mask_list = _do_paste_mask(whole_mask_logits[proposal_counter:proposal_counter + proposals_per_image.pred_boxes.tensor.size()[0],:1,:,:], 
                                    proposals_per_image.pred_boxes.tensor, image_size, image_size, skip_empty = False)[0]
                    
                    rpn_attention_map_per_image = torch.zeros(size = (image_size, image_size), device = cell_whole_mask_list.device)
                    for cell_whole_mask in cell_whole_mask_list:
                        rpn_attention_map_per_image= rpn_attention_map_per_image+torch.clamp(cell_whole_mask,min = 0)

                    batch_image_size = images.tensor.size()[2]
                    rpn_attention_map_per_image=_do_paste_mask(rpn_attention_map_per_image.resize(1, 1, image_size, image_size), torch.tensor([[0,0,image_size, image_size]], device=images.tensor.device),batch_image_size, batch_image_size, skip_empty = False)[0][0]

                    rpn_attention_map_per_image = rpn_attention_map_per_image.sigmoid().resize(1, 1, batch_image_size, batch_image_size)
                    proposal_counter += proposals_per_image.pred_boxes.tensor.size()[0]
                    rpn_attention_map.append(rpn_attention_map_per_image)
                
                features_with_attention = {}
                for key,feature in features.items():
                    features_with_attention[key] = feature.detach().clone()
                for index, rpn_attention_map_per_image in enumerate(rpn_attention_map):
                    for key,feature in features.items():
                        features_with_attention[key][index] = feature[index] * torch.nn.functional.interpolate(rpn_attention_map_per_image, feature.size()[2:])[0]
                        if key == 'p2' and index == 0:
                            pass
                            # imgplot = plt.imshow(torch.Tensor.cpu(feature[0][26]).detach().numpy(), cmap = 'jet')
                            # plt.colorbar()
                            # plt.savefig("/home/jh/zrs_ORCNN/feature.svg")
                            # plt.savefig("/home/jh/zrs_ORCNN/feature.png")
                            # plt.close()
                            # imgplot = plt.imshow(torch.Tensor.cpu(features_with_attention['p2'][0][26]).detach().numpy(), cmap = 'jet')
                            # plt.colorbar()
                            # plt.savefig("/home/jh/zrs_ORCNN/features_with_attention.svg")
                            # plt.savefig("/home/jh/zrs_ORCNN/features_with_attention.png")
                            # plt.close()
                            # imgplot = plt.imshow(torch.Tensor.cpu(rpn_attention_map_per_image[0][0]).detach().numpy(), cmap = 'jet')
                            # plt.colorbar()
                            # plt.savefig("/home/jh/zrs_ORCNN/rpn_attention_map_per_image.svg")
                            # plt.savefig("/home/jh/zrs_ORCNN/rpn_attention_map_per_image.png")
                            # plt.close()
                ###################### 3. going trough an independent path for nuclei ######################
                proposals_with_attention, _ = self.proposal_generator_with_attention(images, features_with_attention, None, with_attention = True)
                nuclei_results, _, _ = self.nuclei_roi_heads(images, features_with_attention, proposals_with_attention, None)

                assert len(results) == 1
                result = results[0]
                nuclei_result = nuclei_results[0]

                result_cell = (result.pred_classes == 0)
                nuclei_result_nuclei = (nuclei_result.pred_classes == 1)
                result.pred_boxes.tensor = torch.cat((result.pred_boxes[result_cell].tensor, nuclei_result.pred_boxes[nuclei_result_nuclei].tensor), 0)
                result.scores = torch.cat((result.scores[result_cell], nuclei_result.scores[nuclei_result_nuclei]), 0)
                result.pred_classes = torch.cat((result.pred_classes[result_cell], nuclei_result.pred_classes[nuclei_result_nuclei]), 0)
                result.pred_masks = torch.cat((result.pred_masks[result_cell], nuclei_result.pred_masks[nuclei_result_nuclei]), 0)
                if result.has('pred_invisible_masks'):
                    result.remove("pred_invisible_masks")
                if result.has('pred_visible_masks'):
                    result.remove("pred_visible_masks")
                results[0] = result
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
