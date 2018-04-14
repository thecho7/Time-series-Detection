from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride # In Faster R-CNN = [16, ]
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales))).float()
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, start, end) specifying an image batch index n and a
        # interval (start, end)
        # top[0].reshape(1, 3)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input): #input : (rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :] # [1, 2 * #anchor, length] -> [1, #anchor, length]
        bbox_deltas = input[1] # [1, #anchor * 2, length]
        im_info = input[2]
        cfg_key = input[3] # 'TRAIN' or 'TEST'

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0) # Normally 1

        feat_length = scores.size(2)
        shift = np.arange(0, feat_length) * self._feat_stride # ANCHORS CENTER
        shifts = torch.from_numpy(shift.transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors # # of bbox per one anchor
        K = shifts.size(0) # total # of anchors

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 2) + shifts.view(K, 1, 2)
        # self._anchors : (num_anchor(?), coordinate(2))
        # shifts : (# of total bbox(?), coordinate(2))
        # anchors : (# of total bbox(?), num_anchor(?), coordinate(2))
        
        anchors = anchors.view(1, K * A, 2).expand(batch_size, K * A, 2)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        # bbox_deltas : [1, #anchor * 2, length] -> [1, length, #anchor * 2] -> [1, -1, coordinate(2)]
        bbox_deltas = bbox_deltas.permute(0, 2, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 2)

        # Same story for the scores:
        scores = scores.permute(0, 2, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)
        
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 3).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

@@@@            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
