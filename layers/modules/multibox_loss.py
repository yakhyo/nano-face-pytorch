import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function

    This class computes the loss for the SSD model by matching ground truth boxes
    with prior boxes, encoding the localization targets, and performing hard negative mining.

    Steps:
    1. Produce confidence target indices by matching ground truth boxes with prior boxes
       that have a Jaccard index greater than a specified threshold (default: 0.5).
    2. Produce localization targets by encoding variances into offsets of ground truth boxes
       and their matched prior boxes.
    3. Perform hard negative mining to filter excessive negative examples, maintaining a
       negative:positive ratio (default: 3:1).

    Objective Loss:
        L(x, c, l, g) = (Lconf(x, c) + αLloc(x, l, g)) / N
        - Lconf is the CrossEntropy Loss
        - Lloc is the SmoothL1 Loss, weighted by α (default: 1)
        - N is the number of matched prior boxes

    Args:
        priors (tensor): Prior boxes, shape: [num_priors, 4].
        threshold (float): Overlap threshold for matching.
        neg_pos_ratio (float): Ratio of negative to positive examples.
        alpha (float): Localization loss weight
        device: Device to keep the priors (default: "cpu")
    """

    def __init__(
        self,
        priors,
        threshold,
        neg_pos_ratio,
        variance=[0.1, 0.2],
        device=torch.device("cpu")
    ):
        super().__init__()
        self.priors = priors  # torch.size(num_priors, 4)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance
        self.device = device

    def forward(self, predictions, ground_truth):
        """
        Multibox Loss
        Args:
            predictions (tuple): A tuple containing localization predictions, 
            confidence predictions, and prior boxes from the SSD network.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 5] (last index is the label).
        """

        loc_preds, conf_preds, landmark_preds = predictions
        batch_size = loc_preds.size(0)
        num_classes = conf_preds.size(2)
        num_priors = self.priors.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_targets = torch.Tensor(batch_size, num_priors, 4).to(self.device)
        lnm_targets = torch.Tensor(batch_size, num_priors, 10).to(self.device)
        conf_targets = torch.LongTensor(batch_size, num_priors).to(self.device)

        for idx in range(batch_size):
            truths = ground_truth[idx][:, :4]
            labels = ground_truth[idx][:, -1]
            landms = ground_truth[idx][:, 4:14]
            defaults = self.priors
            match(
                self.threshold,
                truths,
                defaults,
                self.variance,
                labels,
                landms,
                loc_targets,
                conf_targets,
                lnm_targets,
                idx
            )

        # Landmark Loss (Smooth L1)
        pos1_mask = conf_targets > 0
        num_pos_landm = pos1_mask.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.sum().float(), 1)
        pos_idx1 = pos1_mask.unsqueeze(pos1_mask.dim()).expand_as(landmark_preds)

        landm_p = landmark_preds[pos_idx1].view(-1, 10)
        lnm_targets = lnm_targets[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, lnm_targets, reduction='sum')

        pos_mask = conf_targets != 0
        conf_targets[pos_mask] = 1

        # Localization Loss (Smooth L1)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_preds)
        loc_p = loc_preds[pos_idx].view(-1, 4)
        loc_targets = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_targets, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_preds.view(-1, num_classes)
        conf_loss = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_targets.view(-1, 1))

        # Hard Negative Mining
        conf_loss[pos_mask.view(-1, 1)] = 0  # filter out pos_mask boxes for now
        conf_loss = conf_loss.view(batch_size, -1)
        _, loss_idx = conf_loss.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos_mask.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos_mask.size(1)-1)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos_mask.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg_mask.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
        targets_weighted = conf_targets[(pos_mask+neg_mask).gt(0)]
        conf_loss = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.sum().float(), 1)
        loc_loss /= N
        conf_loss /= N
        loss_landm /= N1

        return loc_loss, conf_loss, loss_landm
