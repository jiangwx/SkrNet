import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def bbox_anchor_iou(bbox, anchor):
    # bbox[0] ground truth width, bbox[1] ground truth hight, anchor[0] anchor width, anchor[1], anchor hight
    inter_area = torch.min(bbox[0], anchor[0]) * torch.min(bbox[1], anchor[1])
    union_area = (bbox[0] * bbox[1] + 1e-16) + anchor[0] * anchor[1] - inter_area
    return inter_area / union_area

def build_targets(pred_boxes, target, anchors, ignore_thres):
    # target.shape [nB,4],(center x, center y, w, h)
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nH = pred_boxes.size(2)
    nW = pred_boxes.size(3)

    obj_mask   = torch.cuda.BoolTensor(nB, nA, nH, nW).fill_(False)
    noobj_mask = torch.cuda.BoolTensor(nB, nA, nH, nW).fill_(True)
    tx         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    ty         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    tw         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    th         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    tconf      = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)

    gt_x = target[:,1]*nW # ground truth x
    gt_y = target[:,2]*nH # ground truth y
    gt_w = target[:,2]*nW # ground truth w
    gt_h = target[:,3]*nH # ground truth h
    grid_x = gt_x.long()  # grid x
    grid_y = gt_y.long()  # grid y

    for b in range(nB):
        ious = torch.stack([bbox_anchor_iou((gt_w[b],gt_h[b]), anchor) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        obj_mask[b, best_n, grid_y[b], grid_x[b]] = True
        noobj_mask[b, best_n, grid_y[b], grid_x[b]] = False
        for i,anchor_ious in enumerate(ious):
            if(anchor_ious > ignore_thres):
                noobj_mask[b,i, grid_y[b], grid_x[b]] = False
        # Coordinates
        tx[b, best_n, grid_y[b], grid_x[b]] = gt_x[b] - gt_x[b].floor()
        ty[b, best_n, grid_y[b], grid_x[b]] = gt_y[b] - gt_y[b].floor()
        # Width and height
        tw[b, best_n, grid_y[b], grid_x[b]] = torch.log(gt_w[b] / anchors[best_n][0] + 1e-16)
        th[b, best_n, grid_y[b], grid_x[b]] = torch.log(gt_h[b] / anchors[best_n][1] + 1e-16)

    tconf = obj_mask.float()

    return obj_mask, noobj_mask, tx, ty, tw, th, tconf


class RegionLoss(nn.Module):
    def __init__(self, anchors=[[1.4940052559648322,2.3598481287086823],[4.0113013115312155,5.760873975661669]]):
        super(RegionLoss, self).__init__()
        self.anchors = torch.cuda.FloatTensor(anchors)
        self.num_anchors = len(anchors)
        self.noobject_scale = 1
        self.object_scale = 5
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
       
        nB = output.data.size(0)
        nA = self.num_anchors
        nH = output.data.size(2)
        nW = output.data.size(3)

        output  = output.view(nB, nA, 5, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        x    = torch.sigmoid(output[...,0])
        y    = torch.sigmoid(output[...,1])
        w    = output[...,2]
        h    = output[...,3]
        conf = torch.sigmoid(output[...,4])
        
        pred_boxes = torch.cuda.FloatTensor(4,nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = self.anchors[:,0]
        anchor_h = self.anchors[:,1]
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

        pred_boxes[0] = x.data.view(nB*nA*nH*nW) + grid_x
        pred_boxes[1] = y.data.view(nB*nA*nH*nW) + grid_y
        pred_boxes[2] = torch.exp(w.data).view(nB*nA*nH*nW) * anchor_w
        pred_boxes[3] = torch.exp(h.data).view(nB*nA*nH*nW) * anchor_h
        pred_boxes = pred_boxes.transpose(0,1).contiguous().view(nB,nA,nH,nW,4)
        #pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(nB,nA,nH,nW,4))
        obj_mask, noobj_mask, tx, ty, tw, th, tconf = build_targets(pred_boxes, target.data, self.anchors, self.thresh)


        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        obj_mask = Variable(obj_mask.cuda())
        noobj_mask  = Variable(noobj_mask.cuda())
        
        loss_scale = (2 - tw*th)*1.5
        loss_x = nn.MSELoss()(x[obj_mask], tx[obj_mask])
        loss_y = nn.MSELoss()(y[obj_mask], ty[obj_mask])
        loss_w = nn.MSELoss()(w[obj_mask], tw[obj_mask])
        loss_h = nn.MSELoss()(h[obj_mask], th[obj_mask])
        loss_conf_obj   = nn.MSELoss()(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = nn.MSELoss()(conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.object_scale * loss_conf_obj + self.noobject_scale * loss_conf_noobj

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf

        print('loss: x %f, y %f, w %f, h %f, conf %f, total %f' % (loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0],  loss.data[0]))

        return loss