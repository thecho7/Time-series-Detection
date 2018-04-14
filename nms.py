import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0]
    x2 = dets[:, 1]

    scores = dets[:, 2]

    areas = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        inter = w
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
