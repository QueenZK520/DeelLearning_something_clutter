import torch
import torch.nn as nn

"""
multi class classification 
smooth label(soft label) CrossEntropy lossfunction
"""
class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        tmp = -true_dist * pred
        if self.weight is not None:
            tmp = torch.sum(tmp, dim=self.dim)
            loss = 0
            weight_sum = 0
            for i in range(tmp.shape[0]):
                loss = loss + tmp[i] * self.weight[target[i]]
                weight_sum = weight_sum + self.weight[target[i]]
            loss = loss / weight_sum
        else:
            loss = torch.mean(torch.sum(tmp, dim=self.dim))
        return loss
      

# test
print("-------")
target = torch.Tensor([2, 1])
input = torch.Tensor([ 0.6698, -0.4865,  0.3717, -2.1004, -1.4200],
        [ 1.1654, -0.0316,  0.4862, -2.2637, -1.7555]])
print(nn.CrossEntropy_criterion()(out_cls, cls_label))
print(LabelSmoothingLossCanonical(smoothing=0)(out_cls, cls_label))

print("-------")
# with weight
weight = [1, 10, 1.44, 25, 8]
weight = torch.from_numpy(np.array(weight)).float()
print(nn.CrossEntropyLoss(weight=weight)(out_cls, cls_label))
print(LabelSmoothingLossCanonical(smoothing=0, weight=weight)(out_cls, cls_label))

print("-------")
# smooth label
print(LabelSmoothingLossCanonical(smoothing=0.1)(out_cls, cls_label))

print("-------")
# smooth label, with weight
print(LabelSmoothingLossCanonical(smoothing=0.1, weight=weight)(out_cls, cls_label))
