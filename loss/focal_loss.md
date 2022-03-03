Focal loss 中两个加权参数的原理和产生的影响
https://blog.csdn.net/yangyehuisw/article/details/106216283

首先需要明确一个在损失函数中的加权细节：想要在损失函数中对样本进行加权，那么加权的思路应该要是逆向的。因为损失函数的优化目标是越小越好，所以你越想保护的部分应该给予小权重，使得这部分可以大。而越想惩罚的部分，应该给予大权重，这样强制让他们只能是小的。
![image](https://user-images.githubusercontent.com/59159306/156333239-7a6fee27-4f4b-408b-b6ad-3b7e1b5b87b8.png)

 

Focal loss ： 

 

其中  \alpha 类似与class weight 给类别加权重。如果 y = 1 类样本个数大于 y = 0， 那么  \alpha 应该小于 0.5，保护样本少的类，而多惩罚样本多的类。结论是样本越不平衡， \alpha 应该越靠近 0 或者 1。

而 \gamma 的作用是竟然把难例分开，这个参数越大，导致的后果是预测的概率值越偏向于0～1的两端。具体推理如下图所示：
![image](https://user-images.githubusercontent.com/59159306/156333482-6642366b-e6e2-44bb-8f5a-690c06c0006a.png)


```python

class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        print("Focal loss: alpha is {}, gamma is {}.".format(self.alpha, self.gamma))
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == "__main__":
    train_label_list = []
    for i in train_dataset.datas:
        train_label_list.append(i[1])
    countTrainD = np.zeros(classes_num)
    for _l in train_label_list:
        countTrainD[_l] += 1
    print("Count of train dataset (not dataloader) :{}".format(countTrainD))
    countTrainD = 1. / countTrainD
    criterion = MultiFocalLoss(num_class=classes_num, alpha=countTrainD, gamma=2)
```
 

