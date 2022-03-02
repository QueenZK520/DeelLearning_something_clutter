Focal loss 中两个加权参数的原理和产生的影响
https://blog.csdn.net/yangyehuisw/article/details/106216283

首先需要明确一个在损失函数中的加权细节：想要在损失函数中对样本进行加权，那么加权的思路应该要是逆向的。因为损失函数的优化目标是越小越好，所以你越想保护的部分应该给予小权重，使得这部分可以大。而越想惩罚的部分，应该给予大权重，这样强制让他们只能是小的。
![image](https://user-images.githubusercontent.com/59159306/156333239-7a6fee27-4f4b-408b-b6ad-3b7e1b5b87b8.png)

 

Focal loss ： 

 

其中  \alpha 类似与class weight 给类别加权重。如果 y = 1 类样本个数大于 y = 0， 那么  \alpha 应该小于 0.5，保护样本少的类，而多惩罚样本多的类。结论是样本越不平衡， \alpha 应该越靠近 0 或者 1。

而 \gamma 的作用是竟然把难例分开，这个参数越大，导致的后果是预测的概率值越偏向于0～1的两端。具体推理如下图所示：
![image](https://user-images.githubusercontent.com/59159306/156333482-6642366b-e6e2-44bb-8f5a-690c06c0006a.png)



 

