# 目标检测中的tricks
## 1.数据增强
### MixUp
Mixup最初用于图像分类，将两张图片a与b按照一定的权重$\lambda$和1-$\lambda$融合得到融合图片，在训练过程中分别计算融合图片的预测与两张原始图片label的loss，然后按照对应的权重$\lambda$和1-$\lambda$对loss进行加权相加得到融合图片的loss，如下图所示。

![img](https://pic3.zhimg.com/v2-4e09d5f5f759fb2015ef72bf15fc9076_b.png)

在目标检测中，图片的融合和分类的处理一样，融合后的图片尺寸分别是两张图片中最大的长宽，保持了在融合后的边界框的绝对位置的不变，融合后的图片的bboxes和labels即为原图的box和label，如下图所示：

<img src="https://pic2.zhimg.com/80/v2-a24e855e639eeb4f3a480ba2b6053789_1440w.jpg?source=1940ef5c" alt="img" style="zoom: 50%;" />

python代码如下：

```python
def mix_up(image, bboxes, labels, image2, bboxes2, labels2):
    # beta(a=1, b=1) = uniform(0, 1)
    lambd = random.uniform(0, 1)

    height = max(image.shape[0], image2.shape[0])
    width = max(image.shape[1], image2.shape[1])
    mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
    mix_img[:image.shape[0], :image.shape[1], :] = image.astype('float32') * lambd
    mix_img[:image2.shape[0], :image2.shape[1], :] += image2.astype('float32') * (1. - lambd)
    mix_img = mix_img.astype(np.uint8)

    mix_bboxes = np.vstack((bboxes, bboxes2))
    mix_labels = np.hstack((labels, labels2))
    mix_weights = np.hstack((np.full(len(bboxes), lambd),
                             np.full(len(bboxes2), (1. - lambd))))

    return mix_img, mix_bboxes, mix_labels, mix_weights
```

MixUp能够增加背景的复杂性，从而使得训练的网络能够更好的适应背景的复杂性，但是如果一个图片背景简单，一个背景复杂，简单背景的图基本就会被覆盖，肉眼看上去就会基本没上就有用信息了，或者如果是一张图有框，一张图没有框，mix在一起，就退化为一张图的loss加一个权重，这些情况下mixup就难以起到很好的效果。

## Cutout

Cutout的提出主要是为了解决目标遮挡问题，通过对训练数据模拟遮挡，一方面能解决现实中遮挡的问题，另一方面也能让模型更好的学习利用上下文的信息，作者描述了两种Cutout的设计理念：

+ 专门从图像的输入中删除图像的重要特征，为了鼓励网络考虑不那么突出的特征。具体操作：在训练的每个epoch过程中，保存每张图片对应的最大激活特征图，在下一个训练回合，对每张图片的最大激活图进行上采样到和原图一样大，然后使用阈值划分为二值图，盖在原图上再输入到cnn中进行训练。因此，这样的操作可以有针对性的对目标进行遮挡。
+ 选择一个固定大小的正方形区域，然后将该区域填充为0即可，为了避免全0区域对训练的影响，需要对数据中心归一化到0。并且以一定概率（50%）允许擦除区域不完全在原图像中。

<img src="https://github.com/Gwencong/PaperNotes/blob/main/imgs/image-20210928220735263.png" alt="image-20210928220735263" style="zoom:70%;" />

## DropBlock

dropout用于全连接层能够提高模型泛化性能和减缓过拟合，但是用于卷积神经网络则不太有效，dropout在卷积层不work的原因可能是由于卷积层的特征图中相邻位置元素在空间上共享语义信息，所以尽管某个单元被dropout掉，但与其相邻的元素依然可以保有该位置的语义信息，信息仍然可以在卷积网络中流通。

DropBlock是一种针对卷积网络的dropout，按块来丢弃，是一种结构化的dropout形式，它将feature map相邻区域中的单元放在一起drop掉。除了卷积层外，在跳跃连接中应用DropbBlock可以提高精确度。

![image-20210929181315166](https://github.com/Gwencong/PaperNotes/blob/main/imgs/image-20210929181315166.png)

## Label Smoothing 

对于损失函数，我们需要用预测概率去拟合真实概率，而拟合one-hot的真实概率函数会带来两个问题：

+ 无法保证模型的泛化能力，容易造成过拟合；
+ one-hot表示会使分类之间的cluster更加紧凑，增加类间距离，减少类内距离，提高泛化性，而由梯度有界可知，这种情况很难适应。会造成模型过于相信预测的类别。

使用 label smoothing 可以缓解这个问题，用更新的标签向量$y_i$来替换传统的ont-hot编码的标签向量$y_{hot}$:

$\hat{y}_i=y_{hot}(1−α)+α/K$

其中K为多分类的类别总个数，α是一个较小的超参数（一般取0.1），即 

$$\hat{y}_i = \begin{cases} 1-\alpha, &i=target \\ \alpha/K, &i \neq target \\ \end{cases}$$

label smoothing后的分布就相当于往真实分布中加入了噪声，避免模型对于正确标签过于自信，使得预测正负样本的输出值差别不那么大，从而避免过拟合，提高模型的泛化能力
