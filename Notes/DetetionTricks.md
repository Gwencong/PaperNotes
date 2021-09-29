# 目标检测中的tricks
## 1.数据增强
### MixUp
Mixup最初用于图像分类，将两张图片a与b按照一定的权重$\lambda$和1-$\lambda$融合得到融合图片，在训练过程中分别计算融合图片的预测与两张原始图片label的loss，然后按照对应的权重$\lambda$和1-$\lambda$对loss进行加权相加得到融合图片的loss，如下图所示。

![img](https://pic3.zhimg.com/v2-4e09d5f5f759fb2015ef72bf15fc9076_b.png)

在目标检测中，图片的融合和分类的处理一样，融合后的图片尺寸分别是两张图片中最大的长宽，保持了在融合后的边界框的绝对位置的不变，融合后的图片的bboxes和labels即为原图的box和label，如下图所示：

![img](https://pic2.zhimg.com/80/v2-a24e855e639eeb4f3a480ba2b6053789_1440w.jpg?source=1940ef5c)

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

<div align=center>
<img src="https://images.gitee.com/uploads/images/2021/0929/091837_bc056072_9801188.png" sytle="zoom:70%;" />
</div>

## CutMix
CutMix将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配。cutmix、cutout、mixup的主要区别在于cutout和cutmix就是填充区域像素值的区别；mixup和cutmix是混合两种样本方式上的区别：mixup是将两张图按比例进行插值来混合样本，cutmix是采用cut部分区域再补丁的形式去混合图像，不会有图像混合后不自然的情形。
<div align=center>
<img src="https://images.gitee.com/uploads/images/2021/0929/093459_e27e02ab_9801188.png" width="70%" height="70%" />
</div>

