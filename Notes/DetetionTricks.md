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