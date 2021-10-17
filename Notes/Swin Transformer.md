# Swin Transformer 阅读笔记

论文地址：https://arxiv.org/pdf/2103.14030.pdf

代码：https://github.com/microsoft/Swin-Transformer

## Abstract

这篇论文是ICCV2021的最佳论文，获得了马尔奖，论文提出了一种能够作为视觉任务一般backbone的视觉transformer：swin transformer。Transformer在NLP领域取得了巨大的成功，但是将其应用到视觉领域却存在许多困难，如视觉领域的实体通常具有较大的尺度变化和较高的图片分辨率，而NLP领域的文本单词的embedding则通常采取固定的尺寸。为了解决这些由于两个领域的差异而产生的困难，作者提出了通过移动窗口（shifted windows）来计算特征表示的分层transformer。shifted windowing方案将self attention的计算限制在不重叠的窗口内，同时又允许跨窗口的连接，极大的提高了计算效率。这种分层架构具有在各种尺度上建模的灵活性，并且相对于图像大小具有线性计算复杂性。这些品质使得swin transformer 能够广泛适用于一系列的视觉任务，如图像分类（*87.3 top-1 accuracy on ImageNet-1K*）、密集预测任务如目标检测（*58.7 box AP and 51.1 mask AP on COCO test dev*）和语义分割（*53.5 mIoU on ADE20K val*），超过之前的SOTA方法，在COCO数据集上增加了*2.7 box AP*和 *2.6 mask AP*，在ADE20K数据集上增加了*3.2 mIoU*，证明了基于transformer的模型作为视觉任务backbone的潜力。分层的结构设计和shifted windows的方法被证明同样适用于all-MLP架构。

## 1.Introduction

视觉领域长期以来一直被CNNs所统治，自AlexNet以来发展到现在，通过更大的规模、更宽泛的连接、更复杂的卷积形式等设计使得CNNs变得越来越powerful，基于CNN设计的backbone广泛应用于各大视觉任务，这些架构上的进步使得性能得到提高，从而广泛提升了整个领域。

而与视觉领域不同的是，NLP领域目前的主流架构是transformer，transformer 专为序列建模和转导任务而设计，以其对数据中长期依赖关系建模的关注而著称。它在语言领域的巨大成功促使研究人员研究其对计算机视觉的适应性，最近它在某些任务上展示富有潜力的结果，特别是在图像分类和联合视觉语言建模任务上。

这篇论文旨在探索出一个像NLP领域的transformer和视觉领域的CNN那样的能够作为视觉中一般任务的backbone。transformer在NLP领域有着很好的表现，但是由于两个领域的模态不同，将其应用到视觉领域存在挑战，transformer在处理语言的过程中以单词token作为基本元素，而视觉元素在尺度上则可能会有很大的差异，因此很难将attention机制应用到如目标检测这样的任务中来。在所有现有的基于transformer的模型中，token的尺寸都被固定，这不适合视觉任务。另一个不同之处是相较于文本中的单词，图片通常具有更高的分辨率，而许多视觉任务都需要在像素级别上进行密集预测，这对与应用在高分辨率图像上的transformer是困难的，因为它的self attention的计算复杂度是图片尺寸的平方。为了解决这些问题，论文提出了swin transformer，如下图所示。

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/swin%20transformer%20campare%20with%20ViT.png)

它构建分层特征图并且对图像大小具有线性计算复杂度。Swin Transformer 通过从小尺寸的patch（灰色轮廓）开始并逐渐合并更深的 Transformer 层中的相邻patch来构建分层表示。通过这些分层特征图，Swin Transformer 模型可以方便地利用先进的技术进行密集预测，例如特征金字塔网络 (FPN)  或 U-Net 。线性计算复杂度是通过在划分图像的非重叠窗口中局部计算自注意力来实现的（图中以红色标出）。每个窗口中的patch数量是固定的，因此复杂度与图像大小成线性关系。这些优点使 Swin Transformer 适合作为各种视觉任务的通用backbone，与之前基于 Transformer 架构的ViT形成对比，后者产生单一分辨率的特征图并具有二次复杂度。 

Swin Transformer 的一个关键设计是它在连续自注意力层之间的窗口分区的移动，如下图所示。

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/shifted%20windowing.png)

shifted window能够连接前一层的相邻窗口，增强它的建模能力。这种策略在现实世界的延迟方面也很有效：窗口内的所有query patch共享相同的key集合，这有助于硬件中的内存访问。相比之下，早期的基于滑动窗口的自注意力方法由于不同query像素对应不同key集合而在通用硬件上存在低延迟问题。论文提出的shifted window方法相比于滑窗的方法有着更低的延迟，但是具有同样强有力的建模能力，shifted window 方法同样适用于all-MLP结构。

Swin Transformer 在图像分类、目标检测和语义分割的识别任务上取得了强大的性能。它在三个任务上以相似的延迟显著优于 ViT / DeiT 和 ResNe(X)t 模型 。它在 COCO test-dev数据集上达到58.7 box AP 和 51.1 mask AP，以 +2.7 box AP高于没有使用额外数据的Copy-paste模型和 +2.6 mask AP高于DetectoRS。在 ADE20K 语义分割上，它在 val 集上获得了 53.5 mIoU，比之前的SOTA模型SETR提高了 +3.2 mIoU。它还在 ImageNet-1K 图像分类上实现了 87.3% 的 top-1 准确率。作者认为，跨计算机视觉和自然语言处理的统一架构可以使这两个领域受益，因为它将促进视觉和文本信号的联合建模，并且可以更深入地共享来自两个领域的建模知识。作者希望 Swin Transformer 在各种视觉问题上的强大表现能够在社区中更深入地推动这种观念，并鼓励视觉和语言信号的统一建模。

## 2.Related Work

### CNN and variants

CNN 作为整个计算机视觉的标准网络模型，尽管在几十年前就有了，但是直到AlexNet被提出才成为视觉领域的主流，自此之后提出了许多更深和更加有效的网络架构如VGG、GoolgleNet、ResNet、DenseNet、HRNet、EfficienNet等，另一方面，也有许多工作是对CNN层本身的改进，如深度可分离卷积，可形变卷积等，尽管CNN和它的变体仍是视觉领域的主流，但是作者注意到transformer结构在联合视觉和NLP建模的强大潜力，而论文提出的swin transformer在几个基础的视觉是被任务上达到了很好的表现，作者希望它能够促进对视觉领域主流模型的转变。 

### Self-attention based backbone architectures

同样受到 NLP 领域自注意力层和 Transformer 架构成功的启发，一些工作采用自注意力层来替换流行的ResNet中的部分或全部空间卷积层。在这些工作中，自注意力是在每个局部窗口内的像素上计算的，以达到加速优化的效果 ，它们实现了比对应的 ResNet 架构稍好一些的准确率/FLOPs的权衡。但是它们昂贵的内存访问导致它们的实际延迟明显大于卷积网络。论文提出的shifted window在连续层之间移动窗口而不是使用滑动窗口，这使得它能在通用硬件中更有效地实现。

### Self-attention/Transformers to complement CNNs

另一项工作是使用自注意力层或 Transformer 来增强标准的 CNN 架构。 自注意力层可以通过提供编码远程依赖或异构交互的能力来补充backbone或head网络。最近，已经有研究将Transformer中的编码器-解码器设计应用于对象检测和实例分割任务 。论文的工作探索了 Transformers 对基本视觉特征提取的适应性，并且是对这些工作的补充。

### **Transformer based vision backbones** 

与我们的工作最相关的是 Vision Transformer (ViT) 及其它的后浪。ViT 的开创性工作直接将 Transformer 架构应用于不重叠的中等大小图像块上进行图像分类。与卷积网络相比，它在图像分类方面实现了令人印象深刻的速度-准确度权衡。虽然ViT需要大规模训练数据集（即 JFT-300M）才能表现良好，但DeiT引入了几种训练策略，允许 ViT 使用较小的 ImageNet-1K 数据集也有效。ViT在图像分类上的结果令人鼓舞，但由于它的特征图分辨率低和随图像尺寸增加而增加的平方计算复杂度，其架构不适合用作密集视觉任务或输入图像分辨率高时的通用骨干网。有一些工作通过直接上采样或反卷积将ViT模型应用于目标检测和语义分割的密集视觉任务，但性能相对较低。同时期的一些其他工作是一些修改 ViT 架构以获得更好的图像分类的研究工作。根据经验，作者发现尽管Swin Transformer侧重于通用性能而不是专门针对分类，Swin Transformer架构可以在这些图像分类方法中实现最佳速度精度权衡。另一项同时期的工作探索了在Transformer上构建多分辨率特征图的类似思路。它的复杂性仍然是图像大小的二次方，而swin transformer的复杂性是线性的并且在局部运算，这已被证明有利于对视觉信号中的高相关性进行建模 。swin transformer在 COCO 目标检测和 ADE20K 语义分割上都达到了SOTA的准确性。

## 3.Method

### 3.1 Overall Architecture

模型结构如下图所示。

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/swin%20transformer%20architecture.png)

首先像ViT中那样将输入的RGB图片通过一个patch分离模块分为不重叠的patches，每个patch作为一个token，然后依次经过4个stage。

第一个stage将每个patch的RGB像素值进行concat作为其特征，比如如果每个patch的是4×4尺寸内的像素点，那么其对应的特征维度就是4×4×3=48维的特征，然后通过一个线性的embedding层将这些原始的特征值（如上述的48维）映射为C维的特征。然后通过两个swin transformer block，从模型结构图中可以看到经过第一个stage后特征图的尺寸变为$$\frac H4\times\frac W4\times C$$ ，每个swin transformer block不改变输入尺寸。

接下来的三个stage相类似，都是由patch merging模块和swin transformer block构成，不同之处在三个stage的swin tansformer block数量有所差异，分别包含$\times2$、$\times6$、$\times2$的block数量。

#### **Patch Merging**

patch merging操作是将$2\times2$范围内的4个patch进行concat得到一个patch，这样使得特征图的宽高减半，通道数变为原来的四倍，然后再通过一个线性的embedding层将4C的维度映射到2C的维数，整个patch merging操作起到两倍下采样的作用。这些stage就像CNN那样能够联合产生分层的特征表示，能够很方便的替换掉各种视觉任务中已有的方法。

#### **Swin Transformer Block**

关于swin transformer block，两个连续的swin transformer block分别包含W-MSA和SW-MAS，如上结构图(b)所示，对于原始transformer block的主要改进在于将multi self attention(MSA)替换为了 window multi self attention(W-MSA) 和shifted window multi self attention(SW-MSA)。

+ **W-MSA**

在ViT中的self attention是对一张图片上的所有patches上进行的，这样的计算复杂度是图片尺寸的平方，因此在swin transformer中采用的是局部self attention计算，将图片中等分为M$\times$M个windows，每个window包含若干个patches，在这些windows内计算self attention，当M的尺寸固定时，self attention的计算复杂度就是线性的，具体而言对于包含$h\times w$个patch的图片来说，其计算复杂灰度公式对比如下，由于涉及具体的self attention计算，公式具体是如何计算得到的不是很清楚，但是通过作者前面定性的分析知道二者一个是平方计算复杂度，一个是线性计算复杂度。
$$
\Omega(MSA)=4hwC^2+2(hw)^2C \\
\Omega(W-MSA)=4hwC^2+2m^2hwC
$$

+ **SW-MSA**

应用上述W-MSA会带来一个问题，就是windows之间互相是没有信息交互的，这限制了模型的建模能力，于是作者提出了SW-MSA来解决这一问题，将包含SW-MSA的swin transformer block加在上述包含W-MSA的swin transformer block后面，使得不同的windows之间能够进行信息交互，如下图所示。

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/shifted%20windowing.png)

第L层是W-MSA中进行的规则patch划分，每个window包含$4\times 4$的patches，第L+1层使用的则是shifted window的划分方式，首先将前一层的每个规则window均分为相同的4个小window，再将不同规则window中的相邻的小window组合成一个window，使得不同层之间的window是进行了移动的，达到一个信息交互的目的。

但是上述shifted window的划分方式又纯在一个问题，就是边缘的window的大小和内部的window的大小是不一样的，四个角上的小window相邻的小window都是来自同一个window，没有跨窗口的连接，而边缘的window只跨了两个window，内部的window则是跨了四个window，从而造成划分的window大小不一的问题，并且得到的window数也不同于上一层，由$\lceil\frac hM\rceil\times\lceil\frac wM\rceil$变为$（\lceil\frac hM\rceil+1）\times（\lceil\frac wM\rceil+1）$，为了解决这个问题，一种解决方法是进行padding使得每个window的尺寸一样，但是作者提出了了一种cycle shift的方式，如下图所示，将左上边沿的window补到右下边沿，得到新的window，使得每个window大小一样，window数量保持和上一层的规则划分方式的数量一致，同时用一个mask机制记录每个window的原来位置来将self attention的计算限制在原window（没做cycle shift时划分的window）内。这就是论文提出的shifted window操作，作者经过实验发现这种方式比其他方法如sliding window有着更加低的延时。

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/shifted%20window%20computation%20manner.png)

#### **Relative position bias**

在相对位置偏置上，作者采取了前人的研究工作中提出的相对位置偏置对特征进行位置编码，其attention公式如下：
$$
Attention(Q,K,V)=SoftMax(QK^T/\sqrt d+B)V
$$
其中d是query的维度除以key的维度。之所以使用该位置编码而不是其他位置编码如绝对位置偏置，是因为作者对比了几种位置编码偏置，发现相对位置偏置最适合swin transformer，能够带来最好的提升。

#### Architecture Variants

除了基础的模型Swin-B外，作者还提供其他几种不同尺寸的模型Swin-T、Swin-S、Swin-L，其中Swin-T和Swin-S的模型复杂度分别和ResNet-50 (DeiT-S) 与ResNet-101相对应。作者在实验中设置M=7，d=32，$\alpha=4$，其中M为window尺寸，d为self attention的head数，$\alpha$为每个MLP层的扩张系数，各个模型的超参数设置如下：

+ Swin-T: *C* = 96, layer numbers = {2*,* 2*,* 6*,* 2}

+ Swin-S: *C* = 96, layer numbers = {2*,* 2*,* 18*,* 2}

+ Swin-B: *C* = 128, layer numbers = {2*,* 2*,* 18*,* 2}

+ Swin-L: *C* = 192, layer numbers = {2*,* 2*,* 18*,* 2}

其中C和前面所述的C一样，是第一个stage输出的channel数量。具体的参数如下图所示：

![img](https://github.com/Gwencong/PaperNotes/blob/main/imgs/Swin-T%E3%80%81S%E3%80%81B%E3%80%81L.png)

## 4. Experiments

作者将Swin Transformer作为backbone分别在ImageNet-1k上进行图像分类、在COCO上进行目标检测、在ADE20K上进行语义分割，得到的结果都超越了当前的SOTA方法，同时还进行了消融实验验证了Shifted windows、Relative position bias、cycle shift带来的精度和速度上的提升。

## 5. Conclusion

本文介绍了Swin Transformer，这是一种新的视觉 Transformer，它产生分层特征表示，并且相对于输入图像大小具有线性计算复杂度。Swin Transformer 在 COCO 目标检测和 ADE20K 语义分割方面达到了SOTA的性能，显著超越了以前的最佳方法。 作者希望Swin Transformer在各种视觉问题上的强大表现能够促进视觉和语言信号的统一建模。作为 Swin Transformer的一个关键元素，基于shifted windows的自注意力在视觉问题上被证明是有效和高效的。

## 6.My point of view

整篇论文看下来，主要改进是对于不同尺度特征的使用和通过shifted windows降低self attention应用在CV领域时带来的计算量问题，这一改进应该是借鉴了CNN，shifted window的思想是很像CNN中通过多个卷积层的堆叠来利用不同尺度的特征的方式。进一步挖掘，swin transformer中的这种shifted windows不同层之间特征没有信息交互与特征融合，类比到CNN中FPN，或许swin transformer还能进一步改进达到更好的性能。同时虽然swin transformer进一步降低了将视觉transformer的计算复杂度，但是对于实时性要求较高的、显存内存有限的工业应用场景而言还是不够的，仍然还有一段路要走。