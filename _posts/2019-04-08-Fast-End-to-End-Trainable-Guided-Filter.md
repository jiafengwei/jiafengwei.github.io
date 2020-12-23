---
layout: article
title: Fast End-to-End Trainable Guided Filter
tags: Image2Image
---

![此处输入图片的描述][1]


<!--more-->


## 1 Citation
Wu H, Zheng S, Zhang J, et al. Fast End-to-End Trainable Guided Filter[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 1838-1847.

## 2 CONTRIBUTIONS
I can learn from this paper:

一、适用于显著性检测的upsampling方法，可以代替DenseCRF取得更好效果


二、任务分类
image processing tasks：
1. $L_0$ smoothing~\cite{Xu/siggraph2011}, 
2. multiscale detail manipulation~\cite{Farbman/tog2008}, 
3. photographic style transfer from a reference image~\cite{Aubry/siggraph2014}, 
4. non-local dehazing~\cite{Berman/cvpr2016} and 
5. image retouching learning from human annotations~\cite{fivek}.

dense prediction approaches=computer vision tasks=from low-level vision to high level-vision：
1. depth estimation from a single image~\cite{Saxena/nips2005}, 
2. saliency object detection~\cite{Liu/cvpr2007} and 
3. semantic segmentation~\cite{He/cvpr2004}.

三、 任务定义
**Depth estimation from a single image**
Depth estimation from a single image is first proposed by Saxena~\etal~\cite{Saxena/nips2005}, which aims at predicting the depth at each pixel of an image with monocular cues, such as texture variations, texture gradients, occlusion, known object sizes, haze, defocus, etc.
首先提出单个图像的深度估计，其目的是用单眼线索预测图像每个像素的深度，例如纹理变化，纹理梯度，遮挡，已知物体尺寸，雾度，散焦，等。

**Saliency Object Detection**
Saliency object detection is used to detect the most salient object in an input image, which is formulated as an image segmentation problem by Liu~\etal~\cite{Liu/cvpr2007}.
They try to separate the salient object from the image background with multi-scale contrast, center-surround histogram, and spatial color distribution.

**Semantic Segmentation**
The task of semantic segmentation is labeling images, in which each pixel is assigned to one of a finite set of labels.
It's first proposed by He~\etal~\cite{He/cvpr2004}, which is solved by combining local and global information in a probabilistic framework.

## 3 Problem&METHOD

* 解决了什么问题？
> Given an image Ih and the corresponding low-resolution output Ol, we seek a full-resolution output Oh, which is visually similar to Ol while preserving the edges and details from Ih. 
给定图像Ih和相应的低分辨率输出Ol，我们寻找全分辨率输出Oh，其在视觉上类似于Ol，同时保留Ih的边缘和细节。

输出时低分辨率转化为高分辨率解决办法是upsampling：
    * Deep Bilateral Learning
    * Joint  Bilateral Upsampling联合双边上采样
        * bilated filter
        * guided filter（作者改进的基础）
        

        
* 提出了怎样的方法？
把另一篇作者的guided filter 变得可以微分求导，因此形成一个可以反向传播训练layer：
1. 参数可学习：适用于图片scale变化的任务
2. 映射函数可学习：适用于输入输出的channel不同的情况

* 为什么会提出这个方法？
反向传播的基础是可微分，沿着梯度下降的方向前进到局部最优，模型参数根据微分结果调节

* 和你已知的解决同一问题的其他方法有什么区别？

- 从Abstract中找

## 4 SOME DETAILS

* 设计了怎样的模型？
1. 将原始图像降低分辨率，通过已有网络，训练得到低分辨率图片
2. 训练Guided Filtering layer的各个参数
3. 原始图像通过Guided Filtering layer得到高分辨率图片

* 这个模型为什么能work？
* 这篇文章的关键点是什么？


### 4.1 整体框架
![此处输入图片的描述][1]
输入$I_l,O_l,I_h$端到端的学习参数$A_l , b_l$，来生成高分辨率的输出

参数|解释
-|-
$C_l(I_l)$| convolutional neural network $C_l(I_l)$ 
$I_l$|a low-resolution image
$I_h$| corresponding high-resolution image
$O_l$|low-resolution output 
$O_h$| producing the high-resolution output，通过local linear model进行linear transformation
$GF(I_l,I_h,O_l)$|Guided Filtering layer 
 



### 4.2 Guided Filtering layer  $GF(I_l,I_h,O_l)$


#### 4.2.1 **原先Guided Filter方法**

$$    O_l^i = a_l^k I_l^i + b_l^k, \forall i \in \omega_k,$$
$$A_l , b_l-> f_\uparrow -> A_h,  b_h $$
$$    O_h = A_h * I_h + b_h$$

参数|解释
-|-
$f_\uparrow$ |bilinear upsampling operator
$A_l$, $b_l$| 用来衡量$I_l$ and $O_l$之间的误差系数
$i$ | the index of a pixel
$k$ | the index of a local square window $\omega$ with radius $r$
$A_h$ ， $b_h$ |由 upsampling $A_l$ and $b_l$生成
$*$ | element-wise multiplication.

原先的具体算法
![此处输入图片的描述][2]
#### 4.2.2 **改进的可微分的Guided Filtering layer** 


![此处输入图片的描述][3]

当输入和输出chanel不相同时，添加转换函数解决，该函数是可学习参数的layer，组成包括 an adaptive normalization layer和 a leaky ReLU layer.

参数|解释
-|-
$F(I)$| a transformation function 
$G_h$ and $G_l$ | guidance map
Even when $n_I = n_O$, a better guidance map than $I_h$ and $I_l$ is required.
改进的具体算法
![此处输入图片的描述][4]
### 4.3 local linear model（linear transformation）& Mean Filter $f_{\mu}$



## 5 ADVANTAGES  


## 6 DISADVANTAGES and Question

- 从Limitation或者Future Work中找

## 7 OTHER 

### 7.1 相关网站 
[[Project]](http://wuhuikai.me/DeepGuidedFilterProject)    [[Paper]](http://wuhuikai.me/DeepGuidedFilterProject/deep_guided_filter.pdf) [[arXiv]](https://arxiv.org/abs/1803.05619)    [[Demo]](http://wuhuikai.me/DeepGuidedFilterProject#demo)    [[Home]](http://wuhuikai.me)

### 7.2 他人笔记 
### 7.3 代码 
[github](https://github.com/wuhuikai/DeepGuidedFilter)

### 7.4 文章  
### 7.5 阅读笔记日志
### 7.6 未解决的疑问


  [1]: /assets/images/2019-04-08-Fast-End-to-End-Trainable-Guided-Filter.png

  [2]: https://github.com/jiafengwei/Picture/raw/master/Fast%20End-to-End%20Trainable%20Guided%20Filter/Alg1.png
  [3]: https://github.com/jiafengwei/Picture/raw/master/Fast%20End-to-End%20Trainable%20Guided%20Filter/GuidedFilterLayer.png
  [4]: https://github.com/jiafengwei/Picture/raw/master/Fast%20End-to-End%20Trainable%20Guided%20Filter/Alg2.png


    https://github.com/jiafengwei/Picture/raw/master/Fast%20End-to-End%20Trainable%20Guided%20Filter/Framework.png