# 信后与系统实验-光学车牌检测识别项目（CCPD数据集）

## 任务要求：
- 第一次汇报要求给出文献级别的方案选择理由和论证，
- 以及具体场景，
- 总体方案设计。 
   - 难点在哪里 
   - 解决方案。

## 方案报告
- 项目背景：交通流量，车位管理，。。。
- 设计要求：实时，准确，鲁棒
- 识别算法比较和选择：
  - 一般步骤：车牌检测，车牌识别
  - 难点：汉字以及字母识别<-独热码，各种各样的车牌<-端对端，拍摄的几何畸变<-端对端或矫正。
  - 非端对端：YOLOv5+LPRNet，HyperLPR（简单的特征提取网络，不明说）
  - 端对端：HyperLPR（复杂的特征提取网络，不明说）
- 总结展望：便携化，嵌入式

## 方案
### 第一套方案（保底方案）
使用 [YOLOv5s和LPRNet](https://blog.csdn.net/qq_38253797/article/details/125054464) 对CCPD车牌进行检测和识别。
用 [YOLOv5s](https://github.com/ultralytics/yolov5) 进行车牌检测，用 [LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch) 进行车牌识别。

主要参考以下四个仓库：

1. Github: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
2. Github: [https://github.com/sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
3. [https://gitee.com/reason1251326862/plate_classification](https://gitee.com/reason1251326862/plate_classification)
4. [https://github.com/kiloGrand/License-Plate-Recognition](https://github.com/kiloGrand/License-Plate-Recognition)

其他资料：
[【YOLOV5-5.x 源码讲解】整体项目文件导航](https://blog.csdn.net/qq_38253797/article/details/119043919)
[HuKai97/yolov5-5.x-annotations](https://github.com/HuKai97/yolov5-5.x-annotations)

#### 关于报告怎么写
- [Yolov5s网络结构](https://blog.csdn.net/qq_38253797/article/details/119684388)

- 概述：考虑到实时性，首先使用YOLOv5检测车牌的位置，抠图后交由专门的车牌识别算法LPRNet识别车牌字符并输出结果
- 内容：简要介绍YOLOv5，其具有实时，多标签优势；LPRNet编码字符，易训练
#### 模型性能

检测模型

model|img_size|epochs|mAP_0.5|mAP_0.5:0.95|size
------ | -----| -----| -----| -----| -----
yolov5s| 640x640| 60 |   0.995|0.825| 14M

识别模型性能

model     | 数据集| epochs| acc    |size
-------- | -----| -----|--------| -----
LPRNet| val | 100 | 94.33% | 1.7M
LPRNet| test | 100 | 94.30% | 1.7M

总体模型速度：（YOLOv5+LPRNet）速度：47.6FPS（970 GPU）

- TODO
- [ ] 根据参数和需求魔改YOLOv5


#### 不足、更多改进空间
1. 数据集缺点，因为算力有限，我使用的只是CCPD2019中的base部分蓝牌和CCPD2020中的全部绿牌，对于一些复杂场景，如：远距离、模糊、复杂场景雪天雨天大雾、
   光线较暗/亮等等，这些其实CCPD2019中都有的，后面如果资源充足的话可以考虑重启这个项目，再优化下数据集；
2. 数据集缺点，无法识别双层车牌
3. 模型方面，可不可以加一些提高图像分辨率的算法，在检测到车牌区域位置，先提高车牌区域分辨率，再进行识别。
4. 模型方面，可不可以加一些图片矫正的算法，在检测到车牌区域位置，先矫正车牌图片，再进行识别。

### 第二套方案（准创新方案）
~使用YOLOv5s和文字OCR方案对CCPD车牌进行检测和识别。~

~用 [YOLOv5s](https://github.com/ultralytics/yolov5) 进行车牌检测，用 [SwinTextSpotter](https://github.com/mxin262/SwinTextSpotter) 进行车牌识别。~
~[using tf](https://gitee.com/guo-kunchang/summeraipro2022/blob/develop/%E6%9C%80%E7%BB%88%E6%8F%90%E4%BA%A4%E7%89%A9/1.%20%E6%BA%90%E7%A0%81%E5%8F%8A%E9%A1%B9%E7%9B%AE%E6%96%87%E4%BB%B6/num_plate_recognision/cnn_chinese.py)~
- [x] [HyperLPR](https://github.com/szad670401/HyperLPR) 的非e2e方案
- [ ] [MobileLPR](https://gitee.com/damone/mobile-lpr)

#### 关于报告怎么写
- 概述：传统CNN暴力方法可能挺慢，需要进行改良：手工提取特征方法，提前介入
- 内容：

#### 模型性能
+ [ ] TODO

#### 不足和改进空间
- 参数量大

### 第三套方案（摆烂方案）
~- [x] 使用[一套端对端方案](https://github.com/chenjun2hao/CLPR.pytorch) 对CCPD车牌进行检测和识别。~
- [x] [HyperLPR](https://github.com/szad670401/HyperLPR) 的e2e方案

#### 关于报告怎么写
- 概述：增强鲁棒性（处理畸变等），增加深层特征数量（网络深度），提高准确性。
- 内容：

#### 模型性能
+ [ ] TODO

#### 不足和改进空间
- 不好改进