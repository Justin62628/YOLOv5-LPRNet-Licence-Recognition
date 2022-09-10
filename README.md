# 信后与系统实验-光学车牌检测识别项目（CCPD数据集）

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


#### 不足、更多改进空间
1. 数据集缺点，因为算力有限，我使用的只是CCPD2019中的base部分蓝牌和CCPD2020中的全部绿牌，对于一些复杂场景，如：远距离、模糊、复杂场景雪天雨天大雾、
   光线较暗/亮等等，这些其实CCPD2019中都有的，后面如果资源充足的话可以考虑重启这个项目，再优化下数据集；
2. 数据集缺点，无法识别双层车牌
3. 模型方面，可不可以加一些提高图像分辨率的算法，在检测到车牌区域位置，先提高车牌区域分辨率，再进行识别。
4. 模型方面，可不可以加一些图片矫正的算法，在检测到车牌区域位置，先矫正车牌图片，再进行识别。

### 第二套方案（准创新方案）
使用YOLOv5s和文字OCR方案对CCPD车牌进行检测和识别。
用 [YOLOv5s](https://github.com/ultralytics/yolov5) 进行车牌检测，用 [SwinTextSpotter](https://github.com/mxin262/SwinTextSpotter) 进行车牌识别。

#### 模型性能
+ [ ] TODO

#### 不足和改进空间
- 参数量大

### 第三套方案（摆烂方案）
使用[一套端对端方案](https://github.com/chenjun2hao/CLPR.pytorch) 对CCPD车牌进行检测和识别。

#### 模型性能
+ [ ] TODO

#### 不足和改进空间
- 不好改进