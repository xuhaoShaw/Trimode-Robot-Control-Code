# 三模式变形轮机器人

此部分为三模式变形轮机器人部分资料，包含：

- 控制部分代码（stair_code)
- 照片素材和处理结果（videos）
- 已发表的IROS论文、IFToMM论文（论文）

## stair_code

### Vision.py

机器人爬楼梯过程中，视觉逻辑部分的实现。

定义了Vision类，包含的函数有

- process_frame：根据前方地形，判断下一步的执行动作
- judge_ground：根据前方地面高度高出or低于当前地面，判断前方地形
- judge_upedge：提取上楼台阶的边缘
- judge_downedge：提取下楼台阶的边缘

### Vision_rs.py

功能实现的主函数。

- 输入：从Intel RealSense深度相机获取到的视频中
- 输出：提取图像每一帧的信息，调用Vision类中的功能，判断前方地形，决策下一步是直接行走，还是让车轮变形翻越台阶

### vision_test.py

测试图像处理逻辑的程序。以从视频中提取到的图片为输入，判断下一步的执行逻辑。



## videos

从相机拍摄的视频中提取到的部分帧的图片。



## 论文

发表的IFToMM论文和IROS论文，及相关材料。



