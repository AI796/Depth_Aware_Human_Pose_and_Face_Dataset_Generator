# Depth_Aware_Human_Pose_and_Face_Dataset_Generator

![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-03.jpg)

Created with DAZ3d, 2048x2048px, 1000key-frame poses.
To train a neural network with 3d aware tasks, we need dataset coming with semantic lables of normal and depth. Such kind of dataset is rare in both real world and nn-generation.
Thus we provide a DAZ3d Generator, which consists of:
- Facial Animations: 1000 key-frames
- Upper Body Poses: 1000 key-frames
- Three R/G/B attenuated lights: to create the depth and normal info simultaneously.

DAZ3d has the ability of rendering img at photo-realistic quality with NVIDIA-IRAY engine, it also provide different kinds of view-ports mode which can also be rendered as img dense/semantic label.

![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-02.jpg)


![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-01.jpg)

https://github.com/ChiCheng123/JointDet
