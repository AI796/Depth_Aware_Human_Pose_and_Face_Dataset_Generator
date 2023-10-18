# Depth_Aware_Human_Pose_and_Face_Dataset_Generator

![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-03.jpg)

Created with DAZ3d, 2048x2048px, 1000key-frame poses.
To train a neural network with 3d aware tasks, we need dataset coming with semantic lables of normal and depth. Such kind of dataset is rare in both real world and nn-generation.
Thus we provide a DAZ3d Generator, which consists of:
- Facial Animations: 1000 key-frames
- Upper Body Poses: 1000 key-frames
- Three R/G/B attenuated lights: to create the depth and normal info simultaneously
- A 360 rotate camera, targeting at center of human-body

DAZ3d has the ability of rendering img at photo-realistic quality with NVIDIA-IRAY engine, it also provide different kinds of viewport modes(1.wireframe 2.texture-shaded 3.photo-real 4.toonify, etc.) which can also be treated as img dense/semantic label.

![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-02.jpg)

Key-frame sequences are listed at ANIMATE2 pannel, so you can play with them like scale-time/flip-animation/swap-sequences, etc. You can also change the render-settings (like resolution) to meet your training task demands.
With outputs of different render-modes, you can scale/rotate/crop your imgs, or interpolate your photo-real img with your rendered label-img. To keep head and body within your ROI, we recommend project: [https://github.com/ChiCheng123/JointDet ](https://github.com/PeterH0323/Smart_Construction), which can detect human head(not only front-face) at fast speed. Sample code is provided at demo.py

![image](https://github.com/AI796/Depth_Aware_Human_Pose_and_Face_Dataset_Generator/blob/main/img/sample-01.jpg)


