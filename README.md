This paper presents a novel method for fully automatic external parameter calibration of 3D lidar and cameras with calibration plates. The proposed method can extract the corners of the calibration plate from the point cloud data of each frame of lidar using the intensity information. The model that divides the checkerboard is optimized by the constraint of the correlation between the reflection intensity of the laser and the color of the checkerboard, so once we know the corners of the checkerboard in the 3D point cloud, then the external calibration between the two sensors is converted into a 3D-2D matching problem. The corresponding 3D-2D point calculation method for the absolute pose between the two sensors is generally UPnP. In addition, the calculated parameters are taken as initial values and perfected using the LM optimization method. The simulation method is used to evaluate the performance of extracting corners in the 3D point cloud. In the paper, the Velodyne HDL 32 radar and Ladybug3 camera are used for experiments, and the accuracy and stability of the external parameter calculation are finally proved. 

 github:https://github.com/icameling/lidar_camera_calibration 

 Main function introduction 

 The calculation of the external parameters of the program is based on the intensity of the laser radar and the calibration of the automatic external parameters of the camera. This code is implemented by pathon, and an extended version of the C++ appears. The main functions of the algorithm: 

 (1) Automatic segmentation of velodyne 3D LIDAR acquisition point cloud. 

 (2) Automatic detection of the calibration board. 

 (3) Detection of the corner points of the calibration board point cloud data. 

 (4) Optimization of external calibration parameters. 

 (5) VTK is used to visualize the point cloud. 

 This function can be adapted to the VLP-16.HDL-32 HDL-64, and the paper uses a 32-wire lidar for data testing. 

 Paper Atlas 

 ![avatar]( 20200524221925984.JPG) 

  overall process 

 ![avatar]( 20200524221942911.JPG) 

  Data from the same scene captured by a LiDAR sensor and a panoramic camera. (a) Points are colored by the intensity of reflections (blue indicates low intensity, red indicates high intensity); (b) A scaled checkerboard. We can see the change in the intensity of reflections from point clouds between black and white modes; (c) A panoramic image of the same scene. 

 ![avatar]( 20200524221949209.JPG) 

  The principles used to estimate the corner points of a point cloud. (a) A checkerboard model; (b) A checkerboard scanning point cloud. The colors represent the intensity (blue indicates low reflection intensity, red indicates high reflection intensity); (c) Find a matrix that can transform most of the three-dimensional points on the corresponding pattern. The green points are the estimated corner points; (d) Consider the corner points of the checkerboard model as the corner points of the point cloud. 

 ![avatar]( 202005242219564.JPG) 

  Distribution of 20 checkerboard positions. The checkerboard is captured by LIDAR from different heights and angles. The length of the coordinate axis is 1 meter. Four groups of colors represent the four positions of each horizontal camera checkerboard. (a) Top view of the checkerboard point cloud; (b) Side view of the checkerboard point cloud. 

 summarize 

 ![avatar]( 20200524222009546.JPG) 

 Segmentation of the checkerboard is performed on the point cloud data obtained from LiDAR. The point cloud of the checkerboard is identified from within the segments based on the segmentation method. The corner points of the checkerboard in the point cloud are estimated by minimizing the defined cost function. On the other hand, the corner points of the checkerboard in the image are detected using the existing method. Correspondence of the corner points is constructed based on the predefined counting order. Then, by solving the absolute pose problem, the initial value of the transformation matrix is estimated using the corresponding pair. Finally, the results are optimized by the method of nonlinear optimization of LM.  



--------------------------------------------------------------------------------

The article is provided by Yan Shouzhi, a member of the official account "Point Cloud PCL". Here is a reprint. If you are interested, you can view his original text https://blog.csdn.net/weixin_43199584/article/details/105256192 

 The latest deep learning method for semantic segmentation of 3D point clouds proposed by researchers at the University of Zaragoza in Spain, the network is divided into two parts, and a new sliding box is proposed to search for "pixels" after spherical projection. Then the improved MiniNetV2 network is used to divide, and then the points with labeled data are backprojected back to the 3D point cloud. Finally, the post-processing process is added, and the network structure is relatively clear. The two networks with different parameter sizes released have refreshed the results on both emanticKITTI and KITTI datasets, becoming the new SoTA. The source code may be open-sourced in April. The author mentioned that the implementation part will refer to the networks of RangeNet ++ and LuNet. 

##  abstract 

>  LIDAR semantic segmentation assigns a semantic label to each 3D point measured by LIDAR, which has become an important task for many robotic applications (e.g. autonomous driving). Fast and efficient semantic segmentation method to meet the requirements of time and computing power in practical applications. This work introduces 3D-MiniNet, a new method for LIDAR semantic segmentation that combines 3D and 2D learning layers. It first learns a 2D representation from the original point through a novel projection, extracting local and global information from the 3D data. The representation is fed into a 2D fully convolutional neural networks (FCNN), which can generate 2D semantic segmentation. The obtained semantic 2D labels are then re-projected back into 3D space and enhanced by a post-processing module. The novelty of the model lies in the projection learning module. The authors conduct a detailed ablation study, showing how each component designed by the module contributes to the final performance of 3D-MiniNet. Validated on the SemanticKITTI and KITTI datasets, records (current SOTA) of 3D real-time (real-time) segmentation and point cloud segmentation were refreshed using 3D-MiniNet-tiny and 3D-MiniNet respectively, faster and more parametric efficient than previous methods. 

##  I. Introduction 

 Robotic autonomous navigation systems use sensors to perceive the world around them. RGB cameras and LIDAR are common and essential components. One of the key components of autonomous robotic systems is semantic segmentation. Semantic segmentation assigns a category label to each LIDAR point or camera pixel. This detailed semantic information is essential for decision-making in dynamic scenarios in the real world. In autonomous robotic systems, LIDAR semantic segmentation provides very useful information for autonomous robots for tasks such as SLAM, autonomous driving, etc., especially for recognizing dynamic objects. 

 Therefore, this task requires a point cloud segmentation model that can provide accurate semantic information quickly and efficiently, which is especially challenging for processing 3D LIDAR data. There are currently two types of segmentation methods: 

 Inspired by Point-based and Projecting-based methods, this paper proposes 3D-MiniNet, a novel and efficient method for semantic segmentation of 3D LIDAR. 3D-MiniNet first performs point-based operations directly on 3D points to learn rich 2D representations, which are then segmented by fast 2D fully convolutional neural networks computation. Finally, the semantic labels are re-projected back to the 3D points, and through a fast post-processing method. 

 The main contribution of this network is the projection learning module, which first learns to group point clouds, extract local and global features, and generate a 2D representation of the learned point cloud. Using ablation studies conducted on this module, it is possible to show how each part improves the performance of 3D-MiniNet. 

 To provide adjacent groups of 3D points to this novel module, the authors implemented a fast point cloud nearest neighbor search that generates structured groups of 3D points. Each of these groups corresponds to a single pixel in a 2D representation. To learn the segmentation from this representation, the authors used a modified version of MiniNetV2 (2D segmented network model). Finally validated on the SemanticKITTI benchmark and the KITTI dataset. The optimal configuration of the network obtained higher mIoU than the previous state-of-the-art in both benchmarks, with a 2x increase in computational speed and parameter reduction of 1/12 of the previous model. 

 ![avatar]( 20200401214309123.png) 

##  II. Related work 

##  A. 2D Semantic Segmentation 

 The current state-of-the-art for 2D semantic segmentation is basically all deep learning methods. The 2D semantic segmentation architecture evolved from convolutional neural networks (CNNs) originally used for classification tasks, requiring only a decoder to be added at the end of the CNN. FCNNs opened the way for modern semantic segmentation architectures. The authors of this work were the first to propose bilinear interpolation upsampling of image features learned by CNNs until the resolution at the time of input is obtained, and to calculate the cross entropy loss for each pixel. Another early approach is SegNet, which proposes a symmetric encoder-decoder architecture using the unpooling operation as the upsampling layer. Recent work has improved these earlier segmented architectures by adding novel operations or modules for classification tasks originally proposed in CNN architectures. 

 FC-DenseNet learns DenseNet using dense modules. PSPNet uses ResNet as its encoder to introduce pyramid pooling modules into the end layer of the CNN to learn valid global context priors. 

 Deeplab-v3 + is one of the best segmentation architectures, and its encoder is based on Xception, which makes use of deep separable convolution and empty convolution, reducing model parameters and computational consumption. 

 As far as the efficiency of the network is concerned, ENet established the basis for subsequent work such as ERFNet, ICNet, etc. The main idea is to work at low resolution, i.e. fast downsampling. MiniNetV2 uses deep separable convolution with multiple extensions, which can efficiently learn local and global spatial relationships. In this work, the author takes MiniNetV2 as the backbone of the network and adjusts it to capture information from the original LIDAR point cloud. 

##  B. 3D Semantic Segmentation 

 There are two main types of point cloud deep learning methods: 

 A point cloud-based approach 

 The disorder of point clouds limits the general convolutional neural networks CNN to process point cloud data. The pioneer method and foundation of point cloud-based processing is PointNet network. PointNet proposes to learn the characteristics of each point through a shared MLP (multi-layer perceptron), and then use the symmetric function maxpooling to deal with the disorder of the point cloud. Later, many new network structures are proposed based on PointNet. Following the idea of point-by-point MLP, PoinNet ++ groups points in a hierarchical manner and learns from larger local areas. At the same time, the author also proposes a multi-scale grouping method to deal with the non-uniformity of the data. On the contrary, there are approaches that follow the idea of convolution and propose different types of operations, such as merging adjacent points into kernel units to be able to perform point-by-point convolution. There are also works that employ graph networks to capture the basic geometry of point clouds, and directed graphs to capture structural and contextual information. To this end, the authors represent point clouds as a set of interconnected hyperpoints. 

 2) Projection-based methods 

 Different intermediate representations of raw point cloud data have been used for 3D semantic segmentation. Felix et al. demonstrate that multi-view representations are very effective. The authors propose projecting 3D points into several 2D images from different captured views, and then performing 2D semantic segmentation independently for each image. Each point calculates its label by fusing different reprojection scores from different views. The disadvantage of this approach is that it requires running neural networks operations multiple times, once for each view. SegCloud utilizes voxel representation, which is a very common method for encoding and discretizing 3D space. This method feeds 3D voxels into 3D-FCNN. The authors then introduce deterministic trilinear interpolation, mapping the bold voxel prediction back to the original point cloud, and adding the application of CRF to optimize the output in the final step. The main disadvantage of this voxel representation is that 3D-FCNN has a very slow execution time for real-time applications. Su et al. proposed SPLATNet, which uses another representation: tetrahedral lattice representation. This method interpolates the 3D point cloud into a tetrahedral sparse lattice, and then applies bilateral convolutional layers to convolve the represented occupancy. LatticeNet was later proposed to improve SPLATNet, and its DeformsSlice module was proposed for re-projecting lattice features back to the point cloud. So far, the representation that allows for more efficient processing is the spherical representation, which is the most common projection for LIDAR semantic segmentation. It is a 2D projection that allows the application of 2D image manipulation, which is very fast and works well on recognition tasks. SqueezeSeg based on SqueezeNet architecture and its successor improvement SqueezeSegV2, both show that with spherical projection, very efficient semantic segmentation tasks can be accomplished. The latest work by Milioto et al. combines DarkNet architecture with GPU-based post-processing methods, which can achieve better results than CRF for real-time semantic segmentation. 

 ![avatar]( 20200401212923679.png) 

 Contrary to projection-based methods, point-based methods operate directly on the original data source without losing any information. But projection-based methods tend to be faster and better suited to the unstructured nature of the data, especially for large inputs like LIDAR scans, which generate hundreds of thousands of points. LuNet is the first work combining projection-based methods with point-based methods. It relies on offline point nearest neighbor search, which makes the method not feasible for real-time applications. In addition, it has only one MLP pool operation and can only learn local information from the original point. In 3D-miniNet, the shortcomings of LuNet are solved by implementing GPU-based fast neighbor search and integrating a novel projection module that learns contextual information from raw 3D points. Through the fast 3D neighbor search algorithm, the input M points (with 

          C 

          1 

        C_1 

    C1 features) are divided into P groups of N points. each point has a  

          C 

          1 

        C_1 

    C1 eigenvector, which is used in this process to expand the data relative to each group to 

          C 

          2 

        C_2 

    C2.3DMiniNet will process the point cloud group and predict a semantic label for each point. Finally, post-processing methods are added to refine the final result. 

##  III. 3D-minNet: LIDAR Point Cloud Segmentation 

 The figure above summarizes the novel and effective LIDAR semantic segmentation method. It consists of three modules: 

 Compared to projection-based methods, there are two main issues that limit the use of point-based models for real-time tasks: 

 This is determined by the characteristics of point clouds. To alleviate these two problems, the approach in this paper involves the use of a fast point nearest neighbor search agent (see Introduction) and a computational module for minimizing point-based operations, which uses raw 3D points as input and outputs a 2D representation that can be processed using a 2D CNN (Introduction). 

###  A. Fast 3D Point Neighbor Search 

 The first step of this method is to project the input original point cloud onto a spherical projection, mapping 3D points (x, y, z) into 2D coordinates (), which is part of the general operation of point cloud spherical projection. 

  The formula for the above spherical projection is the standardized form, and there are many other variants, including the vertical field of view of the sensor and the initial feature number. 

 Perform a point nearest neighbor search in the spherical projection space using the sliding window method. Similar to the convolutional layer, groups of pixels, i.e. projection points, are obtained by sliding the window. The generated groups of points have no intersection, i.e. each point belongs to only one group. This step generates groups of points, each of which is a group of points (), in which all points () from the spherical projection are used. Before providing the actual segmentation module 3D-MiniNet for these groups of points, the features of each point must be enhanced. For each point group obtained, we calculate the average of the five features in the calculation and the average of each feature of the group where each point phase is located to obtain the corresponding (relative) value. In addition, we calculate the coordinate mean 3D Euclidean distance between each point and the group of points in which it is located. Therefore, each point now has 11 features:  

###  B. 3D-MiniNet 

 ![avatar]( 20200401214642782.png) 

 3D-MiniNet consists of two modules, as shown in Figure 3. For the projection module proposed in the paper, it utilizes the original point cloud and calculates the 2D representation, and then the author uses an efficient backbone network based on MiniNetV2 to calculate the semantic segmentation. 1) Projection learning module: 

 The goal of this module is to convert raw 3D points into 2D representations that can be used for efficient segmentation. The input to this module is a set of 3D point groups () that are collected by performing a sliding window search on a spherical projection, as described in the previous subsection. The following three types of features are extracted from the input data (see the left part of Figure 3) and fused in the final module step: 

 2). 2D segmentation module (MiniNet as Backbone): 

 Once the tensor has been computed in the previous module, a valid CNN is used to compute the 2D semantic segmentation (see the MiniNet backbone in Figure 3 for a detailed visual description). The authors primarily use FCNNs rather than operations at multiple MLP layers, considering that multi-layer MLPs are comparatively faster to compute using convolutional operations. 

 FCNN is built based on the MiniNetV2 architecture. Here the encoder uses layer-depth separable convolution and layer-multiple-dilation depth-separable convolution. For the decoder, bilinear interpolation is used as the upsampling layer method. It performs depth-separable convolution at resolution and performs at resolution. Finally, convolution is performed at resolution to obtain 2D semantic segmentation predictions. 

 This paper refers to the MiniNetV2 method to extract fine grain information, i.e. high-resolution underlying features, in the second convolutional branch. The input to the second branch is a spherical projection, the specific details are specified in Section IV-B of the following text. As a final step, the predicted 2D semantic segmentation must be re-projected back into 3D space again. For points already projected into the spherical representation, this is a simple step, as only the semantic labels predicted in the spherical projection need to be assigned. However, points that have not yet been projected into the spherical surface (the resulting 2D coordinates may correspond to more than one 3D point), they do not have semantic labels. For these points, semantic labels of their corresponding 2D coordinates are assigned. This issue can lead to incorrect predictions, so post-processing methods need to be performed to refine the results. 

###  C. Post-processing process 

 In order to cope with the erroneous prediction of non-projected 3D points, this paper follows the post-processing method of Milioto et al. All 3D points will obtain new semantic labels based on their nearest neighbors (KNN). The criteria for selecting the K closest points is not based on the relative Euclidean distance, but on the relative depth value. In addition, the 2-D spherical coordinate distance based on the points narrows the search. The implementation of the method of Milioto et al., is GPU-based and is able to run in 7ms, thus maintaining a low frame rate. 

##  IV. Experimental Section 

###  A. Data sets 

 SemanticKITTI Benchmark: 

 The SemanticKITTI dataset is a large-scale dataset that provides intensive point-by-point annotation for the entire KITTI odometer benchmark test. The dataset contains more than 43,000 scans from which more than 21,000 scan data (sequences 00 to 10) can be used for training, with the remainder (sequences 11 to 21) used as a test set. The dataset distinguishes between 22 different semantic categories, with 19 categories evaluated on the test set via the benchmark's official online platform. Since this is the most relevant and largest single-scan 3D LIDAR semantic segmentation dataset currently available, the authors conducted an ablation study and a more comprehensive evaluation of this dataset. 

 KITTI benchmark: 

 The work of SqueezeSeg provides semantic segmentation labels derived from the 3D object detection challenge of the KITTI dataset. It is a medium-sized dataset divided into 8057 training data and 2791 validation scans. 

###  B. Setting 

 A) 3D point nearest neighbor search parameters: 

 For the SemanticKITTI dataset, the author sets the resolution of the spherical projection to 2048 × 64 (Note: 2048 is (360/horizontal resolution), the image size in the convolutional network is the number of times 2, so it is set to 2048, 64 is the number of lasers, here is 64 lasers, so the image width is 64), and the same for KITTI, the resolution is set to 512 × 64 (the same as the previous network, so that a reasonable comparison can be made). Then the window size stride of 4 x 4 is set to 4, which is to ensure that there is no intersection between the packets. Do not set zero padding when searching for fast point neighbors, so that 8192 sets of 3D points are generated for the SemanticKITTI data and 2048 sets are generated on the KITTI data. Our projection module will receive these groups as input and generate the learned representation for the SemanticKITTI configuration at a resolution of 512 x 16 and for KITTI at a resolution of 128 x 16. 

 B) Network parameters: 

 The complete architecture and all its parameters are depicted in Figure 3. Note here that the author actually proposes three different configurations to evaluate the proposed method: 3D-MiniNet, 3D-MiniNet-small, and 3D-MiniNet-tiny. Each method corresponds to a different number of features on the () feature layer, respectively: 

 The design of the three layers () configured in the FCNN backbone network corresponds to: 

 C) Post-processing parameters: 

 For post-processing methods that use the K-nearest neighbor method, we set the window size of the nearest neighbor search at 2D segmentation to 7 × 7 and set the value to 7. 

 D) The training process: 

 Epochs = 500, for 3D-MiniNet, 3D-MiniNet-small, 3D-MiniNettiny, batch_size set to = 3, 6, 8 respectively 

 (Varies due to memory limitations). The optimizer uses stochastic layer descent (SGD) with an initial learning rate and a decay rate of 0.99 per epoch. Use the cross entropy loss function as an optimization for model loss. 

 Where M is the number of labels for a point and C is the number of categories. Is a binary indicator (with values of 0 or 1) that a point m belongs to a class c, and is the probability that a CNN predicts that a point m belongs to a class c. This probability is calculated by applying a soft-max function to the output of the network. To solve the class imbalance problem, the authors use the median frequency class balance used in SegNet. To smooth the final class weights, the authors propose applying exponentiation, where the frequency of class c is the median of all frequencies, and the authors set i to 0.25. 

 E) Data Enhancement: 

 During training, the entire 3D point cloud is randomly rotated and moved. The authors randomly inverted the signs of the X and Z values of all point clouds, and also deleted some points. 

##  V. Results 

###  A. Ablation study of the projection module 

 ![avatar]( 20200401215640130.png) 

  The projection module is the novelty of this paper. This section shows how each of these sections can help improve the representation of learning. For this experiment, the authors performed using only the 3D-MiniNet-small configuration. The results of the ablation study are recorded in Table 1, measuring the mIoU, velocity, and learning parameters corresponding to each setting. The first line shows the use of only 1 × N convolution in the learning layer as well as the 5-channel input used in RangeNet ( 

          C 

          1 

        C_1 

    The performance of C1), establishing it as a baseline (i.e. spatial feature extractor). The second line shows the performance if 1 × N convolution is replaced with a point-based operation (i.e. local feature extractor). The results show that MLP operations work better for 3D points, but require more execution time. The third line combines convolution and local MLP operations. The results show that the combination of convolution and MLP operations improves performance. The authors argue that this is due to the different types of features learned for each operation type. 

 The attention module improves performance with little additional computational effort. It narrows the feature space down to a specified number of features, thus understanding which features are more important. The fifth line shows the result of adding a contextual feature extractor. Later also learning context via FCNN via convolution, but here, the contextual feature extractor learns different contexts via MLP operations. Background information is often very useful in semantic tasks, for example, for differentiating between cyclists, cyclists, and motorcyclists. This contextual information has a higher boost compared to other feature extractors that show its relevance. Finally, increasing the number of features per point using features relative to point group () will also achieve better performance without an increase in computational time and parameter cost. 

###  B. Benchmark results 

 ![avatar]( 20200401215733390.png) 

 This section presents the quantitative and qualitative results of 3D-MiniNet and compares them with other related works. a) Quantitative analysis: 

 Table II compares the method presented in this paper with several point-based methods (lines 1-6) and projection-based methods (lines 7-12). Measure the mIoU, processing speed (FPS), and number of parameters required for each method. It can be seen that the point-based LIDAR scan semantic segmentation method is slower than the projection method, and it is difficult to continue to improve performance. Current LIDAR sensors such as Velodyne typically operate at a speed of 5-20 FPS. Therefore, the current projection-only method is capable of processing the full amount of data provided by the sensor in real time. 

 ![avatar]( 20200401215801958.png) 

 From the performance of 3D-MiniNet, it uses 12x fewer parameters while 2x faster, so it is almost 3% better than the previous state-of-the-art technology. Interestingly, 3DMiniNet-small can deliver the latest performance more efficiently and faster. If a trade-off can be made between efficiency and performance, the smaller version of Mininet will also get better performance metrics at a higher frame rate. 3D-MiniNet-tiny is capable of running at 98 fps, and the mIoU is only 9% lower (compared to 29% of the SqueezeSeg version at 90 fps, a decrease of 46.9%), and uses fewer parameters (see 3D-MiniNettiny vs. TangentConv). The post-processing method applied in this article shows that it effectively improves segmentation results. This step is crucial for proper handling of points not included in the spherical projection. As shown in Table III, the scan of the KITTI dataset has a lower resolution (64x512). 3D-MiniNet also obtains the latest technology in LIDAR semantic segmentation on this dataset. This method achieves better performance compared to the SqueezeSeg version (+ 10-20 mIoU). 3D-MiniNet also has better performance than LuNet. Note that the authors did not evaluate KNN post-processing in this case, as only 2D labels are available on the KITTI dataset. 

 B) qualitative analysis: 

 ![avatar]( 20200401215846106.png) 

 Figure 4 shows some examples of 3D-MiniNet inference on the test data. Since no test basis is provided for the test set (the evaluation is done externally on the online platform), we can only display the visual results without the need for label comparisons. Note that the method achieves high-quality results both in related categories such as cars and in challenging categories such as traffic signs. The biggest difficulty, predictably, is distinguishing between classes that have similar geometries and structures (e.g. buildings and enclosures) to switch between.  

##  VI. Conclusion 

 The 3D-MiniNet proposed in this paper is a fast and efficient method for semantic segmentation of 3D LIDAR. 3D-MiniNet first projects a 3D point cloud into a two-dimensional space, and then learns semantic segmentation using fully convolutional neural networks. Unlike conventional predefined projection-based methods, 3DMiniNet learns this projection from raw 3D points, achieving excellent results. The ablation research section also illustrates how each part of the method contributes to the learning of representations. 3D-MiniNet also becomes the new SoTA on the SemanticKITTI and KITTI datasets, making it more efficient than previous methods in terms of both real-time and accuracy requirements. 1. 3D-MiniNet original address 2. 3D-MiniNet Github address 



--------------------------------------------------------------------------------

Deep learning (DL) using convolutional neural networks (CNN) architectures is now the standard solution for solving image classification tasks. But the problem becomes more complex when this is applied to 3D data. First, 3D data can be represented using a variety of structures, including: 

 1 Voxel grid 2 Point cloud 3 Multi-view 4 Depth map 

 In the case of multi-view and depth maps, the problem is solved by using 2D CNNs on multiple images. By simply defining 3D convolution kernels, extensions of 2D CNNs can be used for 3D Voxel meshes. However, for the case of 3D point clouds, it is not clear how to apply the DL tool. But there have been several solutions before, please refer to http://www.cnblogs.com/li-yao7758258/p/8182846.html summary for details  

 ![avatar]( 20180322120556564) 

 As well as the recent PointCNN proposed by researchers at Shandong University, this paper proposes a simple and general framework for point cloud-based feature learning. The key to the success of a CNN is to be able to exploit the spatial local correlations (such as images) that are densely represented in grids in the data. However, point clouds are irregular and disordered, so direct convolution of kernels on the features associated with these points results in loss of shape information, while also varying in order. To solve these problems, it is proposed to learn an X transformation from the input points, which is then used to simultaneously weight the input features associated with the points and rearrange them into a potentially implicit canonical order, and then apply the quadrature and sum operations on the elements. Our proposed method is a generalization of typical CNNs to point cloud-based feature learning, and is therefore called PointCNN. Experiments show that PointCNN can achieve comparable or better performance than previous best methods on a variety of challenging benchmark datasets and tasks. Comparison of PointCNN with other methods 

 Secondly, there is more data available for images, despite the recent increase in the number of 3D datasets. However, for 3D cases, synthetic data can be easily generated. 

 Below is a list of papers using DL tools on 3D data 

 Voxel Grid – Volumetric CNN:  Voxnet: A 3D convolutional neural network for real-time object classification  Volumetric and multi-view CNNs for object classification on 3d data – compared volumetric CNNs to Multi-view CNNs for object classification. They showed that the multi-view approach performs better, however, the resolution of the volumetric model was limited  3D shapenetes: A deep representation for volumetric shapes  Multi-View CNNs:  Volumetric and multi-view CNNs for object classification on 3d data  *Multi-View Convolutional Neural Networks for 3D Shape Recognition  Point clouds:*  Pointnet: Deep learning on point sets for 3d classification and segmentation – In this work they applied a convolution kernel on each point separately, creating a higher dimensional representation of each point and then max-pooling over the entire point set (max pooling used as a symmetric function) to get invariance to permutations of the input cloud (since there is no geometrical significance to the point order).  Hand-crafted features + DNN :  3D deep shape descriptor – fed heat kernel signatures (HKS) descriptor into an NN to get an Eigen-shape descriptor and a Fischer shape descriptor.  有问题请指出，同时欢迎大家关注微信公众号 

 ![avatar]( 2018032212070652) 

 Or join the 3D visual WeChat group to communicate and share together   



--------------------------------------------------------------------------------

 Code reference OpenPCDet. 

##  Global transformation 

 The global transformation is to transform the point cloud scene, such as scaling, rotating, and reversing. Both the point cloud and the box are transformed as a whole, and the effect is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 flip 

 ![avatar]( 3058a3d80baf4cc3bd00b84def0c8e85.png) 

 spin 

 ![avatar]( f3eed0aec8b64ae688af46d5052a4bac.png) 

 zoom 

 ![avatar]( df17ba24c97d48049a15f4fe8702c550.png) 

##  Second, target transformation 

 Target-level transformation, random transformation of point clouds and boxes in each target, including rotation, scaling, translation, sparsity and other operations. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 spin 

 ![avatar]( 2ab87f1e97de4688b2a426bcb21729d5.png) 

 Scaling, for obvious contrast, the scaling factor I used here is relatively large, and a smaller scaling factor should be used in actual training. 

 ![avatar]( 6a1b80e35c4f43d0bced4fbe5021d92b.png) 

 Translation, like scaling strategies, should have smaller coefficients when actually used. 

 ![avatar]( 283fbfd7479a4af4951804dc2256d6a4.png) 

##  III. Local transformation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 Sparse, divide the target into 6 regions, and randomly select one of them to sparse. 

 ![avatar]( 127ec78b8df9407e8acdc9407273d202.png) 

 ![avatar]( 438be589f2d04793a8ee01998200b80c.png) 

##  III. Code and test data 

 Link: https://pan.baidu.com/s/1k-GRptGlg0OB6OlacikrJw Extraction Code: j9m6 



--------------------------------------------------------------------------------

Indoor point cloud dataset 

#  1、S3DIS 

 Point Cloud Public Dataset: S3DIS 

#  2、ScanNet 

 outdoor point cloud dataset 

#  3、Semantic3D 

 Point Cloud Public Dataset: Semantic3D 

#  4、SemanticKITTI 

#  5、SensateUrban 

 Point Cloud Public Dataset: SensatUrban 

#  6、Toronto-3D 

 Point Cloud Public Dataset: Toronto-3D 



--------------------------------------------------------------------------------

In the previous article, 3D point cloud transformation (translation, rotation, scaling) and python implementation introduced some basic transformation principles of point cloud in detail, and also used python to achieve, this time we use C++ to do some cloud translation, scaling, rotation transformation. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573576552
  ```  
 ![avatar]( 83ee902178a3432395d6992821dbc60e.png) 



--------------------------------------------------------------------------------

![avatar]( f04cafeae69745b9abfe151cfffc6abd.gif) 

#  First, the basic rules 

##  1.1. Left-handed and right-handed 

 The two-dimensional coordinate system has only two mutually perpendicular axes, the X-axis and the Y-axis, and the positions of the two axes can be changed at will, because no matter how the two two-dimensional coordinate systems are changed, they can be overlapped by rotation or inversion transformation, so different two-dimensional coordinate systems are equivalent. 

 A three-dimensional coordinate system has three axes that pass through the origin and are perpendicular to each other: the X axis, the Y axis, and the Z axis. However, a 3D coordinate system not only has one more axis than a 2D coordinate system, it is much more complex. A 3D coordinate system is not necessarily equivalent, in fact, there are two different coordinate systems: a left-handed coordinate system and a right-handed coordinate system. Only left-handed or right-handed coordinate systems can be rotated to coincide. 

 In the three-dimensional coordinate system, according to the different directions of the Z-axis, there are two coordinate systems: "right-handed system" and "left-handed system". When the direction of the Z-axis is from the eyes to the depths, the coordinate system is left-handed, and vice versa. The so-called right-handed system is a three-dimensional coordinate system expressed by the following methods: 

 That is, think of the thumb as the X axis, the index finger as the Y axis, and the middle finger as the Z axis to consider three-dimensional coordinates. If it is a right-handed system, the middle finger points to your side. If it is a left-handed system, the middle finger points forward 

 ![avatar]( 6e6c8b8a2ee4426a9365f71dba6bba19.png) 

 The following three coordinate systems belong to the right-hand system, and the three coordinate systems can overlap after rotation. They are also the coordinate systems used when we introduce point cloud rotation changes in this blog. 

 ![avatar]( 4f7b92b14c97436ebc2ec43515c28e60.png) 

##  1.2. Left-hand rule and right-hand rule 

 The operation that is often done in three-dimensional space is rotation, and rotation can be decomposed into a combination of sequential rotation around three main axes. Therefore, if you need to know the positive direction of rotation, you can use the left-hand rule and the right-hand rule to determine the positive direction of rotation. 

 For example, for the right-hand coordinate system, after determining a rotation axis, the right hand holds the fist, the thumb points in the positive direction of the rotation axis, and the direction in which the four fingers are bent is the positive direction of rotation. Correspondingly, the left-hand coordinate system is determined by the left hand. After determining the positive direction of rotation, it is easy to know whether to use a positive or negative angle in the formula calculation. The following is an example of the right hand: 

 ![avatar]( cab6aa9a0ebe47f4a2e133fdaede0e69.png) 

 Then according to the left-hand rule and the right-hand rule, the positive direction of rotation with each axis as the main axis is as follows: 

 ![avatar]( 1e2e8d4c43bb4a0c8450b5d9659aedc9.png) 

 As mentioned at the beginning, the rotation of three-dimensional space can be decomposed into a combination of X, Y, and Z as the rotation axis respectively. Then the object in three-dimensional space actually rotates only on the two-dimensional plane perpendicular to the rotation axis each time. If the X axis is used as the rotation axis, the rotation angle, it can be seen that the X coordinate of the rotated point is unchanged, but it is rotated on the Y_Z plane, and the rotation direction can be determined by combining the right-hand rule. The rotation plane is as follows 

 ![avatar]( f50206a26eef432b936dbbf1e6f54c7f.png) 

 Similarly, when the Y axis is used as the rotation axis, the rotation plane is as follows: 

 ![avatar]( 16ef6b0fd983498b8d27db425e660826.png) 

 ![avatar]( 0c0c3c975cc840e5be27a7353e692a3d.png) 

#  Test data 

 Here I use a room in S3DIS (Area3\ office_1) as the test data, in order to facilitate the internal scene of the data, I removed the roof and roof lights, the processing code is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 Check out the following in cloudcompare 

 ![avatar]( 1cb680a67f4744a497b2b0de4e75d9f9.png) 

#  Formula derivation 

 Assuming that the original point coordinates are, and the transformed point coordinates are, several basic transformation formulas are derived as follows: 

##  2.1. Translation 

 Assuming that the translation distance along the x-axis, the translation distance along the y-axis, and the translation distance along the z-axis are: written in matrix form is  

 The code is as follows 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 The results are as follows: 

 ![avatar]( 3d0ebe77b3ba4cb48cccfc1e6987688d.png) 

##  2.2. Rotation 

 Take the X axis as the rotation axis 

 Suppose the rotation angle of point p (x, y, z) to point p (x, y, z), the forward angle between the p and y axes is, 

 ![avatar]( 350227a250584a09bef3934442fdc592.png) 

  Expand to get, because: so: written as a matrix is  

 The formula is applicable to different quadrants and different rotation directions. The key is to judge the positive direction of the rotation 

 Take the Y axis as the rotation axis, or the rotation angle of point p (x, y, z) to point p ' (x, y, z), but now it becomes the positive angle between the p and z axes, 

 ![avatar]( adea1e773b9d43458716a383eac51431.png) 

  Expand to get, because: so: written as a matrix is  

 Take the Z axis as the rotation axis 

 Suppose the rotation angle of point p (x, y, z) to point p (x, y, z), then the forward angle between p and the x axis is, 

 ![avatar]( a2e29d04f08c41f98580571d3cdb448b.png) 

  Expand to get, because: so: written as a matrix is  

 Rotation can be decomposed into sequential x, y, and z rotations, so three rotation matrices can be multiplied to obtain rotations of any axis. 

 The code is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 First look at the rotation transformation with the Z axis as the rotation axis, the parameter settings roate_x = 0, roate_y = 0, in order to verify the rotation direction, I set the roate_z to pi/6 (30 °) and -pi/6 (-30 °) respectively. 

 Let's talk about the problem of zeroing the center point of the point cloud first. You can see that I subtracted the coordinates of the point cloud from the coordinate mean on each axis before performing the rotation operation, and classified the center point of the point cloud as (0,0,0). Here is an explanation of why this is done. The derivation of the above formula for rotation is all carried out around the origin. If our point cloud center point is not at the origin, then the point clouds before and after rotation cannot be superimposed and compared. Take the rotation around the X axis as an example 

 As shown in the figure below, the center of the triangle is not at the origin of the coordinates. After rotating it, we get, 

 ![avatar]( b0b42852aa36491da144f93dbed56d1d.png) 

 If we first translate the center of the coordinate to the origin, and then rotate it to get it, we can see that the center point is unchanged before and after the rotation, so that we can directly compare the rotation transformations before and after. 

 ![avatar]( ec8b2c6e840a4c268f01d0a645459b6c.png) 

 ![avatar]( 6e23fdde167d497ab5103a62188fe23a.gif) 

 The effect of rotating 30 ° around the Y axis, in the parameter settings roate_x = 0, roate_y = pi/6, roate_z = 0. 

 ![avatar]( 35ed5d26ac284a19808cd37c891cb49e.gif) 

 The effect of rotating 30 ° around the X axis, in the parameter settings roate_x = pi/6, roate_y = 0, roate_z = 0. 

 ![avatar]( 75357f0adeca4ab8acebf4e410a2c9b0.gif) 

##  2.3. Zoom 

 Assuming that the scaling coefficients of the point cloud on the x, y, and z axes are,,, respectively, the transformation formula is as follows:  

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 ![avatar]( 8bd69560c2cc429f8b38f48c90e60652.gif) 

#  III. Combination code 

 Now let's combine the rotation, translation, and zoom operations 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 The full code is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573522064
  ```  
 For ease of comparison, we only perform zoom and rotation operations. 

 ![avatar]( d7a95ad5cb1b49c1945806abe3d93658.gif) 

#  IV. Reference 

 4.1. Three-dimensional coordinate rotation matrix 4.2. Fundamentals of 3d transformation: detailed explanation of translation, rotation, and scaling (affine transformation) - formula derivation 4.3. A simple method for judging the positive direction of rotation of a three-dimensional coordinate system 



--------------------------------------------------------------------------------

![avatar]( 20210106181625690.png) 

 Abstract: Although Motion Restoration Structure (SfM) as a mature technique has been widely used in many applications, existing SfM algorithms are still not robust enough in some cases. For example, for example, images are usually taken at close range to obtain detailed textures in order to better reconstruct scene details, which will lead to less overlap between images, thus reducing the accuracy of estimating motion. In this paper, we propose a LiDAR-enhanced SfM process, which jointly processes data from LiDAR and stereo cameras to estimate the motion of sensors. The results show that in large-scale environments, adding LiDAR helps to effectively weed out false matching images and significantly improve the consistency of the model. Experiments were carried out in different environments to test the performance of the algorithm and compare it with the latest SfM algorithms. Related work with major contributions, there is an increasing demand for robot-based inspection, which requires the processing of high-resolution image data of large civil engineering facilities such as bridges, buildings, etc. These applications typically use high-resolution, wide-field-of-view (FOV) cameras, which shoot at close range from the surface of the structure to obtain richer visual details. These characteristics present new challenges to standard SfM algorithms. First, most of the available global or incremental SfM schemes are based on a single camera, so the scale cannot be recovered directly. More importantly, due to the limitation of the field of view, the overlapping area between adjacent images is reduced, resulting in the attitude map can only be locally connected, thus affecting the accuracy of motion estimation. This problem becomes more important in large-scale environments. 

 To address the above challenges, this paper proposes a new scheme that extends the traditional SfM algorithm to apply to stereo cameras and LiDAR sensors. This work is based on the simple idea that the long-range capabilities of LiDAR can be used to suppress relative motion between images. More specifically, we first implement a stereo vision SfM scheme that calculates the motion of the camera and estimates the three-dimensional position of the visual features (structures). The LiDAR point cloud and visual features are then fused into a single optimization function, which is solved iteratively to optimize the motion and structure of the camera. In our scheme, the LiDAR data enhances the SfM algorithm in two ways: 

 1) LiDAR point cloud is used to detect and eliminate invalid image matching, making the stereo camera-based SfM scheme more robust to visual blur. 

 2) LiDAR point clouds are combined with visual features in a joint optimization framework to reduce motion drift. Our scheme enables more consistent and accurate motion estimation than state-of-the-art SfM algorithms. 

 The work of this paper mainly includes the following aspects: 

 1) The global SfM technology is applied to the stereo camera system to realize the motion initialization of the camera at the real scale. 

 2) LiDAR data is used to rule out invalid image matching, further enhancing the reliability of the scheme. 

 3) By combining the data from stereo cameras and lidar, we extend our previously proposed joint optimization scheme and improve the accuracy and consistency of the built model. 

 ![avatar]( 20210106181730507.png) 

 LIDAR-Enhanced Binocular SFM This scheme takes a set of stereoscopic images and associated LiDAR point clouds as inputs to generate a 3D model of the overlay environment in the format of triangulated feature points and merged LiDAR point clouds. The following diagram shows process A, corresponding feature point search, of our LiDAR-enhanced SfM scheme 

 Given a pair of stereo images, computing the correspondence involves feature extraction, matching, and geometric verification. First, we rely on the OpenMVG library to extract SIFT features from the images. Then exhaustive matching of the features is performed using the provided cascade hashing method. Finally, the matching between the two images is verified by geometric verification of the binocular pole constraint. Specifically, RANSAC is used to estimate the fundamental matrix F, which is then used to check the polar error of the matching features. Only geometrically consistent features are retained for further computation. 

 B. Relative motion estimation 

 ![avatar]( 20210106181815711.png) 

 Since stereo image pairs are pre-calibrated, we treat a pair of left and right images as a separate unit. To estimate relative motion, standard stereo matching methods rely on feature points observed in all four images in two pairs of images, while we observe that many points are shared by only three or even two images. Ignoring these points may lose important information for estimating camera motion, especially in cases where image overlap is limited. Therefore, the choice here is to explicitly handle different cases where views are shared between two pose points. Specifically, we consider feature points shared by at least 3 views to ensure reconstruction of the scale. Although points with only 2 views can help estimate rotation and translation directions, since these points usually come from the small overlapping area shown in the figure below, they are ignored here. On the other hand, there may also be multiple types of shared characteristics between two pose points. To simplify the problem, we choose the type with the most correspondence to solve the relative motion. In the three-view case, the feature points are first triangulated with a stereo image pair, and then solved with the RANSAC + P3P algorithm. In the four-view case, we follow the standard processing method by first triangulating the points in the two sites, and then applying the RANSAC + PCA registration algorithm to find the relative motion. In both cases, a nonlinear optimization program is used to optimize the computed pose and triangulation, by minimizing the re-projection error of the inner line. Finally, all poses are transformed to represent the relative motion between the left cameras. C. Verification of relative motion 

 Once the relative motion is found, a pose map can be built where the nodes represent the pose of the image frame and the edges represent the relative motion. The global pose can be solved by the relative motion on the average pose map. However, due to visual ambiguity in the environment (see figure below), there may be invalid edges, and direct average relative motion may produce incorrect global poses. Therefore, a two-step edge verification scheme is designed to remove outliers. 

 (1) In the first step, check the overlap of the lidar point clouds for all image frame pairs and weed out inconsistent point clouds. 

 ![avatar]( 20210106181916570.png) 

 (2) Check the consistency of the loop in the second step. (The specific method can be explained in detail in the paper) D. Global pose initialization 

 ![avatar]( 20210106181939537.png) 

 This part mainly introduces the cost function of optimizing the global frame: E, triangulation and RANSAC 

 In this paper, we adopt the text robust triangulation method and use RANSAC for each three-dimensional feature point to find the best triangulated view. For each trajectory, which is a collection of observations of one feature point in different camera views, two views are randomly sampled and the point is triangulated using the DLT method. A more matching view can be found by projecting the point onto other views and selecting the view with a smaller re-projection error. This process is repeated many times and preserves the largest set of internal views (at least 3 views are required). Finally, the internal view connection is utilized to optimize the pose of the feature point in the global structure by minimizing the re-projection error. 

 F. Joint pose optimization 

 The pose optimization of vision-based SfM algorithm is usually achieved by beam adjustment (BA). However, due to multiple system reasons, such as inaccurate feature position, inaccurate calibration, corresponding outliers, etc., pose estimation may produce large drift over long distances, especially when closed loops cannot be effectively detected. To solve this problem, we consider using the long-range capability of lidar to limit the motion of the camera. This scheme jointly optimizes the camera and lidar observations. This part of the content can be viewed in the original text to understand the formula. 

 Experimental results 

 A. Experimental equipment 

 ![avatar]( 20210106182037229.png) 

 The image below features multiple onboard sensors, including two Ximea color cameras (12 million pixels, global shutter) and a Velodyne Puck LiDAR (VLP-16) mounted on a continuous rotating motor. Using the motor angle measured by the encoder, the scan point of the VLP-16 is converted into a fixed pedestal. B, Relative Motion Estimation C, Relative Motion Verification 

 ![avatar]( 20210106182146996.png) 

 Here we compare the performance of the proposed grid-based check (GC, threshold 0.6) and success rate check (SR) with the outlier exclusion method of rotation cycle check and transform (rotation and translation) cycle check (TC) used by OpenMVG 

 ![avatar]( 20210106182214532.png) 

 Here are the advantages of joint observation modeling in joint optimization. As shown in Figure E, reconstruction 

 ![avatar]( 20210106182243389.png) 

 The reconstruction results of the collected dataset are shown in the figure below. In the first row, the reconstruction of a small concrete structure is shown. The second row compares the reconstruction results using COLMAP, OpenMVG, and our scheme Smith-Hall. In these three tests, left and right images were used for reconstruction. However, neither COLMAP nor OpenMVG were able to handle visual blur caused by stop signs, and limited overlapping images. Therefore, the generated model is either inconsistent or incomplete. Using our scheme helps to effectively rule out invalid motion and allows for the establishment of a more consistent model. In conclusion, this paper proposes a LiDAR-enhanced stereoscopic SfM scheme that uses lidar information to improve the robustness, accuracy, consistency, and completeness of stereoscopic SfM schemes. Experimental results show that this method can effectively find effective motion poses and eliminate visual ambiguity. In addition, the experimental results also show that the combined observation of the camera and lidar helps to completely constrain the external transformation. Finally, the LiDAR-enhanced SfM scheme can produce more consistent reconstruction results than state-of-the-art SfM methods. 



--------------------------------------------------------------------------------

![avatar]( 20200729172303436.png) 

 The loop detection strategy of Lego-LOAM is relatively simple, and it takes into account distance and time at the same time. 1. Using the radius-based nearest neighbor search algorithm in PCL, the current pose of the robot is used as the search point to find several poses within a radius of 7m; 2. Using time as a constraint, if the time difference between the corresponding time of the historical pose and the corresponding time of the current pose is too small, it means that it is a small loop, which is of little significance. The author sets the time difference to be greater than 30s in the program. The following figure expresses two situations: the above figure shows the process of the robot building a map normally, and the following figure shows that the robot returns to the origin and begins to determine whether the conditions for loop detection are met. The corresponding code is this small section 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573542424
  ```  
 Extension: About the usage of radiussearch in PCL 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573542424
  ```  
 ![avatar]( 20200730152344131.png) 

 In addition, the data flow in this piece almost confused me. Here is a picture to understand.  



--------------------------------------------------------------------------------

This part of the code mainly receives the can message of the chassis, and then calculates a wheel speed odometer. Specific: 

#  1. can_status_translator node 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 On the one hand, the received chassis message can_info into the callback function callbackFromCANInfo, mainly published vehicle_status message content 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 On the other hand, the subscription vehicle_status message, into the callbackFromVehicleStatus and then published by calculating the can_velocity and linear_velocity_viz two messages can_velocity message 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 linear_velocity_viz news 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
#  2. can_odometry node 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 ![avatar]( 38769370b9de4ecfa2cee19988ecee96.png) 

 Subscribe to the vehicle_status message, enter the callbackFromVehicleStatus callback, calculate the wheel odometer, the odometer is calculated using the Ackermann steering model: (the corresponding theory can be consulted) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 What needs to be noted here is the angular velocity calculation in the car body coordinate system, and the input is the car body speed and wheel steering angle. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 specific 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 odometer calculation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  
 The odometer motion model can also be written in the following form through the Ackermann steering model theory 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573688939
  ```  


--------------------------------------------------------------------------------

 The BAAF-Net code is based on RandLANet, so the basic structure of the two projects is the same, and the data processing part in the dataet is somewhat different. The modification of this part can be seen in the BAAF-Net training Semantic3D dataset. Here we only introduce the network structure of BAAF-Net. Here is the inference () process. The inference part is mainly divided into two parts: encoding (encoder) and decoding (decoder). The encoder involves gather_neighbour, bilateral_context_block, random_sample components, and the decoder involves nearest_interpolation components. We will introduce the encoder and decoder separately. 

#  First, the encoder 

##  1.1、gather_neighbour() 

 The core here is to use the batch_gather () method of tensorflow to obtain the feature information of each neighborhood point. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  1.2、bilateral_context_block() 

 bilateral context module 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  1.3、random_sample() 

 The down-sampling operation extracts the encoded features corresponding to the sampling points to prepare for the next layer of input. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Let's take a look at the processing process of the encoder 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Before entering the enocer, the feature is upgraded, corresponding to the Feature Extractor in the paper. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 The next step is to calculate the input data for each layer in the network layer by layer, including the location of the point and the ID of the nearest point around that point, the ID of the downsampled point and the ID of the nearest point around that point, and the ID of the point closest to the point in the next layer. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Then there is bilateral context encoding at each layer. 

#  Second, the decoder 

##  2.1、nearest_interpolation() 

 This is the upsampling implementation. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  2.2. Adaptive fusion module 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  


--------------------------------------------------------------------------------

 System: Ubuntu tensorflow: 2.2.0 cuda: 10.2 

##  Data set preprocessing 

 S3DIS Dataset Download and Cleaning Reference Article Training S3DIS Datasets Using RandLA-Net in a TensorFlow 2.0 Environment. 

##  二、BAAF-Net 

###  2.1. Compile downsampling and nearest neighbor search modules 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573662977
  ```  
 The problems encountered during compilation are as follows: 

 ![avatar]( 392de2e4d0c94af5bc1a88c7d623e43f.png) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573662977
  ```  
 ![avatar]( 5f3cbb9f30ec45158cf8b68d3291f5de.png) 

 Then recompile. 

 ![avatar]( fdeca9fe2a5e46c1a65f0a8853864043.png) 

###  2.2. Training 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573662977
  ```  
 ![avatar]( a2853427892142b38948746cb731d50a.png) 

###  2.3. Cross-validation 

 ![avatar]( 998e3edae5744744b7f5f1745c93f15a.png) 



--------------------------------------------------------------------------------

 The author of BAAF-Net only released the training and testing code of the S3DIS dataset on github, and I also asked the author that he has no plans to release the training and testing code of the Semantic3D dataset in the near future, but the author said that he is based on the code of RandLA-Net. You can refer to the code of RandLA-Net training Semantic3D dataset for modification. It just so happens that we are already very familiar with the code of Randla-Net. In this article, we will try to use BAAF-Net to train the Semantic3D dataset. 

 ![avatar]( ce21f2f62e0945e6844aac06217a1946.png) 

#  First, data preprocessing 

 Refer to the data preprocessing procedure of training Semantic3D dataset Semantic3D using RandLA-Net in tensorflow 2.0 environment. 

#  Training and testing code modification 

 Copy the main_semantic.py script of RandLaNet to the BAAF-Net directory. Two modifications 1. Network reference 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 change to 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 2、get_tf_mapping 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 Remove the layer-by-layer downsampling and nearest neighbor search operations and replace them with 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
#  III. Training 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 ![avatar]( fd1207d182804eb1ad688945cff6d997.png) 

#  IV. Testing 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 ![avatar]( ff7fe0db0bf64a449d3428586b03cf52.png) 



--------------------------------------------------------------------------------

CloudCompare supports multi-window linkage to display multiple point cloud data, which can be used to compare the original point cloud data with the semantic segmentation results. 

#  Create multiple windows 

 ![avatar]( 2435655808fe48a0a3152bf950fc5f26.png) 

 After opening CloudCompare, there will be a default view. If we want to open two data at the same time, we need to create another view. Select 3D Views in the menu bar and click New in the drop-down menu, thus creating a new view. At this time, there are two views, 3D View 1 and 3D View 2, in the drop-down menu of 3D Views, and then click Title in the 3D Views drop-down menu. At this time, two parallel windows will appear in the data display area.  

 ![avatar]( 5b5c3e7f6e52413f9cc105a1c8b68017.png) 

#  Load point cloud data separately 

 ![avatar]( 3f13f6548a1e4535ad8fce06ac401ae6.png) 

 Drag the data into the two windows respectively. Generally, the initialization state of the two windows is the same, as shown in the following figure  

 ![avatar]( e541e11effa24ac093523797ee3201a0.png) 

 If you accidentally drag and scale one of the data before doing the linkage, the initialization state of the two data is inconsistent, as shown below  

 ![avatar]( d381fc215d6d48e080e016a348532ea3.png) 

 You can reset the initialization state of the point cloud data through Camera settings in the Display menu. First, adjust the scale of the two views through the mouse wheel. As shown in the figure below, then open the Camera settings panel, adjust X1, Y1, and Z1 in Current mode to 0, and do the same for both windows (click the mouse directly to select the window), and then you can see that the two data states are exactly the same.  

#  Three, two windows linkage 

 Select the Camera link in the Display drop-down menu to link the two data, drag any data, and the other data will move with it, so that you can compare the two data. 

 ![avatar]( c9283c7e26af4571ab34fd6dd027840b.gif) 



--------------------------------------------------------------------------------

There are often people ask about the learning of CMake, and there are many blogs on the Internet that introduce the usage of learning CMake, but I don't think learning needs to be so rigid. I will learn by the way when I use it, that is, learning by doing, from shallow to deep, and slowly I will be familiar with it. This learning process will encounter many problems. Drive yourself to learn CMake by solving problems. First, summarize the benefits of CMake. CMake is a cross-platform compilation tool, so there is no need to toss the platform. For example, Windows needs to create Visual Studio project files, configure the environment and other issues, Linux create Makefiles, and OS X creates Xcode project files. In fact, most of your configuration will be the same. Using CMake will give you good project maintenance and reduce your maintenance costs. 

 Cmake is derived from Kitware and several open-source developers in the process of developing several tool kits (VTKs) 

 The raw product will eventually form a system and become an independent open-source project. The official website is www.cmake.org, you can get more information about cmake by visiting the official website, 

 Features of Cmake 

 1. Open source code, released with a BSD-like license. http://cmake.org/HTML/Copyright.html 2. Cross-platform, and can generate native compilation configuration files. On the Linux/Unix platform, generate makefiles. On the Apple platform, you can generate xcode. On the Windows platform, you can generate MSVC engineering files. 3. Able to manage large-scale projects. 4. Simplify the compilation and construction process. The toolchain of Cmake is very simple: cmake + make. 5. Efficient and extensible. Modules with specific functions can be written for cmake to expand the cmake function. 

 Tip: 1. If you don't have actual project requirements, then you can stop here, because the learning process of cmake is a practical process. Without practice, you will forget it after reading for a few more days. 2. If your project only has a few files, writing Makefiles directly is the best choice. 3. If you are using a language other than C/C++/Java, please do not use cmake (at least for now) 

 Then we will explain the usage of CMake according to the CMakeLists.txt file of each level in the CMake and PCL libraries. 

 For example, I now want to combine the PCL library to write a point cloud library-based code that does not need to convert between data formats and parse the CMake file: the .cpp file of the function is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573643556
  ```  
 The corresponding CMakeLists.txt file is written as follows, with some basic understanding 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573643556
  ```  
 Compile this code command: 

 mkdir build cd build cmake … make 

 The above process is the so-called out-of-source external compilation. One of the biggest advantages is that it has no impact on the original project, and all actions take place in the compilation directory. This is enough to convince us to use external compilation to build the project. 

 Summarize the basic syntax: 1. Variables are valued in the way of ${}, but in the IF control statement, the variable name 2 is directly used. The instruction (parameter 1, parameter 2...) parameters are enclosed in parentheses, and the parameters are separated by spaces or semicolons. Take the above ADD_EXECUTABLE instruction as an example. If there is another func.cpp source file, it should be written as: ADD_EXECUTABLE (hello main.cpp func.cpp) 3. The instruction is case-independent, and the parameters and variables are case-dependent 

 Confusion about grammar 

 SET (SRC_LIST main.c) can also be written as SET (SRC_LIST "main.cpp") There is no difference, but suppose a source file name is fu nc.c (the file name contains spaces in the middle). At this time, you must use double quotes. If you write SET (SRC_LIST fu nc.c), you will get an error, indicating that you cannot find the fu file and the nc.cpp file. In this case, you must write: SET (SRC_LIST "fu nc.cpp") 

 Clean up the project: Run: make clean to clean up the build results. 

 ![avatar]( 20191220194938367.png) 

 The above is the whole content. There may be some wrong welcome instructions, and you can send emails to communicate. You can follow the WeChat official account. Join our translation team or join the WeChat official account group, and also join the technical exchange group to communicate with more friends.  



--------------------------------------------------------------------------------

Following the previous article 

 There are interdependencies between the various modules in the PCL library 

 Among them, the Common module is the most basic module, which is the header file that defines various data structures, so the Common module does not need dependencies, but the IO module needs the support of the two major modules, common and Octree. At this time, how should we refer to their dependencies? Here we need to explain how to build static libraries and dynamic libraries. Then static libraries and dynamic libraries generally provide various functions for the implementation of other programming algorithms. 

 Here is a simple explanation of the way to create CMAKE required to create a project file. " 

 ![avatar]( 20191220195102966.png) 

 (1) Create a new folder HEllo_cmake file (2) First create the include file and create the header file libHelloCMAKE.h file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 3) Create a src file and create a new libHelloCMAKE.cpp, which implements the void printHello (); function declared in the .h file. The details are as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (4) Create a new main file, which is to create the main function, implement the print hello CMAKE function written above, and create a new useHello.cpp file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (4) The next step is to write a cmake file to compile and generate a dynamic link library, and write a function to apply the link library we generated. We create a new CMakeList.txt file in the hello_cmake file, the content of the file is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (5) Also create a new CMakeLists.txt file in src with the following content: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (6) Create a new CMakeLists.txt file under the main file, the file content: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 ![avatar]( 20191220195340502.png) 

 (7) At this time, a project file has been created, we need to compile it: Execute the following command under the hello_cmake file: mkdir build cd build cmake... make Execution result: You can see that we have generated a dynamic link library file libhello_shared.so and static link library libhello.a can directly run the program 

 How to add non-standard header file search path by INCLUDE_DIRECTORIES directive. How to add non-standard library file search path by LINK_DIRECTORIES directive. If the library is linked by TARGET_LINK_LIBRARIES or executable binary add library. And how to link to static library. 

 Regarding some common variables in CMake: 

 (1) CMAKE_SOURCE_DIR, PROJECT_SOURCE_DIR, _SOURCE_DIR all represent the top-level directory of the project 

 (2) CMAKE_CURRENT_SOURCE_DIR refers to the path where CMakeLists.txt is currently processed 

 (3) CMAKE_CURRENT_LIST_FILE output the full path to the CMakeLists.txt that calls this variable 

 (4) CMAKE_MODULE_PATH This variable is used to define the path where your own cmake module is located. If your project is more complex, you may write some cmake modules by yourself. These cmake modules are released with your project. In order for cmake to find these modules when processing CMakeLists.txt, you need to set your own cmake module path through the SET directive. For example, SET (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake) At this time, you can call your own module through the INCLUDE directive. This way of defining the CMAKE module is also available in the PCL (5) EXECUTABLE_OUTPUT_PATH and LIBRARY_OUTPUT_PATH are used to redefine the final result directory, we have already mentioned these two variables. (6) ROJECT_NAME return the project name defined by the PROJECT directive. 

 The way to call environment variables in CMAKE (1) The way to set environment variables is: SET (ENV {variable name} value). For example, the above example is used (2) CMAKE_INCLUDE_CURRENT_DIR automatically add CMAKE_CURRENT_BINARY_DIR and CMAKE_CURRENT_SOURCE_DIR to the current process, CMakeLists.txt. It is equivalent to adding: INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}) to each CMakeLists.txt. 

 The switch option in CMAKE (1) CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS, used to control the writing method of IF ELSE statement (2) BUILD_SHARED_LIBS This switch is used to control the default library compilation method. If it is not set, the use of ADD_LIBRARY does not specify the library. In the case of type, the default compiled and generated libraries are static libraries. If SET (BUILD_SHARED_LIBS ON), the default generated is the dynamic library. (3) CMAKE_C_FLAGS set the C compilation options, you can also add it through the instruction ADD_DEFINITIONS (). (4) CMAKE_CXX_FLAGS set C++ compilation options, or you can add it through the instruction ADD_DEFINITIONS (). This is a simple engineering tutorial for building CMake, and more content in PCL will be introduced in detail later 

 ![avatar]( 20191220195431876.png) 

 If the above content is wrong or needs to be added, please leave a message! At the same time, everyone is welcome to pay attention to the WeChat official account, actively share the submission, so that everyone can share together, refuse to just be a hand-reaching party! Or join the 3D visual WeChat group or QQ exchange group, communicate and share together! Submit or contact the group owner Email: dianyunpcl@163.com original is not easy, please contact the group owner for reprinting, indicate the source  



--------------------------------------------------------------------------------

image_encodings file is a source file about the image encoding mode, which specifies the RGB image and depth map encoding mode 

 The encoded file image_encodings header diagram that .cpp depends on 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwMzIwNDU1OTgzMi0xNjk3NDIzMDA3LnBuZw) 

 Functions in Command Space sensor_msgs :: image_encodings 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573625404
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573625404
  ```  
 The most important thing is that the correct coding method can realize the display of the depth map 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwMzIxMjk1MDQ4OC00NDI0MjU0MDgucG5n) 

  From this, we can see that the depth map uses cv_bridge to convert and the conversion between RGB graphs is basically similar to the structure that OPENCV can handle, but the most important thing is the correct encoding mode, so this is very critical, in order to use depth map and RGB graph to generate point clouds, so we need to use the correct encoding mode for depth map, I will not show the specific code, then we can take a look, for the difference between different encoding modes to generate point clouds, It looks like a fault, but if you cooperate with the correct encoding mode effect is not like this, so it is very important to choose the correct encoding mode when using cv_bridge, temporarily updated here, if you have any questions, you can directly comment, or follow WeChat official account, or Join the QQ exchange group to communicate with more people 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwNDE0NDkwNjM2My04NTg2ODY3ODgucG5n) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573625404
  ```  


--------------------------------------------------------------------------------

directory 

 Princeton ModelNet (CAD files) 

 Princeton Shape 

 PCD files used in PCL  

 Stanford 3D Scanning Library (PLY format) 

 Georgia Tech Scan Library (PLY format) 

 ALS Airborne LiDAR (CSV format) 

 Digital Michelangelo Project 

 Virtual Terrain Project (GIS related) 

 Technical University of Munich dataset 

 Berkeley 3D Object Dataset 

 RGB-D data 

 New York University Dataset 

 Building data 

 MIT (Urban Large-Scale Data) 

 ShapeNet dataset 

 Sydney City Dataset 

#  Princeton ModelNet (CAD files) 

 The goal of the Princeton ModelNet project is to provide researchers in computer vision, computer graphics, robotics, and cognitive science with comprehensive 3D CAD models of objects. 

>  https://modelnet.cs.princeton.edu/ 

#  Princeton Shape 

 The dataset contains a database of 3D polygon models collected from the World Wide Web. For each 3D model, an Object File Format (.off) file with the polygon geometry of the model, a model information file (e.g., the location from which it came), and a JPEG image file with a thumbnail view, the model. Benchmark Version 1 contains 1,814 models. 

 It also provides a library of software tools corresponding to the datasets for processing these datasets. 

 ![avatar]( 20210808110019853.png) 

>  https://shape.cs.princeton.edu/benchmark/ 

#   PCD files used in PCL  

 ![avatar]( 20210808152847818.png) 

>  Point Cloud Library - Browse /PCD datasets at SourceForge.net 

#   Stanford 3D Scanning Library (PLY format) 

>  http://graphics.stanford.edu/data/3Dscanrep/ 

 The purpose of this repository is to provide the public with some in-depth data and detailed reconstruction.  

 ![avatar]( 20210808153001557.png) 

 #  

 ![avatar]( 20210808154300851.png) 

  In addition, it also provides methods for converting PLY data format files to other formats: 

>  Our utility for converting PLY files to Inventor files. Click here to download the executable.Richard Harding of the Sony Playstation group has contributed a ply-to-Maya plugin. It supports import from PLY to Maya, and export from Maya to PLY, for versions 6.0 and 7.0 of Maya. Starting with Maya 8.5, this exporter can be downloaded from http://sites.google.com/site/mayaplyimportexport/.Bruce Merry of South Africa has written a script to import PLY files to Blender. Click here to download it.A shareware program written by Zoltan Karpati for converting between many 3D formats, including PLY.For converting PLY to OBJ/3DS formats, there used to be a free demo version of Deep Exploration, available here, but we hear it is no longer available.Diego Nehab (Princeton) has also written a toolkit for manipulating PLY files.Another site with information about PLY files is the PLY File Format page of the Georgia Institute of Technology's Large Geometric Models Archive.Jo�o Oliveira of University College London has modified the Georgia Tech libraries so that reading of PLY files is robust to the line breaks inserted when editing them on various platforms. Here is his package.Okino's PolyTrans package includes a PLY importer and exporter.Paolo Cignoni's MeshLab system, available from SourceForge.A C++ library for parsing PLY files, written by Ares Lagae, is available here. 

#  Georgia Tech Scan Library (PLY format) 

>  https://www.cc.gatech.edu/projects/large_models/ 

 The purpose of this website is to provide large models for researchers in computer graphics and related fields. There are thousands of geometric models available online, but the vast majority are small and therefore do not provide adequate challenges for creators of new geometric algorithms and techniques. Very large models are a challenge for techniques of rendering, auto-simplification, geometric compression, visibility techniques, surface reconstruction and surface fitting. In the digital age, there are many sources of very large geometric datasets, but many researchers do not have access to this data. This website attempts to rectify this situation.  

 ![avatar]( 20210808154410806.png) 

 ![avatar]( 20210808154426932.png) 

#  ALS Airborne LiDAR (CSV format) 

 The website is dedicated to providing datasets to the robotics community, with the aim of facilitating the evaluation and comparison of results. 

>  https://projects.asl.ethz.ch/datasets/ 

 ![avatar]( 20210808154907803.png) 

  The dataset is in CSV format, and the website provides matlab scripts to read and display the data. 

#   Digital Michelangelo Project 

 ![avatar]( 20210808160620473.png) 

 http://graphics.stanford.edu/data/mich/ 

#  Virtual Terrain Project (GIS related) 

>  http://vterrain.org/ 

#   Technical University of Munich dataset 

>  https://vision.in.tum.de/data/datasets

https://ias.cs.tum.edu/software/semantic-3d 

#  Berkeley 3D Object Dataset 

>  http://kinectdata.com/ 

#  RGB-D data 

>  http://rgbd-dataset.cs.washington.edu/index.html 

#  New York University Dataset 

>  https://cs.nyu.edu/~silberman/datasets/ 

#  Building data 

>  http://www.digital210king.org/ 

#  MIT (Urban Large-Scale Data) 

>  https://modelnet.cs.princeton.edu/

https://shape.cs.princeton.edu/benchmark/ 

#  ShapeNet dataset 

>  http://graphics.stanford.edu/data/3Dscanrep/ 

#  Sydney City Dataset 

>  https://www.cc.gatech.edu/projects/large_models/ 



--------------------------------------------------------------------------------

(1) About pcl :: PCL Point Cloud 2 :: Ptr and pcl :: Point Cloud < pcl :: Point XYZ > The difference between the two data structures 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 Difference:     

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 Then to achieve data conversion between them, 

 For example? 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 The red part of the program is a sentence that realizes the data conversion between the two. We can see that 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 Then according to this naming style, we can see more about converting between data formats of the members of the class 

 （1） 

 Function uses field_map implementation to blob a PCL :: Point Cloud 2 binary data into a PCL :: Point Cloud < pointT > object 

 Generate your own MsgFieldMap using PCLPointCloud2 (PCLPointCloud2, PointCloud < T >) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 （2） 

 void pcl::fromPCLPointCloud2 ( 

                                                   ) 

 Convert PCL :: PCL Point Cloud data format to PCL :: Point Cloud < pointT > format 

 （3） 

  void pcl::fromROSMsg（ 

                                      ） 

 （4） 

  void pcl::fromROSMsg（ 

                                      ） 

 In the use of fromROSMsg is a kind of data conversion under ROS, we take an example to achieve subscription using kinect publish /camera/depth/points from the program we can see how to use this function to achieve data conversion. And I added in the program if the use of PCL library implementation in ROS call and visualization, 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 So for this section of Mini Program, it can be converted from a published node to a visualization function that can use PCL, not necessarily RVIZ, so we analyze the following steps, in the callback function of the subscription topic, 

 Void cloud_cb (const sensor_msgs :: PointCloud2ConstPtr & input)//This sets a data type as sensor_msgs :: PointCloud2ConstPtr & input parameter {sensor_msgs :: PointCloud2 output;//The data format of the point cloud in ROS (or the data type of the point cloud in the release topic) pcl :: Point Cloud < pcl :: Point XYZ RGB >:: Ptr cloud (new pcl :: Point Cloud < pcl :: Point XYZ RGB >);//The type of data stored after conversion output =* input; pcl :: from ROS Msg (output, * cloud);//The most important step is to realize the conversion of data from ROS to PCL, and you can also directly use the PCL library to realize visualization 

   Viewer.showCloud (cloud);//visualization of the PCL library pub.publish (output);//then the original output type is still sensor_msgs :: PointCloud2, which can be visualized by RVIZ} 

 Then you can also use 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957373624
  ```  
 This piece of code to achieve the role of saving. Then see that and take a look at the visual result 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwMzExNDA1NTE3NS0yMDU3MTc5ODE3LnBuZw) 

 Visualize saved PCD files using pcl_viewer 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwMzExNDE0NDI2OS0xNjkzNDE4NzE5LnBuZw) 

 It may be messy to write, but you can refer to the conversion and visualization of point cloud data types in PCL. At the same time, interested parties are welcome to scan the QR code or QQ group below. 

 Communicate with me and submit articles, everyone learns together, makes progress and shares together 

 ![avatar]( aHR0cDovL2ltYWdlczIwMTUuY25ibG9ncy5jb20vYmxvZy85NzYzOTQvMjAxNzA0Lzk3NjM5NC0yMDE3MDQwMzEyMzQxODU4Mi05MzkyOTczODUucG5n) 



--------------------------------------------------------------------------------

This is the first draft of the article has not been perfected, there should be some problems, wait for the time to continue to update later, the original article, without permission, please do not reprint!!! 

 First, we introduce the conversion between two kinds of point clouds that are often used in PCL libraries. Here, based on the experience in engineering, we will analyze how to realize the conversion between various point cloud data defined in the program from the code level, and introduce how PCL should be used in ROS. How to transform data structures. 

 (1) PCL :: PCL Point Cloud 2 :: Ptr and pcl :: Point Cloud pcl :: Point XYZ 

 PCL :: Point XYZ is a data structure, PCL :: Point Cloud is a constructor, such as 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 The constructor pcl :: Point Cloud also contains other properties of the point cloud, such as the width height is_dense sensor_origin_ sensor_orientation_ 

 PCL :: PCL Point Cloud 2 is a structure that also contains the basic properties of the point cloud, which is defined in PCL as 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Then add Ptr: pcl :: PCL Point Cloud 2 :: Ptr to this structure to represent the smart pointer. The following program implements the function of filtering and illustrates the transformation between the two 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 PCL :: from PCL Point Cloud 2 (* cloud_filtered_blob, * cloud_filtered); This sentence implements the point cloud of PCL :: PCL Point Cloud data format to pcl :: Point Cloud format 

 (2) Smart pointer Ptr type point cloud data and non-Ptr type conversion 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 For example, in the following program, if we define a point cloud in the form of a non-smart pointer, the implementation of a segmentation code is as follows. At this time, we need to pay attention to the point cloud represented as a pointer in the form of cloud.makeShared () in setInputCloud, because the function input requires a point cloud of smart pointers. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Summarize the conversion between point cloud data formats provided in PCL 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 (3) How can the point cloud defined in ROS be converted to the point cloud data defined in PCL in ROS? 

 First of all, we have the following two point cloud data formats in ROS sensor_msgs :: PointCloud sensor_msgs :: PointCloud2 At the same time, we can also use the point cloud data type defined in PCL in ROS pcl :: PointCloud 

 So here's a summary. 

 Convert ROS to PCL :: Point Cloud data format: 

 sensor_msgs :: conversion between PointCloud2 and pcl :: Point Cloud using pcl :: from ROS Msg and pcl :: toROS Msg 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 ROS and PCL in PCL :: PCL Point Cloud 2 point cloud data conversion: 

 sensor_msgs :: conversion between PointCloud2ConstPtr and PCL :: PCL Point Cloud 2 using pcl_conversions :: toPCL pcl_conversions :: fromPCL 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Conversion between two point cloud data formats in ROS: sensor_msgs :: PointCloud and sensor_msgs :: PointCloud2, using sensor_msgs :: convertPointCloud2ToPointCloud and sensor_msgs :: convertPointCloudToPointCloud2. 

 (Here why ROS has two point cloud data formats, because in the iteration of ROS, the point cloud just defined is sensor_msgs :: PointCloud only contains the point cloud's XYZ and multiple channel point clouds, and with the development of ROS, the form can no longer meet the needs, so there is sensor_msgs :: PointCloud2 not only contains sensor_msgs :: PointCloud2 in the multi-channel point cloud data, but also adds other properties of the point cloud, such as width, height, whether dense, etc.) 

 Here is an example. For example, we want to publish a topic of point cloud data through ROS. How should we write a program? 

 The following functions are implemented to convert sensor_msgs :: PointCloud2ConstPtr to sensor_msgs :: PointCloud2 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 The above case is the simplest. In order to realize the collection of point clouds through ROS, if we want to go through a filtering process in the callback function cloud_cb in the above program, how to convert the data? We can analyze the following, need to go through the following conversion 

 sensor_msgs :: PointCloud2ConstPtr -- > pcl :: PCL PointCloud2 (here we give an example of this type as the input of the filter function, of course, it can also be the point cloud form of other PCLs) -- > sensor_msgs :: PointCloud2 (this is the most need to publish the data form of the point cloud, why this form, because this form does not appear when the RVIZ visualization in ROS warning) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Analysis: The above function uses the conversion function of ROS to convert the point cloud data twice 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 ![avatar]( 20200824224159303.PNG) 

 Here is a kinect point cloud data visualization in ROS  

 sensor_msgs :: the conversion between PointCloud2 and PCL :: Point Cloud, here is a direct callback function to achieve plane segmentation as an example, using the interface provided by PCL to achieve the conversion to ROS: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Here is no longer a program example, to summarize the ROS provided in the pcl_conversions namespace point cloud transformation relationship 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 An instance of point cloud data conversion to ROS using the functions provided in PCL 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573763869
  ```  
 Final summary: Both ROS and PCL provide each other with the conversion of PCL to ROS and ROS to PCL point cloud data. 



--------------------------------------------------------------------------------

