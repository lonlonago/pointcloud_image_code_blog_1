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

