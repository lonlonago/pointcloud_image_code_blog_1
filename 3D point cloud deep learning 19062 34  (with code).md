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

