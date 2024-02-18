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

