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

