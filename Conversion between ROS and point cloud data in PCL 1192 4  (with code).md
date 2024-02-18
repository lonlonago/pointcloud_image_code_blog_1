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

