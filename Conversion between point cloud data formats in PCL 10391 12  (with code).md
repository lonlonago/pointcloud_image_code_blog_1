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

