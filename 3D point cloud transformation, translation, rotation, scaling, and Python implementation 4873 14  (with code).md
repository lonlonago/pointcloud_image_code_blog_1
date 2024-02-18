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

