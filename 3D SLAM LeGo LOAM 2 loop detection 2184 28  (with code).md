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

