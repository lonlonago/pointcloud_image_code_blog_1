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
