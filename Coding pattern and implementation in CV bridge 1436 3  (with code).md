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
