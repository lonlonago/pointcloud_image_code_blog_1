 The BAAF-Net code is based on RandLANet, so the basic structure of the two projects is the same, and the data processing part in the dataet is somewhat different. The modification of this part can be seen in the BAAF-Net training Semantic3D dataset. Here we only introduce the network structure of BAAF-Net. Here is the inference () process. The inference part is mainly divided into two parts: encoding (encoder) and decoding (decoder). The encoder involves gather_neighbour, bilateral_context_block, random_sample components, and the decoder involves nearest_interpolation components. We will introduce the encoder and decoder separately. 

#  First, the encoder 

##  1.1、gather_neighbour() 

 The core here is to use the batch_gather () method of tensorflow to obtain the feature information of each neighborhood point. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  1.2、bilateral_context_block() 

 bilateral context module 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  1.3、random_sample() 

 The down-sampling operation extracts the encoded features corresponding to the sampling points to prepare for the next layer of input. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Let's take a look at the processing process of the encoder 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Before entering the enocer, the feature is upgraded, corresponding to the Feature Extractor in the paper. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 The next step is to calculate the input data for each layer in the network layer by layer, including the location of the point and the ID of the nearest point around that point, the ID of the downsampled point and the ID of the nearest point around that point, and the ID of the point closest to the point in the next layer. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
 Then there is bilateral context encoding at each layer. 

#  Second, the decoder 

##  2.1、nearest_interpolation() 

 This is the upsampling implementation. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
##  2.2. Adaptive fusion module 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573698056
  ```  
