 The author of BAAF-Net only released the training and testing code of the S3DIS dataset on github, and I also asked the author that he has no plans to release the training and testing code of the Semantic3D dataset in the near future, but the author said that he is based on the code of RandLA-Net. You can refer to the code of RandLA-Net training Semantic3D dataset for modification. It just so happens that we are already very familiar with the code of Randla-Net. In this article, we will try to use BAAF-Net to train the Semantic3D dataset. 

 ![avatar]( ce21f2f62e0945e6844aac06217a1946.png) 

#  First, data preprocessing 

 Refer to the data preprocessing procedure of training Semantic3D dataset Semantic3D using RandLA-Net in tensorflow 2.0 environment. 

#  Training and testing code modification 

 Copy the main_semantic.py script of RandLaNet to the BAAF-Net directory. Two modifications 1. Network reference 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 change to 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 2、get_tf_mapping 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 Remove the layer-by-layer downsampling and nearest neighbor search operations and replace them with 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
#  III. Training 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 ![avatar]( fd1207d182804eb1ad688945cff6d997.png) 

#  IV. Testing 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573636203
  ```  
 ![avatar]( ff7fe0db0bf64a449d3428586b03cf52.png) 

