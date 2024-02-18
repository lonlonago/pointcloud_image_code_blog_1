 Code reference OpenPCDet. 

##  Global transformation 

 The global transformation is to transform the point cloud scene, such as scaling, rotating, and reversing. Both the point cloud and the box are transformed as a whole, and the effect is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 flip 

 ![avatar]( 3058a3d80baf4cc3bd00b84def0c8e85.png) 

 spin 

 ![avatar]( f3eed0aec8b64ae688af46d5052a4bac.png) 

 zoom 

 ![avatar]( df17ba24c97d48049a15f4fe8702c550.png) 

##  Second, target transformation 

 Target-level transformation, random transformation of point clouds and boxes in each target, including rotation, scaling, translation, sparsity and other operations. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 spin 

 ![avatar]( 2ab87f1e97de4688b2a426bcb21729d5.png) 

 Scaling, for obvious contrast, the scaling factor I used here is relatively large, and a smaller scaling factor should be used in actual training. 

 ![avatar]( 6a1b80e35c4f43d0bced4fbe5021d92b.png) 

 Translation, like scaling strategies, should have smaller coefficients when actually used. 

 ![avatar]( 283fbfd7479a4af4951804dc2256d6a4.png) 

##  III. Local transformation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573558958
  ```  
 Sparse, divide the target into 6 regions, and randomly select one of them to sparse. 

 ![avatar]( 127ec78b8df9407e8acdc9407273d202.png) 

 ![avatar]( 438be589f2d04793a8ee01998200b80c.png) 

##  III. Code and test data 

 Link: https://pan.baidu.com/s/1k-GRptGlg0OB6OlacikrJw Extraction Code: j9m6 

