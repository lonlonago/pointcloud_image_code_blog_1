Following the previous article 

 There are interdependencies between the various modules in the PCL library 

 Among them, the Common module is the most basic module, which is the header file that defines various data structures, so the Common module does not need dependencies, but the IO module needs the support of the two major modules, common and Octree. At this time, how should we refer to their dependencies? Here we need to explain how to build static libraries and dynamic libraries. Then static libraries and dynamic libraries generally provide various functions for the implementation of other programming algorithms. 

 Here is a simple explanation of the way to create CMAKE required to create a project file. " 

 ![avatar]( 20191220195102966.png) 

 (1) Create a new folder HEllo_cmake file (2) First create the include file and create the header file libHelloCMAKE.h file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 3) Create a src file and create a new libHelloCMAKE.cpp, which implements the void printHello (); function declared in the .h file. The details are as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (4) Create a new main file, which is to create the main function, implement the print hello CMAKE function written above, and create a new useHello.cpp file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (4) The next step is to write a cmake file to compile and generate a dynamic link library, and write a function to apply the link library we generated. We create a new CMakeList.txt file in the hello_cmake file, the content of the file is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (5) Also create a new CMakeLists.txt file in src with the following content: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 (6) Create a new CMakeLists.txt file under the main file, the file content: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573622655
  ```  
 ![avatar]( 20191220195340502.png) 

 (7) At this time, a project file has been created, we need to compile it: Execute the following command under the hello_cmake file: mkdir build cd build cmake... make Execution result: You can see that we have generated a dynamic link library file libhello_shared.so and static link library libhello.a can directly run the program 

 How to add non-standard header file search path by INCLUDE_DIRECTORIES directive. How to add non-standard library file search path by LINK_DIRECTORIES directive. If the library is linked by TARGET_LINK_LIBRARIES or executable binary add library. And how to link to static library. 

 Regarding some common variables in CMake: 

 (1) CMAKE_SOURCE_DIR, PROJECT_SOURCE_DIR, _SOURCE_DIR all represent the top-level directory of the project 

 (2) CMAKE_CURRENT_SOURCE_DIR refers to the path where CMakeLists.txt is currently processed 

 (3) CMAKE_CURRENT_LIST_FILE output the full path to the CMakeLists.txt that calls this variable 

 (4) CMAKE_MODULE_PATH This variable is used to define the path where your own cmake module is located. If your project is more complex, you may write some cmake modules by yourself. These cmake modules are released with your project. In order for cmake to find these modules when processing CMakeLists.txt, you need to set your own cmake module path through the SET directive. For example, SET (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake) At this time, you can call your own module through the INCLUDE directive. This way of defining the CMAKE module is also available in the PCL (5) EXECUTABLE_OUTPUT_PATH and LIBRARY_OUTPUT_PATH are used to redefine the final result directory, we have already mentioned these two variables. (6) ROJECT_NAME return the project name defined by the PROJECT directive. 

 The way to call environment variables in CMAKE (1) The way to set environment variables is: SET (ENV {variable name} value). For example, the above example is used (2) CMAKE_INCLUDE_CURRENT_DIR automatically add CMAKE_CURRENT_BINARY_DIR and CMAKE_CURRENT_SOURCE_DIR to the current process, CMakeLists.txt. It is equivalent to adding: INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}) to each CMakeLists.txt. 

 The switch option in CMAKE (1) CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS, used to control the writing method of IF ELSE statement (2) BUILD_SHARED_LIBS This switch is used to control the default library compilation method. If it is not set, the use of ADD_LIBRARY does not specify the library. In the case of type, the default compiled and generated libraries are static libraries. If SET (BUILD_SHARED_LIBS ON), the default generated is the dynamic library. (3) CMAKE_C_FLAGS set the C compilation options, you can also add it through the instruction ADD_DEFINITIONS (). (4) CMAKE_CXX_FLAGS set C++ compilation options, or you can add it through the instruction ADD_DEFINITIONS (). This is a simple engineering tutorial for building CMake, and more content in PCL will be introduced in detail later 

 ![avatar]( 20191220195431876.png) 

 If the above content is wrong or needs to be added, please leave a message! At the same time, everyone is welcome to pay attention to the WeChat official account, actively share the submission, so that everyone can share together, refuse to just be a hand-reaching party! Or join the 3D visual WeChat group or QQ exchange group, communicate and share together! Submit or contact the group owner Email: dianyunpcl@163.com original is not easy, please contact the group owner for reprinting, indicate the source  

