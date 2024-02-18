There are often people ask about the learning of CMake, and there are many blogs on the Internet that introduce the usage of learning CMake, but I don't think learning needs to be so rigid. I will learn by the way when I use it, that is, learning by doing, from shallow to deep, and slowly I will be familiar with it. This learning process will encounter many problems. Drive yourself to learn CMake by solving problems. First, summarize the benefits of CMake. CMake is a cross-platform compilation tool, so there is no need to toss the platform. For example, Windows needs to create Visual Studio project files, configure the environment and other issues, Linux create Makefiles, and OS X creates Xcode project files. In fact, most of your configuration will be the same. Using CMake will give you good project maintenance and reduce your maintenance costs. 

 Cmake is derived from Kitware and several open-source developers in the process of developing several tool kits (VTKs) 

 The raw product will eventually form a system and become an independent open-source project. The official website is www.cmake.org, you can get more information about cmake by visiting the official website, 

 Features of Cmake 

 1. Open source code, released with a BSD-like license. http://cmake.org/HTML/Copyright.html 2. Cross-platform, and can generate native compilation configuration files. On the Linux/Unix platform, generate makefiles. On the Apple platform, you can generate xcode. On the Windows platform, you can generate MSVC engineering files. 3. Able to manage large-scale projects. 4. Simplify the compilation and construction process. The toolchain of Cmake is very simple: cmake + make. 5. Efficient and extensible. Modules with specific functions can be written for cmake to expand the cmake function. 

 Tip: 1. If you don't have actual project requirements, then you can stop here, because the learning process of cmake is a practical process. Without practice, you will forget it after reading for a few more days. 2. If your project only has a few files, writing Makefiles directly is the best choice. 3. If you are using a language other than C/C++/Java, please do not use cmake (at least for now) 

 Then we will explain the usage of CMake according to the CMakeLists.txt file of each level in the CMake and PCL libraries. 

 For example, I now want to combine the PCL library to write a point cloud library-based code that does not need to convert between data formats and parse the CMake file: the .cpp file of the function is as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573643556
  ```  
 The corresponding CMakeLists.txt file is written as follows, with some basic understanding 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573643556
  ```  
 Compile this code command: 

 mkdir build cd build cmake … make 

 The above process is the so-called out-of-source external compilation. One of the biggest advantages is that it has no impact on the original project, and all actions take place in the compilation directory. This is enough to convince us to use external compilation to build the project. 

 Summarize the basic syntax: 1. Variables are valued in the way of ${}, but in the IF control statement, the variable name 2 is directly used. The instruction (parameter 1, parameter 2...) parameters are enclosed in parentheses, and the parameters are separated by spaces or semicolons. Take the above ADD_EXECUTABLE instruction as an example. If there is another func.cpp source file, it should be written as: ADD_EXECUTABLE (hello main.cpp func.cpp) 3. The instruction is case-independent, and the parameters and variables are case-dependent 

 Confusion about grammar 

 SET (SRC_LIST main.c) can also be written as SET (SRC_LIST "main.cpp") There is no difference, but suppose a source file name is fu nc.c (the file name contains spaces in the middle). At this time, you must use double quotes. If you write SET (SRC_LIST fu nc.c), you will get an error, indicating that you cannot find the fu file and the nc.cpp file. In this case, you must write: SET (SRC_LIST "fu nc.cpp") 

 Clean up the project: Run: make clean to clean up the build results. 

 ![avatar]( 20191220194938367.png) 

 The above is the whole content. There may be some wrong welcome instructions, and you can send emails to communicate. You can follow the WeChat official account. Join our translation team or join the WeChat official account group, and also join the technical exchange group to communicate with more friends.  

