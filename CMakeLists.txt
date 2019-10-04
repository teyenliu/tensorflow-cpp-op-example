# FIXME: I still cannot use CMake to build so that I use g++ command instead.
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#g++ -std=c++11 -shared inner_product.cc -o libinner_product.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
#g++ -std=c++11 -shared inner_product_grad.cc -o libinner_product_grad.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2


cmake_minimum_required(VERSION 2.8)

project (demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

#if(DEFINED ENV{TensorFlow_GIT_REPO})
#  message(STATUS "FOUND TENSORFLOW GIT-REPO in $ENV{TensorFlow_GIT_REPO}")
#else()
#    message(FATAL_ERROR "TensorFlow_GIT_REPO must be defined, see README.md")
#endif()


# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -O2 ${CMAKE_CXX_FLAGS} ")

#TARGET_LINK_LIBRARIES(demo tensorflow_framework)
# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories("/usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public")
link_directories(${Tensorflow_LIB_DIRS})
add_library(inner_product SHARED inner_product.cc)

# build the gradient operation which is used in inner_product_grad.py
# to register it
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories("/usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public")
link_directories(${Tensorflow_LIB_DIRS})
add_library(inner_product_grad SHARED inner_product_grad.cc)

