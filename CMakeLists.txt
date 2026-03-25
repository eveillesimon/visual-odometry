cmake_minimum_required(VERSION 3.10)
project(visual_odometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET_NAME visual_odometry)

find_package(OpenCV REQUIRED)

add_executable(${TARGET_NAME} src/main.cpp)

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()