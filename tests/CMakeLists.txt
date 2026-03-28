include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(test_image_sequence
    test_image_sequence.cpp
    ../src/image_sequence.cpp
)

target_include_directories(test_image_sequence PRIVATE
    ../include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_image_sequence PRIVATE
    GTest::gtest_main
    ${OpenCV_LIBS}
)

target_compile_definitions(test_image_sequence PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(MSVC)
  target_compile_options(test_image_sequence PRIVATE /W4 /WX)
else()
  target_compile_options(test_image_sequence PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_test(NAME test_image_sequence COMMAND test_image_sequence)