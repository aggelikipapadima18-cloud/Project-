cmake_minimum_required(VERSION 3.15)
project(ikona_assignments_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(askisi1_erotima1 askisi1_erotima1.cpp)
target_link_libraries(askisi1_erotima1 PRIVATE ${OpenCV_LIBS})

add_executable(askisi2_histogram_equalization askisi2_histogram_equalization.cpp)
target_link_libraries(askisi2_histogram_equalization PRIVATE ${OpenCV_LIBS})

add_executable(askisi3_denoising askisi3_denoising.cpp)
target_link_libraries(askisi3_denoising PRIVATE ${OpenCV_LIBS})

add_executable(askisi4_restoration askisi4_restoration.cpp)
target_link_libraries(askisi4_restoration PRIVATE ${OpenCV_LIBS})

add_executable(askisi5_dct_compression askisi5_dct_compression.cpp)
target_link_libraries(askisi5_dct_compression PRIVATE ${OpenCV_LIBS})

add_executable(askisi6_edges askisi6_edges.cpp)
target_link_libraries(askisi6_edges PRIVATE ${OpenCV_LIBS})

add_executable(run_all_assignments_c run_all_assignments.cpp)
