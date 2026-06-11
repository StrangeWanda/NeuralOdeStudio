# LAYER 1: The Core Math & Inference Engine

add_library(ncore STATIC 
    src/neural_network.cpp
    include/core/matrix.h
    include/core/neural_network.h
    include/core/rk4_solver.h
)

target_include_directories(ncore PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ncore PUBLIC cxx_std_20)