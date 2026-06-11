# LAYER 2: The Training & GPU Engine

set(BACKEND_SRC
    include/backend/i_trainer.h
    src/cpu_trainer.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND BACKEND_SRC 
        include/backend/adjoint_solver.cuh
        src/cuda_trainer.cu
    )
endif()



add_library(nbackend STATIC ${BACKEND_SRC})

target_include_directories(nbackend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nbackend PUBLIC ncore)

target_compile_features(nbackend PUBLIC cxx_std_20)