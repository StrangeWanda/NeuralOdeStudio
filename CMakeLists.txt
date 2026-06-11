cmake_minimum_required(VERSION 3.22)

# Start by declaring only CXX. We probe for CUDA later.
project(NeuralOdeStudio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 0. Hardware Probing (CUDA Detection)
# -----------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "[NeuralOdeStudio] CUDA compiler found! Enabling GPU acceleration.")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89) # Targets Turing, Ampere, Ada
    
    # This creates a global macro so your C++ code knows the GPU is active
    add_compile_definitions(USE_CUDA_BACKEND) 
else()
    message(WARNING "[NeuralOdeStudio] CUDA compiler NOT found. Falling back to pure CPU execution.")
endif()

# -----------------------------------------------------------------------------
# 1. Fetch Dependencies (GUI Only)
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking  # <-- Pulls the docking branch directly
)
FetchContent_MakeAvailable(imgui)

# -----------------------------------------------------------------------------
# 2. Define Targets (The Subdirectories)
# -----------------------------------------------------------------------------
add_subdirectory(src/core)     # The Math Engine
add_subdirectory(src/backend)  # The Training Engine
# add_subdirectory(src/datagen)

# -----------------------------------------------------------------------------
# 3. Layer 4: The App Studio (GUI)
# -----------------------------------------------------------------------------
set(IMGUI_SRC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp    # Needed for ImGui::ShowDemoWindow()
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# We define the GUI executable target
add_executable(app_studio 
    src/gui/src/main.cpp 
    ${IMGUI_SRC}
)

# Include directories so the GUI can see ImGui headers
target_include_directories(app_studio PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

# Link the windowing libraries and our custom engines
target_link_libraries(app_studio PRIVATE
    glfw
    opengl32 
    ncore
    nbackend
)