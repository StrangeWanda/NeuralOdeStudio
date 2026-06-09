cmake_minimum_required(VERSION 3.22)

# For now, we only require CXX (C++). We will add CUDA here later.
project(NeuralOdeStudio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
# 2. Define Targets (The Stubbing Strategy)
# -----------------------------------------------------------------------------

# TODO: Uncomment these as we build the layers
# add_subdirectory(src/core)
# add_subdirectory(src/backend)
# add_subdirectory(src/datagen)

# -----------------------------------------------------------------------------
# 3. Layer 4: The App Studio (GUI)
# -----------------------------------------------------------------------------
set(IMGUI_SRC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp    # <-- Add this line
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

# Include directories
target_include_directories(app_studio PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

# Link the windowing libraries
target_link_libraries(app_studio PRIVATE
    glfw
    opengl32 
)