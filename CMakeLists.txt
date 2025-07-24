cmake_minimum_required(VERSION 3.16)
project(underactuated-arm)

set(CMAKE_CXX_STANDARD 17)

# Set MuJoCo root
set(MUJOCO_PATH /opt/mujoco)
include_directories(${MUJOCO_PATH}/include)

# Link directories
link_directories(
  /opt/mujoco/build/lib
  /usr/lib/aarch64-linux-gnu  # (or wherever libglfw.so is)
)

# Core source
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS core/*.cpp)

# -------- Executable --------
add_executable(underactuated-arm ${CORE_SOURCES})

# -------- Link Libraries --------
target_link_libraries(underactuated-arm
    mujoco
    glfw
    pthread
    dl
    m
    X11
)
