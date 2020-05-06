# Scripts want to rely on stock FindOpenCL.cmake
cmake_minimum_required (VERSION 3.1)

set (Target_Name SYCL-Raycaster)

# Always include current directory for all compile commands 
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

# Generate the configuration file for application to locate shader files
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/${Target_Name}-Config.in.hpp"
                "${CMAKE_CURRENT_BINARY_DIR}/include/${Target_Name}-Config.hpp")

# Specify source and header files
set (Sources source/Main.cpp
             source/${Target_Name}.cpp
             source/InteropWindow.cpp
             source/LatticeBoltzmann2D.cpp)

set (Headers include/InteropWindow.hpp
             include/LatticeBoltzmann2D.hpp
             include/${Target_Name}.hpp
             ${CMAKE_CURRENT_BINARY_DIR}/include/${Target_Name}-Config.hpp)

set (Shaders shader/Vertex.glsl
             shader/Fragment.glsl)

find_package(glm)
find_package (Qt5 5.5 CONFIG REQUIRED Core Gui OpenGL)
find_package (OpenGL REQUIRED)

# Append to module path for locating FindComputeCpp.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# Find dependencies
find_package (ComputeCpp REQUIRED)

# Declare executable target
add_executable(${Target_Name} ${Sources} ${Headers} ${Shaders})

# Target include directories
target_include_directories (${Target_Name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                  ${CMAKE_CURRENT_BINARY_DIR}/include
                                                  ${Qt5Gui_PRIVATE_INCLUDE_DIRS}
                                                  ${ComputeCpp_INCLUDE_DIRS})

# Target link directives
target_link_libraries(${Target_Name} PUBLIC Qt5::Core Qt5::Gui Qt5::OpenGL $<$<PLATFORM_ID:Windows>:Qt5::WinMain>
                                            OpenGL::GL glm)

# Automatically invoke Qt tools on source files
set_target_properties(${Target_Name} PROPERTIES CXX_STANDARD 14
                                                CXX_STANDARD_REQUIRED ON
                                                AUTOMOC ON)

# Apply the ComputeCpp compiler to select source files (containing SYCL code)
add_sycl_to_target(TARGET ${Target_Name}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/${Target_Name}.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/source/LatticeBoltzmann2D.cpp)

# Create source groups for IDE
source_group ("Sources" FILES ${Sources})
source_group ("Headers" FILES ${Headers})
source_group ("Shaders" FILES ${Shaders})
