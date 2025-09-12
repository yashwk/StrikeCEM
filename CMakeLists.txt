cmake_minimum_required(VERSION 3.24)
project(StrikeCEM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Fetch nlohmann_json
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

# Sources
set(SOURCES
        src/strikecem/main.cpp
        src/strikecem/core/Config.cpp
        src/strikecem/io/MeshLoader.cpp
        src/strikecem/solvers/PhysicalOptics.cpp
)

# Public headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(StrikeCEM ${SOURCES})
target_link_libraries(StrikeCEM PRIVATE nlohmann_json::nlohmann_json)

# Shared library target for later StrikeEngine integration
add_library(strikecem_lib SHARED ${SOURCES})
target_link_libraries(strikecem_lib PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(strikecem_lib PROPERTIES OUTPUT_NAME "StrikeCEM")

# Installation rules
include(GNUInstallDirs)
install(TARGETS StrikeCEM strikecem_lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/strikecem)
install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/strikecem/examples)
install(DIRECTORY schemas/ DESTINATION ${CMAKE_INSTALL_DATADIR}/strikecem/schemas)
