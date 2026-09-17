cmake_minimum_required(VERSION 3.24)
project(StrikeCEM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Pinned dependencies (IMPLEMENTATION.md keeps this surface small on purpose).
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
        json_schema_validator
        GIT_REPOSITORY https://github.com/pboettch/json-schema-validator.git
        GIT_TAG 2.4.0
)
set(JSON_VALIDATOR_BUILD_TESTS OFF)
set(JSON_VALIDATOR_BUILD_EXAMPLES OFF)
set(JSON_VALIDATOR_INSTALL OFF)
FetchContent_MakeAvailable(json_schema_validator)

find_package(OpenSSL REQUIRED)

# Public headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library: config loader, conventions, sample plans.
add_library(scem_core STATIC src/strikecem/core/Config.cpp)
target_link_libraries(scem_core PUBLIC
        nlohmann_json::nlohmann_json
        nlohmann_json_schema_validator
        OpenSSL::Crypto)

# Executable (SPEC FR-11: invoked as `strikecem`).
add_executable(strikecem src/strikecem/main.cpp src/strikecem/app/CLI.cpp)
target_link_libraries(strikecem PRIVATE scem_core)

# Shared library target for later StrikeEngine integration
add_library(strikecem_lib SHARED src/strikecem/app/CLI.cpp)
target_link_libraries(strikecem_lib PUBLIC scem_core)
set_target_properties(strikecem_lib PROPERTIES OUTPUT_NAME "StrikeCEM")

# Tests
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
enable_testing()
add_executable(scem_tests tests/test_conventions.cpp tests/test_config.cpp)
target_link_libraries(scem_tests PRIVATE scem_core GTest::gtest_main)
target_compile_definitions(scem_tests PRIVATE
        SCEM_SCHEMA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/schemas/config.schema.json"
        SCEM_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
gtest_discover_tests(scem_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Installation rules
include(GNUInstallDirs)
install(TARGETS strikecem strikecem_lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/strikecem)
install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/strikecem/examples)
install(DIRECTORY schemas/ DESTINATION ${CMAKE_INSTALL_DATADIR}/strikecem/schemas)
