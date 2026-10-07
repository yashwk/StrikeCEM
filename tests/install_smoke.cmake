file(REMOVE_RECURSE "${SCEM_INSTALL_PREFIX}")
execute_process(
        COMMAND "${CMAKE_COMMAND}" --install "${SCEM_BUILD_DIR}" --prefix "${SCEM_INSTALL_PREFIX}"
        RESULT_VARIABLE install_result
        OUTPUT_VARIABLE install_output
        ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "install failed: ${install_output}${install_error}")
endif()

if(IS_ABSOLUTE "${SCEM_BINDIR}")
    set(binary "${SCEM_BINDIR}/strikecem")
else()
    set(binary "${SCEM_INSTALL_PREFIX}/${SCEM_BINDIR}/strikecem")
endif()
if(IS_ABSOLUTE "${SCEM_INCLUDEDIR}")
    set(header "${SCEM_INCLUDEDIR}/strikecem/io/MeshLoader.hpp")
else()
    set(header "${SCEM_INSTALL_PREFIX}/${SCEM_INCLUDEDIR}/strikecem/io/MeshLoader.hpp")
endif()
if(NOT EXISTS "${header}" OR EXISTS "${SCEM_INSTALL_PREFIX}/${SCEM_INCLUDEDIR}/strikecem/strikecem")
    message(FATAL_ERROR "installed header layout is incorrect: ${header}")
endif()

execute_process(
        COMMAND "${CMAKE_COMMAND}" -E chdir "${SCEM_INSTALL_PREFIX}"
                "${binary}" validate "${SCEM_SOURCE_DIR}/tests/fixtures/valid_minimal.json"
        RESULT_VARIABLE run_result
        OUTPUT_VARIABLE run_output
        ERROR_VARIABLE run_error)
if(NOT run_result EQUAL 0 OR NOT run_output MATCHES "status: ok")
    message(FATAL_ERROR "installed out-of-tree CLI failed: ${run_output}${run_error}")
endif()
