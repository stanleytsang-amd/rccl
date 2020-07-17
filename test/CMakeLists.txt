# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  message("Going to build unit tests (Installed in /test/UnitTests)")

  # chrpath is required to properly set rpath for the UnitTests executable
  find_program(CHRPATH chrpath)
  if(NOT CHRPATH)
      message(FATAL_ERROR "chrpath is required for UnitTests. Please install (e.g. sudo apt-get install chrpath)")
  endif()

  # Download and unpack googletest at configure time
  # configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
  # execute_process(
    # COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    # RESULT_VARIABLE result
    # WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
  # if(result)
    # message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  # endif()
  # execute_process(
    # COMMAND ${CMAKE_COMMAND} --build .
    # RESULT_VARIABLE result
    # WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
  # if(result)
    # message(FATAL_ERROR "Build step for googletest failed: ${result}")
  # endif()

  # Add googletest directly to our build. This adds the following targets:
  # gtest, gtest_main, gmock and gmock_main
  add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                   "${CMAKE_BINARY_DIR}/googletest-build")

  # Add googletest directly to our build. This defines the gtest and gtest_main
  # targets. add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
  # ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)

  # Collect source files for tests
  set(TEST_SOURCES_SINGLE_PROCESS
    test_AllGather.cpp
    test_AllReduce.cpp
    test_Broadcast.cpp
    test_Reduce.cpp
    test_ReduceScatter.cpp
    test_GroupCalls.cpp
    test_CombinedCalls.cpp
    test_AllReduceAbort.cpp
    test_BroadcastAbort.cpp
    test_Scatter.cpp
    test_Gather.cpp
    test_AllToAll.cpp
  )

  set(TEST_SOURCES_MULTI_PROCESS
    test_AllGatherMultiProcess.cpp
    test_AllReduceMultiProcess.cpp
    test_AllReduceAbortMultiProcess.cpp
    test_AllToAllMultiProcess.cpp
    test_BroadcastMultiProcess.cpp
    test_BroadcastAbortMultiProcess.cpp
    test_CombinedCallsMultiProcess.cpp
    test_GatherMultiProcess.cpp
    test_GroupCallsMultiProcess.cpp
    test_ReduceMultiProcess.cpp
    test_ReduceScatterMultiProcess.cpp
    test_ScatterMultiProcess.cpp
  )

  add_executable(UnitTestsSingleProcess ${TEST_SOURCES_SINGLE_PROCESS})
  add_executable(UnitTestsMultiProcess ${TEST_SOURCES_MULTI_PROCESS})
  target_include_directories(UnitTestsSingleProcess PRIVATE /opt/rocm)
  target_link_libraries(UnitTestsSingleProcess PRIVATE gtest_main PRIVATE rccl)
  install(TARGETS UnitTestsSingleProcess RUNTIME DESTINATION test)
  target_include_directories(UnitTestsMultiProcess PRIVATE /opt/rocm)
  target_link_libraries(UnitTestsMultiProcess PRIVATE gtest_main PRIVATE rccl PRIVATE rt)
  install(TARGETS UnitTestsMultiProcess RUNTIME DESTINATION test)

  # HCC adds /opt/rocm/lib as RPATH, even though the install process is supposed to
  # remove RPATH.  As a work-around, set the correct RPATH for the unit test executable
  # as a post-install step
  install(
    CODE
      "execute_process(COMMAND chrpath -r ${CMAKE_INSTALL_PREFIX}/lib:/opt/rocm/lib ${CMAKE_INSTALL_PREFIX}/test/UnitTests)"
    )
else()
  message("Not building unit tests")
endif()
