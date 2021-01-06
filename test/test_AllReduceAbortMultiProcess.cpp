/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllReduceAbortMultiProcess.hpp"

#define NUM_ITER 8
#define FAKE_OP_COUNT NUM_ITER+1

namespace CorrectnessTests
{
    TEST_P(AllReduceAbortMultiProcessTest, Correctness) {
        Dataset* dataset = (Dataset*)mmap(NULL, sizeof(Dataset), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);

        int* done = (int*)mmap(NULL, numDevices*sizeof(int), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        int* remaining = (int*)mmap(NULL, sizeof(int), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        ncclResult_t* ncclAsyncErr = (ncclResult_t*)mmap(NULL, sizeof(ncclResult_t), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        memset(done, 0, numDevices*sizeof(int));
        *remaining = numDevices;
        Barrier::ClearShmFiles(std::atoi(getenv("NCCL_COMM_ID")));

        int pid1 = 0;
        int pid2 = 0;
        int pid3 = 0;
        pid1 = fork();

        // From this point on, ignore original process as we cannot have it create a HIP context
        if (pid1 == 0)
        {
            pid2 = fork();
            if (numDevices > 2)
            {
                pid3 = fork();
            }
            if ((pid2 > 0 && pid3 == 0 && numDevices == 2)  || (pid2 > 0 && pid3 > 0 && numDevices > 2))
            {
                // Process 0
                TestAllReduceAbort(0, *dataset, done, *remaining, *ncclAsyncErr);
                if (pid3 > 0)
                {
                    waitpid(pid3, NULL, 0);
                }
            }
            else if ((pid2 == 0 && pid3 == 0 && numDevices == 2) || (pid2 == 0 && pid3 > 0 && numDevices > 2))
            {
                // Process 1
                TestAllReduceAbort(1, *dataset, done, *remaining, *ncclAsyncErr);
                if (numDevices > 2)
                {
                    waitpid(pid3, NULL, 0);
                }
                exit(0);
            }
            else if (pid2 > 0 && pid3 == 0 && numDevices > 2)
            {
                // Process 2 (available when numDevices > 2)
                TestAllReduceAbort(2, *dataset, done, *remaining, *ncclAsyncErr);
                exit(0);
            }
            else if (pid2 == 0 && pid3 == 0 && numDevices == 4)
            {
                // Process 3 (available when numDevices == 4)
                TestAllReduceAbort(3, *dataset, done, *remaining, *ncclAsyncErr);
                exit(0);
            }
            else
            {
                exit(0);
            }
            waitpid(pid2, NULL, 0);
            exit(0);
        }
        waitpid(pid1, NULL, 0);
    }

    INSTANTIATE_TEST_CASE_P(AllReduceAbortMultiProcessSweep,
                            AllReduceAbortMultiProcessTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclFloat32),
                                // Number of elements
                                testing::Values(1024, 1048576),
                                // Number of devices
                                testing::Values(2, 4),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
