/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_BroadcastAbortMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(BroadcastAbortMultiProcessTest, Correctness)
    {
        Dataset* dataset = (Dataset*)mmap(NULL, sizeof(Dataset), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollBroadcast);

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
            if (pid2 > 0)
            {
                printf("pid2: %d\n", pid2);
                fflush(stdout);
            }
            if (numDevices > 2)
            {
                pid3 = fork();
                if (pid3 > 0)
                {
                    printf("pid3: %d\n", pid3);
                    fflush(stdout);
                }
            }
            if ((pid2 > 0 && pid3 == 0 && numDevices == 2)  || (pid2 > 0 && pid3 > 0 && numDevices > 2))
            {
                // Process 0
                printf("Process 0 in\n");
                TestBroadcastAbort(0, *dataset, done, *remaining, *ncclAsyncErr);
                if (pid3 > 0)
                {
                    waitpid(pid3, NULL, 0);
                    printf("pid3 close: %d\n", pid3);
                    fflush(stdout);
                }
            }
            else if ((pid2 == 0 && pid3 == 0 && numDevices == 2) || (pid2 == 0 && pid3 > 0 && numDevices > 2))
            {
                // Process 1
                printf("Process 1 in\n");
                fflush(stdout);
                TestBroadcastAbort(1, *dataset, done, *remaining, *ncclAsyncErr);
                if (numDevices > 2)
                {
                    waitpid(pid3, NULL, 0);
                    printf("pid3 close: %d\n", pid3);
                    fflush(stdout);
                }
                exit(0);
            }
            else if (pid2 > 0 && pid3 == 0 && numDevices > 2)
            {
                // Process 2 (available when numDevices > 2)
                printf("Process 2 in\n");
                fflush(stdout);
                TestBroadcastAbort(2, *dataset, done, *remaining, *ncclAsyncErr);
                exit(0);
            }
            else if (pid2 == 0 && pid3 == 0 && numDevices == 4)
            {
                // Process 3 (available when numDevices == 4)
                printf("Process 3 in\n");
                fflush(stdout);
                TestBroadcastAbort(3, *dataset, done, *remaining, *ncclAsyncErr);
                exit(0);
            }
            else
            {
                exit(0);
            }
            waitpid(pid2, NULL, 0);
            printf("pid2 close: %d\n", pid2);
            fflush(stdout);
            exit(0);
        }
        printf("pid1: %d\n", pid1);
        waitpid(pid1, NULL, 0);
        printf("pid1 close: %d\n", pid1);
        fflush(stdout);
    }

    INSTANTIATE_TEST_CASE_P(BroadcastAbortMultiProcessSweep,
                            BroadcastAbortMultiProcessTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclFloat32),
                                // Number of elements
                                testing::Values(1048576),
                                // Number of devices
                                testing::Values(2, 4),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
