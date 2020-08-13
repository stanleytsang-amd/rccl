/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_ReduceScatterMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(ReduceScatterMultiProcessCorrectnessTest, Correctness)
    {
        Dataset* dataset = (Dataset*)mmap(NULL, sizeof(Dataset), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollReduceScatter);
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
                fflush(stdout);
                TestReduceScatter(0, *dataset);
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
                TestReduceScatter(1, *dataset);
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
                printf("Process 2 in\n");
                fflush(stdout);
                // Process 2 (available when numDevices > 2)
                TestReduceScatter(2, *dataset);
                exit(0);
            }
            else if (pid2 == 0 && pid3 == 0 && numDevices == 4)
            {
                printf("Process 3 in\n");
                fflush(stdout);
                // Process 3 (available when numDevices == 4)
                TestReduceScatter(3, *dataset);
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

    INSTANTIATE_TEST_CASE_P(ReduceScatterMultiProcessCorrectnessSweep,
                            ReduceScatterMultiProcessCorrectnessTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum, ncclProd, ncclMax, ncclMin),
                                // Data types
                                testing::Values(ncclInt8,
                                                ncclUint8,
                                                ncclInt32,
                                                ncclUint32,
                                                ncclInt64,
                                                ncclUint64,
                                                //ncclFloat16,
                                                ncclFloat32,
                                                ncclFloat64,
                                                ncclBfloat16),
                                // Number of elements
                                testing::Values(3072, 3145728),
                                // Number of devices
                                testing::Values(2,3,4),
                                // In-place or not
                                testing::Values(false, true),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
