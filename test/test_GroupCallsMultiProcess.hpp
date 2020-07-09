/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TEST_GROUPCALLS_MULTI_PROCESS_HPP
#define TEST_GROUPCALLS_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"
#include "test_AllToAllMultiProcess.hpp"
#include "test_BroadcastMultiProcess.hpp"
#include "test_GatherMultiProcess.hpp"
#include "test_ScatterMultiProcess.hpp"

#include <string>

namespace CorrectnessTests
{
    class GroupCallsMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        void TestGroupCalls(int process, std::vector<int> const& ranks, std::vector<Dataset*>& datasets, std::vector<ncclFunc_t> const& funcs)
        {
            ncclGroupStart();
            for (int i = 0; i < ranks.size(); i++)
            {
                SetUpPerProcess(ranks[i], funcs, comms[ranks[i]], streams[ranks[i]], datasets);
            }
            ncclGroupEnd();

            if (numDevices > numDevicesAvailable) return;

            int numProcesses = numDevices / ranks.size();
            Barrier barrier(process, numProcesses, std::atoi(getenv("NCCL_COMM_ID")));

            int const root = 0;
            for (int i = 0; i < ranks.size(); i++)
            {
                AllToAllMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[0], ranks[i]);
                BroadcastMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[1], root, ranks[i]);
                GatherMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[2], root, ranks[i]);
                ScatterMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[3], root, ranks[i]);
            }
            barrier.Wait();

            ncclGroupStart();

            // AllToAll
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                ncclAllToAll(datasets[0]->inputs[rank],
                             datasets[0]->outputs[rank],
                             numElements, dataType,
                             comms[rank], streams[rank]);
            }

            // Broadcast
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                ncclBroadcast(datasets[1]->inputs[rank],
                              datasets[1]->outputs[rank],
                              numElements, dataType,
                              root, comms[rank], streams[rank]);
            }

            // Gather
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                ncclGather(datasets[2]->inputs[rank],
                           datasets[2]->outputs[rank],
                           numElements, dataType,
                           root, comms[rank], streams[rank]);
            }

            // Scatter
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                ncclScatter(datasets[3]->inputs[rank],
                            datasets[3]->outputs[rank],
                            numElements, dataType,
                            root, comms[rank], streams[rank]);
            }

            // Signal end of group call
            ncclGroupEnd();

            for (int i = 0; i < ranks.size(); i++)
            {
                HIP_CALL(hipSetDevice(ranks[i]));
                HIP_CALL(hipStreamSynchronize(streams[ranks[i]]));
            }

            for (int i = 0; i < funcs.size(); i++)
            {
                for (int j = 0; j < ranks.size(); j++)
                {
                    ValidateResults(*datasets[i], ranks[j]);
                }
                barrier.Wait();
                for (int j = 0; j < ranks.size(); j++)
                {
                    datasets[i]->Release(ranks[j]);
                }
            }

            for (int i = 0; i < ranks.size(); i++)
            {
                TearDownPerProcess(comms[ranks[i]], streams[ranks[i]]);
            }
        }
    };
}

#endif
