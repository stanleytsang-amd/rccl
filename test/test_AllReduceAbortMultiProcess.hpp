/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLREDUCE_MULTI_PROCESS_HPP
#define TEST_ALLREDUCE_MULTI_PROCESS_HPP

#include "../include/comm.h"

#define NUM_ITER 8
#define FAKE_OP_COUNT NUM_ITER+1

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    #define HIPCHECK(cmd)                                                          \
    do {                                                                           \
      hipError_t error = (cmd);                                                    \
      if (error != hipSuccess) {                                                   \
        std::cerr << "Encountered HIP error (" << error << ") at line "            \
                  << __LINE__ << " in file " << __FILE__ << "\n";                  \
        exit(-1);                                                                  \
      }                                                                            \
    } while (0)

    #define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
    #define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

    class AllReduceAbortMultiProcessTest : public MultiProcessCorrectnessTest
    {
    public:
        void TestAllReduceAbort(int rank, Dataset& dataset, int*& done, int& remaining, ncclResult_t& ncclAsyncErr)
        {
            SetUpPerProcess(rank, ncclCollAllReduce, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable) return;

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            FillDatasetWithPattern(dataset, rank);

            uint64_t *fake_opCount, *fake_head;
            hipStream_t stream;
            if (rank == 0)
            {
                int gpu = 0; // GPU number to trigger abort
                ncclComm_t comm = comms[gpu];

                HIPCHECK(hipSetDevice(gpu));
                HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                struct ncclChannel* channel = comm->channels;
                uint64_t **p_dev_opCount = (uint64_t **)((uint8_t*)(channel->devPeers + channel->ring.next) + offsetof(struct ncclPeer, send.conn.opCountRem));
                uint64_t **p_dev_head = (uint64_t **)((uint8_t*)(channel->devPeers + channel->ring.next) + offsetof(struct ncclPeer, send.conn.head));
                uint64_t *real_opCount, *fake_o;
                uint64_t *real_head, *fake_h;

                // get original opCount and head
                HIPCHECK(hipMemcpy(&real_opCount, p_dev_opCount, sizeof(uint64_t*), hipMemcpyDefault));
                HIPCHECK(hipMemcpy(&real_head, p_dev_head, sizeof(uint64_t*), hipMemcpyDefault));
                // allocate and install fakes
                HIPCHECK(hipHostMalloc(&fake_opCount, sizeof(uint64_t*), hipHostMallocMapped));
                HIPCHECK(hipMemcpy(p_dev_opCount, &fake_opCount, sizeof(uint64_t*), hipMemcpyDefault));
                *fake_opCount = FAKE_OP_COUNT;
                HIPCHECK(hipHostMalloc(&fake_head, sizeof(uint64_t*), hipHostMallocMapped));
                HIPCHECK(hipMemcpy(p_dev_head, &fake_head, sizeof(uint64_t*), hipMemcpyDefault));
                *fake_head = 0;
                // read back fakes to confirm
                HIPCHECK(hipMemcpy(&fake_o, p_dev_opCount, sizeof(uint64_t*), hipMemcpyDefault));
                HIPCHECK(hipMemcpy(&fake_h, p_dev_head, sizeof(uint64_t*), hipMemcpyDefault));
            }

            ncclAllReduce(dataset.inputs[rank], dataset.outputs[rank],
                          numElements, dataType, op, comms[rank], streams[rank]);

            auto start = std::chrono::high_resolution_clock::now();
            hipError_t hipErr;
            bool timeout = false, abort_called = false;

            while (remaining) {
                int idle = 1;
                for (int i=0; i<numDevices; i++) {
                    barrier.Wait();
                    if (rank == i && !done[i])
                    {
                        hipErr = hipStreamQuery(streams[i]);
                        if (hipErr == hipSuccess) {
                            done[i] = 1;
                            remaining--;
                            //printf("Rank %d put success at %d, remaining %d\n", rank, i, remaining);
                            //for (int j = 0; j < numDevices; j++) printf("%d ", done[j]);
                            //printf("\n");
                            //fflush(stdout);
                        }
                    }
                    barrier.Wait();
                    if (done[i])
                    {
                        idle = 0;
                        continue;
                    }
     #if NCCL_MAJOR >= 2
     #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
                    auto delta = std::chrono::high_resolution_clock::now() - start;
                    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
                    if (deltaSec > 10.0 && !timeout) {
                        fprintf(stderr, "[          ] timeout condition, calling ncclCommAbort ...\n");
                        timeout = true;
                    }

                    if (i == rank) ncclCommGetAsyncError(comms[i], &ncclAsyncErr);
                    barrier.Wait();

                    if ((ncclAsyncErr != ncclSuccess || timeout) && !abort_called) {
                        // An asynchronous error happened. Stop the operation and destroy
                        // the communicator
                        fprintf(stderr, "[          ] ncclAsyncErr = %d\n", ncclAsyncErr);
                        ncclCommAbort(comms[rank]);
                        // Abort the perf test
                        abort_called = true;
                        barrier.Wait();
                        break;
                    }
    #endif
    #endif
                }
                // We might want to let other threads (including NCCL threads) use the CPU.
                if (idle) pthread_yield();
            }

            if (rank == 0)
            {
                HIPCHECK(hipHostFree(fake_opCount));
                HIPCHECK(hipHostFree(fake_head));
                HIPCHECK(hipStreamDestroy(stream));
            }
            dataset.Release(rank);
        }
    };
}

#endif
