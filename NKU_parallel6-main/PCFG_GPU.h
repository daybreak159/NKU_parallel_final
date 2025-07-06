#ifndef PCFG_GPU_H
#define PCFG_GPU_H

#include <cuda_runtime.h>
#include "PCFG.h"
#include <vector>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 批量处理的大小
#define BATCH_SIZE 128

// GPU上的Segment数据结构
struct GPUSegment {
    char** values;   // 指向所有可能值的指针数组
    int count;       // 值的数量
};

// GPU上的PT数据结构
struct GPUPTData {
    char* prefix;           // 前缀（所有已确定的segment）
    GPUSegment lastSegment; // 最后一个segment（需要并行处理）
};

// 为PriorityQueue类声明额外的GPU相关函数
// 注意：PriorityQueue类已在PCFG.h中定义，这里只是声明新增的方法
namespace PriorityQueueExtension {
    // 新增GPU相关功能声明
    void GenerateCPU(PriorityQueue* queue, PT pt);
    void GenerateGPU(PriorityQueue* queue, PT pt);
    void GenerateHybrid(PriorityQueue* queue, PT pt);
    void GenerateBatch(PriorityQueue* queue, vector<PT>& pts, int batchSize);
    bool isGPUSuitable(PriorityQueue* queue, PT& pt);
}

// CUDA核函数声明
__global__ void generatePasswordsKernel(const GPUPTData ptData, char** results, int maxResultSize);

#endif // PCFG_GPU_H