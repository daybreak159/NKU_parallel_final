#include "PCFG_GPU.h"
#include <cuda_runtime.h>
#include <string>
#include <thread>
#include <mutex>
#include <cstring>

using namespace std;
//nvcc main.cpp train.cpp guessing_GPU.cu md5.cpp -o GPU.exe

// 用于线程间同步的互斥锁 - 使用读写锁优化
static std::mutex gpu_mutex;

// 预计算的PT工作量缓存，避免重复计算
static std::unordered_map<int, int> pt_workload_cache;
static std::mutex cache_mutex;

// 单段口令生成 kernel - 采用guessing.cu的简洁设计
__global__ void kernel_single(const char *d_values, char *d_out, int num, int maxlen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        const char *src = d_values + idx * maxlen;
        char *dst = d_out + idx * maxlen;
        int i = 0;
        for (; i < maxlen - 1 && src[i] != '\0'; ++i) dst[i] = src[i];
        dst[i] = '\0';
    }
}

// 多段口令生成 kernel - 采用guessing.cu的简洁设计
__global__ void kernel_multi(const char *prefix, int prefix_len, const char *d_values, char *d_out, int num, int maxlen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        char *dst = d_out + idx * maxlen;
        int i = 0;
        for (; i < prefix_len && i < maxlen - 1; ++i) dst[i] = prefix[i];
        const char *src = d_values + idx * maxlen;
        int j = 0;
        for (; i < maxlen - 1 && src[j] != '\0'; ++i, ++j) dst[i] = src[j];
        dst[i] = '\0';
    }
}

// GPU并行单段生成 - 完全采用guessing.cu的实现
void gpu_generate_single(const vector<string> &values, vector<string> &guesses) {
    int num = values.size();
    if (num == 0) return;
    int maxlen = 0;
    for (auto &s : values) maxlen = max(maxlen, (int)s.size());
    maxlen += 1;
    vector<char> h_values(num * maxlen, 0);
    for (int i = 0; i < num; ++i)
        strncpy(&h_values[i * maxlen], values[i].c_str(), maxlen);

    char *d_values, *d_out;
    cudaMalloc(&d_values, num * maxlen);
    cudaMalloc(&d_out, num * maxlen);
    cudaMemcpy(d_values, h_values.data(), num * maxlen, cudaMemcpyHostToDevice);

    int block = 256, grid = (num + block - 1) / block;
    kernel_single<<<grid, block>>>(d_values, d_out, num, maxlen);
    cudaDeviceSynchronize();

    vector<char> h_out(num * maxlen, 0);
    cudaMemcpy(h_out.data(), d_out, num * maxlen, cudaMemcpyDeviceToHost);

    guesses.clear();
    for (int i = 0; i < num; ++i)
        guesses.emplace_back(&h_out[i * maxlen]);

    cudaFree(d_values);
    cudaFree(d_out);
}

// GPU并行多段生成 - 完全采用guessing.cu的实现
void gpu_generate_multi(const string &prefix, const vector<string> &values, vector<string> &guesses) {
    int num = values.size();
    if (num == 0) return;
    int maxlen = prefix.size();
    for (auto &s : values) maxlen = max(maxlen, (int)(prefix.size() + s.size()));
    maxlen += 1;
    vector<char> h_values(num * maxlen, 0);
    for (int i = 0; i < num; ++i)
        strncpy(&h_values[i * maxlen], values[i].c_str(), maxlen);

    char *d_values, *d_out, *d_prefix;
    cudaMalloc(&d_values, num * maxlen);
    cudaMalloc(&d_out, num * maxlen);
    cudaMalloc(&d_prefix, maxlen);
    cudaMemcpy(d_values, h_values.data(), num * maxlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix.c_str(), prefix.size(), cudaMemcpyHostToDevice);

    int block = 256, grid = (num + block - 1) / block;
    kernel_multi<<<grid, block>>>(d_prefix, prefix.size(), d_values, d_out, num, maxlen);
    cudaDeviceSynchronize();

    vector<char> h_out(num * maxlen, 0);
    cudaMemcpy(h_out.data(), d_out, num * maxlen, cudaMemcpyDeviceToHost);

    guesses.clear();
    for (int i = 0; i < num; ++i)
        guesses.emplace_back(&h_out[i * maxlen]);

    cudaFree(d_values);
    cudaFree(d_out);
    cudaFree(d_prefix);
}

// 添加缺失的函数实现

void PriorityQueue::CalProb(PT &pt)
{
    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                PT new_pt = *this;
                new_pt.pivot = i;
                res.emplace_back(new_pt);
                curr_indices[i] -= 1;
                break;
            }
            else
            {
                curr_indices[i] = 0;
                if (i == curr_indices.size() - 2)
                {
                    pivot = init_pivot;
                    return res;
                }
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 优化：缓存工作量计算，避免重复查找
int getLastSegmentWorkload(PriorityQueue* queue, PT& pt) {
    // 简单的缓存策略：使用PT内容的哈希作为缓存键
    int cache_key = pt.content.size() * 1000 + pt.content.back().type * 100 + pt.content.back().length;
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = pt_workload_cache.find(cache_key);
        if (it != pt_workload_cache.end()) {
            return it->second;
        }
    }
    
    // 计算工作量
    int lastSegmentIndex = pt.content.size() - 1;
    int valueCount = 0;
    
    if (lastSegmentIndex >= 0) {
        segment* lastSeg;
        if (pt.content[lastSegmentIndex].type == 1) {
            lastSeg = &queue->m.letters[queue->m.FindLetter(pt.content[lastSegmentIndex])];
        } else if (pt.content[lastSegmentIndex].type == 2) {
            lastSeg = &queue->m.digits[queue->m.FindDigit(pt.content[lastSegmentIndex])];
        } else {
            lastSeg = &queue->m.symbols[queue->m.FindSymbol(pt.content[lastSegmentIndex])];
        }
        valueCount = lastSeg->ordered_values.size();
    }
    
    // 缓存结果
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        pt_workload_cache[cache_key] = valueCount;
    }
    
    return valueCount;
}

// 优化：降低GPU阈值，并加入简单的自适应机制
bool PriorityQueueExtension::isGPUSuitable(PriorityQueue* queue, PT& pt) {
    int valueCount = getLastSegmentWorkload(queue, pt);
    
    // 动态阈值：根据GPU利用率调整
    static int gpu_threshold = 500;  // 初始阈值降低
    static int call_count = 0;
    static int gpu_calls = 0;
    
    call_count++;
    if (valueCount > gpu_threshold) {
        gpu_calls++;
    }
    
    // 每1000次调用调整一次阈值
    if (call_count % 1000 == 0) {
        double gpu_ratio = (double)gpu_calls / call_count;
        if (gpu_ratio < 0.1) {  // GPU使用率太低，降低阈值
            gpu_threshold = max(200, gpu_threshold - 100);
        } else if (gpu_ratio > 0.5) {  // GPU使用率太高，提高阈值
            gpu_threshold = min(2000, gpu_threshold + 100);
        }
    }
    
    return valueCount > gpu_threshold;
}

// 优化：减少锁争用，使用批量插入
void PriorityQueueExtension::GenerateCPU(PriorityQueue* queue, PT pt) {
    // 计算PT的概率
    queue->CalProb(pt);

    // 对于只有一个segment的PT
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) {
            a = &queue->m.letters[queue->m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &queue->m.digits[queue->m.FindDigit(pt.content[0])];
        } else {
            a = &queue->m.symbols[queue->m.FindSymbol(pt.content[0])];
        }
        
        // 关键：完全按照guessing.cu的思路，直接使用所有ordered_values
        const vector<string>& values = a->ordered_values;  // 使用引用避免拷贝
        
        // 优化：批量插入，减少锁争用
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            queue->guesses.reserve(queue->guesses.size() + values.size());  // 预留空间
            queue->guesses.insert(queue->guesses.end(), values.begin(), values.end());
            queue->total_guesses += values.size();
        }
    } else {
        string guess;
        int seg_idx = 0;
        // 构建前缀
        for (size_t k = 0; k < pt.curr_indices.size() && seg_idx < (int)pt.content.size() - 1; k++) {
            int idx = pt.curr_indices[k];
            if (pt.content[seg_idx].type == 1) {
                guess += queue->m.letters[queue->m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += queue->m.digits[queue->m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += queue->m.symbols[queue->m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
        }

        // 获取最后一个segment
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &queue->m.letters[queue->m.FindLetter(pt.content[pt.content.size() - 1])];
        } else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &queue->m.digits[queue->m.FindDigit(pt.content[pt.content.size() - 1])];
        } else {
            a = &queue->m.symbols[queue->m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // 关键：完全按照guessing.cu的思路，直接使用所有ordered_values
        const vector<string>& values = a->ordered_values;  // 使用引用避免拷贝
        vector<string> localGuesses;
        localGuesses.reserve(values.size());  // 预留空间
        
        // 生成所有组合
        for (const string& value : values) {
            localGuesses.emplace_back(guess + value);
        }
        
        // 优化：批量插入
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            queue->guesses.reserve(queue->guesses.size() + localGuesses.size());  // 预留空间
            queue->guesses.insert(queue->guesses.end(), localGuesses.begin(), localGuesses.end());
            queue->total_guesses += localGuesses.size();
        }
    }
}

// 优化：GPU处理也使用类似优化
void PriorityQueueExtension::GenerateGPU(PriorityQueue* queue, PT pt) {
    // 计算PT的概率
    queue->CalProb(pt);
    
    // 对于只有一个segment的PT
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) {
            a = &queue->m.letters[queue->m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &queue->m.digits[queue->m.FindDigit(pt.content[0])];
        } else {
            a = &queue->m.symbols[queue->m.FindSymbol(pt.content[0])];
        }
        
        // 关键：完全按照guessing.cu的思路，直接使用所有ordered_values
        const vector<string>& values = a->ordered_values;  // 使用引用避免拷贝
        vector<string> local_guesses;
        gpu_generate_single(values, local_guesses);
        
        // 优化：批量插入
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            queue->guesses.reserve(queue->guesses.size() + local_guesses.size());  // 预留空间
            queue->guesses.insert(queue->guesses.end(), local_guesses.begin(), local_guesses.end());
            queue->total_guesses += local_guesses.size();
        }
    } else {
        string guess;
        int seg_idx = 0;
        // 构建前缀
        for (size_t k = 0; k < pt.curr_indices.size() && seg_idx < (int)pt.content.size() - 1; k++) {
            int idx = pt.curr_indices[k];
            if (pt.content[seg_idx].type == 1) {
                guess += queue->m.letters[queue->m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += queue->m.digits[queue->m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += queue->m.symbols[queue->m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
        }

        // 获取最后一个segment
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &queue->m.letters[queue->m.FindLetter(pt.content[pt.content.size() - 1])];
        } else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &queue->m.digits[queue->m.FindDigit(pt.content[pt.content.size() - 1])];
        } else {
            a = &queue->m.symbols[queue->m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // 关键：完全按照guessing.cu的思路，直接使用所有ordered_values
        const vector<string>& values = a->ordered_values;  // 使用引用避免拷贝
        vector<string> local_guesses;
        gpu_generate_multi(guess, values, local_guesses);
        
        // 优化：批量插入
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            queue->guesses.reserve(queue->guesses.size() + local_guesses.size());  // 预留空间
            queue->guesses.insert(queue->guesses.end(), local_guesses.begin(), local_guesses.end());
            queue->total_guesses += local_guesses.size();
        }
    }
}

// 混合方式生成口令（GPU+CPU协同工作）
void PriorityQueueExtension::GenerateHybrid(PriorityQueue* queue, PT pt) {
    // 根据工作量决定使用GPU还是CPU
    if (isGPUSuitable(queue, pt)) {
        // 使用GPU处理
        GenerateGPU(queue, pt);
    } else {
        // 使用CPU处理
        GenerateCPU(queue, pt);
    }
}

// 进阶功能：批量处理优化
void PriorityQueueExtension::GenerateBatch(PriorityQueue* queue, vector<PT>& pts, int batchSize) {
    // 预分类和预排序优化
    vector<PT> gpuPTs, cpuPTs;
    gpuPTs.reserve(pts.size() / 2);  // 预留空间
    cpuPTs.reserve(pts.size() / 2);
    
    for (PT& pt : pts) {  // 使用引用避免拷贝
        if (isGPUSuitable(queue, pt)) {
            gpuPTs.push_back(std::move(pt));  // 使用移动语义
        } else {
            cpuPTs.push_back(std::move(pt));
        }
    }
    
    // 优化：并行处理GPU和CPU任务
    std::thread gpu_thread([&]() {
        for (PT& pt : gpuPTs) {
            GenerateGPU(queue, pt);
        }
    });
    
    // CPU任务在主线程处理
    for (PT& pt : cpuPTs) {
        GenerateCPU(queue, pt);
    }
    
    gpu_thread.join();
}

// 原始的Generate函数，保留向后兼容性
void PriorityQueue::Generate(PT pt) {
    // 使用混合模式
    PriorityQueueExtension::GenerateHybrid(this, pt);
}

// 扩展原有的PopNext函数 - 使用简化版本
void PriorityQueue::PopNext() {
    // 恢复原始逻辑：完全处理单个PT
    PriorityQueueExtension::GenerateHybrid(this, priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (size_t i = 0; i < new_pts.size(); i++) {
        PT pt = new_pts[i];
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (iter->prob < pt.prob && (iter - 1)->prob >= pt.prob) {
                    priority.insert(iter, pt);
                    inserted = true;
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                if (iter->prob < pt.prob) {
                    priority.insert(iter, pt);
                } else {
                    priority.emplace_back(pt);
                }
                inserted = true;
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.insert(iter, pt);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            priority.emplace_back(pt);
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}