#include "md5.h"
#include <iostream>
#include <cstring>
#include <arm_neon.h>

#include <iomanip>
#include <cassert>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace chrono;

// --- 使用 md5_provided.cpp 中的串行 StringProcess ---
Byte* StringProcess(string input, int* n_byte) {
    // 将输入的字符串转换为Byte为单位的数组
    Byte* blocks = (Byte*)input.c_str();
    int length = input.length();

    // 计算原始消息长度（以比特为单位）
    int bitLength = length * 8;

    // paddingBits: 原始消息需要的padding长度（以bit为单位）
    int paddingBits = bitLength % 512;
    if (paddingBits > 448) {
        paddingBits = 512 - (paddingBits - 448);
    } else if (paddingBits < 448) {
        paddingBits = 448 - paddingBits;
    } else if (paddingBits == 448) {
        paddingBits = 512;
    }

    // 原始消息需要的padding长度（以Byte为单位）
    int paddingBytes = paddingBits / 8;
    // 创建最终的字节数组
    int paddedLength = length + paddingBytes + 8;
    Byte* paddedMessage = nullptr; // 初始化为 nullptr
    try {
        paddedMessage = new Byte[paddedLength];
    } catch (const std::bad_alloc& e) {
        cerr << "Memory allocation failed in StringProcess: " << e.what() << endl;
        *n_byte = 0; // 表示失败
        return nullptr; // 返回 nullptr 表示失败
    }

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);

    // 添加消息长度
    for (int i = 0; i < 8; ++i) {
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    *n_byte = paddedLength;
    return paddedMessage;
}

// --- 修改并行字符串预处理函数以使用预分配缓冲区和 memcpy 优化 ---
void StringProcess_Parallel(const std::vector<std::string>& inputs, size_t input_count, Byte** paddedMessagePointers, int* messageLengths, size_t alignment) {
    // 检查输入大小是否匹配
    if (inputs.size() != input_count) {
        throw std::runtime_error("Input vector size does not match input_count in StringProcess_Parallel");
    }

    // #pragma omp parallel for if(input_count > 1000) // 注释掉 OpenMP 并行 for 指令
    for (size_t i = 0; i < input_count; ++i) {
        // 直接从 vector 访问输入字符串
        const string& input = inputs[i];
        const int length = input.length();
        const int bitLength = length * 8;

        int paddingBits = bitLength % 512;
        if (paddingBits >= 448) {
            paddingBits = 512 - paddingBits + 448;
        } else {
            paddingBits = 448 - paddingBits;
        }
        if (paddingBits == 0) paddingBits = 512;

        const int paddingBytes = paddingBits / 8;
        const int paddedLength = length + paddingBytes + 8;
        messageLengths[i] = paddedLength; // 存储计算出的长度

        // 获取预分配的指针
        Byte* paddedMessage = paddedMessagePointers[i];

        // 检查指针是否有效 (由调用者保证)
        if (paddedMessage == nullptr) {
            cerr << "Error: Pre-allocated buffer pointer is null for input " << i << endl;
            messageLengths[i] = 0; // 标记失败
            continue; // 继续处理下一个输入
        }

        // 直接使用 input.c_str() 和预分配的缓冲区
        memcpy(paddedMessage, input.c_str(), length);
        paddedMessage[length] = 0x80;

        // --- 使用 memcpy 优化零填充 ---
        size_t zero_fill_size = paddingBytes - 1;
        if (zero_fill_size > 0) {
            if (zero_fill_size <= ZERO_BUFFER_SIZE) {
                // 如果填充量不大，使用 memcpy 从预定义缓冲区复制
                memcpy(paddedMessage + length + 1, zero_padding_buffer, zero_fill_size);
            } else {
                // 如果填充量较大，仍然使用 memset
                memset(paddedMessage + length + 1, 0, zero_fill_size);
            }
        }
        // --- 结束 memcpy 优化 ---

        uint64_t len64 = static_cast<uint64_t>(length) * 8;
        for (int j = 0; j < 8; ++j) {
            paddedMessage[length + paddingBytes + j] = (len64 >> (j * 8)) & 0xFF;
        }
    }
}

// --- 使用 md5_provided.cpp 中的串行 MD5Hash ---
void MD5Hash(string input, bit32* state) {
    Byte* paddedMessage = nullptr;
    int messageLength = 0; // 使用单个 int

    // 调用新的 StringProcess
    paddedMessage = StringProcess(input, &messageLength);

    // 检查 StringProcess 是否成功分配内存
    if (paddedMessage == nullptr || messageLength == 0) {
        cerr << "StringProcess failed for input: " << input << endl;
        state[0] = state[1] = state[2] = state[3] = 0xFFFFFFFF; // 例如，设置为全 F
        return;
    }

    int n_blocks = messageLength / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    for (int i = 0; i < n_blocks; i += 1) {
        bit32 x[16];
        const Byte* current_block = paddedMessage + i * 64; // 直接计算块指针

        for (int i1 = 0; i1 < 16; ++i1) {
            x[i1] = (current_block[4 * i1]) |
                    (current_block[4 * i1 + 1] << 8) |
                    (current_block[4 * i1 + 2] << 16) |
                    (current_block[4 * i1 + 3] << 24);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        /* Round 1 */
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x02441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x04881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    for (int i = 0; i < 4; i++) {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |
                   ((value & 0xff00) << 8) |
                   ((value & 0xff0000) >> 8) |
                   ((value & 0xff000000) >> 24);
    }

    delete[] paddedMessage; // 释放 StringProcess 分配的内存
}

// --- 修改 SIMD 批量哈希函数以移除预取和 M_static，简化加载 ---
void MD5Hash_SIMD_Batch(const Byte** paddedMessages, const int* messageLengths, size_t input_count, bit32** states) {
    constexpr size_t SIMD_LANES = 4;
    // --- 临时加载缓冲区，仍然需要 ---
    alignas(16) uint32_t load_buffer[SIMD_LANES];

    for (size_t batch_start = 0; batch_start < input_count; batch_start += SIMD_LANES) {
        size_t current_batch_size = std::min(SIMD_LANES, input_count - batch_start);

        int max_blocks = 0;
        for (size_t i = 0; i < current_batch_size; ++i) {
            if (paddedMessages[batch_start + i] != nullptr) {
                max_blocks = std::max(max_blocks, messageLengths[batch_start + i] / 64);
            }
        }

        if (max_blocks == 0 && current_batch_size > 0) continue;

        uint32x4_t state_a = vdupq_n_u32(0x67452301);
        uint32x4_t state_b = vdupq_n_u32(0xefcdab89);
        uint32x4_t state_c = vdupq_n_u32(0x98badcfe);
        uint32x4_t state_d = vdupq_n_u32(0x10325476);

        for (int block_idx = 0; block_idx < max_blocks; ++block_idx) {
            // --- 在循环内部直接加载数据 ---
            // 定义临时向量存储当前块的16个消息字
            uint32x4_t x[16];
            for (int j = 0; j < 16; ++j) {
                memset(load_buffer, 0, sizeof(load_buffer));

                for (size_t lane = 0; lane < current_batch_size; ++lane) {
                    size_t global_idx = batch_start + lane;
                    if (paddedMessages[global_idx] != nullptr && block_idx < (messageLengths[global_idx] / 64)) {
                        const Byte* block_ptr = paddedMessages[global_idx] + block_idx * 64;
                        // 直接从内存加载，仍然需要 reinterpret_cast 来访问 uint32_t
                        load_buffer[lane] = *(reinterpret_cast<const uint32_t*>(block_ptr + j * 4));
                    }
                }
                // 加载到临时向量 x[j]
                x[j] = vld1q_u32(load_buffer);
            }
            // --- 结束数据加载 ---

            uint32x4_t aa = state_a;
            uint32x4_t bb = state_b;
            uint32x4_t cc = state_c;
            uint32x4_t dd = state_d;

            // --- 使用临时向量 x[j] 进行计算 ---
            FF_NEON(aa, bb, cc, dd, x[0], s11, 0xd76aa478);
            FF_NEON(dd, aa, bb, cc, x[1], s12, 0xe8c7b756);
            FF_NEON(cc, dd, aa, bb, x[2], s13, 0x242070db);
            FF_NEON(bb, cc, dd, aa, x[3], s14, 0xc1bdceee);
            FF_NEON(aa, bb, cc, dd, x[4], s11, 0xf57c0faf);
            FF_NEON(dd, aa, bb, cc, x[5], s12, 0x4787c62a);
            FF_NEON(cc, dd, aa, bb, x[6], s13, 0xa8304613);
            FF_NEON(bb, cc, dd, aa, x[7], s14, 0xfd469501);
            FF_NEON(aa, bb, cc, dd, x[8], s11, 0x698098d8);
            FF_NEON(dd, aa, bb, cc, x[9], s12, 0x8b44f7af);
            FF_NEON(cc, dd, aa, bb, x[10], s13, 0xffff5bb1);
            FF_NEON(bb, cc, dd, aa, x[11], s14, 0x895cd7be);
            FF_NEON(aa, bb, cc, dd, x[12], s11, 0x6b901122);
            FF_NEON(dd, aa, bb, cc, x[13], s12, 0xfd987193);
            FF_NEON(cc, dd, aa, bb, x[14], s13, 0xa679438e);
            FF_NEON(bb, cc, dd, aa, x[15], s14, 0x49b40821);

            GG_NEON(aa, bb, cc, dd, x[1], s21, 0xf61e2562);
            GG_NEON(dd, aa, bb, cc, x[6], s22, 0xc040b340);
            GG_NEON(cc, dd, aa, bb, x[11], s23, 0x265e5a51);
            GG_NEON(bb, cc, dd, aa, x[0], s24, 0xe9b6c7aa);
            GG_NEON(aa, bb, cc, dd, x[5], s21, 0xd62f105d);
            GG_NEON(dd, aa, bb, cc, x[10], s22, 0x02441453);
            GG_NEON(cc, dd, aa, bb, x[15], s23, 0xd8a1e681);
            GG_NEON(bb, cc, dd, aa, x[4], s24, 0xe7d3fbc8);
            GG_NEON(aa, bb, cc, dd, x[9], s21, 0x21e1cde6);
            GG_NEON(dd, aa, bb, cc, x[14], s22, 0xc33707d6);
            GG_NEON(cc, dd, aa, bb, x[3], s23, 0xf4d50d87);
            GG_NEON(bb, cc, dd, aa, x[8], s24, 0x455a14ed);
            GG_NEON(aa, bb, cc, dd, x[13], s21, 0xa9e3e905);
            GG_NEON(dd, aa, bb, cc, x[2], s22, 0xfcefa3f8);
            GG_NEON(cc, dd, aa, bb, x[7], s23, 0x676f02d9);
            GG_NEON(bb, cc, dd, aa, x[12], s24, 0x8d2a4c8a);

            HH_NEON(aa, bb, cc, dd, x[5], s31, 0xfffa3942);
            HH_NEON(dd, aa, bb, cc, x[8], s32, 0x8771f681);
            HH_NEON(cc, dd, aa, bb, x[11], s33, 0x6d9d6122);
            HH_NEON(bb, cc, dd, aa, x[14], s34, 0xfde5380c);
            HH_NEON(aa, bb, cc, dd, x[1], s31, 0xa4beea44);
            HH_NEON(dd, aa, bb, cc, x[4], s32, 0x4bdecfa9);
            HH_NEON(cc, dd, aa, bb, x[7], s33, 0xf6bb4b60);
            HH_NEON(bb, cc, dd, aa, x[10], s34, 0xbebfbc70);
            HH_NEON(aa, bb, cc, dd, x[13], s31, 0x289b7ec6);
            HH_NEON(dd, aa, bb, cc, x[0], s32, 0xeaa127fa);
            HH_NEON(cc, dd, aa, bb, x[3], s33, 0xd4ef3085);
            HH_NEON(bb, cc, dd, aa, x[6], s34, 0x04881d05);
            HH_NEON(aa, bb, cc, dd, x[9], s31, 0xd9d4d039);
            HH_NEON(dd, aa, bb, cc, x[12], s32, 0xe6db99e5);
            HH_NEON(cc, dd, aa, bb, x[15], s33, 0x1fa27cf8);
            HH_NEON(bb, cc, dd, aa, x[2], s34, 0xc4ac5665);

            II_NEON(aa, bb, cc, dd, x[0], s41, 0xf4292244);
            II_NEON(dd, aa, bb, cc, x[7], s42, 0x432aff97);
            II_NEON(cc, dd, aa, bb, x[14], s43, 0xab9423a7);
            II_NEON(bb, cc, dd, aa, x[5], s44, 0xfc93a039);
            II_NEON(aa, bb, cc, dd, x[12], s41, 0x655b59c3);
            II_NEON(dd, aa, bb, cc, x[3], s42, 0x8f0ccc92);
            II_NEON(cc, dd, aa, bb, x[10], s43, 0xffeff47d);
            II_NEON(bb, cc, dd, aa, x[1], s44, 0x85845dd1);
            II_NEON(aa, bb, cc, dd, x[8], s41, 0x6fa87e4f);
            II_NEON(dd, aa, bb, cc, x[15], s42, 0xfe2ce6e0);
            II_NEON(cc, dd, aa, bb, x[6], s43, 0xa3014314);
            II_NEON(bb, cc, dd, aa, x[13], s44, 0x4e0811a1);
            II_NEON(aa, bb, cc, dd, x[4], s41, 0xf7537e82);
            II_NEON(dd, aa, bb, cc, x[11], s42, 0xbd3af235);
            II_NEON(cc, dd, aa, bb, x[2], s43, 0x2ad7d2bb);
            II_NEON(bb, cc, dd, aa, x[9], s44, 0xeb86d391);
            // --- 结束使用 x[j] ---

            state_a = vaddq_u32(state_a, aa);
            state_b = vaddq_u32(state_b, bb);
            state_c = vaddq_u32(state_c, cc);
            state_d = vaddq_u32(state_d, dd);
        }

        // --- 存储结果和字节序转换保持不变 ---
        alignas(16) uint32_t final_states_a[SIMD_LANES];
        alignas(16) uint32_t final_states_b[SIMD_LANES];
        alignas(16) uint32_t final_states_c[SIMD_LANES];
        alignas(16) uint32_t final_states_d[SIMD_LANES];

        vst1q_u32(final_states_a, state_a);
        vst1q_u32(final_states_b, state_b);
        vst1q_u32(final_states_c, state_c);
        vst1q_u32(final_states_d, state_d);

        for (size_t lane = 0; lane < current_batch_size; ++lane) {
            size_t global_idx = batch_start + lane;
            if (states[global_idx] != nullptr) {
                states[global_idx][0] = __builtin_bswap32(final_states_a[lane]);
                states[global_idx][1] = __builtin_bswap32(final_states_b[lane]);
                states[global_idx][2] = __builtin_bswap32(final_states_c[lane]);
                states[global_idx][3] = __builtin_bswap32(final_states_d[lane]);
            }
        }
    }
}