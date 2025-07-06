/**
 * md5_sse.cpp
 * 
 * 使用SSE指令集的高性能MD5哈希实现
 * 结合了内存池技术和高效SIMD操作
 */

 #pragma GCC optimize(2)

 #include <iostream>
 #include <vector>
 #include <string>
 #include <cstring>
 #include <cstdint>
 #include <ctime>
 #include <chrono>
 #include <algorithm>
 #include <immintrin.h> // SSE指令集
 #include <random>
 #include <iomanip>
 
 // 类型定义
// 类型定义
using u32 = uint32_t;
using u8 = uint8_t;
using u64 = uint64_t;  // 添加 u64 类型定义
 
 // MD5算法常量
 #define s11 7
 #define s12 12
 #define s13 17
 #define s14 22
 #define s21 5
 #define s22 9
 #define s23 14
 #define s24 20
 #define s31 4
 #define s32 11
 #define s33 16
 #define s34 23
 #define s41 6
 #define s42 10
 #define s43 15
 #define s44 21
 
 // 内存池配置
 #define MAX_BATCH_SIZE 10000
 #define MAX_MESSAGE_LENGTH 128
 #define ALIGNMENT 16  // SSE要求16字节对齐
 
 // 内联SSE函数实现
 inline __m128i simd_F(__m128i x, __m128i y, __m128i z) {
     return _mm_or_si128(_mm_and_si128(x, y), _mm_and_si128(_mm_xor_si128(x, _mm_set1_epi32(0xFFFFFFFF)), z));
 }
 
 inline __m128i simd_G(__m128i x, __m128i y, __m128i z) {
     return _mm_or_si128(_mm_and_si128(x, z), _mm_and_si128(y, _mm_xor_si128(z, _mm_set1_epi32(0xFFFFFFFF))));
 }
 
 inline __m128i simd_H(__m128i x, __m128i y, __m128i z) {
     return _mm_xor_si128(_mm_xor_si128(x, y), z);
 }
 
 inline __m128i simd_I(__m128i x, __m128i y, __m128i z) {
     return _mm_xor_si128(y, _mm_or_si128(x, _mm_xor_si128(z, _mm_set1_epi32(0xFFFFFFFF))));
 }
 
 inline __m128i simd_ROTATELEFT(__m128i val, int shift) {
     return _mm_or_si128(_mm_slli_epi32(val, shift), _mm_srli_epi32(val, 32 - shift));
 }
 
 // 高效实现的MD5变换步骤
 #define STEP_F(a, b, c, d, x, s, t) \
     (a) = _mm_add_epi32(b, simd_ROTATELEFT(_mm_add_epi32(_mm_add_epi32(_mm_add_epi32(a, simd_F(b,c,d)), x), _mm_set1_epi32(t)), s))
 
 #define STEP_G(a, b, c, d, x, s, t) \
     (a) = _mm_add_epi32(b, simd_ROTATELEFT(_mm_add_epi32(_mm_add_epi32(_mm_add_epi32(a, simd_G(b,c,d)), x), _mm_set1_epi32(t)), s))
 
 #define STEP_H(a, b, c, d, x, s, t) \
     (a) = _mm_add_epi32(b, simd_ROTATELEFT(_mm_add_epi32(_mm_add_epi32(_mm_add_epi32(a, simd_H(b,c,d)), x), _mm_set1_epi32(t)), s))
 
 #define STEP_I(a, b, c, d, x, s, t) \
     (a) = _mm_add_epi32(b, simd_ROTATELEFT(_mm_add_epi32(_mm_add_epi32(_mm_add_epi32(a, simd_I(b,c,d)), x), _mm_set1_epi32(t)), s))
 
 // 串行标准MD5实现
 void md5_serial(const std::string& input, u32* digest) {
     // 初始状态
     u32 a = 0x67452301;
     u32 b = 0xEFCDAB89;
     u32 c = 0x98BADCFE;
     u32 d = 0x10325476;
     
     // 计算需要的字节数
     u32 originalLength = input.length();
     u32 zeroPadding = (56 - (originalLength % 64)) % 64;
     u32 totalLength = originalLength + zeroPadding + 8;
     
     // 准备处理后的数据
     u8* processedData = new u8[totalLength];
     memcpy(processedData, input.c_str(), originalLength);
     
     // 添加1位
     processedData[originalLength] = 0x80;
     
     // 添加0填充
     memset(processedData + originalLength + 1, 0, zeroPadding - 1);
     
     // 添加原始长度（以位为单位）
     u64 bitLength = originalLength * 8;
     memcpy(processedData + originalLength + zeroPadding, &bitLength, 8);
     
     // 处理每个64字节块
     for (u32 offset = 0; offset < totalLength; offset += 64) {
         u32 M[16];
         for (int j = 0; j < 16; j++) {
             M[j] = (processedData[offset + j*4]) |
                   (processedData[offset + j*4 + 1] << 8) |
                   (processedData[offset + j*4 + 2] << 16) |
                   (processedData[offset + j*4 + 3] << 24);
         }
         
         u32 AA = a;
         u32 BB = b;
         u32 CC = c;
         u32 DD = d;
         
         // Round 1
         #define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
         #define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
         
         #define FF(a, b, c, d, x, s, ac) { \
             (a) += F(b, c, d) + (x) + (ac); \
             (a) = ROTATE_LEFT(a, s); \
             (a) += (b); \
         }
         
         FF(a, b, c, d, M[0],  s11, 0xd76aa478);
         FF(d, a, b, c, M[1],  s12, 0xe8c7b756);
         FF(c, d, a, b, M[2],  s13, 0x242070db);
         FF(b, c, d, a, M[3],  s14, 0xc1bdceee);
         FF(a, b, c, d, M[4],  s11, 0xf57c0faf);
         FF(d, a, b, c, M[5],  s12, 0x4787c62a);
         FF(c, d, a, b, M[6],  s13, 0xa8304613);
         FF(b, c, d, a, M[7],  s14, 0xfd469501);
         FF(a, b, c, d, M[8],  s11, 0x698098d8);
         FF(d, a, b, c, M[9],  s12, 0x8b44f7af);
         FF(c, d, a, b, M[10], s13, 0xffff5bb1);
         FF(b, c, d, a, M[11], s14, 0x895cd7be);
         FF(a, b, c, d, M[12], s11, 0x6b901122);
         FF(d, a, b, c, M[13], s12, 0xfd987193);
         FF(c, d, a, b, M[14], s13, 0xa679438e);
         FF(b, c, d, a, M[15], s14, 0x49b40821);
         
         // Round 2
         #define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
         
         #define GG(a, b, c, d, x, s, ac) { \
             (a) += G(b, c, d) + (x) + (ac); \
             (a) = ROTATE_LEFT(a, s); \
             (a) += (b); \
         }
         
         GG(a, b, c, d, M[1],  s21, 0xf61e2562);
         GG(d, a, b, c, M[6],  s22, 0xc040b340);
         GG(c, d, a, b, M[11], s23, 0x265e5a51);
         GG(b, c, d, a, M[0],  s24, 0xe9b6c7aa);
         GG(a, b, c, d, M[5],  s21, 0xd62f105d);
         GG(d, a, b, c, M[10], s22, 0x02441453);
         GG(c, d, a, b, M[15], s23, 0xd8a1e681);
         GG(b, c, d, a, M[4],  s24, 0xe7d3fbc8);
         GG(a, b, c, d, M[9],  s21, 0x21e1cde6);
         GG(d, a, b, c, M[14], s22, 0xc33707d6);
         GG(c, d, a, b, M[3],  s23, 0xf4d50d87);
         GG(b, c, d, a, M[8],  s24, 0x455a14ed);
         GG(a, b, c, d, M[13], s21, 0xa9e3e905);
         GG(d, a, b, c, M[2],  s22, 0xfcefa3f8);
         GG(c, d, a, b, M[7],  s23, 0x676f02d9);
         GG(b, c, d, a, M[12], s24, 0x8d2a4c8a);
         
         // Round 3
         #define H(x, y, z) ((x) ^ (y) ^ (z))
         
         #define HH(a, b, c, d, x, s, ac) { \
             (a) += H(b, c, d) + (x) + (ac); \
             (a) = ROTATE_LEFT(a, s); \
             (a) += (b); \
         }
         
         HH(a, b, c, d, M[5],  s31, 0xfffa3942);
         HH(d, a, b, c, M[8],  s32, 0x8771f681);
         HH(c, d, a, b, M[11], s33, 0x6d9d6122);
         HH(b, c, d, a, M[14], s34, 0xfde5380c);
         HH(a, b, c, d, M[1],  s31, 0xa4beea44);
         HH(d, a, b, c, M[4],  s32, 0x4bdecfa9);
         HH(c, d, a, b, M[7],  s33, 0xf6bb4b60);
         HH(b, c, d, a, M[10], s34, 0xbebfbc70);
         HH(a, b, c, d, M[13], s31, 0x289b7ec6);
         HH(d, a, b, c, M[0],  s32, 0xeaa127fa);
         HH(c, d, a, b, M[3],  s33, 0xd4ef3085);
         HH(b, c, d, a, M[6],  s34, 0x04881d05);
         HH(a, b, c, d, M[9],  s31, 0xd9d4d039);
         HH(d, a, b, c, M[12], s32, 0xe6db99e5);
         HH(c, d, a, b, M[15], s33, 0x1fa27cf8);
         HH(b, c, d, a, M[2],  s34, 0xc4ac5665);
         
         // Round 4
         #define I(x, y, z) ((y) ^ ((x) | (~z)))
         
         #define II(a, b, c, d, x, s, ac) { \
             (a) += I(b, c, d) + (x) + (ac); \
             (a) = ROTATE_LEFT(a, s); \
             (a) += (b); \
         }
         
         II(a, b, c, d, M[0],  s41, 0xf4292244);
         II(d, a, b, c, M[7],  s42, 0x432aff97);
         II(c, d, a, b, M[14], s43, 0xab9423a7);
         II(b, c, d, a, M[5],  s44, 0xfc93a039);
         II(a, b, c, d, M[12], s41, 0x655b59c3);
         II(d, a, b, c, M[3],  s42, 0x8f0ccc92);
         II(c, d, a, b, M[10], s43, 0xffeff47d);
         II(b, c, d, a, M[1],  s44, 0x85845dd1);
         II(a, b, c, d, M[8],  s41, 0x6fa87e4f);
         II(d, a, b, c, M[15], s42, 0xfe2ce6e0);
         II(c, d, a, b, M[6],  s43, 0xa3014314);
         II(b, c, d, a, M[13], s44, 0x4e0811a1);
         II(a, b, c, d, M[4],  s41, 0xf7537e82);
         II(d, a, b, c, M[11], s42, 0xbd3af235);
         II(c, d, a, b, M[2],  s43, 0x2ad7d2bb);
         II(b, c, d, a, M[9],  s44, 0xeb86d391);
         
         a += AA;
         b += BB;
         c += CC;
         d += DD;
     }
     
     // 输出结果
     digest[0] = a;
     digest[1] = b;
     digest[2] = c;
     digest[3] = d;
     
     delete[] processedData;
 }
 
 // 预处理函数 - 为批处理准备消息
 void preprocess_messages(const std::vector<std::string>& inputs, 
                          u8* message_pool, 
                          u8** message_pointers,
                          u32* message_lengths) {
     
     const size_t pool_stride = MAX_MESSAGE_LENGTH;
     
     // 并行预处理所有输入
     for (size_t i = 0; i < inputs.size(); i++) {
         u8* current_buffer = message_pool + i * pool_stride;
         const std::string& input = inputs[i];
         
         // 计算需要的字节数
         u32 originalLength = input.length();
         u32 zeroPadding = (56 - (originalLength % 64)) % 64;
         u32 totalLength = originalLength + zeroPadding + 8;
         
         if (totalLength > pool_stride) {
             std::cerr << "Warning: Message length exceeds buffer size!" << std::endl;
             totalLength = pool_stride;  // 安全截断
         }
         
         // 复制原始字符串
         memcpy(current_buffer, input.c_str(), originalLength);
         
         // 添加1位
         current_buffer[originalLength] = 0x80;
         
         // 添加0填充
         memset(current_buffer + originalLength + 1, 0, zeroPadding - 1);
         
         // 添加原始长度（以位为单位）
         u64 bitLength = originalLength * 8;
         memcpy(current_buffer + originalLength + zeroPadding, &bitLength, 8);
         
         // 设置指针和长度
         message_pointers[i] = current_buffer;
         message_lengths[i] = totalLength;
     }
 }
 
 // 使用SSE的并行批处理MD5实现 - 一次处理4个消息
 void md5_sse_batch(const std::vector<std::string>& inputs, 
                    std::vector<std::vector<u32>>& digests,
                    u8* message_pool) {
                    
     const size_t num_inputs = inputs.size();
     digests.resize(num_inputs, std::vector<u32>(4));
     
     // 准备内存池和指针数组
     u8** message_pointers = new u8*[num_inputs];
     u32* message_lengths = new u32[num_inputs];
     
     // 预处理所有输入
     preprocess_messages(inputs, message_pool, message_pointers, message_lengths);
     
     // 每次处理4条输入
     const size_t SIMD_WIDTH = 4;
     
     for (size_t batch = 0; batch < num_inputs; batch += SIMD_WIDTH) {
         size_t current_batch = std::min(SIMD_WIDTH, num_inputs - batch);
         
         // 计算每条消息处理的最大块数
         size_t max_blocks = 0;
         for (size_t i = 0; i < current_batch; ++i) {
             size_t blocks = message_lengths[batch + i] / 64;
             max_blocks = std::max(max_blocks, blocks);
         }
         
         // 初始化MD5状态
         __m128i state0 = _mm_set1_epi32(0x67452301);
         __m128i state1 = _mm_set1_epi32(0xEFCDAB89);
         __m128i state2 = _mm_set1_epi32(0x98BADCFE);
         __m128i state3 = _mm_set1_epi32(0x10325476);
         
         // 处理每个64字节块
         for (size_t block = 0; block < max_blocks; ++block) {
             __m128i a = state0;
             __m128i b = state1;
             __m128i c = state2;
             __m128i d = state3;
             
             // 准备当前块的16个32位字
             __m128i x[16];
             for (int i = 0; i < 16; i++) {
                 u32 values[4] = {0, 0, 0, 0};
                 
                 for (size_t j = 0; j < current_batch; ++j) {
                     if (block * 64 + i * 4 + 3 < message_lengths[batch + j]) {
                         const u8* ptr = message_pointers[batch + j] + block * 64;
                         values[j] = (ptr[i*4]) |
                                    (ptr[i*4 + 1] << 8) |
                                    (ptr[i*4 + 2] << 16) |
                                    (ptr[i*4 + 3] << 24);
                     }
                 }
                 
                 // 将4个值加载到一个SIMD寄存器
                 x[i] = _mm_set_epi32(values[3], values[2], values[1], values[0]);
             }
             
             // Round 1
             STEP_F(a, b, c, d, x[0],  s11, 0xd76aa478);
             STEP_F(d, a, b, c, x[1],  s12, 0xe8c7b756);
             STEP_F(c, d, a, b, x[2],  s13, 0x242070db);
             STEP_F(b, c, d, a, x[3],  s14, 0xc1bdceee);
             STEP_F(a, b, c, d, x[4],  s11, 0xf57c0faf);
             STEP_F(d, a, b, c, x[5],  s12, 0x4787c62a);
             STEP_F(c, d, a, b, x[6],  s13, 0xa8304613);
             STEP_F(b, c, d, a, x[7],  s14, 0xfd469501);
             STEP_F(a, b, c, d, x[8],  s11, 0x698098d8);
             STEP_F(d, a, b, c, x[9],  s12, 0x8b44f7af);
             STEP_F(c, d, a, b, x[10], s13, 0xffff5bb1);
             STEP_F(b, c, d, a, x[11], s14, 0x895cd7be);
             STEP_F(a, b, c, d, x[12], s11, 0x6b901122);
             STEP_F(d, a, b, c, x[13], s12, 0xfd987193);
             STEP_F(c, d, a, b, x[14], s13, 0xa679438e);
             STEP_F(b, c, d, a, x[15], s14, 0x49b40821);
             
             // Round 2
             STEP_G(a, b, c, d, x[1],  s21, 0xf61e2562);
             STEP_G(d, a, b, c, x[6],  s22, 0xc040b340);
             STEP_G(c, d, a, b, x[11], s23, 0x265e5a51);
             STEP_G(b, c, d, a, x[0],  s24, 0xe9b6c7aa);
             STEP_G(a, b, c, d, x[5],  s21, 0xd62f105d);
             STEP_G(d, a, b, c, x[10], s22, 0x02441453);
             STEP_G(c, d, a, b, x[15], s23, 0xd8a1e681);
             STEP_G(b, c, d, a, x[4],  s24, 0xe7d3fbc8);
             STEP_G(a, b, c, d, x[9],  s21, 0x21e1cde6);
             STEP_G(d, a, b, c, x[14], s22, 0xc33707d6);
             STEP_G(c, d, a, b, x[3],  s23, 0xf4d50d87);
             STEP_G(b, c, d, a, x[8],  s24, 0x455a14ed);
             STEP_G(a, b, c, d, x[13], s21, 0xa9e3e905);
             STEP_G(d, a, b, c, x[2],  s22, 0xfcefa3f8);
             STEP_G(c, d, a, b, x[7],  s23, 0x676f02d9);
             STEP_G(b, c, d, a, x[12], s24, 0x8d2a4c8a);
             
             // Round 3
             STEP_H(a, b, c, d, x[5],  s31, 0xfffa3942);
             STEP_H(d, a, b, c, x[8],  s32, 0x8771f681);
             STEP_H(c, d, a, b, x[11], s33, 0x6d9d6122);
             STEP_H(b, c, d, a, x[14], s34, 0xfde5380c);
             STEP_H(a, b, c, d, x[1],  s31, 0xa4beea44);
             STEP_H(d, a, b, c, x[4],  s32, 0x4bdecfa9);
             STEP_H(c, d, a, b, x[7],  s33, 0xf6bb4b60);
             STEP_H(b, c, d, a, x[10], s34, 0xbebfbc70);
             STEP_H(a, b, c, d, x[13], s31, 0x289b7ec6);
             STEP_H(d, a, b, c, x[0],  s32, 0xeaa127fa);
             STEP_H(c, d, a, b, x[3],  s33, 0xd4ef3085);
             STEP_H(b, c, d, a, x[6],  s34, 0x04881d05);
             STEP_H(a, b, c, d, x[9],  s31, 0xd9d4d039);
             STEP_H(d, a, b, c, x[12], s32, 0xe6db99e5);
             STEP_H(c, d, a, b, x[15], s33, 0x1fa27cf8);
             STEP_H(b, c, d, a, x[2],  s34, 0xc4ac5665);
             
             // Round 4
             STEP_I(a, b, c, d, x[0],  s41, 0xf4292244);
             STEP_I(d, a, b, c, x[7],  s42, 0x432aff97);
             STEP_I(c, d, a, b, x[14], s43, 0xab9423a7);
             STEP_I(b, c, d, a, x[5],  s44, 0xfc93a039);
             STEP_I(a, b, c, d, x[12], s41, 0x655b59c3);
             STEP_I(d, a, b, c, x[3],  s42, 0x8f0ccc92);
             STEP_I(c, d, a, b, x[10], s43, 0xffeff47d);
             STEP_I(b, c, d, a, x[1],  s44, 0x85845dd1);
             STEP_I(a, b, c, d, x[8],  s41, 0x6fa87e4f);
             STEP_I(d, a, b, c, x[15], s42, 0xfe2ce6e0);
             STEP_I(c, d, a, b, x[6],  s43, 0xa3014314);
             STEP_I(b, c, d, a, x[13], s44, 0x4e0811a1);
             STEP_I(a, b, c, d, x[4],  s41, 0xf7537e82);
             STEP_I(d, a, b, c, x[11], s42, 0xbd3af235);
             STEP_I(c, d, a, b, x[2],  s43, 0x2ad7d2bb);
             STEP_I(b, c, d, a, x[9],  s44, 0xeb86d391);
             
             // 更新状态
             state0 = _mm_add_epi32(state0, a);
             state1 = _mm_add_epi32(state1, b);
             state2 = _mm_add_epi32(state2, c);
             state3 = _mm_add_epi32(state3, d);
         }
         
         // 提取结果
         alignas(16) u32 state0_arr[4], state1_arr[4], state2_arr[4], state3_arr[4];
         
         _mm_store_si128((__m128i*)state0_arr, state0);
         _mm_store_si128((__m128i*)state1_arr, state1);
         _mm_store_si128((__m128i*)state2_arr, state2);
         _mm_store_si128((__m128i*)state3_arr, state3);
         
         // 保存结果到输出数组
         for (size_t i = 0; i < current_batch; ++i) {
             digests[batch + i][0] = state0_arr[i];
             digests[batch + i][1] = state1_arr[i];
             digests[batch + i][2] = state2_arr[i];
             digests[batch + i][3] = state3_arr[i];
         }
     }
     
     // 清理
     delete[] message_pointers;
     delete[] message_lengths;
 }
 
 // 将MD5哈希值转为十六进制字符串
 std::string md5_to_string(const u32* digest) {
     std::stringstream ss;
     for (int i = 0; i < 4; ++i) {
         ss << std::hex << std::setfill('0') << std::setw(8) << digest[i];
     }
     return ss.str();
 }
 
 // 测试MD5实现的正确性
 void verify_md5_implementation() {
     std::cout << "Testing SSE MD5 implementation correctness:" << std::endl;
     std::cout << "------------------------------------------" << std::endl;
     
     std::vector<std::string> test_cases = {
         "",
         "a",
         "abc",
         "message digest",
         "abcdefghijklmnopqrstuvwxyz",
         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", // 62字节输入
         "12345678901234567890123456789012345678901234567890123456789012345678901234567890" // 80字节输入
     };
     
     // 分配内存池
     u8* message_pool = new u8[test_cases.size() * MAX_MESSAGE_LENGTH];
     
     bool all_correct = true;
     
     for (const auto& input : test_cases) {
         // 计算串行MD5
         u32 serial_digest[4];
         md5_serial(input, serial_digest);
         std::string serial_result = md5_to_string(serial_digest);
         
         // 计算并行MD5
         std::vector<std::vector<u32>> parallel_digests;
         md5_sse_batch({input}, parallel_digests, message_pool);
         std::string parallel_result = md5_to_string(parallel_digests[0].data());
         
         // 比较结果
         bool match = (serial_result == parallel_result);
         if (!match) all_correct = false;
         
         std::cout << "Input: \"" << (input.empty() ? "[empty]" : input) << "\"" << std::endl;
         std::cout << "  Serial MD5:   " << serial_result << std::endl;
         std::cout << "  Parallel MD5: " << parallel_result << std::endl;
         std::cout << "  Match: " << (match ? "YES ✓" : "NO ✗") << std::endl;
         std::cout << std::endl;
     }
     
     if (all_correct) {
         std::cout << "All tests passed! SSE implementation is correct." << std::endl;
     } else {
         std::cout << "Some tests failed! Please check the implementation." << std::endl;
     }
     std::cout << std::endl;
     
     delete[] message_pool;
 }

 void performance_test() {
    std::cout << "Performance Test Results:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout.flush(); // 确保输出立即显示
    
    const int NUM_BATCHES = 10;
    const int BATCH_SIZE = 100000;  // 降低批次大小到1万进行测试
    
    double total_serial_time = 0.0;
    double total_parallel_time = 0.0;
    size_t total_messages = 0;
    
    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        try {
            std::cout << "Starting batch " << batch + 1 << "/" << NUM_BATCHES << std::endl;
            std::cout.flush();
            
            // 动态调整批次大小，但减小波动范围
            int actual_batch_size = BATCH_SIZE + (rand() % 2000 - 1000);
            std::vector<std::string> messages(actual_batch_size);
            total_messages += actual_batch_size;
            
            std::cout << "Generating " << actual_batch_size << " random messages..." << std::endl;
            std::cout.flush();
            
            // 生成随机消息
            for (int i = 0; i < actual_batch_size; ++i) {
                int len = 10 + (rand() % 30);  // 简化长度范围10-40
                messages[i].resize(len);
                for (int j = 0; j < len; ++j) {
                    messages[i][j] = 'a' + (rand() % 26);  // 使用简单的字符集
                }
            }
            
            std::cout << "Processing batch of " << actual_batch_size << " guesses..." << std::endl;
            std::cout << "Starting serial hashing..." << std::endl;
            std::cout.flush();
            
            // 串行处理
            auto start_serial = std::chrono::high_resolution_clock::now();
            
            std::vector<u32> serial_digest(4);
            for (const auto& msg : messages) {
                md5_serial(msg, serial_digest.data());
                if ((&msg - &messages[0]) % 1000 == 0) {
                    std::cout << "." << std::flush;  // 显示进度点
                }
            }
            std::cout << std::endl;
            
            auto end_serial = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> serial_duration = end_serial - start_serial;
            total_serial_time += serial_duration.count();
            
            std::cout << "Starting parallel hashing..." << std::endl;
            std::cout.flush();
            
            // 并行处理 - 确保内存分配成功
            u8* message_pool = nullptr;
            try {
                message_pool = new u8[actual_batch_size * MAX_MESSAGE_LENGTH];
                std::cout << "Memory pool allocated successfully." << std::endl;
            }
            catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed: " << e.what() << std::endl;
                continue; // 跳过此批次
            }
            
            std::vector<std::vector<u32>> parallel_digests;
            
            auto start_parallel = std::chrono::high_resolution_clock::now();
            
            md5_sse_batch(messages, parallel_digests, message_pool);
            
            auto end_parallel = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> parallel_duration = end_parallel - start_parallel;
            total_parallel_time += parallel_duration.count();
            
            delete[] message_pool;
            
            // 输出批次结果
            double batch_serial_time = serial_duration.count();
            double batch_parallel_time = parallel_duration.count();
            double batch_speedup = batch_serial_time / batch_parallel_time;
            
            std::cout << "Serial Hash Time (Batch): " << std::fixed << std::setprecision(6) 
                     << batch_serial_time << " s" << std::endl;
            std::cout << "Parallel Hash Time (Batch): " << batch_parallel_time << " s" << std::endl;
            std::cout << "Batch Speedup (Serial / Parallel): " << std::setprecision(3) 
                     << batch_speedup << "x" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            std::cout.flush();
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in batch " << batch + 1 << ": " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception in batch " << batch + 1 << std::endl;
        }
    }
    
    // 输出总体结果
    double overall_speedup = total_serial_time / total_parallel_time;
    
    std::cout << "================ Summary =================" << std::endl;
    std::cout << "Total Messages Processed: " << total_messages << std::endl;
    std::cout << "Total Serial Hash Time: " << std::fixed << std::setprecision(6) 
             << total_serial_time << " seconds" << std::endl;
    std::cout << "Total Parallel Hash Time: " << total_parallel_time << " seconds" << std::endl;
    std::cout << "Overall Hash Speedup (Serial / Parallel): " << std::setprecision(3) 
             << overall_speedup << "x" << std::endl;
    std::cout << "=========================================" << std::endl;
}
 
 int main() {
     // 验证实现的正确性
     verify_md5_implementation();
     
     // 性能测试
     performance_test();
     
     return 0;
 }