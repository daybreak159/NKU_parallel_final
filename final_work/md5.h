#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h> // 添加 NEON 头文件
#include <vector>     // 添加 vector 头文件

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
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

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// --- 使用参考实现的 SIMD 宏定义 ---
// SIMD版本的基础函数 - 使用宏定义 (保持不变，因为下面的宏依赖它们)
#define F_NEON(x, y, z) vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32(x), (z)))
#define G_NEON(x, y, z) vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32(z)))
#define H_NEON(x, y, z) veorq_u32((x), veorq_u32((y), (z)))
#define I_NEON(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32(z)))

// SIMD版本的旋转左移宏 (保持不变)
#define ROTATELEFT_NEON(num, n) vorrq_u32(vshlq_n_u32(num, n), vshrq_n_u32(num, 32 - n))

// --- 使用通用的 MD5_STEP_NEON 宏 ---
#define MD5_STEP_NEON(func, a, b, c, d, x, s, ac) do { \
  uint32x4_t _term = func((b), (c), (d));       \
  uint32x4_t _add = vaddq_u32(_term, (x));      \
  _add = vaddq_u32(_add, vdupq_n_u32(ac));      \
  (a) = vaddq_u32((a), _add);                     \
  (a) = ROTATELEFT_NEON((a), (s));                  \
  (a) = vaddq_u32((a), (b));                        \
} while(0)

// 使用通用宏定义 FF_NEON, GG_NEON, HH_NEON, II_NEON
#define FF_NEON(a, b, c, d, x, s, ac) MD5_STEP_NEON(F_NEON, a, b, c, d, x, s, ac)
#define GG_NEON(a, b, c, d, x, s, ac) MD5_STEP_NEON(G_NEON, a, b, c, d, x, s, ac)
#define HH_NEON(a, b, c, d, x, s, ac) MD5_STEP_NEON(H_NEON, a, b, c, d, x, s, ac)
#define II_NEON(a, b, c, d, x, s, ac) MD5_STEP_NEON(I_NEON, a, b, c, d, x, s, ac)
// --- 结束使用通用的 MD5_STEP_NEON 宏 ---

// --- 添加静态/全局变量声明 ---
#define ZERO_BUFFER_SIZE 64 // 用于 memcpy 优化的零缓冲区大小
static Byte zero_padding_buffer[ZERO_BUFFER_SIZE] = {0}; // 定义零缓冲区

// 串行 MD5 哈希函数声明
void MD5Hash(string input, bit32 *state);

// 并行 SIMD 批量哈希函数声明 (保持不变)
void MD5Hash_SIMD_Batch(const Byte** paddedMessages, const int* messageLengths, size_t input_count, bit32** states);

// 串行字符串处理函数声明 (保持不变)
Byte* StringProcess(string input, int* n_byte);

// 并行字符串处理函数声明 (修改签名以接受预分配的缓冲区)
// 不再返回 Byte**& 和 int*&，而是接收它们作为参数
void StringProcess_Parallel(const std::vector<std::string>& inputs, size_t input_count, Byte** paddedMessagePointers, int* messageLengths, size_t alignment = 16);