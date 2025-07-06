#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <new> // 包含 <new> 以使用 std::bad_alloc
#include <cstring> // 包含 <cstring> 以使用 memset

using namespace std;
using namespace chrono;

// 辅助函数：将 bit32 数组转换为十六进制字符串
string format_hash(const bit32* state) {
    stringstream ss;
    for (int i = 0; i < 4; ++i) {
        // 确保 state 指针有效
        if (!state) return "Error: Null state pointer";
        ss << std::setw(8) << std::setfill('0') << hex << state[i];
    }
    return ss.str();
}

int main()
{
    // 1. 准备输入
    vector<string> test_strings = {
        "", // 空字符串
        "a", // 字符串 "a"
        "abc", // 字符串 "abc"
        "message digest", // 字符串 "message digest"
        "abcdefghijklmnopqrstuvwxyz", // 字符串 "abcdefghijklmnopqrstuvwxyz"
        "1234567890123456789012345678901234567890", // 较长的字符串 "1234567890123456789012345678901234567890"
    };

    size_t input_count = test_strings.size();
    const size_t alignment = 16;
    const size_t MAX_PADDED_LEN_PER_MSG = 256; // 与 main.cpp 保持一致

    cout << "Testing " << input_count << " strings..." << endl;

    // 2. 串行计算哈希进行比较
    vector<string> serial_hashes;
    cout << "--- Serial MD5 Calculation ---" << endl;
    for (const auto& str : test_strings) {
        bit32 serial_state[4];
        MD5Hash(str, serial_state);
        serial_hashes.push_back(format_hash(serial_state));
        cout << "String: \"" << (str.length() > 20 ? str.substr(0, 20) + "..." : str) << "\"" << endl;
        cout << "Serial MD5: " << serial_hashes.back() << endl;
    }
    cout << "-----------------------------" << endl;

    // 3. 预分配 SIMD 计算所需的内存 (模仿 main.cpp)
    bit32** parallel_results = nullptr;
    Byte** padded_message_pointers = nullptr;
    Byte* padded_message_pool = nullptr;
    int* messageLengths = nullptr;
    bool prealloc_ok = true;

    try {
        parallel_results = new bit32*[input_count];
        for (size_t i = 0; i < input_count; ++i) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, 4 * sizeof(bit32)) != 0) {
                throw std::runtime_error("Memory pre-allocation failed for parallel_results.");
            }
            parallel_results[i] = static_cast<bit32*>(ptr);
            memset(parallel_results[i], 0, 4 * sizeof(bit32)); // 初始化为 0
        }

        padded_message_pointers = new Byte*[input_count];

        void* pool_ptr = nullptr;
        if (posix_memalign(&pool_ptr, alignment, input_count * MAX_PADDED_LEN_PER_MSG) != 0) {
             throw std::runtime_error("Memory pre-allocation failed for padded_message_pool.");
        }
        padded_message_pool = static_cast<Byte*>(pool_ptr);

        for (size_t i = 0; i < input_count; ++i) {
            padded_message_pointers[i] = padded_message_pool + i * MAX_PADDED_LEN_PER_MSG;
        }

        messageLengths = new int[input_count];

    } catch (const std::bad_alloc& e) {
        cerr << "Error during memory pre-allocation (new): " << e.what() << endl;
        prealloc_ok = false;
    } catch (const std::runtime_error& e) {
         cerr << "Error during memory pre-allocation (posix_memalign): " << e.what() << endl;
         prealloc_ok = false;
    }

    // 如果预分配失败，则提前退出
    if (!prealloc_ok) {
        cerr << "Critical error: Failed to pre-allocate memory buffers. Exiting correctness test." << endl;
        // 清理可能已部分分配的内存
        if (parallel_results) {
            for (size_t i = 0; i < input_count; ++i) { if (parallel_results[i]) free(parallel_results[i]); }
            delete[] parallel_results;
        }
        if (padded_message_pool) free(padded_message_pool);
        delete[] padded_message_pointers;
        delete[] messageLengths;
        return 1;
    }

    // 4. 调用并行预处理和 SIMD 哈希函数
    bool success = true;
    cout << "\n--- Parallel SIMD Calculation ---" << endl;
    try {
        // 4a. 调用并行预处理
        cout << "Calling StringProcess_Parallel..." << endl;
        StringProcess_Parallel(test_strings, input_count, padded_message_pointers, messageLengths, alignment);
        cout << "StringProcess_Parallel finished." << endl;

        // 4b. 调用 SIMD 批量哈希
        cout << "Calling MD5Hash_SIMD_Batch..." << endl;
        MD5Hash_SIMD_Batch(const_cast<const Byte**>(padded_message_pointers), messageLengths, input_count, parallel_results);
        cout << "MD5Hash_SIMD_Batch finished." << endl;

        // 5. 比较结果
        bool all_correct = true;
        for (size_t i = 0; i < input_count; ++i) {
            if (parallel_results[i]) { // 检查指针是否有效
                string parallel_hash = format_hash(parallel_results[i]);
                cout << "SIMD MD5[" << i << "]: " << parallel_hash << endl;

                if (parallel_hash != serial_hashes[i]) {
                    cerr << "Hash mismatch for string " << i << "!" << endl;
                    cerr << "  Serial:   " << serial_hashes[i] << endl;
                    cerr << "  Parallel: " << parallel_hash << endl;
                    all_correct = false;
                }
            } else {
                cerr << "Error: SIMD result buffer for string " << i << " is null." << endl;
                all_correct = false;
            }
        }
        cout << "--------------------------------" << endl;

        if (all_correct) {
            cout << "\nResult: All SIMD hashes match serial results. Test PASSED!" << endl;
        } else {
            cout << "\nResult: Some SIMD hashes DON'T match serial results. Test FAILED!" << endl;
            success = false;
        }
    }
    catch (const std::exception& e) {
        cerr << "Error during parallel processing: " << e.what() << endl;
        success = false;
    }

    // 6. 清理预分配的内存
    cout << "\nCleaning up memory..." << endl;
    if (parallel_results) {
        for (size_t i = 0; i < input_count; ++i) {
            if (parallel_results[i]) free(parallel_results[i]);
        }
        delete[] parallel_results;
    }
    if (padded_message_pool) {
        free(padded_message_pool);
    }
    delete[] padded_message_pointers;
    delete[] messageLengths;
    cout << "Cleanup finished." << endl;

    return success ? 0 : 1; // 返回 0 表示成功，非 0 表示失败
}
