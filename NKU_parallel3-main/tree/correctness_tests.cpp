#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace chrono;

// 辅助函数：将MD5哈希结果转换为十六进制字符串
string hashToString(bit32* state) {
    stringstream ss;
    for (int i = 0; i < 4; i++) {
        ss << std::setw(8) << std::setfill('0') << hex << state[i];
    }
    return ss.str();
}

// 验证MD5哈希函数正确性的主函数
int main(int argc, char* argv[])
{
    // 测试字符串数组，包含多个不同长度和内容的测试样例
    vector<string> testInputs = {
        // 简短字符串
        "helloworlddd",
        // 较短字符串
        "thequickbrownfoxjumpsoverthelazydog",
        // 中等长度字符串
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva",
        // 非常长的字符串 (重复内容)
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva"
    };

    cout << "同时运行串行和并行MD5实现，并比较结果..." << endl;

    // 存储串行结果
    vector<string> serialResults;
    // 存储并行结果
    vector<string> parallelResults;
    // 动态分配的状态数组，需要后续释放
    vector<bit32*> serialStates;
    vector<bit32*> parallelStates;

    // 1. 计算所有测试字符串的串行MD5结果
    cout << "\n运行串行MD5..." << endl;
    for (const auto& input : testInputs) {
        bit32* state = new bit32[4];
        MD5Hash(input, state);

        string result = hashToString(state);
        serialResults.push_back(result);
        serialStates.push_back(state);

        cout << "测试字符串: " << (input.length() <= 30 ? input : input.substr(0, 27) + "...") << endl;
        cout << "串行MD5结果: " << result << endl << endl;
    }

    // 2. 计算所有测试字符串的并行MD5结果
    cout << "\n运行并行MD5..." << endl;
    MD5Hash_NEON(testInputs, parallelStates);

    for (size_t i = 0; i < testInputs.size(); i++) {
        string result = hashToString(parallelStates[i]);
        parallelResults.push_back(result);

        cout << "测试字符串: " << (testInputs[i].length() <= 30 ? testInputs[i] : testInputs[i].substr(0, 27) + "...") << endl;
        cout << "并行MD5结果: " << result << endl << endl;
    }

    // 3. 比较串行和并行结果
    cout << "\n结果比较:" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    cout << setw(10) << left << "测试编号" 
         << setw(35) << "串行MD5结果" 
         << setw(35) << "并行MD5结果" 
         << setw(10) << "是否一致" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;

    bool allCorrect = true;
    for (size_t i = 0; i < serialResults.size(); i++) {
        bool match = serialResults[i] == parallelResults[i];
        cout << setw(10) << left << i+1
             << setw(35) << serialResults[i] 
             << setw(35) << parallelResults[i]
             << setw(10) << (match ? "✓" : "✗") << endl;

        if (!match) allCorrect = false;
    }
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    cout << "总体结果: " << (allCorrect ? "全部通过 ✓" : "存在不匹配 ✗") << endl;

    // 并行与串行性能对比
    cout << "\n性能对比 (多字符串哈希):" << endl;

    // 串行版本 (依次处理每个字符串)
    auto start_serial = high_resolution_clock::now();
    for (const auto& input : testInputs) {
        bit32 state[4];
        MD5Hash(input, state);
    }
    auto end_serial = high_resolution_clock::now();
    auto serial_duration = duration_cast<microseconds>(end_serial - start_serial);

    // 并行版本 (一次性处理所有字符串)
    auto start_parallel = high_resolution_clock::now();
    vector<bit32*> states;
    MD5Hash_NEON(testInputs, states);
    auto end_parallel = high_resolution_clock::now();
    auto parallel_duration = duration_cast<microseconds>(end_parallel - start_parallel);

    cout << "串行处理时间: " << serial_duration.count() << " 微秒" << endl;
    cout << "并行处理时间: " << parallel_duration.count() << " 微秒" << endl;
    cout << "加速比: " << fixed << setprecision(2) 
         << static_cast<double>(serial_duration.count()) / parallel_duration.count() << "x" << endl;

    // 释放动态分配的内存
    for (auto& state : serialStates) {
        delete[] state;
    }

    for (auto& state : parallelStates) {
        delete[] state;
    }

    // 清理用于性能测试的states
    for (auto& state : states) {
        delete[] state;
    }

    return 0;
}

//g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main