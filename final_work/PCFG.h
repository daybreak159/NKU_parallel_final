#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include <vector> // 确保包含 vector
#include <utility> // 确保包含 pair
#include <functional> // 确保包含 hash

// #include <chrono>
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg(); // 声明

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;

    void insert(string value); // 声明
    void order(); // 声明
    void PrintValues(); // 声明
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content; // 确保声明

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg); // 声明
    void PrintPT(); // 声明

    // 导出新的PT
    vector<PT> NewPTs(); // 声明

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices; // 确保声明

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices; // 确保声明
    // void init();
    float preterm_prob = 0.0f; // 确保声明并初始化
    float prob = 0.0f; // 确保声明并初始化
};

// --- 新增 PTKey 类型定义 ---
using PTKey = std::vector<std::pair<int, int>>;

// --- 新增 PTKey 的哈希函数结构体 ---
struct PTKeyHash {
    std::size_t operator()(const PTKey& key) const {
        std::size_t seed = key.size();
        for(const auto& p : key) {
            // 结合 pair 中两个 int 的哈希值
            std::size_t h1 = std::hash<int>{}(p.first);
            std::size_t h2 = std::hash<int>{}(p.second);
            // 使用类似 boost::hash_combine 的方式混合哈希值
            seed ^= h1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0; // 确保声明
    vector<PT> preterminals; // 确保声明
    int FindPT(PT pt); // 声明

    vector<segment> letters; // 确保声明
    vector<segment> digits; // 确保声明
    vector<segment> symbols; // 确保声明
    int FindLetter(segment seg); // 声明
    int FindDigit(segment seg); // 声明
    int FindSymbol(segment seg); // 声明

    unordered_map<int, int> preterm_freq; // 确保声明
    unordered_map<int, int> letters_freq; // 确保声明
    unordered_map<int, int> digits_freq; // 确保声明
    unordered_map<int, int> symbols_freq; // 确保声明

    vector<PT> ordered_pts; // 确保声明

    // 给定一个训练集，对模型进行训练
    void train(string train_path); // 声明

    // 对已经训练的模型进行保存
    // void store(string store_path); // 如果有实现，需要声明

    // 从现有的模型文件中加载模型
    // void load(string load_path); // 如果有实现，需要声明

    // 对一个给定的口令进行切分
    void parse(string pw); // 声明

    void order(); // 声明

    // 打印模型
    void print(); // 声明

    // --- 新增 PT 索引映射 ---
    std::unordered_map<PTKey, int, PTKeyHash> pt_index_map;
};

class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
    size_t total_guesses_batch_delta = 0; // 确保此行存在
    vector<string> guesses;
};
