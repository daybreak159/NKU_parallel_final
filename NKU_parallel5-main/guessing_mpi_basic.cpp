#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstring> 
#include <iterator>
using namespace std;
//mpic++ -o main guessing_mpi_basic.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11
// 字符串数据打包工具 - 将字符串集合转换为MPI传输格式
void pack_string_data(const vector<string>& string_collection, vector<char>& packed_buffer, vector<int>& string_lengths) {
    string_lengths.clear();
    string_lengths.reserve(string_collection.size());
    
    // 计算总缓冲区大小并记录每个字符串长度
    int total_buffer_size = 0;
    for (const auto& str : string_collection) {
        int len = str.length();
        string_lengths.push_back(len);
        total_buffer_size += len;
    }
    
    // 分配并填充连续缓冲区
    packed_buffer.resize(total_buffer_size);
    int current_offset = 0;
    for (const auto& str : string_collection) {
        memcpy(packed_buffer.data() + current_offset, str.c_str(), str.length());
        current_offset += str.length();
    }
}

// 字符串数据解包工具 - 从MPI传输格式重构字符串集合
vector<string> unpack_string_data(const vector<char>& packed_buffer, const vector<int>& string_lengths) {
    vector<string> result_strings;
    result_strings.reserve(string_lengths.size());
    
    int current_offset = 0;
    for (int length : string_lengths) {
        if (length > 0) {
            result_strings.emplace_back(packed_buffer.data() + current_offset, length);
        } else {
            result_strings.emplace_back("");
        }
        current_offset += length;
    }
    
    return result_strings;
}

// 保持原有函数不变
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int idx = 0;

    for (int currIdx : pt.curr_indices)
    {
        if (pt.content[idx].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.letters[m.FindLetter(pt.content[idx])].total_freq;
        }
        if (pt.content[idx].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.digits[m.FindDigit(pt.content[idx])].total_freq;
        }
        if (pt.content[idx].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[idx])].total_freq;
        }
        idx += 1;
    }
}

void PriorityQueue::init()
{
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

        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    GenerateParallelMPI(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 主要修改：MPI并行密码生成实现
void PriorityQueue::GenerateParallelMPI(PT pt) {
    // 获取MPI环境信息
    int current_rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    
    // 计算PT概率
    CalProb(pt);
    
    // 处理单segment和多segment的统一框架
    segment* target_seg_ptr = nullptr;
    string base_prefix = "";
    int total_work_items = 0;
    
    // 根据PT结构确定处理方式
    if (pt.content.size() == 1) {
        // 单segment情况：直接定位目标segment
        const segment& single_seg = pt.content[0];
        switch (single_seg.type) {
            case 1: target_seg_ptr = &m.letters[m.FindLetter(single_seg)]; break;
            case 2: target_seg_ptr = &m.digits[m.FindDigit(single_seg)]; break;
            case 3: target_seg_ptr = &m.symbols[m.FindSymbol(single_seg)]; break;
        }
        total_work_items = pt.max_indices[0];
    } else {
        // 多segment情况：构建前缀并定位最后segment
        int prefix_seg_index = 0;
        for (int segment_value_idx : pt.curr_indices) {
            if (prefix_seg_index >= (int)pt.content.size() - 1) break;
            
            const segment& current_seg = pt.content[prefix_seg_index];
            switch (current_seg.type) {
                case 1: 
                    base_prefix += m.letters[m.FindLetter(current_seg)].ordered_values[segment_value_idx];
                    break;
                case 2:
                    base_prefix += m.digits[m.FindDigit(current_seg)].ordered_values[segment_value_idx];
                    break;
                case 3:
                    base_prefix += m.symbols[m.FindSymbol(current_seg)].ordered_values[segment_value_idx];
                    break;
            }
            prefix_seg_index++;
        }
        
        // 定位最后的segment
        const segment& final_seg = pt.content.back();
        switch (final_seg.type) {
            case 1: target_seg_ptr = &m.letters[m.FindLetter(final_seg)]; break;
            case 2: target_seg_ptr = &m.digits[m.FindDigit(final_seg)]; break;
            case 3: target_seg_ptr = &m.symbols[m.FindSymbol(final_seg)]; break;
        }
        total_work_items = pt.max_indices.back();
    }
    
    // 分布式负载分配策略
    struct WorkDistribution {
        int base_load;
        int extra_load;
        int start_index;
        int end_index;
    } work_dist;
    
    work_dist.base_load = total_work_items / process_count;
    work_dist.extra_load = total_work_items % process_count;
    work_dist.start_index = current_rank * work_dist.base_load + min(current_rank, work_dist.extra_load);
    work_dist.end_index = work_dist.start_index + work_dist.base_load + (current_rank < work_dist.extra_load ? 1 : 0);
    
    // 本地密码生成
    vector<string> local_password_batch;
    local_password_batch.reserve(work_dist.end_index - work_dist.start_index);
    
    for (int work_idx = work_dist.start_index; work_idx < work_dist.end_index; work_idx++) {
        string generated_password = base_prefix + target_seg_ptr->ordered_values[work_idx];
        local_password_batch.push_back(move(generated_password));
    }
    
    // 第一阶段：收集各进程的工作量统计
    int local_batch_size = local_password_batch.size();
    vector<int> process_batch_sizes(process_count);
    
    MPI_Gather(&local_batch_size, 1, MPI_INT, 
               process_batch_sizes.data(), 1, MPI_INT, 
               0, MPI_COMM_WORLD);
    
    // 主进程更新总计数
    if (current_rank == 0) {
        for (int batch_size : process_batch_sizes) {
            total_guesses += batch_size;
        }
    }
    
    // 第二阶段：数据打包和传输准备
    vector<char> local_packed_data;
    vector<int> local_string_lengths;
    pack_string_data(local_password_batch, local_packed_data, local_string_lengths);
    
    // 收集数据大小信息用于变长数据传输
    int local_data_size = local_packed_data.size();
    vector<int> process_data_sizes(process_count);
    vector<int> data_displacements(process_count);
    
    MPI_Gather(&local_data_size, 1, MPI_INT,
               process_data_sizes.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);
    
    // 计算数据偏移和总大小
    int total_data_size = 0;
    if (current_rank == 0) {
        data_displacements[0] = 0;
        for (int i = 0; i < process_count; i++) {
            if (i > 0) {
                data_displacements[i] = data_displacements[i-1] + process_data_sizes[i-1];
            }
            total_data_size += process_data_sizes[i];
        }
    }
    
    // 第三阶段：收集字符串数据
    vector<char> aggregated_data(total_data_size);
    MPI_Gatherv(local_packed_data.data(), local_packed_data.size(), MPI_CHAR,
                aggregated_data.data(), process_data_sizes.data(), data_displacements.data(),
                MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // 第四阶段：收集长度信息
    int local_length_count = local_string_lengths.size();
    vector<int> process_length_counts(process_count);
    vector<int> length_displacements(process_count);
    
    MPI_Gather(&local_length_count, 1, MPI_INT,
               process_length_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);
    
    int total_length_count = 0;
    if (current_rank == 0) {
        length_displacements[0] = 0;
        for (int i = 0; i < process_count; i++) {
            if (i > 0) {
                length_displacements[i] = length_displacements[i-1] + process_length_counts[i-1];
            }
            total_length_count += process_length_counts[i];
        }
    }
    
    vector<int> aggregated_lengths(total_length_count);
    MPI_Gatherv(local_string_lengths.data(), local_string_lengths.size(), MPI_INT,
                aggregated_lengths.data(), process_length_counts.data(), length_displacements.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
    
    // 最终阶段：主进程解包和结果整合
    if (current_rank == 0) {
        vector<string> final_password_collection = unpack_string_data(aggregated_data, aggregated_lengths);
        guesses.reserve(guesses.size() + final_password_collection.size());
        guesses.insert(guesses.end(), 
                      make_move_iterator(final_password_collection.begin()),
                      make_move_iterator(final_password_collection.end()));
    }
}

// 保持原有Generate函数作为串行备份
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}