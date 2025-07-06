#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstring>  // 添加这个头文件
using namespace std;

// ===== 数据结构优化 =====
//mpic++ -o main guessing_mpi_jinjie1.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11
// 统一的MPI工作单元
struct MPIWorkUnit {
    int unit_id;
    vector<char> pt_data;
    int data_size;
    int assigned_process;
    
    MPIWorkUnit() : unit_id(-1), data_size(0), assigned_process(-1) {}
};

// 结果收集器
struct ResultCollector {
    vector<string> passwords;
    vector<PT> new_pts;
    int total_count;
    
    ResultCollector() : total_count(0) {}
    
    void merge_results(const ResultCollector& other) {
        passwords.insert(passwords.end(), other.passwords.begin(), other.passwords.end());
        new_pts.insert(new_pts.end(), other.new_pts.begin(), other.new_pts.end());
        total_count += other.total_count;
    }
};

// 内存缓冲区管理器
class BufferManager {
private:
    vector<char> reusable_buffer;
    vector<int> size_buffer;
    
public:
    void prepare_buffer(size_t required_size) {
        if (reusable_buffer.size() < required_size) {
            reusable_buffer.resize(required_size * 1.5); // 预留额外空间
        }
    }
    
    vector<char>& get_char_buffer() { return reusable_buffer; }
    vector<int>& get_size_buffer() { return size_buffer; }
    
    void reset_buffers() {
        // 保持容量，只清空内容
        reusable_buffer.clear();
        size_buffer.clear();
    }
};

// ===== 核心函数保持不变的部分 =====
void PriorityQueue::CalProb(PT &pt) {
    // 保持原有实现
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init() {
    // 保持原有实现
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

vector<PT> PT::NewPTs() {
    // 保持原有实现
    vector<PT> res;
    if (content.size() == 1) {
        return res;
    } else {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i]) {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// ===== 重新设计的序列化函数 =====

class DataSerializer {
public:
    // 改进的字符串序列化 - 使用单次遍历
    static void pack_strings_optimized(const vector<string>& strings, 
                                     vector<char>& buffer, 
                                     vector<int>& lengths,
                                     BufferManager& buffer_mgr) {
        if (strings.empty()) {
            buffer_mgr.reset_buffers();
            return;
        }
        
        lengths.resize(strings.size());
        size_t total_size = 0;
        
        // 单次遍历计算总大小和长度
        for (size_t i = 0; i < strings.size(); ++i) {
            lengths[i] = static_cast<int>(strings[i].size());
            total_size += lengths[i];
        }
        
        buffer_mgr.prepare_buffer(total_size);
        buffer = buffer_mgr.get_char_buffer();
        buffer.resize(total_size);
        
        // 批量复制数据
        size_t write_offset = 0;
        for (const string& str : strings) {
            if (!str.empty()) {
                memcpy(&buffer[write_offset], str.data(), str.size());
                write_offset += str.size();
            }
        }
    }
    
    // 改进的字符串反序列化 - 使用预分配
    static vector<string> unpack_strings_optimized(const vector<char>& buffer, 
                                                  const vector<int>& lengths) {
        vector<string> result;
        result.reserve(lengths.size()); // 预分配容量
        
        size_t read_offset = 0;
        for (int length : lengths) {
            if (length > 0 && read_offset + length <= buffer.size()) {
                result.emplace_back(&buffer[read_offset], length);
                read_offset += length;
            } else {
                result.emplace_back(); // 空字符串
            }
        }
        
        return result;
    }
    
    // 改进的PT序列化 - 使用结构化写入
    static bool encode_pt_with_validation(const PT& pt, vector<char>& buffer) {
        // 数据有效性检查
        if (pt.content.empty() || pt.max_indices.empty() || pt.curr_indices.empty()) {
            return false;
        }
        
        // 计算确切大小
        size_t required_size = sizeof(int) + sizeof(float) * 2; // 基础字段
        required_size += sizeof(int) + pt.content.size() * sizeof(int) * 2; // content
        required_size += sizeof(int) * 2; // 向量大小
        required_size += pt.max_indices.size() * sizeof(int);
        required_size += pt.curr_indices.size() * sizeof(int);
        
        buffer.resize(required_size);
        
        // 使用结构化写入
        char* write_ptr = buffer.data();
        
        // 写入基础数据
        *reinterpret_cast<int*>(write_ptr) = pt.pivot;
        write_ptr += sizeof(int);
        *reinterpret_cast<float*>(write_ptr) = pt.prob;
        write_ptr += sizeof(float);
        *reinterpret_cast<float*>(write_ptr) = pt.preterm_prob;
        write_ptr += sizeof(float);
        
        // 写入content
        *reinterpret_cast<int*>(write_ptr) = static_cast<int>(pt.content.size());
        write_ptr += sizeof(int);
        for (const segment& seg : pt.content) {
            *reinterpret_cast<int*>(write_ptr) = seg.type;
            write_ptr += sizeof(int);
            *reinterpret_cast<int*>(write_ptr) = seg.length;
            write_ptr += sizeof(int);
        }
        
        // 写入max_indices
        *reinterpret_cast<int*>(write_ptr) = static_cast<int>(pt.max_indices.size());
        write_ptr += sizeof(int);
        memcpy(write_ptr, pt.max_indices.data(), pt.max_indices.size() * sizeof(int));
        write_ptr += pt.max_indices.size() * sizeof(int);
        
        // 写入curr_indices
        *reinterpret_cast<int*>(write_ptr) = static_cast<int>(pt.curr_indices.size());
        write_ptr += sizeof(int);
        memcpy(write_ptr, pt.curr_indices.data(), pt.curr_indices.size() * sizeof(int));
        
        return true;
    }
    
    // 改进的PT反序列化 - 添加边界检查
    static bool decode_pt_with_validation(const vector<char>& buffer, PT& result) {
        if (buffer.size() < sizeof(int) + sizeof(float) * 2 + sizeof(int)) {
            return false; // 缓冲区太小
        }
        
        const char* read_ptr = buffer.data();
        
        // 读取基础数据
        result.pivot = *reinterpret_cast<const int*>(read_ptr);
        read_ptr += sizeof(int);
        result.prob = *reinterpret_cast<const float*>(read_ptr);
        read_ptr += sizeof(float);
        result.preterm_prob = *reinterpret_cast<const float*>(read_ptr);
        read_ptr += sizeof(float);
        
        // 读取content
        int content_size = *reinterpret_cast<const int*>(read_ptr);
        read_ptr += sizeof(int);
        
        if (content_size < 0 || content_size > 100) { // 合理性检查
            return false;
        }
        
        result.content.clear();
        result.content.reserve(content_size);
        
        for (int i = 0; i < content_size; ++i) {
            if (read_ptr + sizeof(int) * 2 > buffer.data() + buffer.size()) {
                return false; // 边界检查
            }
            
            int type = *reinterpret_cast<const int*>(read_ptr);
            read_ptr += sizeof(int);
            int length = *reinterpret_cast<const int*>(read_ptr);
            read_ptr += sizeof(int);
            
            result.content.emplace_back(type, length);
        }
        
        // 读取max_indices
        if (read_ptr + sizeof(int) > buffer.data() + buffer.size()) {
            return false;
        }
        
        int max_indices_size = *reinterpret_cast<const int*>(read_ptr);
        read_ptr += sizeof(int);
        
        if (max_indices_size < 0 || max_indices_size > 100) {
            return false;
        }
        
        result.max_indices.resize(max_indices_size);
        if (max_indices_size > 0) {
            if (read_ptr + max_indices_size * sizeof(int) > buffer.data() + buffer.size()) {
                return false;
            }
            memcpy(result.max_indices.data(), read_ptr, max_indices_size * sizeof(int));
            read_ptr += max_indices_size * sizeof(int);
        }
        
        // 读取curr_indices
        if (read_ptr + sizeof(int) > buffer.data() + buffer.size()) {
            return false;
        }
        
        int curr_indices_size = *reinterpret_cast<const int*>(read_ptr);
        read_ptr += sizeof(int);
        
        if (curr_indices_size < 0 || curr_indices_size > 100) {
            return false;
        }
        
        result.curr_indices.resize(curr_indices_size);
        if (curr_indices_size > 0) {
            if (read_ptr + curr_indices_size * sizeof(int) > buffer.data() + buffer.size()) {
                return false;
            }
            memcpy(result.curr_indices.data(), read_ptr, curr_indices_size * sizeof(int));
        }
        
        return true;
    }
};

// ===== 重新设计的PT处理函数 =====

class PTProcessor {
public:
    static ResultCollector process_pt_batch(const vector<PT>& pt_batch, PriorityQueue& queue) {
        ResultCollector collector;
        
        for (const PT& pt : pt_batch) {
            // 复制PT以避免修改原始数据
            PT working_pt = pt;
            
            // 生成密码
            vector<string> passwords = generate_passwords_for_pt(working_pt, queue);
            collector.passwords.insert(collector.passwords.end(), passwords.begin(), passwords.end());
            
            // 生成新PT
            vector<PT> new_pts = working_pt.NewPTs();
            for (PT& new_pt : new_pts) {
                queue.CalProb(new_pt);
                collector.new_pts.push_back(new_pt);
            }
        }
        
        collector.total_count = collector.passwords.size();
        return collector;
    }
    
private:
    static vector<string> generate_passwords_for_pt(const PT& pt, PriorityQueue& queue) {
        vector<string> passwords;
        
        if (pt.content.size() == 1) {
            // 单segment处理
            segment* target_seg = get_target_segment(pt.content[0], queue);
            if (target_seg && pt.max_indices.size() > 0) {
                passwords.reserve(pt.max_indices[0]); // 预分配
                for (int i = 0; i < pt.max_indices[0]; ++i) {
                    if (i < target_seg->ordered_values.size()) {
                        passwords.push_back(target_seg->ordered_values[i]);
                    }
                }
            }
        } else {
            // 多segment处理
            string prefix = build_prefix_string(pt, queue);
            segment* last_seg = get_target_segment(pt.content.back(), queue);
            
            if (last_seg && !pt.max_indices.empty()) {
                int last_max = pt.max_indices.back();
                passwords.reserve(last_max); // 预分配
                
                for (int i = 0; i < last_max; ++i) {
                    if (i < last_seg->ordered_values.size()) {
                        passwords.push_back(prefix + last_seg->ordered_values[i]);
                    }
                }
            }
        }
        
        return passwords;
    }
    
    static segment* get_target_segment(const segment& seg, PriorityQueue& queue) {
        switch (seg.type) {
            case 1: return &queue.m.letters[queue.m.FindLetter(seg)];
            case 2: return &queue.m.digits[queue.m.FindDigit(seg)];
            case 3: return &queue.m.symbols[queue.m.FindSymbol(seg)];
            default: return nullptr;
        }
    }
    
    static string build_prefix_string(const PT& pt, PriorityQueue& queue) {
        string prefix;
        
        for (size_t i = 0; i < pt.curr_indices.size() && i < pt.content.size() - 1; ++i) {
            segment* seg = get_target_segment(pt.content[i], queue);
            if (seg && pt.curr_indices[i] < seg->ordered_values.size()) {
                prefix += seg->ordered_values[pt.curr_indices[i]];
            }
        }
        
        return prefix;
    }
};

// ===== 重新设计的主函数 - 按需分发策略 =====

void PriorityQueue::PopNext() {
    int process_rank, total_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    
    // 使用静态缓冲区管理器
    static BufferManager buffer_mgr;
    
    // 确定工作分配策略
    int available_pts = static_cast<int>(priority.size());
    int pts_to_process = min(total_processes, available_pts);
    
    if (pts_to_process == 0) {
        return; // 提前退出
    }
    
    // 只在必要时进行广播
    if (process_rank == 0) {
        MPI_Bcast(&pts_to_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&pts_to_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // 创建工作分配表
    vector<MPIWorkUnit> work_units;
    if (process_rank == 0) {
        work_units.resize(pts_to_process);
        
        for (int i = 0; i < pts_to_process && !priority.empty(); ++i) {
            work_units[i].unit_id = i;
            work_units[i].assigned_process = i % total_processes;
            
            PT current_pt = priority.front();
            priority.erase(priority.begin());
            
            if (!DataSerializer::encode_pt_with_validation(current_pt, work_units[i].pt_data)) {
                // 编码失败，跳过这个PT
                continue;
            }
            work_units[i].data_size = work_units[i].pt_data.size();
        }
    }
    
    // 按需分发工作单元
    vector<PT> local_pts;
    for (int unit_idx = 0; unit_idx < pts_to_process; ++unit_idx) {
        int data_size = 0;
        
        if (process_rank == 0) {
            data_size = work_units[unit_idx].data_size;
        }
        
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (data_size <= 0) continue;
        
        vector<char> pt_buffer(data_size);
        
        if (process_rank == 0) {
            pt_buffer = work_units[unit_idx].pt_data;
        }
        
        MPI_Bcast(pt_buffer.data(), data_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // 只有分配到的进程处理数据
        if (unit_idx % total_processes == process_rank) {
            PT decoded_pt;
            if (DataSerializer::decode_pt_with_validation(pt_buffer, decoded_pt)) {
                local_pts.push_back(decoded_pt);
            }
        }
    }
    
    // 批量处理PT
    ResultCollector local_results = PTProcessor::process_pt_batch(local_pts, *this);
    
    // 收集结果统计
    int local_password_count = local_results.total_count;
    int local_new_pt_count = local_results.new_pts.size();
    
    vector<int> all_password_counts(total_processes);
    vector<int> all_new_pt_counts(total_processes);
    
    MPI_Gather(&local_password_count, 1, MPI_INT, all_password_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_new_pt_count, 1, MPI_INT, all_new_pt_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 准备密码数据序列化
    vector<char> password_buffer;
    vector<int> password_lengths;
    DataSerializer::pack_strings_optimized(local_results.passwords, password_buffer, password_lengths, buffer_mgr);
    
    // 准备新PT数据序列化
    vector<char> my_new_pt_data;
    for (const PT& pt : local_results.new_pts) {
        vector<char> pt_encoded;
        if (DataSerializer::encode_pt_with_validation(pt, pt_encoded)) {
            int encoded_size = pt_encoded.size();
            my_new_pt_data.insert(my_new_pt_data.end(), (char*)&encoded_size, (char*)&encoded_size + sizeof(int));
            my_new_pt_data.insert(my_new_pt_data.end(), pt_encoded.begin(), pt_encoded.end());
        }
    }
    
    // 收集数据大小信息
    int my_password_data_size = password_buffer.size();
    int my_password_length_size = password_lengths.size();
    int my_new_pt_data_size = my_new_pt_data.size();
    
    vector<int> password_data_sizes(total_processes);
    vector<int> password_length_sizes(total_processes);
    vector<int> new_pt_data_sizes(total_processes);
    
    MPI_Gather(&my_password_data_size, 1, MPI_INT, password_data_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_password_length_size, 1, MPI_INT, password_length_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_new_pt_data_size, 1, MPI_INT, new_pt_data_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 计算偏移量
    vector<int> password_data_disps(total_processes);
    vector<int> password_length_disps(total_processes);
    vector<int> new_pt_data_disps(total_processes);
    
    if (process_rank == 0) {
        password_data_disps[0] = 0;
        password_length_disps[0] = 0;
        new_pt_data_disps[0] = 0;
        
        for (int i = 1; i < total_processes; i++) {
            password_data_disps[i] = password_data_disps[i-1] + password_data_sizes[i-1];
            password_length_disps[i] = password_length_disps[i-1] + password_length_sizes[i-1];
            new_pt_data_disps[i] = new_pt_data_disps[i-1] + new_pt_data_sizes[i-1];
        }
    }
    
    // 准备接收缓冲区
    int total_password_data_size = 0;
    int total_password_length_size = 0;
    int total_new_pt_data_size = 0;
    
    if (process_rank == 0) {
        for (int i = 0; i < total_processes; i++) {
            total_password_data_size += password_data_sizes[i];
            total_password_length_size += password_length_sizes[i];
            total_new_pt_data_size += new_pt_data_sizes[i];
        }
    }
    
    vector<char> all_password_data(total_password_data_size);
    vector<int> all_password_lengths(total_password_length_size);
    vector<char> all_new_pt_data(total_new_pt_data_size);
    
    // 收集所有数据
    MPI_Gatherv(password_buffer.data(), password_buffer.size(), MPI_CHAR,
               all_password_data.data(), password_data_sizes.data(), password_data_disps.data(),
               MPI_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(password_lengths.data(), password_lengths.size(), MPI_INT,
               all_password_lengths.data(), password_length_sizes.data(), password_length_disps.data(),
               MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(my_new_pt_data.data(), my_new_pt_data.size(), MPI_CHAR,
               all_new_pt_data.data(), new_pt_data_sizes.data(), new_pt_data_disps.data(),
               MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // 主进程处理收集到的所有结果
    if (process_rank == 0) {
        // 处理本地结果（主进程自己的结果）
        guesses.insert(guesses.end(), local_results.passwords.begin(), local_results.passwords.end());
        total_guesses += local_results.total_count;
        
        // 将本地新PT插入优先队列
        for (const PT& new_pt : local_results.new_pts) {
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (iter != priority.end() - 1 && iter != priority.begin()) {
                    if (new_pt.prob <= iter->prob && new_pt.prob > (iter + 1)->prob) {
                        priority.emplace(iter + 1, new_pt);
                        inserted = true;
                        break;
                    }
                }
                if (iter == priority.end() - 1) {
                    priority.emplace_back(new_pt);
                    inserted = true;
                    break;
                }
                if (iter == priority.begin() && iter->prob < new_pt.prob) {
                    priority.emplace(iter, new_pt);
                    inserted = true;
                    break;
                }
            }
            
            if (!inserted && priority.empty()) {
                priority.push_back(new_pt);
            }
        }
        
        // 处理其他进程的密码结果
        for (int i = 0; i < total_processes; i++) {
            if (i == 0) continue; // 跳过主进程，已经处理过了
            
            if (all_password_counts[i] > 0) {
                vector<string> proc_passwords = DataSerializer::unpack_strings_optimized(
                    vector<char>(all_password_data.begin() + password_data_disps[i],
                                all_password_data.begin() + password_data_disps[i] + password_data_sizes[i]),
                    vector<int>(all_password_lengths.begin() + password_length_disps[i],
                               all_password_lengths.begin() + password_length_disps[i] + all_password_counts[i])
                );
                
                guesses.insert(guesses.end(), proc_passwords.begin(), proc_passwords.end());
                total_guesses += all_password_counts[i];
            }
        }
        
        // 处理收集到的所有新PT数据
        int offset = 0;
        while (offset < total_new_pt_data_size) {
            if (offset + sizeof(int) > total_new_pt_data_size) break;
            
            int pt_buffer_size = *(int*)(&all_new_pt_data[offset]);
            offset += sizeof(int);
            
            if (offset + pt_buffer_size > total_new_pt_data_size) break;
            
            vector<char> pt_encoded_data(all_new_pt_data.begin() + offset, 
                                        all_new_pt_data.begin() + offset + pt_buffer_size);
            offset += pt_buffer_size;
            
            PT new_pt;
            if (DataSerializer::decode_pt_with_validation(pt_encoded_data, new_pt)) {
                // 插入新PT到优先队列
                bool inserted = false;
                for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                    if (iter != priority.end() - 1 && iter != priority.begin()) {
                        if (new_pt.prob <= iter->prob && new_pt.prob > (iter + 1)->prob) {
                            priority.emplace(iter + 1, new_pt);
                            inserted = true;
                            break;
                        }
                    }
                    if (iter == priority.end() - 1) {
                        priority.emplace_back(new_pt);
                        inserted = true;
                        break;
                    }
                    if (iter == priority.begin() && iter->prob < new_pt.prob) {
                        priority.emplace(iter, new_pt);
                        inserted = true;
                        break;
                    }
                }
                
                if (!inserted && priority.empty()) {
                    priority.push_back(new_pt);
                }
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}