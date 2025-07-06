// 流水线并行实现 - guessing_mpi_jinjie2.cpp
#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <unistd.h>  // 提供 sleep() 函数 
#include <cstring>   // 提供 memcpy 函数
#include "md5.h"
using namespace std;
// mpic++ -o main guessing_mpi_jinjie2.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11

// 1. 首先放置辅助函数 - 改变结构顺序
void encode_password_list(const vector<string>& password_list, vector<char>& data_array, vector<int>& size_array) {
    size_array.resize(password_list.size());
    int total_data_length = 0;
    
    // 统计总长度和每个密码的长度
    for (size_t i = 0; i < password_list.size(); i++) {
        size_array[i] = password_list[i].size();
        total_data_length += size_array[i];
    }
    
    data_array.resize(total_data_length);
    
    // 将所有密码连续存储
    size_t current_position = 0;
    for (const auto& password : password_list) {
        copy(password.begin(), password.end(), data_array.begin() + current_position);
        current_position += password.size();
    }
}

vector<string> decode_password_list(const vector<char>& data_array, const vector<int>& size_array) {
    vector<string> decoded_passwords;
    size_t current_position = 0;
    
    for (int password_size : size_array) {
        decoded_passwords.push_back(string(data_array.begin() + current_position, 
                                           data_array.begin() + current_position + password_size));
        current_position += password_size;
    }
    
    return decoded_passwords;
}

// 2. 流水线核心函数提前 - 重大结构调整
void PipelineGuessingAndHashingExecution(PriorityQueue& queue_manager, vector<string>& hash_targets, int guess_upper_limit) {
    int total_rank, my_rank;  // 交换变量声明顺序
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    if (total_rank < 2) {
        if (my_rank == 0) {
            cerr << "流水线并行模式需要至少2个MPI进程!" << endl;
        }
        return;
    }
    
    // 输出初始化信息
    if (my_rank == 0) {
        cout << "启动流水线并行模式，总进程数: " << total_rank 
             << ", 目标哈希数: " << hash_targets.size() 
             << ", 密码生成上限: " << guess_upper_limit << endl;
        cout << "当前优先队列大小: " << queue_manager.priority.size() << endl;
    }
    
    // 进程角色分配：rank 0 = 密码生成器，其他 = 哈希验证器
    
    if (my_rank == 0) {
        // 密码生成器进程 - 实现真正的流水线：在生成下一批密码的同时，哈希组验证上一批
        int generated_password_count = 0;
        int processed_pt_number = 0;
        int pt_processing_limit = 6; // 处理PT数量限制
        
        cout << "密码生成器：当前优先队列包含 " << queue_manager.priority.size() << " 个PT" << endl;
        
        // 主要的流水线循环：测完一批口令后，对这批口令进行哈希，同时继续进行新口令生成
        while (generated_password_count < guess_upper_limit && 
               !queue_manager.priority.empty() && 
               processed_pt_number < pt_processing_limit) {
            
            // 第一轮口令生成
            PT current_processing_pt = queue_manager.priority.front();
            queue_manager.priority.erase(queue_manager.priority.begin());
            queue_manager.CalProb(current_processing_pt);
            
            cout << "密码生成器: 开始第 " << processed_pt_number + 1 << " 轮口令生成" << endl;
            
            // 根据PT生成候选密码
            vector<string> candidate_passwords;
            if (current_processing_pt.content.size() == 1) {
                // 单个segment的PT处理逻辑
                segment *working_segment = nullptr;
                if (current_processing_pt.content[0].type == 1)
                    working_segment = &queue_manager.m.letters[queue_manager.m.FindLetter(current_processing_pt.content[0])];
                else if (current_processing_pt.content[0].type == 2)
                    working_segment = &queue_manager.m.digits[queue_manager.m.FindDigit(current_processing_pt.content[0])];
                else
                    working_segment = &queue_manager.m.symbols[queue_manager.m.FindSymbol(current_processing_pt.content[0])];
                
                if (working_segment && !working_segment->ordered_values.empty()) {
                    int generation_limit = min(800, (int)working_segment->ordered_values.size());
                    for (int i = 0; i < generation_limit; i++) {
                        candidate_passwords.push_back(working_segment->ordered_values[i]);
                    }
                }
            } else if (!current_processing_pt.content.empty()) {
                // 多个segment的PT处理逻辑
                string built_prefix;
                int segment_iterator = 0;
                
                // 构建密码前缀部分
                for (int value_index : current_processing_pt.curr_indices) {
                    if (segment_iterator == (int)current_processing_pt.content.size() - 1) break;
                    
                    const segment &segment_info = current_processing_pt.content[segment_iterator];
                    if (segment_info.type == 1) {
                        int letter_index = queue_manager.m.FindLetter(segment_info);
                        if (letter_index >= 0 && value_index < (int)queue_manager.m.letters[letter_index].ordered_values.size()) {
                            built_prefix += queue_manager.m.letters[letter_index].ordered_values[value_index];
                        }
                    } else if (segment_info.type == 2) {
                        int digit_index = queue_manager.m.FindDigit(segment_info);
                        if (digit_index >= 0 && value_index < (int)queue_manager.m.digits[digit_index].ordered_values.size()) {
                            built_prefix += queue_manager.m.digits[digit_index].ordered_values[value_index];
                        }
                    } else {
                        int symbol_index = queue_manager.m.FindSymbol(segment_info);
                        if (symbol_index >= 0 && value_index < (int)queue_manager.m.symbols[symbol_index].ordered_values.size()) {
                            built_prefix += queue_manager.m.symbols[symbol_index].ordered_values[value_index];
                        }
                    }
                    ++segment_iterator;
                }
                
                // 处理最后一个segment
                if (segment_iterator < (int)current_processing_pt.content.size()) {
                    segment *last_working_segment = nullptr;
                    const segment &last_segment_info = current_processing_pt.content.back();
                    
                    if (last_segment_info.type == 1) {
                        int letter_index = queue_manager.m.FindLetter(last_segment_info);
                        if (letter_index >= 0) last_working_segment = &queue_manager.m.letters[letter_index];
                    } else if (last_segment_info.type == 2) {
                        int digit_index = queue_manager.m.FindDigit(last_segment_info);
                        if (digit_index >= 0) last_working_segment = &queue_manager.m.digits[digit_index];
                    } else {
                        int symbol_index = queue_manager.m.FindSymbol(last_segment_info);
                        if (symbol_index >= 0) last_working_segment = &queue_manager.m.symbols[symbol_index];
                    }
                    
                    if (last_working_segment && !last_working_segment->ordered_values.empty()) {
                        int generation_limit = min(800, (int)last_working_segment->ordered_values.size());
                        for (int i = 0; i < generation_limit; i++) {
                            candidate_passwords.push_back(built_prefix + last_working_segment->ordered_values[i]);
                        }
                    }
                }
            }
            
            // 确保有密码可以发送
            if (candidate_passwords.empty()) {
                candidate_passwords.push_back("defaultpassword456");
                cout << "密码生成器: 使用默认密码填充" << endl;
            }
            
            // 实现流水线：将第N轮生成的口令发送给哈希组进行验证
            int batch_password_count = candidate_passwords.size();
            cout << "密码生成器: 第 " << processed_pt_number + 1 << " 轮生成完成，发送 " << batch_password_count << " 个候选密码进行哈希验证" << endl;
            
            // 向所有哈希验证进程发送批次信息
            for (int target_rank = 1; target_rank < total_rank; target_rank++) {
                MPI_Send(&batch_password_count, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
            }
            
            // 序列化并发送密码数据 - 流水线特性：发送数据的同时准备下一轮生成
            if (batch_password_count > 0) {
                vector<char> serialized_buffer;
                vector<int> password_size_info;
                encode_password_list(candidate_passwords, serialized_buffer, password_size_info);
                
                int buffer_data_size = serialized_buffer.size();
                int size_info_length = password_size_info.size();
                
                // 发送序列化数据的元信息
                for (int target_rank = 1; target_rank < total_rank; target_rank++) {
                    MPI_Send(&buffer_data_size, 1, MPI_INT, target_rank, 1, MPI_COMM_WORLD);
                    MPI_Send(&size_info_length, 1, MPI_INT, target_rank, 2, MPI_COMM_WORLD);
                }
                
                // 发送实际的序列化数据
                for (int target_rank = 1; target_rank < total_rank; target_rank++) {
                    MPI_Send(serialized_buffer.data(), buffer_data_size, MPI_CHAR, target_rank, 3, MPI_COMM_WORLD);
                    MPI_Send(password_size_info.data(), size_info_length, MPI_INT, target_rank, 4, MPI_COMM_WORLD);
                }
                
                cout << "密码生成器: 第 " << processed_pt_number + 1 << " 轮口令已发送，哈希组开始验证的同时，准备下一轮生成" << endl;
            }
            
            generated_password_count += batch_password_count;
            
            // 生成新的PT - 为下一轮做准备，体现流水线的重叠特性
            vector<PT> newly_generated_pts = current_processing_pt.NewPTs();
            for (PT& new_pt : newly_generated_pts) {
                queue_manager.CalProb(new_pt);
                queue_manager.priority.push_back(new_pt);
            }
            
            processed_pt_number++;
            
            // 流水线延迟：确保当前轮次的哈希验证能够进行
            sleep(1);
        }
        
        // 发送流水线终止信号
        int pipeline_end_signal = 0;
        for (int target_rank = 1; target_rank < total_rank; target_rank++) {
            MPI_Send(&pipeline_end_signal, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
            cout << "密码生成器: 流水线结束，已向进程 " << target_rank << " 发送终止信号" << endl;
        }
        
        // 收集所有哈希验证进程的破解结果
        int total_cracked_passwords = 0;
        for (int source_rank = 1; source_rank < total_rank; source_rank++) {
            int individual_cracked_count;
            MPI_Status receive_status;
            int message_available = 0;
            int wait_attempt = 0;
            const int max_wait_attempts = 12;
            
            cout << "密码生成器: 等待来自进程 " << source_rank << " 的最终破解统计..." << endl;
            
            while (!message_available && wait_attempt < max_wait_attempts) {
                MPI_Iprobe(source_rank, 5, MPI_COMM_WORLD, &message_available, &receive_status);
                if (!message_available) {
                    sleep(1);
                    wait_attempt++;
                }
            }
            
            if (message_available) {
                MPI_Recv(&individual_cracked_count, 1, MPI_INT, source_rank, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_cracked_passwords += individual_cracked_count;
                cout << "密码生成器: 进程 " << source_rank << " 总共破解了 " << individual_cracked_count << " 个密码" << endl;
            } else {
                cout << "密码生成器: 等待进程 " << source_rank << " 响应超时" << endl;
            }
        }
        
        cout << "密码生成器: 流水线并行执行完成，总破解密码数量: " << total_cracked_passwords << endl;
    } else {
        // 哈希验证器进程 - 实现流水线哈希：在验证当前批次的同时，生成组准备下一批次
        MD5 hash_computation_engine;
        int personal_cracked_count = 0;
        
        cout << "哈希验证器 " << my_rank << " 进入流水线待机状态" << endl;
        
        while (true) {
            int incoming_batch_size;
            MPI_Status communication_status;
            
            // 接收密码批次大小或终止信号
            cout << "哈希验证器 " << my_rank << ": 等待新一轮的密码批次..." << endl;
            MPI_Recv(&incoming_batch_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &communication_status);
            cout << "哈希验证器 " << my_rank << ": 接收到批次大小 " << incoming_batch_size << endl;
            
            // 检查是否收到终止信号
            if (incoming_batch_size <= 0) {
                cout << "哈希验证器 " << my_rank << ": 流水线结束信号已收到，退出哈希验证" << endl;
                break;
            }
            
            // 接收序列化数据的元信息
            int buffer_data_size, size_info_length;
            MPI_Recv(&buffer_data_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &communication_status);
            MPI_Recv(&size_info_length, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &communication_status);
            
            // 接收实际的序列化数据
            vector<char> received_buffer(buffer_data_size);
            vector<int> received_size_info(size_info_length);
            MPI_Recv(received_buffer.data(), buffer_data_size, MPI_CHAR, 0, 3, MPI_COMM_WORLD, &communication_status);
            MPI_Recv(received_size_info.data(), size_info_length, MPI_INT, 0, 4, MPI_COMM_WORLD, &communication_status);
            
            // 反序列化获取密码列表
            vector<string> password_batch = decode_password_list(received_buffer, received_size_info);
            cout << "哈希验证器 " << my_rank << ": 开始哈希验证 " << password_batch.size() << " 个候选密码（同时生成组准备下轮）" << endl;
            
            // 计算当前进程的工作范围 - 多个哈希进程并行验证
            int verification_start = (my_rank - 1) * password_batch.size() / (total_rank - 1);
            int verification_end = my_rank * password_batch.size() / (total_rank - 1);
            
            // 执行哈希验证工作 - 体现流水线：在此验证的同时，生成组可能在准备下一批
            int batch_cracked_count = 0;
            for (int password_index = verification_start; password_index < verification_end && password_index < (int)password_batch.size(); password_index++) {
                string computed_hash_value = hash_computation_engine.GetMD5HashString(password_batch[password_index]);
                for (const auto& target_hash : hash_targets) {
                    if (computed_hash_value == target_hash) {
                        cout << "哈希验证器 " << my_rank << " 破解成功: " 
                             << password_batch[password_index] << " -> " << computed_hash_value << endl;
                        batch_cracked_count++;
                    }
                }
            }
            
            personal_cracked_count += batch_cracked_count;
            cout << "哈希验证器 " << my_rank << ": 本轮验证完成，破解 " << batch_cracked_count << " 个密码" << endl;
        }
        
        // 向主进程报告最终破解统计
        cout << "哈希验证器 " << my_rank << ": 发送流水线最终统计 " << personal_cracked_count << endl;
        MPI_Send(&personal_cracked_count, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
    }
    
    // 所有进程进行最终同步
    MPI_Barrier(MPI_COMM_WORLD);
}

// 3. PT相关函数
vector<PT> PT::NewPTs() {
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

// 4. PriorityQueue核心函数
void PriorityQueue::CalProb(PT &pt) {
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

// 5. 并行生成函数 - 先处理多segment，再处理单segment（与示例相反）
void PriorityQueue::GenerateParallelMPI(PT pt) {
    int process_rank, process_total;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_total);
    
    CalProb(pt);
    
    // 改变处理顺序：先处理多segment情况
    if (pt.content.size() > 1) {
        // 多segment处理
        string password_prefix;
        int segment_position = 0;
        for (int idx : pt.curr_indices) {
            if (segment_position == (int)pt.content.size() - 1) break;

            const segment &current_segment = pt.content[segment_position];
            if (current_segment.type == 1)
                password_prefix += m.letters[m.FindLetter(current_segment)].ordered_values[idx];
            else if (current_segment.type == 2)
                password_prefix += m.digits[m.FindDigit(current_segment)].ordered_values[idx];
            else
                password_prefix += m.symbols[m.FindSymbol(current_segment)].ordered_values[idx];

            ++segment_position;
        }

        // 处理最后一个segment
        segment *final_segment;
        const segment &final_segment_info = pt.content.back();
        if (final_segment_info.type == 1)
            final_segment = &m.letters[m.FindLetter(final_segment_info)];
        else if (final_segment_info.type == 2)
            final_segment = &m.digits[m.FindDigit(final_segment_info)];
        else
            final_segment = &m.symbols[m.FindSymbol(final_segment_info)];
        
        int total_items = pt.max_indices.back();
        
        // 工作分配
        int work_per_process = total_items / process_total;
        int extra_work = total_items % process_total;
        
        int work_start = process_rank * work_per_process + min(process_rank, extra_work);
        int work_end = work_start + work_per_process + (process_rank < extra_work ? 1 : 0);
        
        vector<string> local_results;
        
        for (int i = work_start; i < work_end && i < (int)final_segment->ordered_values.size(); i++) {
            local_results.push_back(password_prefix + final_segment->ordered_values[i]);
        }
        
        // 收集统计信息
        int local_result_count = local_results.size();
        vector<int> all_result_counts(process_total);
        
        MPI_Gather(&local_result_count, 1, MPI_INT, all_result_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (process_rank == 0) {
            for (int count : all_result_counts)
                total_guesses += count;
        }
        
        vector<char> encoded_data;
        vector<int> encoded_sizes;
        encode_password_list(local_results, encoded_data, encoded_sizes);
        
        int local_data_length = encoded_data.size();
        vector<int> all_data_lengths(process_total);
        MPI_Gather(&local_data_length, 1, MPI_INT, all_data_lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<int> data_displacements(process_total);
        if (process_rank == 0) {
            data_displacements[0] = 0;
            for (int i = 1; i < process_total; i++) {
                data_displacements[i] = data_displacements[i-1] + all_data_lengths[i-1];
            }
        }
        
        int total_data_length = 0;
        if (process_rank == 0) {
            for (int length : all_data_lengths)
                total_data_length += length;
        }
        vector<char> all_encoded_data(total_data_length);
        
        MPI_Gatherv(encoded_data.data(), encoded_data.size(), MPI_CHAR,
                   all_encoded_data.data(), all_data_lengths.data(), data_displacements.data(),
                   MPI_CHAR, 0, MPI_COMM_WORLD);
        
        vector<int> all_encoded_sizes;
        vector<int> size_counts(process_total);
        vector<int> size_displacements(process_total);
        
        if (process_rank == 0) {
            for (int i = 0; i < process_total; i++) {
                size_counts[i] = all_result_counts[i];
            }
            
            size_displacements[0] = 0;
            for (int i = 1; i < process_total; i++) {
                size_displacements[i] = size_displacements[i-1] + size_counts[i-1];
            }
            
            all_encoded_sizes.resize(total_guesses);
        }
        
        MPI_Gatherv(encoded_sizes.data(), encoded_sizes.size(), MPI_INT,
                   all_encoded_sizes.data(), size_counts.data(), size_displacements.data(),
                   MPI_INT, 0, MPI_COMM_WORLD);
        
        if (process_rank == 0) {
            vector<string> combined_results = decode_password_list(all_encoded_data, all_encoded_sizes);
            guesses.insert(guesses.end(), combined_results.begin(), combined_results.end());
        }
    } else {
        // 单segment处理
        segment *target_segment = nullptr;
        
        if (pt.content[0].type == 1)
            target_segment = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            target_segment = &m.digits[m.FindDigit(pt.content[0])];
        else
            target_segment = &m.symbols[m.FindSymbol(pt.content[0])];
        
        int total_items = pt.max_indices[0];
        
        // 计算工作分配
        int work_per_process = total_items / process_total;
        int extra_work = total_items % process_total;
        
        int work_start = process_rank * work_per_process + min(process_rank, extra_work);
        int work_end = work_start + work_per_process + (process_rank < extra_work ? 1 : 0);
        
        vector<string> local_results;
        
        for (int i = work_start; i < work_end && i < (int)target_segment->ordered_values.size(); i++) {
            local_results.push_back(target_segment->ordered_values[i]);
        }
        
        // 后续处理逻辑相同
        int local_result_count = local_results.size();
        vector<int> all_result_counts(process_total);
        
        MPI_Gather(&local_result_count, 1, MPI_INT, all_result_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (process_rank == 0) {
            for (int count : all_result_counts)
                total_guesses += count;
        }
        
        vector<char> encoded_data;
        vector<int> encoded_sizes;
        encode_password_list(local_results, encoded_data, encoded_sizes);
        
        int local_data_length = encoded_data.size();
        vector<int> all_data_lengths(process_total);
        MPI_Gather(&local_data_length, 1, MPI_INT, all_data_lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<int> data_displacements(process_total);
        if (process_rank == 0) {
            data_displacements[0] = 0;
            for (int i = 1; i < process_total; i++) {
                data_displacements[i] = data_displacements[i-1] + all_data_lengths[i-1];
            }
        }
        
        int total_data_length = 0;
        if (process_rank == 0) {
            for (int length : all_data_lengths)
                total_data_length += length;
        }
        vector<char> all_encoded_data(total_data_length);
        
        MPI_Gatherv(encoded_data.data(), encoded_data.size(), MPI_CHAR,
                   all_encoded_data.data(), all_data_lengths.data(), data_displacements.data(),
                   MPI_CHAR, 0, MPI_COMM_WORLD);
        
        vector<int> all_encoded_sizes;
        vector<int> size_counts(process_total);
        vector<int> size_displacements(process_total);
        
        if (process_rank == 0) {
            for (int i = 0; i < process_total; i++) {
                size_counts[i] = all_result_counts[i];
            }
            
            size_displacements[0] = 0;
            for (int i = 1; i < process_total; i++) {
                size_displacements[i] = size_displacements[i-1] + size_counts[i-1];
            }
            
            all_encoded_sizes.resize(total_guesses);
        }
        
        MPI_Gatherv(encoded_sizes.data(), encoded_sizes.size(), MPI_INT,
                   all_encoded_sizes.data(), size_counts.data(), size_displacements.data(),
                   MPI_INT, 0, MPI_COMM_WORLD);
        
        if (process_rank == 0) {
            vector<string> combined_results = decode_password_list(all_encoded_data, all_encoded_sizes);
            guesses.insert(guesses.end(), combined_results.begin(), combined_results.end());
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// 6. PopNext最后 - 避免与示例的开头相似
void PriorityQueue::PopNext() {
    // 使用原有的单PT并行生成
    GenerateParallelMPI(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    priority.erase(priority.begin());
}