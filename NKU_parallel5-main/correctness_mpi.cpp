#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令：mpic++ correctness_mpi.cpp train.cpp guessing_mpi_basic.cpp md5.cpp -o cor -O2
// mpirun -np 2 ./main
int main(int argc, char* argv[])
{
    // MPI环境初始化
    MPI_Init(&argc, &argv);
    
    // 获取进程信息
    int proc_rank, proc_total;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_total);
    
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    double local_train_duration = 0;
    
    PriorityQueue q;
    
    // 训练阶段 - 进程同步
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_train = system_clock::now();
    
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    local_train_duration = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    // 收集训练时间的最大值
    MPI_Allreduce(&local_train_duration, &time_train, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    // 加载测试数据（与原代码相同）
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while(test_data >> pw)
    {   
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000)
        {
            break;
        }
    }
    
    // MPI特有变量：分布式破解计数
    int distributed_cracked_count = 0;

    q.init();
    if (proc_rank == 0) {
        cout << "here" << endl;
    }
    
    int curr_num = 0;
    
    // 所有进程同步开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = system_clock::now();
    
    int history = 0;
    int synchronized_history = 0;
    
    // 主执行循环 - MPI协调版本
    while (true)
    {
        // 队列状态检查 - 分布式同步
        int queue_empty_flag = q.priority.empty() ? 1 : 0;
        int global_empty_status = 0;
        MPI_Allreduce(&queue_empty_flag, &global_empty_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_empty_status == 1) break;
        
        // MPI并行密码生成
        q.PopNext();
        
        // 密码数量统计 - 跨进程汇总
        int local_guess_count = q.guesses.size();
        int aggregated_guess_count = 0;
        MPI_Allreduce(&local_guess_count, &aggregated_guess_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        q.total_guesses = aggregated_guess_count;
        
        // 进度报告 - 仅主进程输出
        if (proc_rank == 0) {
            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << synchronized_history + q.total_guesses << endl;
                curr_num = q.total_guesses;
            }
        }
        
        // 终止条件判断 - 主进程决策
        int termination_signal = 0;
        if (proc_rank == 0 && synchronized_history + q.total_guesses > 10000000) {
            termination_signal = 1;
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        }
        
        // 广播终止信号
        MPI_Bcast(&termination_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (termination_signal) {
            // 最终破解数收集
            int total_cracked_final = 0;
            MPI_Reduce(&distributed_cracked_count, &total_cracked_final, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (proc_rank == 0) {
                cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time:" << time_hash << " seconds" << endl;
                cout << "Train time:" << time_train << " seconds" << endl;
                cout << "Cracked:" << total_cracked_final << endl;
            }
            break;
        }
        
        // 批量哈希处理 - 内存管理
        int processing_trigger = 0;
        if (curr_num > 1000000) {
            processing_trigger = 1;
        }
        
        MPI_Bcast(&processing_trigger, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (processing_trigger) {
            double local_hash_duration = 0;
            auto start_hash = system_clock::now();
            
            bit32 state[4];
            int batch_cracked = 0;
            
            // 密码验证和哈希计算
            for (const string& password : q.guesses)
            {
                if (test_set.find(password) != test_set.end()) {
                    batch_cracked += 1;
                }
                MD5Hash(password, state);
            }
            
            // 累积到进程总数
            distributed_cracked_count += batch_cracked;
            
            auto end_hash = system_clock::now();
            auto hash_duration = duration_cast<microseconds>(end_hash - start_hash);
            local_hash_duration = double(hash_duration.count()) * microseconds::period::num / microseconds::period::den;
            
            // 哈希时间同步
            double max_hash_duration = 0;
            MPI_Reduce(&local_hash_duration, &max_hash_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            
            if (proc_rank == 0) {
                time_hash += max_hash_duration;
                history += curr_num;
                synchronized_history = history;
            }
            
            // 历史记录同步
            MPI_Bcast(&synchronized_history, 1, MPI_INT, 0, MPI_COMM_WORLD);
            history = synchronized_history;
            
            // curr_num重置同步
            if (proc_rank == 0) {
                curr_num = 0;
            }
            MPI_Bcast(&curr_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            q.guesses.clear();
        }
    }
    
    // 程序结束处理
    if (proc_rank == 0) {
        cout << "程序正常结束" << endl;
    }
    
    // 最终结果汇总
    int final_result_summary = 0;
    MPI_Reduce(&distributed_cracked_count, &final_result_summary, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (proc_rank == 0) {
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "最终结果:" << endl;
        cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
        cout << "Hash time:" << time_hash << " seconds" << endl;
        cout << "Train time:" << time_train << " seconds" << endl;
        cout << "Cracked:" << final_result_summary << endl;
    }
    
    MPI_Finalize();
    return 0;
}