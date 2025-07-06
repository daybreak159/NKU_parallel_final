#include "PCFG_openmp.h"
#include <algorithm>
#include <functional>

using namespace std;

const int PriorityQueue::PARALLEL_TASK_MIN;

segment* PriorityQueue::getSegmentPtr(const segment& seg) {
    if (seg.isLetter()) {
        return &m.letters[m.FindLetter(seg)];
    } else if (seg.isDigit()) {
        return &m.digits[m.FindDigit(seg)];
    } else if (seg.isSymbol()) {
        return &m.symbols[m.FindSymbol(seg)];
    }
    return nullptr;
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int idx = 0;

    for (int currIdx : pt.curr_indices)
    {
        if (pt.content[idx].isLetter())
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.letters[m.FindLetter(pt.content[idx])].total_freq;
        }
        else if (pt.content[idx].isDigit())
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.digits[m.FindDigit(pt.content[idx])].total_freq;
        }
        else if (pt.content[idx].isSymbol())
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[idx])].ordered_freqs[currIdx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[idx])].total_freq;
        }
        idx++;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.isLetter())
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            else if (seg.isDigit())
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            else if (seg.isSymbol())
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *segPtr = getSegmentPtr(pt.content[0]);
        if (!segPtr) return;
        
        for (int i = 0; i < pt.max_indices[0]; i++)
        {
            string guess = segPtr->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses++;
        }
    }
    else
    {
        if (pt.content.empty() || pt.curr_indices.empty()) return;
        
        string prefix;
        int idx = 0;
        for (int currIdx : pt.curr_indices)
        {
            if (idx >= pt.content.size() - 1) break;
            
            if (pt.content[idx].isLetter())
            {
                prefix += m.letters[m.FindLetter(pt.content[idx])].ordered_values[currIdx];
            }
            else if (pt.content[idx].isDigit())
            {
                prefix += m.digits[m.FindDigit(pt.content[idx])].ordered_values[currIdx];
            }
            else if (pt.content[idx].isSymbol())
            {
                prefix += m.symbols[m.FindSymbol(pt.content[idx])].ordered_values[currIdx];
            }
            idx++;
        }

        segment *lastSegPtr = getSegmentPtr(pt.content.back());
        if (!lastSegPtr) return;
        
        for (int i = 0; i < pt.max_indices.back(); i++)
        {
            string fullGuess = prefix + lastSegPtr->ordered_values[i];
            guesses.emplace_back(fullGuess);
            total_guesses++;
        }
    }
}

void PriorityQueue::GenerateParallelOMP(PT pt, int num_threads)
{
    // 计算PT的概率
    CalProb(pt);

    // 设置线程数，如果未指定则使用系统默认
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // 单segment处理
    if (pt.content.size() == 1)
    {
        if (pt.content.empty() || pt.max_indices.empty()) {
            return;
        }

        segment *target_segment_data = getSegmentPtr(pt.content[0]);
        if (!target_segment_data) return;
        
        int max_idx_val = pt.max_indices[0];
        int actual_values_count = target_segment_data->ordered_values.size();
        int loop_bound = std::min(max_idx_val, actual_values_count);
        
        if (loop_bound <= 0) return;
        
        size_t result_count = 0;
        
        if (loop_bound > PARALLEL_TASK_MIN && omp_get_max_threads() > 1) {
            #pragma omp parallel
            {
                // 线程私有结果容器
                std::vector<std::string> private_results;
                private_results.reserve(loop_bound / omp_get_num_threads() + 1);
                
                #pragma omp for nowait
                for (int i = 0; i < loop_bound; ++i)
                {
                    private_results.emplace_back(target_segment_data->ordered_values[i]);
                }
                
                #pragma omp critical(MergeThreadResults)
                {
                    guesses.insert(guesses.end(), 
                                  std::make_move_iterator(private_results.begin()), 
                                  std::make_move_iterator(private_results.end()));
                    result_count += private_results.size();
                }
            }
        } else { 
            for (int i = 0; i < loop_bound; ++i) {
                guesses.emplace_back(target_segment_data->ordered_values[i]);
            }
            result_count = loop_bound;
        }
        
        total_guesses += result_count;
    }
    // 多segment处理
    else 
    {
        if (pt.content.empty() || pt.curr_indices.empty() || 
            pt.max_indices.size() != pt.content.size()) {
            return;
        }
        
        // 构建前缀
        string prefix_str;
        int seg_idx = 0;
        
        for (int idx : pt.curr_indices) 
        {
            if (seg_idx >= pt.content.size() - 1) break;
            
            const segment& current_seg = pt.content[seg_idx];
            const segment* seg_data = nullptr;
            
            if (current_seg.isLetter()) {
                seg_data = &m.letters[m.FindLetter(current_seg)];
            }
            else if (current_seg.isDigit()) {
                seg_data = &m.digits[m.FindDigit(current_seg)];
            }
            else if (current_seg.isSymbol()) {
                seg_data = &m.symbols[m.FindSymbol(current_seg)];
            }

            if (seg_data && idx < seg_data->ordered_values.size()) {
                prefix_str += seg_data->ordered_values[idx];
            } else {
                return;
            }
            
            seg_idx++;
        }
        
        // 获取最后一个segment
        int last_idx = pt.content.size() - 1;
        if (last_idx < 0) return;
        
        segment *last_seg_data = getSegmentPtr(pt.content[last_idx]);
        if (!last_seg_data) return;
        
        int max_idx_val = pt.max_indices[last_idx];
        int actual_values_count = last_seg_data->ordered_values.size();
        int loop_bound = std::min(max_idx_val, actual_values_count);
        
        if (loop_bound <= 0) return;
        
        size_t result_count = 0;
        
        if (loop_bound > PARALLEL_TASK_MIN && omp_get_max_threads() > 1) {
            #pragma omp parallel
            {
                std::vector<std::string> private_results;
                private_results.reserve(loop_bound / omp_get_num_threads() + 1);
                
                #pragma omp for nowait
                for (int i = 0; i < loop_bound; ++i)
                {
                    private_results.emplace_back(prefix_str + last_seg_data->ordered_values[i]);
                }
                
                #pragma omp critical(MergeThreadResults)
                {
                    guesses.insert(guesses.end(), 
                                  std::make_move_iterator(private_results.begin()), 
                                  std::make_move_iterator(private_results.end()));
                    result_count += private_results.size();
                }
            }
        } else { 
            for (int i = 0; i < loop_bound; ++i) {
                guesses.emplace_back(prefix_str + last_seg_data->ordered_values[i]);
            }
            result_count = loop_bound;
        }
        
        total_guesses += result_count;
    }
}

void PriorityQueue::PopNext(int num_threads)
{
    if (priority.empty()) return;

    // 使用OpenMP并行版本生成猜测
    GenerateParallelOMP(priority.front(), num_threads);

    // 生成新的PT列表
    vector<PT> newPTs = priority.front().NewPTs();
    for (PT& newPt : newPTs)
    {
        CalProb(newPt);
        
        // 按概率大小插入到队列
        auto iter = priority.begin();
        bool inserted = false;
        
        while (iter != priority.end() && !inserted)
        {
            // 中间位置
            if (iter != priority.begin() && iter != priority.end() - 1)
            {
                if (newPt.prob <= iter->prob && newPt.prob > (iter + 1)->prob)
                {
                    priority.insert(iter + 1, newPt);
                    inserted = true;
                }
            }
            // 队尾插入
            else if (iter == priority.end() - 1)
            {
                priority.push_back(newPt);
                inserted = true;
            }
            // 队首插入
            else if (iter == priority.begin() && iter->prob < newPt.prob)
            {
                priority.insert(iter, newPt);
                inserted = true;
            }
            
            if (!inserted) ++iter;
        }
    }

    // 移除首元素
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> result;

    if (content.size() <= 1)
    {
        return result;
    }
    else
    {
        int originalPivot = pivot;
        
        for (int i = pivot; i < curr_indices.size() - 1; i++)
        {
            curr_indices[i]++;
            
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                result.push_back(*this);
            }
            
            curr_indices[i]--;
        }
        
        pivot = originalPivot;
        return result;
    }
}