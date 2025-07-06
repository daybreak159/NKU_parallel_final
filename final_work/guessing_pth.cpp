#include "PCFG_pth.h"
#include <pthread.h>
#include <queue>
#include <functional>
#include <algorithm>

using namespace std;

// 初始化静态成员
ThreadPoolManager* ThreadPoolManager::currentInstance = nullptr;

// 静态清理函数
void ThreadPoolManager::cleanupThreadPool() {
    if (currentInstance) {
        pthread_mutex_lock(&currentInstance->taskMutex);
        currentInstance->shouldTerminate = true;
        pthread_cond_broadcast(&currentInstance->taskCond);
        pthread_mutex_unlock(&currentInstance->taskMutex);

        for (int i = 0; i < 4; i++) {
            pthread_join(currentInstance->workers[i], NULL);
        }

        pthread_mutex_destroy(&currentInstance->taskMutex);
        pthread_cond_destroy(&currentInstance->taskCond);
        pthread_mutex_destroy(&currentInstance->completionMutex);
        pthread_cond_destroy(&currentInstance->completionCond);
    }
}

void* ThreadPoolManager::workerFunction(void* arg) {
    ThreadPoolManager* pool = static_cast<ThreadPoolManager*>(arg);
    
    while (true) {
        Task currentTask;
        bool hasTask = false;

        pthread_mutex_lock(&pool->taskMutex);
        while (pool->taskQueue.empty() && !pool->shouldTerminate) {
            pthread_cond_wait(&pool->taskCond, &pool->taskMutex);
        }

        if (pool->shouldTerminate && pool->taskQueue.empty()) {
            pthread_mutex_unlock(&pool->taskMutex);
            break;
        }

        if (!pool->taskQueue.empty()) {
            currentTask = pool->taskQueue.front();
            pool->taskQueue.pop();
            hasTask = true;
        }
        pthread_mutex_unlock(&pool->taskMutex);

        if (hasTask) {
            currentTask.execute();

            pthread_mutex_lock(&pool->completionMutex);
            pool->runningTasks--;
            if (pool->runningTasks == 0) {
                pthread_cond_signal(&pool->completionCond);
            }
            pthread_mutex_unlock(&pool->completionMutex);
        }
    }
    return NULL;
}

void ThreadPoolManager::initialize() {
    if (isInitialized) return;

    pthread_mutex_init(&taskMutex, NULL);
    pthread_cond_init(&taskCond, NULL);
    pthread_mutex_init(&completionMutex, NULL);
    pthread_cond_init(&completionCond, NULL);

    for (int i = 0; i < 4; i++) {
        pthread_create(&workers[i], NULL, workerFunction, this);
    }

    isInitialized = true;
    shouldTerminate = false;
    
    // 设置当前实例引用并注册静态函数用于清理
    currentInstance = this;
    atexit(cleanupThreadPool);
}

void ThreadPoolManager::addTask(std::function<void()> func) {
    Task newTask;
    newTask.execute = func;
    
    pthread_mutex_lock(&taskMutex);
    taskQueue.push(newTask);
    runningTasks++;
    pthread_cond_signal(&taskCond);
    pthread_mutex_unlock(&taskMutex);
}

void ThreadPoolManager::waitForCompletion() {
    pthread_mutex_lock(&completionMutex);
    if (runningTasks > 0) {
        pthread_cond_wait(&completionCond, &completionMutex);
    }
    pthread_mutex_unlock(&completionMutex);
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
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

void PriorityQueue::GenerateParallel(PT pt)
{
    threadPool.initialize();
    CalProb(pt);

    // 单段PT处理
    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // 获取工作量和线程数
        int work_size = pt.max_indices[0];
        int num_threads = min(4, max(1, work_size / 5000));

        // 小任务直接串行处理
        if (work_size <= 1000) {
            for (int i = 0; i < work_size; i++) {
                string guess = a->ordered_values[i];
                guesses.push_back(guess);
                total_guesses++;
            }
            return;
        }

        // 预分配结果空间
        guesses.reserve(guesses.size() + work_size);
        pthread_mutex_t result_mutex;
        pthread_mutex_init(&result_mutex, NULL);

        // 分配工作
        int items_per_thread = (work_size + num_threads - 1) / num_threads;

        // 创建任务
        for (int t = 0; t < num_threads; t++) {
            int start = t * items_per_thread;
            int end = min(start + items_per_thread, work_size);
            
            if (start >= end) continue;

            // 创建任务闭包
            auto task = [this, a, start, end, &result_mutex]() {
                vector<string> local_results;
                local_results.reserve(end - start);
                
                // 本地处理
                for (int i = start; i < end; i++) {
                    local_results.push_back(a->ordered_values[i]);
                }
                
                // 合并结果
                pthread_mutex_lock(&result_mutex);
                this->guesses.insert(this->guesses.end(),
                                   make_move_iterator(local_results.begin()),
                                   make_move_iterator(local_results.end()));
                this->total_guesses += local_results.size();
                pthread_mutex_unlock(&result_mutex);
            };
            
            threadPool.addTask(task);
        }
        
        // 等待完成
        threadPool.waitForCompletion();
        pthread_mutex_destroy(&result_mutex);
    }
    // 多段PT处理
    else
    {
        string guess;
        int seg_idx = 0;
        
        // 构建前缀
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        
        // 获取最后一个segment
        segment *a = nullptr;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // 获取工作量和线程数
        int work_size = pt.max_indices[pt.content.size() - 1];
        int num_threads = min(4, max(1, work_size / 5000));

        // 小任务直接串行处理
        if (work_size <= 1000) {
            for (int i = 0; i < work_size; i++) {
                string temp = guess + a->ordered_values[i];
                guesses.push_back(temp);
                total_guesses++;
            }
            return;
        }
        
        // 预分配结果空间
        guesses.reserve(guesses.size() + work_size);
        pthread_mutex_t result_mutex;
        pthread_mutex_init(&result_mutex, NULL);

        // 分配工作
        int items_per_thread = (work_size + num_threads - 1) / num_threads;
        
        // 创建任务
        for (int t = 0; t < num_threads; t++) {
            int start = t * items_per_thread;
            int end = min(start + items_per_thread, work_size);
            
            if (start >= end) continue;
            
            // 创建任务闭包
            auto task = [this, a, start, end, guess, &result_mutex]() {
                vector<string> local_results;
                local_results.reserve(end - start);
                
                // 本地处理
                for (int i = start; i < end; i++) {
                    local_results.push_back(guess + a->ordered_values[i]);
                }
                
                // 合并结果
                pthread_mutex_lock(&result_mutex);
                this->guesses.insert(this->guesses.end(),
                                   make_move_iterator(local_results.begin()),
                                   make_move_iterator(local_results.end()));
                this->total_guesses += local_results.size();
                pthread_mutex_unlock(&result_mutex);
            };
            
            threadPool.addTask(task);
        }
        
        // 等待完成
        threadPool.waitForCompletion();
        pthread_mutex_destroy(&result_mutex);
    }
}

void PriorityQueue::PopNext()
{
    if (priority.empty()) return;

    // 使用并行版本生成当前队列头元素的所有猜测
    GenerateParallel(priority.front());

    // 根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
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

    // 将队首PT出队
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
}