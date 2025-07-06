#include "PCFG.h"      
#include <vector>      
#include <iostream>    
#include <iterator>    
#include <omp.h>       
#include <algorithm>   

using namespace std; 

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;

    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            if (m.letters[m.FindLetter(pt.content[index])].total_freq == 0) {
                pt.prob = 0;
                break;
            }
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 2)
        {
            if (m.digits[m.FindDigit(pt.content[index])].total_freq == 0) {
                pt.prob = 0;
                break;
            }
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 3)
        {
            if (m.symbols[m.FindSymbol(pt.content[index])].total_freq == 0) {
                pt.prob = 0;
                break;
            }
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        else {
            pt.prob = 0;
            break;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            else if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            else if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
            else {
                cerr << "Warning: Unknown segment type in init for PT: "; pt.PrintPT(); cerr << endl;
                pt.max_indices.emplace_back(0);
            }
        }

        int pt_index = m.FindPT(pt);
        if (pt_index != -1 && m.preterm_freq.count(pt_index)) {
            if (m.total_preterm == 0) {
                pt.preterm_prob = 0.0f;
                cerr << "Warning: total_preterm is zero during init." << endl;
            } else {
                pt.preterm_prob = static_cast<float>(m.preterm_freq[pt_index]) / m.total_preterm;
            }
        } else {
            pt.preterm_prob = 0.0f;
        }

        CalProb(pt);
        priority.emplace_back(std::move(pt));
    }

    std::sort(priority.begin(), priority.end(), [](const PT& a, const PT& b) {
        return a.prob > b.prob;
    });
}

void PriorityQueue::PopNext()
{
    if (priority.empty()) {
        return;
    }

    Generate(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();

    for (PT pt : new_pts)
    {
        CalProb(pt);
        bool inserted = false;
        auto it = priority.begin();
        while (it != priority.end()) {
            if (pt.prob > it->prob) {
                priority.emplace(it, std::move(pt));
                inserted = true;
                break;
            }
            ++it;
        }
        if (!inserted) {
            priority.emplace_back(std::move(pt));
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

void PriorityQueue::Generate(PT pt)
{
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
        else {
            cerr << "Error: Invalid segment type or segment not found in model." << endl;
            return;
        }

        std::vector<std::string> local_guesses_combined;
        long long local_total_guesses = 0;

        #pragma omp parallel reduction(+:local_total_guesses) default(none) firstprivate(a) shared(pt, local_guesses_combined, cerr)
        {
            std::vector<std::string> local_guesses_private;
            #pragma omp for nowait
            for (int i = 0; i < pt.max_indices[0]; i += 1)
            {
                string guess = a->ordered_values[i];
                local_guesses_private.emplace_back(std::move(guess));
                local_total_guesses += 1;
            }

            #pragma omp critical
            {
                local_guesses_combined.insert(local_guesses_combined.end(),
                                              std::make_move_iterator(local_guesses_private.begin()),
                                              std::make_move_iterator(local_guesses_private.end()));
            }
        }

        guesses.insert(guesses.end(),
                       std::make_move_iterator(local_guesses_combined.begin()),
                       std::make_move_iterator(local_guesses_combined.end()));
        total_guesses += local_total_guesses;

    }
    else
    {
        string guess_prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2)
            {
                guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3)
            {
                guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            else {
                cerr << "Error: Invalid segment type or segment not found in model during prefix generation." << endl;
                return;
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a = nullptr;
        const size_t last_seg_idx = pt.content.size() - 1;
        if (pt.content[last_seg_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_seg_idx])];
        }
        else if (pt.content[last_seg_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_seg_idx])];
        }
        else if (pt.content[last_seg_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_seg_idx])];
        }
        else {
            cerr << "Error: Invalid segment type or segment not found in model for the last segment." << endl;
            return;
        }

        std::vector<std::string> local_guesses_combined;
        long long local_total_guesses = 0;

        #pragma omp parallel reduction(+:local_total_guesses) default(none) firstprivate(a, guess_prefix) shared(pt, last_seg_idx, local_guesses_combined, cerr)
        {
            std::vector<std::string> local_guesses_private;
            #pragma omp for nowait
            for (int i = 0; i < pt.max_indices[last_seg_idx]; i += 1)
            {
                string temp = guess_prefix + a->ordered_values[i];
                local_guesses_private.emplace_back(std::move(temp));
                local_total_guesses += 1;
            }

            #pragma omp critical
            {
                local_guesses_combined.insert(local_guesses_combined.end(),
                                              std::make_move_iterator(local_guesses_private.begin()),
                                              std::make_move_iterator(local_guesses_private.end()));
            }
        }

        guesses.insert(guesses.end(),
                       std::make_move_iterator(local_guesses_combined.begin()),
                       std::make_move_iterator(local_guesses_combined.end()));
        total_guesses += local_total_guesses;
    }
}
