#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include <cstdlib>

#include <stdexcept>
#include <new>

using namespace std;
using namespace std::chrono;


void dual_output(std::ofstream& file, const std::string& message) {
    std::cout << message << std::endl;
    std::cout.flush();
    file << message << std::endl;
    file.flush();
}

int main()
{

    std::ofstream output_file("./files/results.txt");
    if (!output_file.is_open()) {

        #ifdef __cpp_lib_filesystem
            #include <filesystem>
            try {
                std::filesystem::create_directory("./files");
                output_file.open("./files/results.txt");
                if (!output_file.is_open()) {
                     std::cerr << "Error: Could not create ./files directory or open results.txt." << std::endl;
                     return 1;
                }
            } catch (const std::exception& e) {
                 std::cerr << "Error creating ./files directory: " << e.what() << std::endl;
                 return 1;
            }
        #else

            std::cerr << "Error: Could not open ./files/results.txt. Ensure ./files directory exists." << std::endl;
            return 1;
        #endif
    }


    double time_hash_serial = 0;
    double time_hash_parallel = 0;
    double time_guess_generation = 0;
    double time_train = 0;
    PriorityQueue q;


    auto start_train = high_resolution_clock::now();
    dual_output(output_file, "Training...");
    dual_output(output_file, "Training phase 1: reading and parsing passwords...");
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    dual_output(output_file, "Training phase 2: Ordering segment values and PTs...");
    q.m.order();
    auto end_train = high_resolution_clock::now();
    auto duration_train_ns = duration_cast<nanoseconds>(end_train - start_train);
    time_train = duration_train_ns.count() * 1e-9;
    dual_output(output_file, "Train time: " + std::to_string(time_train) + " seconds");


    q.init();
    dual_output(output_file, "Initialization complete. Starting guessing...");
    int curr_num = 0;
    int history = 0;
    auto start_guess_hash_loop = high_resolution_clock::now();


    const size_t MAX_BATCH_SIZE = 1500000;
    const size_t alignment = 16;
    const size_t MAX_PADDED_LEN_PER_MSG = 256;

    bit32** parallel_results = nullptr;
    Byte** padded_message_pointers = nullptr;
    Byte* padded_message_pool = nullptr;
    int* messageLengths = nullptr;
    bool prealloc_ok = true;

    try {
        parallel_results = new bit32*[MAX_BATCH_SIZE];
        for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, 4 * sizeof(bit32)) != 0) {
                throw std::runtime_error("Memory pre-allocation failed for parallel_results.");
            }
            parallel_results[i] = static_cast<bit32*>(ptr);
        }

        padded_message_pointers = new Byte*[MAX_BATCH_SIZE];

        void* pool_ptr = nullptr;
        if (posix_memalign(&pool_ptr, alignment, MAX_BATCH_SIZE * MAX_PADDED_LEN_PER_MSG) != 0) {
             throw std::runtime_error("Memory pre-allocation failed for padded_message_pool.");
        }
        padded_message_pool = static_cast<Byte*>(pool_ptr);

        for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) {
            padded_message_pointers[i] = padded_message_pool + i * MAX_PADDED_LEN_PER_MSG;
        }

        messageLengths = new int[MAX_BATCH_SIZE];

    } catch (const std::bad_alloc& e) {
        cerr << "Error during memory pre-allocation (new): " << e.what() << endl;
        prealloc_ok = false;
    } catch (const std::runtime_error& e) {
         cerr << "Error during memory pre-allocation (posix_memalign): " << e.what() << endl;
         prealloc_ok = false;
    }

    if (!prealloc_ok) {
        dual_output(output_file, "Critical error: Failed to pre-allocate memory buffers. Exiting.");
        if (parallel_results) {
            for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) { if (parallel_results[i]) free(parallel_results[i]); }
            delete[] parallel_results;
        }
        if (padded_message_pool) free(padded_message_pool);
        delete[] padded_message_pointers;
        delete[] messageLengths;
        return 1;
    }


    while (!q.priority.empty())
    {
        auto start_pop = high_resolution_clock::now();
        q.PopNext();
        auto end_pop = high_resolution_clock::now();
        auto duration_pop_ns = duration_cast<nanoseconds>(end_pop - start_pop);
        time_guess_generation += duration_pop_ns.count() * 1e-9;

        curr_num = q.guesses.size();


        if (curr_num >= 100000 && (curr_num / 100000) > ((curr_num - q.total_guesses_batch_delta) / 100000) )
        {
             dual_output(output_file, "Guesses generated (approx): " + std::to_string(history + curr_num));
             q.total_guesses_batch_delta = 0;
        }

        int generate_n = 10000000;
        if (history + curr_num >= generate_n)
        {
            dual_output(output_file, "Target number of guesses reached.");
            break;
        }


        if (curr_num >= 1000000)
        {
            dual_output(output_file, "Processing batch of " + std::to_string(curr_num) + " guesses...");
            const size_t pw_count = q.guesses.size();

            if (pw_count > MAX_BATCH_SIZE) {
                cerr << "Error: Batch size (" << pw_count << ") exceeds MAX_BATCH_SIZE (" << MAX_BATCH_SIZE << "). Skipping batch." << endl;
                history += curr_num;
                q.guesses.clear();
                q.guesses.shrink_to_fit();
                curr_num = 0;
                q.total_guesses_batch_delta = 0;
                continue;
            }

            auto start_serial_hash = high_resolution_clock::now();
            vector<array<bit32, 4>> serial_results(pw_count);
            for (size_t i = 0; i < pw_count; ++i) {
                 MD5Hash(q.guesses[i], serial_results[i].data());
            }
            auto end_serial_hash = high_resolution_clock::now();
            auto duration_serial_ns = duration_cast<nanoseconds>(end_serial_hash - start_serial_hash);
            double time_serial_batch = duration_serial_ns.count() * 1e-9;
            time_hash_serial += time_serial_batch;
            dual_output(output_file, "Serial Hash Time (Batch): " + std::to_string(time_serial_batch) + " s");


            chrono::high_resolution_clock::time_point end_parallel_hash;
            chrono::nanoseconds duration_parallel_ns;
            double time_parallel_batch = 0.0;
            bool parallel_run_successful = true;

            auto start_parallel_hash = high_resolution_clock::now();

            try {
                StringProcess_Parallel(q.guesses, pw_count, padded_message_pointers, messageLengths, alignment);
            } catch (const std::runtime_error& e) {
                 cerr << "Error during parallel preprocessing: " << e.what() << endl;
                 parallel_run_successful = false;
                 time_hash_parallel += time_serial_batch;
                 dual_output(output_file, "Parallel Hash Time (Batch, Preproc+SIMD): SKIPPED DUE TO PREPROC ERROR");
            }

            if (parallel_run_successful) {
                MD5Hash_SIMD_Batch(const_cast<const Byte**>(padded_message_pointers), messageLengths, pw_count, parallel_results);

                end_parallel_hash = high_resolution_clock::now();
                duration_parallel_ns = duration_cast<nanoseconds>(end_parallel_hash - start_parallel_hash);
                time_parallel_batch = duration_parallel_ns.count() * 1e-9;

                time_hash_parallel += time_parallel_batch;
                dual_output(output_file, "Parallel Hash Time (Batch, Preproc+SIMD): " + std::to_string(time_parallel_batch) + " s");
            }

            if (parallel_run_successful && time_parallel_batch > 1e-12 && time_serial_batch > 1e-12) {
                 dual_output(output_file, "Batch Speedup (Serial / Parallel): " + std::to_string(time_serial_batch / time_parallel_batch) + "x");
            } else {
                 dual_output(output_file, "Batch time(s) too small or parallel run skipped.");
            }
            dual_output(output_file, "----------------------------------------");

            history += curr_num;
            q.guesses.clear();
            q.guesses.shrink_to_fit();
            curr_num = 0;
            q.total_guesses_batch_delta = 0;
        }
    }

    auto end_guess_hash_loop = high_resolution_clock::now();
    auto duration_loop_ns = duration_cast<nanoseconds>(end_guess_hash_loop - start_guess_hash_loop);
    double total_loop_time = duration_loop_ns.count() * 1e-9;

    dual_output(output_file, "\n================ Summary ================");
    dual_output(output_file, "Total Guesses Generated: " + std::to_string(history + curr_num));
    dual_output(output_file, "Train Time: " + std::to_string(time_train) + " seconds");
    dual_output(output_file, "Guess Generation Time: " + std::to_string(time_guess_generation) + " seconds");
    dual_output(output_file, "Total Serial Hash Time: " + std::to_string(time_hash_serial) + " seconds");
    dual_output(output_file, "Total Parallel Hash Time (Preproc+SIMD): " + std::to_string(time_hash_parallel) + " seconds");
    dual_output(output_file, "Total Loop Time (Guess + Hash): " + std::to_string(total_loop_time) + " seconds");

    if (time_hash_parallel > 1e-12 && time_hash_serial > 1e-12) {
        dual_output(output_file, "Overall Hash Speedup (Serial / Parallel): " + std::to_string(time_hash_serial / time_hash_parallel) + "x");
    } else {
        dual_output(output_file, "Overall hash times too small to calculate speedup reliably.");
    }
    dual_output(output_file, "========================================");


    if (parallel_results) {
        for (size_t i = 0; i < MAX_BATCH_SIZE; ++i) {
            if (parallel_results[i]) {
                free(parallel_results[i]);
            }
        }
        delete[] parallel_results;
    }
    if (padded_message_pool) {
        free(padded_message_pool);
    }
    delete[] padded_message_pointers;
    delete[] messageLengths;

    dual_output(output_file, "Program finished successfully.");
    output_file.close();

    return 0;
}
