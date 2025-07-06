# 口令猜测算法MPI并行化实验

## 项目概述

本项目基于 PCFG（Probabilistic Context-Free Grammar）口令猜测算法，使用 MPI 消息传递接口实现了多种并行化策略。项目旨在探索 MPI 编程的过程和细节，并在工程上尝试对口令猜测算法进行并行化优化。

## 文件结构

```
guess/
├── PCFG.h                          # PCFG 算法核心头文件
├── md5.h                           # MD5 哈希计算头文件
├── md5.cpp                         # MD5 哈希计算实现
├── train.cpp                       # 模型训练实现
├── correctness_mpi.cpp             # 正确性验证程序
├── main.cpp                        # 主程序入口
├── guessing_mpi_base.cpp           # 基础 MPI 并行实现
├── guessing_mpi_jinjie1.cpp        # 进阶要求1：PT层面并行
├── guessing_mpi_jinjie2.cpp        # 进阶要求2：流水线并行
└── 作业要求.md                      # 实验要求文档
```

## 功能特性

### 1. 基础 MPI 并行化（`guessing_mpi_base.cpp`）

- 实现了核心 `Generate` 函数的并行化版本 `GenerateParallelMPI`
- 采用四阶段数据收集架构
- 支持自定义密码序列化和反序列化
- 验证了并行算法的功能正确性

### 2. 进阶要求1：PT 层面并行计算（`guessing_mpi_jinjie1.cpp`）

- 实现了 `PopNext_MultiPT` 函数
- 支持从优先队列一次性取出多个 PT 并行处理
- 自定义 PT 对象序列化机制
- 解决了复杂数据结构的进程间传输问题

### 3. 进阶要求2：密码生成与哈希流水线并行（`guessing_mpi_jinjie2.cpp`）

- 实现了 `PipelineGuessingAndHashingExecution` 流水线框架
- 密码生成控制器与哈希验证集群分离设计
- 支持计算重叠执行的流水线架构
- 多阶段通信协议保证数据完整性

## 编译与运行


### 编译命令

#### 1. 基础 MPI 并行版本

```bash
mpic++ -o main guessing_mpi_basic.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11
```

#### 2. PT 层面并行版本

```bash
mpic++ -o main guessing_mpi_jinjie1.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11
```

#### 3. 流水线并行版本

```bash
mpic++ -o main guessing_mpi_jinjie2.cpp correctness_mpi.cpp train.cpp md5.cpp -std=c++11
```







