# GPU加速密码猜测系统 README

## 项目简介

基于PCFG（Probabilistic Context-Free Grammar）的密码猜测系统，采用CUDA GPU并行化技术加速密码生成过程。本项目实现了基础的GPU并行化功能以及进阶的批量处理、异步协同和自适应调度机制。

## 文件说明

| 文件名 | 类型 | 功能描述 |
|--------|------|----------|
| `main.cpp` | C++源文件 | 主程序入口，负责整体流程控制和测试 |
| `train.cpp` | C++源文件 | PCFG模型训练模块，处理密码语法学习 |
| `guessing_GPU.cu` | CUDA源文件 | GPU并行化核心实现，包含CUDA内核函数 |
| `md5.cpp` | C++源文件 | MD5哈希计算模块，用于密码验证 |
| `PCFG.h` | C头文件 | PCFG核心数据结构和算法定义 |
| `PCFG_GPU.h` | C头文件 | GPU并行化相关的头文件声明 |
| `md5.h` | C头文件 | MD5哈希函数接口声明 |
| `correctness.cpp` | C++源文件 | 功能正确性验证模块 |

## 编译运行

### 环境要求
- NVIDIA GPU（支持CUDA）
- CUDA Toolkit 11.0+
- GCC/G++ 编译器

### 编译命令
```bash
nvcc main.cpp train.cpp guessing_GPU.cu md5.cpp -o GPU.exe -O2
```
