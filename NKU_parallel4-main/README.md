
# NKU 2025年春季并行程序设计第四次实验作业

## 项目简介

本项目是南开大学2025年春季学期并行程序设计课程的第四次实验作业，实现了基于PCFG（概率上下文无关文法）的密码猜测算法的并行化优化。

## 代码文件说明

项目包含三个不同的实现版本，文件命名规则如下：

- **openmp后缀**：OpenMP并行实现版本
- **2后缀**：原始串行代码实现版本  
- **pth后缀**：Pthread并行代码实现版本

## 主要文件结构

```
├── main.cpp                    # Pthread版本主程序
├── correctness2.cpp           # 串行版本正确性验证程序
├── guessing_pth.cpp           # Pthread并行实现核心算法
├── guessing_openmp.cpp        # OpenMP并行实现核心算法
├── guessing2.cpp              # 原始串行实现核心算法
├── PCFG_pth.h                 # Pthread版本头文件
├── PCFG_openmp.h              # OpenMP版本头文件
├── PCFG2.h                    # 串行版本头文件
├── train.cpp / train2.cpp     # 模型训练模块
├── md5.cpp / md5.h            # MD5哈希计算模块
└── 我的main.tex               # 实验报告LaTeX源码
```

## 编译说明

根据代码中的编译指令：

```bash
# Pthread版本
g++ main.cpp train.cpp guessing_pth.cpp md5.cpp -o main -O2

# 串行版本
g++ correctness2.cpp train2.cpp guessing2.cpp md5.cpp -o main2 -O2

# OpenMP版本（需要相应的openmp后缀文件）
g++ main_openmp.cpp train_openmp.cpp guessing_openmp.cpp md5.cpp -o main_omp -O2 -fopenmp
```




## 补充说明

在完成本次并行作业的过程中，因为代码经历了多次修改，很多测试数据没有保留或者无法与某个对应版本的代码相对应，我只上传了部分相关数据，具体数据请以报告为准，非常抱歉
