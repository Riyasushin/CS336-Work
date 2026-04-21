# CSE 234 W25: PA 2
Hi everyone, welcome to PA 2 of CSE234 WINTER 2025! We very much recommend you to start early on this assignment. The assignment is due Tuesday, March 4th, 2025 at 11:59 pm.

Remember, you have 5 late days in total for all PAs.

Academic integrity is key! You may discuss and collaborate with friends, but do not blatantly cheat and copy off other people for this assignment.

### **Important** 
You may work with groups of up to 3 on this assignment and submit your assignment together. Just add your group members to your Gradescope submission.

## Part 1: Matmul Kernel Optimization (40 pts + extra credit :D)
The file you must complete in this part is `matmul_triton.ipynb`. You will find detailed TODOs in this file as well as an overview in the writeup below.

In this part of the assignment, you will implement Triton-based implementation of a matrix multiplication + ReLU + add kernel. The kernel computes the matrix function **D = ReLU(A × B + C)** where **A** is of shape *(M, K)*, **B** is of shape *(K, N)*, and **C & D** are of shape *(M, N)*. We will break the kernel down into four main steps:

1. **Tile Assignment**  
2. **Shared Memory Tiling + Cooperative Fetching**  
3. **Register Tiling (Accumulator)**  
4. **Operator Fusion**
5. **Write Cache/Epilogue (Store Results)**

Each section below explains the purpose and the implementation details of each step.

### Setup
#### **VERY IMPORTANT:** 
Please use Google Colab's free T4 GPU to run the notebook. You will be graded based on your performance and speedup when running on the T4 GPU. Different GPUs have different accelerations based on floating point precision, tensor cores, etc, so please use the Colab GPU when coding and evaluating your performance.

We also understand that there will be some issues in this assignment regarding precision and performance on fp32 tensors. Since T4's have known issues with fp32 precision (https://github.com/NVIDIA/TensorRT-LLM/issues/396#issuecomment-1817806895) and their tensor cores do not have compatibility with fp32 acceleration, your speedup on matmul may not be as good as you expect. However, please still use T4 GPU on Colab to complete this assignment as we have calculated our benchmarks, accuracy tolerance, and implementation based on Colab. We want you to be able to finish this assignment for free and not have to pay for a GPU.

**Note:** Since T4s do not support fp32 precision well, please ALWAYS do matmul (and accumulation) in fp16 precision for the purposes of this assignment. Changing it will lead to different performance metrics or broken code, which you do not want.

### 1.1 Tile Assignment
**Tile Assignment** is the process by which the overall matrix **C** is divided into smaller submatrices (tiles). Each kernel instance (a GPU “program” or thread block) is responsible for computing one tile of **C**. This division allows the computation to be parallelized across many threads.

Each kernel launch instance (denoted by a unique program id `pid`) should be mapped to a specific tile in **C**. The tile is defined by two indices: one for the row (denoted `pid_m`) and one for the column (`pid_n`).

### 1.2 Shared Memory Tiling + Cooperative Fetching
**Shared Memory Tiling** is used to load sub-tiles (smaller blocks) of matrices A and B into faster, on-chip memory. Threads within a kernel instance load these sub-tiles, reducing the number of global memory accesses.

Some things to keep in mind:
- You may need to use `tl.arange` to help compute offsets for the rows and columns.
- Use masks to make sure that out-of-bound memory accesses do not occur.

### 1.3 Register Tiling (Accumulator)
**Register Tiling** is the use of registers to hold intermediate results (partial sums) during the matrix multiplication. For this section, you will need to use an accumulator (a BLOCK_SIZE_M by BLOCK_SIZE_N matrix initialized to zeros) to accumulate results of dot products computed over tiles.

After accumulation you can optionally choose to fuse an activation function (like leaky ReLU) - this is used in practice to make architectures that use lots of matmuls and activation functions together (like transformers) much much faster!

### 1.4 Add and ReLU Fusion
In this step, we fuse the element-wise addition of matrix C and the final ReLU activation directly into the matmul kernel to optimize performance by reducing memory traffic and kernel launch overhead. After computing the matrix multiplication and storing the results in the accumulator, you must load the corresponding tile from matrix C and add it element-wise to the activated accumulator. Then apply the ReLU function using an element-wise maximum operation to set all negative values to zero. 

This fusion of operations avoids writing intermediate results back to global memory and then reloading them for further processing, minimizing latency and making more efficient use of the GPU’s memory hierarchy.

### 1.5 Write Cache/Epilogue
In this step, we write the tile of C back to global memory. This final step ensures that the computed results are stored in the correct locations in the output matrix C. Be sure to also use a mask to prevent invalid indices from being written to. (Use `tl.store` to store your tiles.)

### 1.6 Grid Search
To achieve full credit on part 1, you will have to perform grid search or manually find values for the hyperparameter block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`). We have provided parameters that should result in you achieving around 0.9-1x speedup, but you will have to search for different block sizes to achieve better speedup.

### Part 1 Grading
This entire section combined will be 40 points total. We will grade you on completion of each subpart, as well as overall performance. We provide a list of thresholds below for speedup and how many points you will get for each speedup multiplier with respect to the reference PyTorch implementation provided in `reference_matmul` in the notebook.

- <1.0x speedup: grade subject to your implementation
- 1.0x speedup: 35 / 40 points
- 1.1x speedup: 40 / 40 points
- 1.25x speedup: 45 / 40 points
- 1.4x speedup: 50 / 40 points

Please make sure to test on the GPUs that we have mentioned here. Speedup is much much different on different GPUs! **AND DO NOT CHANGE THE CODE** in the performance benchmark section of the notebook. We are only testing on 2048x2048 fp16 matmul + add + relu, with 5000 iterations.

For submission, **please run your jupyter notebook fully with your best submission speedup**. This way we can have an easier time grading and looking at your implementation!

## Part 2: Tensor Parallel Communication (60 pts)

In this part of the programming assignment, you will work on developing communication protocols for Data Parallel and Tensor Model Parallel training from the ground up, utilizing the Message Passing Interface ([MPI](https://mpi4py.readthedocs.io/en/stable/)) and NumPy.

Since the main focus will not be on the actual forward computation or backward propagation, your task will be to implement only the communication aspects.

**Note: we will only focus on the forward pass of the training loop for this PA.**

### Setup Guide

You'll need need a multi-core machine with at least 8 cores for this assignment.

#### Installing MPI
You'll need to have MPI installed for this assignment. 

##### Linux
```bash
sudo apt install libopenmpi-dev
```
##### MacOS
```bash
brew install openmpi
```
##### Windows
Go to the following and install mpiexec. Note, you'll have to use `mpiexec` instead of `mpirun` in all the commands in this assignment.
https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

#### Conda
The easiest way is to install an environment manager such as [Miniconda](https://docs.anaconda.com/free/miniconda/) and MPI for your OS (supported on MacOS, Linux, and Windows).

Once you have Conda installed and initialzied, create a new env with the following command:
```bash
conda create -n cse234pa2 python=3.10 -y
```
Then activate it:
```bash
conda activate cse234pa2
```
Now you should be able to install the requirements.txt
```bash
git clone https://github.com/hao-ai-lab/cse234-w25-PA
cd cse234-w25-PA/pa2
pip install -r requirements.txt
```

**You can also create and activate Conda environments through VSCode.**


With your environment active you can clone the repo and install the dependencies:

```bash
git clone https://github.com/hao-ai-lab/cse234-w25-PA
cd cse234-w25-PA/pa2
pip install -r requirements.txt
```

Make sure your platform contains MPI support with at least 8 cores.

### 2.0. Warm-up

This assignment aims to guide you step by step through a 2D parallel training pipeline,incorporating both tensor model and data parallel training. For tensor model parallel training, we will delve into the naive approach. 

To become acquainted with our communication protocols, we will begin by experimenting with the MPI package that has been installed.

#### MPI Test
To verify that *mpi4py* has been setup correctly for distributed workloads, run:
```bash
mpirun -n 8 python mpi-test.py
```

Or if you are using Windows, use `mpiexec` instead of `mpirun` for all of follow commands:
```bash
mpiexec -n 8 python mpi-test.py
```

Depending on your machine, you can control the number of processes lanched with the `-n` argument. 

Additionally, we have included some simple examples of MPI functions in mpi-test.py, such as Allreduce(), Allgather(), Reduce_scatter(), Split() and Alltoall(). Please note that these five MPI functions are the only ones you need to use and are permitted for this assignment.

- ##### All-Reduce

<p align="center">
<img src="figs/allreduce.png" alt="image" width="450" height="auto">
</p>

You can see an all-reduce example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case allreduce
```

- ##### All-Gather

<p align="center">
<img src="figs/allgather.png" alt="image" width="450" height="auto">
</p>

You can see an all-gather example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case allgather
```

- ##### Reduce-Scatter

<p align="center">
<img src="figs/reduce_scatter.png" alt="image" width="450" height="auto">
</p>

You can see a reduce-scatter example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case reduce_scatter
```

- ##### Split

<p align="center">
<img src="figs/split.jpeg" alt="image" width="350" height="auto">
</p>


The Split function is particularly useful when applying MPI functions on a group basis. You can observe an example of group-wise reduction with the split function by running: 

```bash
mpirun -n 8 python mpi-test.py --test_case split
```

When playing with different test cases, try to get yourself familiar with the underline mpi functions 
and think about whether the output meets your expectation. 


#### Node Indexing Specifications

With a given data and model parallel size, we will assign nodes in a model parallel major for this assignment.
For instance, for `mp_size=2, dp_size=4` on 8 nodes we will group the nodes as shown below:

<p align="center">
<img src="figs/group.png" alt="image" width="350" height="auto">
</p>

### 2.1 Implementation of Collective Communication Primitives (20 pts + 10 pts bonus)

In previous part, we directly use primitives from the MPI library. In this task, you need to implement your own version of two primitives: all-reduce and all-to-all. 

You only need to fill in the function `myAllreduce` and `myAlltoall` in the `comm.py` file!

After implementing both function, you can use 

```bash
mpirun -n 8 python mpi-test.py --test_case myallreduce
```

and 

```bash
mpirun -n 8 python mpi-test.py --test_case myalltoall
```

to check the correctness of your implementation. We can also see the time consumption of both implementation. 
Discuss the time difference between your implementation and the MPI one, and give some possible reason if there is a gap. 
If your implementation is fast (within 150% time consumption of MPI version), you can get a bonus (+2.5 pts for each function)!
If your implementation is as fast as the MPI implementation (within 105% time consumption of MPI version), you can get a bonus (+5 pts for each function)!

Put your discussion (2-3 sentences) in the file discussion2-1.txt.

### 2.2 Data Split for Data Parallel Training (5 pts)

For this part, your task is to implement the `split_train` function in `data/data_parallel_preprocess.py`.

The function takes in the training data and returns the data split according to the given `mp_size, dp_size` 
and `rank`. You should split data uniformly across data parallel groups while the model parallel groups can share the 
same data split within the same data parallel group. The data length is guaranteed to be divided equally by the
`dp_size` in all our test cases.

Hints: 
For `mp_size=2, dp_size=4`, you should split the data this way:
 
<p align="center">
<img src="figs/part1-hint.png" alt="image" width="350" height="auto">
</p>

To test your implementation, please run
```bash
python3 -m pytest -l -v tests/test_data_split.py
```

### 2.3 Layer Initialization (15 pts)

In this part, your task is to get necessary information for model and data parallel training, which is then
used to initialize the corresponding layers in your model.

For this assignment we will work with a transformer layer.

You are only required to implement the communications within four fully connective layers within a transformer layer for forward and backward.
We have already taken care of the other stuffs i.e. the forward/backward computations and the training pipeline as these
are not relevant to the goal of this assignment.

For data parallel, we simply just split the batch of data equally across different data parallel groups.

For naive tensor model parallel training, we split the weight matrix of all of the fully connective layers (fc_q, fc_k, fc_v, fc_o)
along the output dimension (partition output) and shard them across different nodes. (Note that we don't shard different layers 
to different node as we don't consider pipeline parallelism here)

Given the above information, you need to implement the `get_info` function in `model/func_impl.py`.
The function gets essential information for later parts, including model/data parallel indexing,
model/data parallel communication groups, in/out dimensions for four FC layers. Please refers to the function for more information and hints.

To test your implementation, please run
```bash
mpirun -n 8 python3 -m pytest -l -v --with-mpi tests/test_get_info.py
```

### 2.4 Naive Model Parallel Forward Communication (10 pts)

Your task in this part is to implement the forward communications in W_o layer for the naive model parallel.
You need to implement the `naive_collect_forward_input` and `naive_collect_forward_output` functions in
`model/func_impl.py`. Please refer to the code for more details.

To test your implementations, please run
```bash
mpirun -n 4 python3 -m pytest -l -v --with-mpi tests/test_transformer_forward.py
```


### 2.5 Naive Model Parallel Backward Communication (10 pts)

Your task in this part is to implement the backward communications in W_o layer for the naive model parallel.
You need to implement the `naive_collect_backward_input` and `naive_collect_backward_output` functions in
`model/func_impl.py`. Please refer to the code for more details.

To test your implementations, please run
```bash
mpirun -n 4 python3 -m pytest -l -v --with-mpi tests/test_transformer_backward.py
```

## How to Submit Your Homework (Important!)

In your pa2 root directory run
```bash
make handin.tar
```
Then you will see a `handin.tar` file under your root directory, please go to Gradescope and submit the tar.

#### References 

- some images credited to https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
