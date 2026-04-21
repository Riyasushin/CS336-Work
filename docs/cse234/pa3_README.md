# CSE 234 Programming Assignment 3

Hi everyone, welcome to PA 3 of CSE234 WINTER 2025! We very much recommend you to start early on this assignment. The assignment is due Thursday, March 20th, 2025 at 11:59 pm PST.

Remember, you have 5 late days in total for all PAs.

Academic integrity is key! You may discuss with friends, but do not blatantly cheat and copy off other people for this assignment.


### **Collaboration** 
- You can collaborate on Part 1, 2 and 3 (with up to 3 people).
- You need to finish Part 4 individually, instead of in groups.

## How to Submit Your Homework (Important!)

In your pa3 root directory run
```bash
make handin.tar
```
Then you will see a `handin.tar` file under your root directory, please go to Gradescope and submit the tar.


## Part 1: Mixture of Experts (50 points)

Your first task is to implement an Mixture of Expert (MoE) model with two different variants of communication patterns for the MoE layer: using tensor parallel (TP) and expert parallel (EP). 

As a refererence implementation, we provide the `SimpleMoE` class in the `moe.py` file. We provide the skeleton code for the `MoE_TP` and `MoE_EP` classes. You can test your implementation by running the `test_moe.py` file.

This part is a natural extension to PA2 Part2. In particular, to implement EP, you will need to use all-to-all communication. 

### 1.1. Tensor Parallel (TP) (20 points)

Every process holds all experts, but only a portion of each experts' weights. At expert forward pass, each process compute partial expert output, and then perform all-reduce to get the full expert output.

- Implement `ShardedLinear` class in the `moe.py` file. (10 points)
- Implement `MoE_TP` class in the `moe.py` file. (10 points)

### 1.2. Expert Parallel (EP) (20 points)

Each process holds a subset of experts in its entirety. At expert forward pass, each process compute the output for its assigned slice of the input, and then perform all-to-all communication to get the full expert output.

- Implement `MoE_EP` class in the `moe.py` file. (20 points)


### 1.3. Benchmark your implementation (10 points)


- Write a script to benchmark the performance of your implementation. We provide a `benchmark.py` file to help you get started, but you need to modify it to further benchmark the performance of your implementation. (5 points)
- Compare the two implementations, and write a short paragraph in `analysis.md` file to analyze the performance of your implementation. Why is one better than the other? It is optional to include plots and long paragraphs, but you are encouraged to do so. (5 points)




## Part 2: Scaling Laws and Training Cost Analysis (30 points + 25 points extra)

Your second task would be analyze the training cost of Llama-7B. You will be asked to estimate the number of model parameters and the amount of flops and memory required for training. In addition, you will also have the chance to design your optimal model training schema given a scaling law and compute budget.


### 2.1. Model Training Cost Analysis (15 pts)
In the file `llama_7b_config.json`, the model configurations of Llama-7B is provided. Your task is to implement the `model_training_cost_analysis_llama` function in the `model_training_cost_analysis.py` file. This function takes the path to the Llama-7B model configuration file and outputs the number of trainable model parameters, number of TFLOPs and the peak memory cost required during training. 

The total model parameters should include:
* the parameters for word embedding
* the parameters for positional embedding
* the parameters for the transformer layers (Attention, MLP, and Layernorm)

The number of TFLOPs refers to the amount of computation required for the forward pass of a single transformer layer in the Llama-7B model. The peak memory cost is the amount of GPU memory required for the forward pass of a single transformer layer using fixed `fp16` precision training. You can also assume we do checkpoint rematerialization at each of the transformer boundary.

We will use the following command to check the output of your implementation:
```bash
python3 model_training_cost_analysis.py --model_config <path-to-llama_7b-config.json>
```

### 2.2. Design your own model training schema (15 pts)
In this part, you will be required to determine the optimal model size and the amount of training tokens given the following scaling law:

$$
L (N, D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.29}} + 1.69
$$

and a training budget of 5 million dollars. You will have three GPU options:

- NVIDIA A100: cost per hour = $4.0, peak FP16 performance = 312 TFLOPs
- NVIDIA V100: cost per hour = $2.5, peak FP16 performance = 125 TFLOPs
- NVIDIA T4:  cost per hour = $1.0, peak FP16 performance = 65 TFLOPs

Assume MFU = 40% for all 3 types of GPUs. 

Please implement the `get_optimal_N_D_from_cost` function in the file `model_training_cost_analysis.py`. You need to select the best GPU type, compute the best total effective training FLOPs, and get optimal value of N and D using the scaling law. 

We will use the following command to check the output of your implementation:
```bash
python3 model_training_cost_analysis.py --training_budget <training_budget>
```

After getting optimal model size, design your own model architecture and create a configuration file named `my_model_config.json`, following similar format of `llama_7b_config.json`.

### 2.3. MoE Model Cost Analysis (bonus 25 points)

Now let's work on a new popular model: `DeepSeek-V3`. The configuration of this model is provided in the file `deepseek_v3_config.json`. Can you use your previous function to analyze this model? 

If not, write a new function `model_training_cost_analysis_deepseek` to analyze the cost of `DeepSeek-V3`. Based on your analysis, give some advantages of MoE models (submit as a Markdown file named `part2/moe.md`). 

Just a reference: Deepseek *claims* that they can train such a good model using 5 million dollar. 


## Part 3: Speculative Decoding (50 points + 15 extra)

### Overview
In this assignment, you will implement and optimize a speculative decoding algorithm for language models. Speculative decoding is a technique that accelerates text generation by using a smaller, faster "draft" model to propose tokens that a larger "target" model then verifies, potentially reducing the number of expensive forward passes needed during generation.

Traditional autoregressive text generation requires one forward pass through a language model for each generated token, which can be slow for large models. Speculative decoding accelerates this process by:

Using a smaller, faster model to draft a sequence of tokens
- Verifying these tokens with the larger model in a single forward pass
- Accepting correct tokens and regenerating from the first error

This approach can significantly reduce latency, especially when the draft model's predictions align well with the target model.



### 3.1. Complete the Implementation (25 points)

Fill in the placeholders in the provided template:
- initialize_target_model and initialize_draft_model (5 points): Initialize the target and draft language model
- generate_draft_tokens (5 points): Generate speculative tokens with the draft model
- verify_tokens_vectorized (5 points): Efficiently verify draft tokens with the target model
- speculative_decode (10 points): Implement the main speculative decoding algorithm

### 3.2. Performance Optimization (20 points)

- Achieve a minimum 1x speedup over baseline decoding (10 points)
- Achieve >75% draft token acceptance rate (10 points)

### 3.3. Analysis and Evaluation (5 points)

- Benchmark your implementation against baseline decoding.
- Analyze the impact of different parameter settings on performance.
- Document all optimizations and their effects in your report.

### **Bonus Opportunities (up to 15 extra points)**

- Achieve >1.7x speedup over baseline (5 points)
- Achieve >85% token acceptance rate (5 points)
- Successfully implement and benchmark with different model pairs beyond the default ones (5 points)

### Implementation Details
You are provided with a template that includes the basic structure of a speculative decoder. Your task is to implement the core functionality in the marked placeholders.
The implementation should work with batch size 1 (single sequence generation) and be efficient for generating sequences of moderate length (up to a few hundred tokens).

### Submission Instructions
Your submission should include:

- The completed Python file with your implementation
- A brief report (maximum two pages) documenting:
  - Your approach to implementing speculative decoding
  - Optimizations you implemented
  - Performance results and analysis
  - Any challenges encountered and how you addressed them
  
**PLEASE run your Jupyter notebook with your best result before submission.**

## Part 4: What is the future of LLM and AI?

### 4.1. Argumentative Essay (40 points)

In this section, you will write a 500-word argumentative essay on a controversial topic related to the future of AI and technology markets.


Your task is to write a persuasive essay arguing that this conviction will be realized in the future. This requires you to:


1. Research current market positions, technological developments, and industry trends
2. Develop a clear thesis statement supporting the conviction
3. Provide specific, precise arguments with concrete predictions
4. Support your claims with evidence and logical reasoning
5. Address potential counterarguments


**Example Conviction statements**

You can choose one of the following statements as your conviction. Alternatively, you can also come up with your own statement.

- "NVIDIA's data center revenue will decline by 50% in 2027 because the scaling law of LLM training will not sustain."
- The open-source AI ecosystem will surpass closed models in commercial adoption by 2028, driven by superior customizability and the diminishing returns of scale beyond 100B parameter models.
- The economic value of foundation model APIs will decrease by 99.9% by 2027 as commoditization and open-source alternatives drive a shift toward value creation in domain-specific fine-tuning and deployment.
- Multimodal small language models (SLMs) will replace large language models for 80% of enterprise applications by 2027, due to their superior cost-efficiency and reduced hallucination rates.


**Requirements:**
- Essay must be approximately 500 words (±10%)
- Submit this separately as a PDF to gradescope under the assignment PA3 - Essay
- Must include specific, time-bound predictions (e.g., "NVIDIA's data center revenue will decline by X% in Q3 2025 because...")
- Cite the sources you used in the essay to support your arguments.


**Grading Criteria:**
- **Precision of arguments (40%):** Specific, measurable predictions with clear timelines will receive higher scores than vague generalizations. For example, "Qualcomm's AI chip revenue will grow by 75% in 2026 due to mobile AI adoption" is better than "Qualcomm will grow in the future."
- **Evidence and reasoning (30%):** Strong supporting evidence and logical connections between claims
- **Counterargument handling (15%):** Acknowledging and effectively addressing opposing viewpoints
- **Writing quality (15%):** Clear organization, professional tone, and proper citations

**Note:** The purpose of this assignment is to develop your analytical and argumentative skills in the context of technology trends. Your grade is based on the quality of your reasoning and specificity of your arguments, not on whether your predictions ultimately prove correct.

