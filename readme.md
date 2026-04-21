# 底层目标

搭一个能跑起来的 LLM training 框架，最后你能说得出：
- 对 training 本身有理解
- 有 training / RL 优化加速的经验
- 有 MoE 的经验

学习素材是 CS336 assignments 1/2/4/5 + CSE234 PA2/PA3 的脚手架（PDF、测试、fixtures）。

## 希望最后的效果

1. **一个 uv 环境，一个整体仓库** —— 不要每个 assignment 各自 pyproject + venv。
2. **模块化** —— 可以一个 stub 一个 stub 地实现、测试；不是"把整个 assignment 做完才能跑"。
3. **Adapter 模式** —— 因为 cs336 的测试是这样写的，保留
4. **PA2 PA3 中的算子优化、TP in training、PP in training、MoE等能整合进来成为一整个项目的内容（最后是一个相对完善的训练框架）** —— 测试需改正没实现时必须 FAIL，不要因为 stub 返回 zeros 就假绿。
5. **`assignment1-basics/` 等原代码repo的角色**
   - 集成 tests / 代码前留作只读参考
   - 需要完成这一部分的内容时，先把tests、初始代码按照合适的方式组合进入整个 tiny-training repo, 然后就可以删除原来的的代码文件夹
