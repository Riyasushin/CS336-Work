# tiny-training-basic

LLM 训练框架的基础层：tokenizer / transformer / optimizer / training loop。

来源：迁移自 `assignment1-basics/`（CS336 Assignment 1 脚手架）。

## 当前状态

- `src/tiny_training_basic/`：仅 `__init__.py` + `pretokenization_example.py`（BPE chunk 边界查找辅助）
- `tests/`：tests + adapters + fixtures + 快照已全量迁入，**adapter 函数全是 `raise NotImplementedError`**，测试必然 FAIL —— 这是预期的起点
- 实现方式：逐个 stub 实现、对着 `tests/adapters.py` 把函数体里的 `NotImplementedError` 替换为对自己实现的调用

## 运行测试

```bash
cd Tiny-Training-basic
uv run pytest
```

（或从仓库根运行 `uv run --package tiny-training-basic pytest`）
