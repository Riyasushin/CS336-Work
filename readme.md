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

## 环境与测试

所有包共用顶层 `pyproject.toml` + `uv.lock`，不要在子包各自起 venv。

### 首次同步

```bash
uv sync
```

### 跑测试

pytest 的 rootdir 要落在子包里（否则会被带进 `assignment*/`、`cse234-w25-PA/` 这些只读参考目录）。两种等价办法：

```bash
# A. 进子包再跑
cd Tiny-Training-basic && uv run pytest

# B. 在仓库根用 uv 的 --directory
uv run --directory Tiny-Training-basic pytest
```

其它子包同理，把 `Tiny-Training-basic` 换成 `Tiny-Training-System` / `Tiny-Training-Data` / `Tiny-Training-RL` 即可。

常用变种（以 basic 为例）：

```bash
# 跑某个文件 / 某个用例
uv run --directory Tiny-Training-basic pytest tests/test_model.py
uv run --directory Tiny-Training-basic pytest tests/test_model.py::test_linear -x

# 只 collect 不执行，确认测试被发现
uv run --directory Tiny-Training-basic pytest --collect-only -q
```

### 预期状态

当前 `tiny-training-basic` 里 **25 个测试全部 FAIL on `NotImplementedError`**（包含 2 个 GQA 测试：`test_grouped_query_self_attention` 和 `test_grouped_query_self_attention_with_rope`，用 property-based 方式对 `F.scaled_dot_product_attention(..., enable_gqa=True)` 比较，不依赖 snapshot） —— 这是预期起点：adapter 函数还没实现。每实现一项就把 `tests/adapters.py` 里对应 `raise NotImplementedError` 换成对自己实现的调用，那一批测试就会变绿。

**注意**：测试必须在未实现时真 FAIL，不允许 stub 返回 zeros 造假绿。

### 其它常用

```bash
# 加依赖（例如给 system 包加 triton）
uv add --package tiny-training-system triton

# 看环境里装了什么
uv pip list

# 只验证 import 能通
uv run python -c "import tiny_training_basic, tiny_training_system, tiny_training_data, tiny_training_rl"
```
