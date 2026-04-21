- docs/cse234/pa3_part1_analysis.md 和 pa3_part2_moe.md 是空文件 —— 这俩在 PA3                                
  里就是学生写作的占位，原样保留。
- alignment 测试需重量级 extra —— uv sync --extra alignment 会装 transformers/accelerate/math-verify。原      
  assignment5 的 vllm==0.7.2 和 flash-attn==2.7.4.post1 因为钉 torch==2.5.1 和基础 torch~=2.11 冲突，我从 extra 
  里拿掉了，pyproject 注释里写了单独 venv 的做法。Week 8 真需要 vLLM 时再按注释单起一个。
- 测试二进制 fixtures (76M) —— tests/basics/fixtures 12M、tests/alignment/fixtures 61M（tiny-gpt2 权重+Llama-3
   tokenizer），_snapshots ~3M。你可能要加 .gitignore 或 Git LFS，否则 git commit 会给历史压个                  
  75M。最简单："fixtures 和 _snapshots 不入 git，用 scripts/fetch_fixtures.sh 
  按需恢复"——不过这工作量不小，建议先跟 Git LFS 试试，或者直接入 git（今天的仓库 40M 还不算大）。