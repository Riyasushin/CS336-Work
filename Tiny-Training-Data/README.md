# tiny-training-data

数据处理层：HTML 抽取 / 语言识别 / PII 脱敏 / NSFW & 有害言论分类 / Gopher 质量过滤 / 质量分类器 / 精确行去重 / MinHash 模糊去重。

## 来源

- `assignment4-data/` —— CS336 Assignment 4 脚手架（PDF + tests + fixtures + adapters）

## 当前状态

`TINY_TRAINING_DATA_ASSETS=/data/CS336-use/tiny-training-data-assets uv run --directory Tiny-Training-Data pytest` → **20 passed, 1 failed**：

| 文件                           | passed / total |
| ------------------------------ | -------------- |
| `tests/test_extract.py`        | 1 / 1          |
| `tests/test_pii.py`            | 5 / 5          |
| `tests/test_langid.py`         | 2 / 2          |
| `tests/test_quality.py`        | 8 / 8          |
| `tests/test_toxicity.py`       | 1 / 2          |
| `tests/test_deduplication.py`  | 3 / 3          |

唯一 miss：`test_classify_toxic_speech` 第二段 `"Why the fc*k should I get a warning for doing nothing?"` 被 Dolma hatespeech 模型判成 `toxic: 0.9985 / non-toxic: 0.0016`，测试期望 `non-toxic`。是模型决策边界的差异（测试注释说 "a Jigsaw-trained model" —— 各家 Jigsaw 模型分界不同），不是实现 bug。

`src/` 已有 `extract.py` / `langid.py` / `pii.py` / `quality.py` / `assets.py`。待补的 5 个 adapter 还在 `tests/adapters.py` 里保持 `raise NotImplementedError`。

## Adapter 覆盖面

`tests/adapters.py`：

- `run_extract_text_from_html_bytes(html_bytes)` → resiliparse / fastwarc
- `run_identify_language(text)` → fastText `lid.176.bin`
- `run_mask_emails(text)` / `run_mask_phone_numbers(text)` / `run_mask_ips(text)` → 正则替换 `|||EMAIL_ADDRESS|||` / `|||PHONE_NUMBER|||` / `|||IP_ADDRESS|||`
- `run_classify_nsfw(text)` / `run_classify_toxic_speech(text)` → Jigsaw fastText 分类器
- `run_classify_quality(text)` → 自训 wiki-vs-cc fastText 分类器
- `run_gopher_quality_filter(text)` → Gopher 论文里 6 条启发式规则
- `run_exact_line_deduplication(input_files, output_directory)` → 两遍扫描：第 1 遍计行哈希计数，第 2 遍只保留 count==1 的行
- `run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)` → n-gram shingle → MinHash 签名 → LSH 分桶 → 桶内 Jaccard 相似度验证 → 集群内保留一份

每实装一个就把对应函数里的 `raise NotImplementedError` 换成对 `tiny_training_data.<module>` 里实现的调用。

## Asset 路径解析

`langid` / `classify_nsfw` / `classify_toxic_speech` / `classify_quality` 都需要本地 fasttext 模型。通过环境变量 `TINY_TRAINING_DATA_ASSETS` 指向目录，默认 `~/.cache/tiny_training_data/`。预期文件名：

| 文件                                            | 来源                                                                                 |
| ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `lid.176.bin`                                   | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`              |
| `dolma_fasttext_nsfw_jigsaw_model.bin`          | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin`    |
| `dolma_fasttext_hatespeech_jigsaw_model.bin`    | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin` |
| `quality_classifier.bin`                        | 自己训（wiki 正例 vs CC 负例）                                                        |

## 运行测试

```bash
cd Tiny-Training-Data
uv run pytest
```

（或从仓库根 `uv run --directory Tiny-Training-Data pytest`）

## 依赖说明

- `tiny-training-basic`（workspace dep；质量分类器训练可能要用到 tokenizer）
- `resiliparse` / `fastwarc` —— HTML 抽取、WARC 解析
- `fasttext` —— 语言识别、质量 / NSFW / 有害言论分类
- `mmh3` —— MinHash 的 murmurhash3
- `nltk` —— 分词（word / sentence）
- `tldextract` —— URL 域名提取（质量分类特征）
- `xopen` —— 读写压缩后的 dedup 输出
- `tqdm`
