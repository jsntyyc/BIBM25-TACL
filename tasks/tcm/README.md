# TCM-SD / ZY-BERT 实验

本目录迁自 `ZY-BERT-main/TCM Code/TCM`。这一轮整理将实验放进 Python 包、隔离入口执行并参数化路径；保留原模型结构、超参数默认值、课程阶段、β 判断、早停和 checkpoint 保存顺序。两份 `modelling_tcm.py` 仍独立，现代根级入口使用根级版本，`model/` 入口使用同目录旧版本。没有抽取跨任务的公共 TACL 实现。

## 先查看入口与输入

从仓库根目录运行，无需安装 PyTorch 即可查看帮助或检查输入。示例中的 `python` 需指向你的 Python 环境：

```powershell
python -B -m tasks.tcm.run --help
python -B -m tasks.tcm.run_cl_v2 --check-paths
python -B -m tasks.tcm.curriculum_learning.kmeans_distance_apw --help
python -B -m tasks.tcm.check_entrypoints
```

`--check-paths` 检查本地数据路径、现代 JSONL 字段与本地模型目录是否存在，输出解析后的参数；发现缺失时返回非零码，不加载模型、不创建输出目录。它不验证模型权重完整性、依赖兼容性或训练效果。正常运行也会在加载训练依赖前检查输入。

现代入口共同参数：`--data-dir`、`--train-file` / `--dev-file` / `--test-file`（适用时）、`--model-dir`（别名 `--model-name-or-path`）、`--output-dir`、`--device auto|cpu|cuda|cuda:N`、可选 `--cuda-visible-devices`。课程生成入口另有 `--output-file`；嵌入入口使用 `--input-path`。具体训练参数见各入口 `--help` 与文件顶部 `DEFAULTS`。

相对路径统一相对于仓库根目录；默认数据位于 `data/tcm/<原实验文件名>`，模型位于 `models/tcm/<原模型名>`，输出位于 `runs/tcm/<入口名>`。根级与嵌套同名入口的输出分别隔离。数据目录中的默认文件名保持原来的 `V2`、`mapped` 等区分，不会自动拿其他文件补缺。

默认不再写死原服务器的 GPU 编号，沿用当前进程的可见 GPU；`auto` 在多卡时仍采用原入口的 DataParallel 分支。显式 `cpu` 或 `cuda:N` 使用指定单设备。所有原批量大小保留，单卡使用者需通过 CLI 按实际显存调整；未在 L40 上运行训练或测量显存。

```powershell
python -m tasks.tcm.run_cl --data-dir data/tcm/prepared --model-dir models/tcm/ZY-BERT --device cuda:0 --train-batch-size 16
python -m tasks.tcm.curriculum_learning.kmeans_distance_apw --train-file data/tcm/prepared/train.json --model-dir models/tcm/ZY-BERT --device cuda:0 --batch-size 16
```

这些命令需要先准备输入与权重；示例没有隐含的数据下载或字段转换。

## 入口索引与原行为

| `python -m tasks.tcm.<模块>` | 作用与保留的差异 |
|---|---|
| `run` | ZY-BERT 分类基线，148 类，max length 256，开发集 Macro-F1、patience=3 |
| `run_cl` | ZY-BERT 固定阶段：epoch<2 仅 easy、epoch<6 加 middle，其后全量；patience=3 |
| `run_cl_v2` | ZY-BERT 动态阶段，window_size=3，β=0.8 |
| `run_tcmbert_cl` | TCM-BERT；β=0.9 及增长率代码仍被注释；实际为单次不优于全程 best 即切前两阶段，最终阶段 patience=3 |
| `runroberta` | 原特定 RoBERTa 模型分类基线，max length 512；保持原模型目录名称 |
| `runroberta_cl` | 原中文 RoBERTa/BERT 接口实验，固定阶段课程；没有按文件名替换 tokenizer 或模型类 |
| `runroberta_cl_v2` | 原中文 RoBERTa/BERT 接口实验，window_size=3，β=0.9 |
| `curriculum_learning.apw_kmeans` | 原 `APW+KMEANS.py`；APW 4 epochs、q=1，写出 easy/middle/hard 字符串 |
| `curriculum_learning.kmeans_distance_apw` | 原 `kmeans+distance+apw.py`；APW 5 epochs、q=10、距离分组，写出 difficulty 与 PCA/t-SNE 图 |
| `curriculumlearning` | 原 CLS + PCA/KMeans 可视化实验；保留 4 个整数簇号，不能直接当作字符串课程标签 |
| `model.run_tcmbert_cl` | 旧子目录中的另一个 ZY-BERT 动态课程实验，β=0.5；仍使用旧 `model/modelling_tcm.py` |
| `model.run_tcm` | 历史知识增强分类入口；原 underscore 参数保留，常用参数另有 hyphen 别名 |
| `model.run_tcm_mrc` | 历史 MRC 入口，要求旧数据结构与旧 transformers API |
| `model.traiditonal_classifer` | 历史未完成分类模板；保留原拼写与缺少调用参数的问题；仅帮助入口可独立检查 |

v2 课程比较的是绝对 F1 差分：最近 3 个 F1 形成 2 个差分，最后差分低于 β×差分均值时切阶段，并清空阶段历史。最终 hardest 阶段触发停止时，原代码的 `break` 在本轮最佳 checkpoint 保存之前；这里未改写其行为。此处说明是代码事实，论文对照以仓库文档为准。

## 数据与权重前提

仓库提供的数据已经归入 `data/tcm/samples` 与 `data/tcm/provided`。提供内容包含原始病历样例、开发集、证候知识与词表；完整训练集、测试集、现代实验的 difficulty 标注文件、V2 文件和预训练模型权重没有随此实验代码提供。完整数据/模型的上游获取链接和许可见迁移保留的上游说明。

现代分类和 APW 输入为 **JSON Lines**，每行一个对象，包含 `TEXT` 字符串与可转整数的 `LABEL`，标签需与训练模型的 148 类映射对应。课程训练还读取 `difficulty`，值应为 `easy`、`middle`、`hard`。原代码缺失 difficulty 时将其当作 easy 的逻辑保留。原始样例字段是 `chief_complaint`、`medical_history`、`FDMR`、`norm_syndrome` 等，不能直接传给现代分类入口；文本拼接、标签映射和切分策略必须显式确定，不能用样例或开发集补成训练/测试集。

APW 入口保持原先从观察到的标签数推断分类数、再检查标签范围的行为，要求输入标签连续编号。`curriculumlearning` 只要求 `TEXT`，输出的整数簇号需要另行解释，整理未自动映射为 easy/middle/hard。

旧分类入口需要 `train_v3_knowledge_tfidf_3k_shuffle.json`、`dev_v3_knowledge_tfidf_3k.json`、`test_v3_knowledge_tfidf_3k.json` 与 `v2_tcm.xlsx`。可用 `--knowledge-file` 和 `--label-vocab` 指定资源；默认词表通过包内 `__file__` 定位。旧 MRC 默认文件为 `train-v1.1.json`、`dev-v1.1.json`，也可用 `--train_file` / `--predict_file` 指定。MRC 模块不再在 import 时读取未提供的 Excel；其原全局知识库未被任何活动代码使用。

```powershell
python -B -m tasks.tcm.model.run_tcm --help
python -B -m tasks.tcm.model.run_tcm --model-type bert --differentiation-element full --do-train --check-paths
python -B -m tasks.tcm.model.run_tcm_mrc --model-type bert --do-train --check-paths
```

## 环境与历史限制

`requirements.txt` 是根据 import 整理的依赖清单，不是锁文件。上游未提供可验证的版本组合；现代入口与旧 `model/` 使用不同时代的 transformers API，可能需要分开的环境。PyTorch 的 CUDA wheel 也必须根据实际主机选择。本轮未安装 GPU 依赖，未运行训练、模型下载或指标复现。

旧版仍引用顶层 `transformers.AdamW`、`pretrained_config_archive_map`、旧 tokenizer 路径等已变化的 API。旧 `model/modelling_tcm.py` 中未定义的 `ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP` 与 RoBERTa 类未设置 `self.num_labels` 的问题保留，不能把 `--help` 通过等同于该入口可训练。`traiditonal_classifer.py` 与 `data_preprocess/tfidf.py` 还包含缺少必需调用参数的历史模板代码，不作为完整运行配方。

`data_preprocess` 中的词表、标准化表与停用词作为包内资源保留；`make_word_cloud` 的字体与输出路径可由调用者传入，中文字体需自行准备。`speed.ipynb` 与 `data_preprocess/text_process.ipynb` 是原始历史 Notebook，内容、输出及旧路径未重新执行或改写，不能直接视作新的模块入口示例。

验证范围：`check_entrypoints` 使用 Python 标准库检查 14 个入口的 `--help`、导入不加载训练依赖/不改 GPU 环境、JSONL 输入契约、路径检查不生成输出、关键原默认值与全部 Python 语法。它不替代有真实数据和匹配环境的训练验收。
