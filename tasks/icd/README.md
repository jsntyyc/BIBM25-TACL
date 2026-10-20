# ICD 任务：源码整理与运行说明

本目录来自原仓库 `ICD/` 及其六个压缩包。`src`、`configs`、`reports`、测试和课程实验已展开；数据切分文件移至仓库的 `data/icd/splits/<dataset>/`。原始压缩包保留在 `archives/`，可用于追溯。

## 运行入口与环境

在仓库根目录运行，不需要安装训练依赖即可查看简要帮助：

```console
python -m tasks.icd.main --help
python -m unittest tasks.icd.tests.test_paths -v
```

Python 需要至少 3.10。建议给 ICD 单独建立环境；先按实际 CUDA/CPU 平台安装 PyTorch，再按 `tasks/icd/requirements.txt` 安装其余已知依赖。原作者没有提供版本锁；该文件不代表所有最新版已验证兼容，尤其需要核对 Vaex、NumPy、SciPy、Gensim 和 Transformers 的组合。本次没有安装训练环境或运行完整训练。

```console
python -m pip install --no-deps -e .
python -m pip install -r tasks/icd/requirements.txt
```

第一条在仓库根目录执行，只安装项目命名空间；PyTorch 需在第二条之前自行安装。editable 安装后可从其他工作目录使用 `python -m tasks.icd.main`。任务内旧 `setup.py` 现在只显示迁移说明，不再执行不完整的独立安装。

装好 Hydra 后可检查实际组合配置而不加载模型或训练依赖：

```console
python -m tasks.icd.main experiment=mimiciii_full/plm_icd --cfg job --resolve
python -m tasks.icd.main --hydra-help
```

训练示例（需先备齐下列数据与权重）：

```console
python -m tasks.icd.main experiment=mimiciii_full/plm_icd gpu=[0] debug=true
python -m tasks.icd.main experiment=mimiciv_icd10/plm_icd data.dir=/path/to/prepared/icd10 data.split_dir=/path/to/splits gpu=[0] debug=true
```

`debug=true` 沿用原代码的 W&B disabled 模式。原配置 GPU `[6]` 保留，示例显式覆盖为 `[0]`；训练仍会使用原来的 CUDA/AMP 逻辑。主 `data: defaults` 为空，实际运行必须选择实验或具体数据配置。

## 路径与数据约定

所有默认路径相对于仓库根目录解析，不依赖启动时的工作目录：

| 用途 | 默认路径 | Hydra 覆盖 |
| --- | --- | --- |
| 预处理语料 | `data/icd/datasets/<dataset>/` | `paths.data_root` 或 `data.dir` |
| 切分/子集文件 | `data/icd/splits/<dataset>/` | `paths.split_root` 或 `data.split_dir` |
| 模型权重 | `models/icd/` | `paths.model_root`、`model.configs.model_path`、`text_transform.configs.model_path` |
| 训练结果 | `runs/icd/` | `experiment_dir` 或 `paths.run_dir` |

也可设置 `BIBM25_ICD_DATA_ROOT`、`BIBM25_ICD_SPLIT_ROOT`、`BIBM25_ICD_MODEL_ROOT`、`BIBM25_ICD_RUNS_DIR`；相对值仍以仓库根目录为基准。报表脚本使用这些环境变量，独立于训练进程的 Hydra 覆盖。MIMIC 原始数据目录可用 `BIBM25_MIMICIII_RAW`、`BIBM25_MIMICIV_RAW`、`BIBM25_MIMICIV_NOTE_RAW` 覆盖。W&B 报表项目用 `BIBM25_WANDB_PROJECT`；训练的 W&B 项目仍由原 callbacks 配置指定。

原训练文件名保留：

| 配置 | 语料文件 | 默认切分文件 |
| --- | --- | --- |
| mimiciii_50 | mimiciii_cluster3.feather | mimiciii_50_splits.feather |
| mimiciii_full | mimiciii_cluster3.feather | mimiciii_full_splits.feather |
| mimiciii_clean | mimiciii_clean.feather | mimiciii_clean_splits.feather |
| mimiciv_icd9 | mimiciv_icd9_clusters3.feather | mimiciv_icd9_split.feather |
| mimiciv_icd10 | mimiciv_icd10_clusters3.feather | mimiciv_icd10_split.feather |

仓库只提供切分/子集文件；不包含完整临床语料和 RoBERTa 权重。默认 PLM 模型目录名仍为 `RoBERTa-base-PM-M3-Voc-distill-align-hf`。语料必须包括 `_id`、`text`、`target`、`num_words`、`num_targets`、`difficulty` 及配置中的 ICD 列；数据管道沿用 `_id` inner join。`difficulty` 的训练类别沿用 `easy/middle/hard`。旧外部配置未提供 `split_dir` 时，底层读取函数仍回退到 `data.dir`。

## 实验、报表与保留差异

`experiments/curriculum/` 保存原研究脚本、notebook 和图片。这些是 legacy 实验记录：原主机绝对路径、部分 conda `sys.path`、历史输出及人工处理流程保留，不纳入活动入口可运行性的承诺。不要直接对受保护原始数据运行这些脚本。迁移没有重新执行 notebook 或修改其算法。

活动源码的导入改为 `tasks.icd.src.*`，报表改为 `tasks.icd.reports.*`。W&B sweep 的程序入口改为 `python -m tasks.icd.main`；报表图输出至 `runs/icd/images/`。报表可按模块运行，例如 `python -m tasks.icd.reports.plots.mimiciii_clean.training_data_plot`，但仍需要原 W&B 运行记录和语料。原报表中 `mimiciii_50` 使用 `mimiciii_full.feather` 的文件名差异保留，未推定改成其他文件。

本次仅整理工程结构和路径。`trainer.py` 的 hard 初始化、Micro-F1、β=0.7、窗口长度3、`<=` 判断，以及 callbacks 的原指标和早停条件均保留；它们与论文的差异应另行记录和验证。`trainer-test.py`、`trainer_old.py` 也保留。checkpoint 仍不包含课程阶段或窗口历史，因此不应声称现有续训完全等价于未中断训练。

## 验证边界

标准库路径测试验证导入无重依赖、跨工作目录路径、独立 split 解析和帮助入口。历史 `pytest` 测试需要训练依赖，`test_factories.py` 部分用例无条件要求 CUDA；缺失的 `prepare_data.utils` 上游源码使对应预处理测试显式跳过。未凭空补写预处理逻辑，未声称原科学测试或模型训练已通过。
