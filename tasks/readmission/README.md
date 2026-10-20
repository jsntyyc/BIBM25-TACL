# 再入院预测与历史基线

本目录保留 ClinicalBERT 再入院代码、Word2Vec/CNN/传统分类器及课程评分变体。原说明在 `docs/upstream/readmission/README.md`。原始训练执行体位于各目录的 `_impl/`，同名公共模块负责轻量参数解析、路径预检及显式启动。请使用公共模块，避免直接运行 `_impl`。

## 环境与启动

建议使用独立 Python 3.10 环境，先在仓库根安装命名空间 `python -m pip install -e . --no-deps`，再按 `requirements.txt` 准备任务依赖。依赖清单来自静态 import 审计，尚未锁定或验证完整训练环境。

```console
python -m tasks.readmission.run_readmission --help
python -m tasks.readmission.run_readmission_CL --help
python -m tasks.readmission.curriculum_learning.distance_discharge --help
python -m tasks.readmission.CNN --help
python -m tasks.readmission.preprocess --help
```

完整 BERT 入口保留原参数名及训练默认值。例如先检查资源：

```console
python -m tasks.readmission.run_readmission --task_name readmission --readmission_mode discharge --data_dir data/readmission/discharge --bert_model models/readmission/pretraining --do_train --check-paths
```

准备完资源后移除 `--check-paths` 启动原训练流程。CL 版额外支持 `--difficulty-file`，默认仍为原来的 `train_with_difficulty_distance_discharge.csv`。这项覆盖只改变输入路径，不统一不同评分脚本的难度算法。

未带原 argparse 的基线增加 `--train-file`、`--val-file`、`--test-file`（原脚本具有相应变量时）、`--word2vec-model`、`--output-dir` 和 `--device`。`--device cpu` 或 `--device cuda:0` 可覆盖旧设备字符串；不传时保留原设备默认值。现有 BERT 参数使用 `--no_cuda`、`--gpu`。

预处理输入为 MIMIC-III 1.4 的 `ADMISSIONS.csv` 和 `NOTEEVENTS.csv`：

```console
python -m tasks.readmission.preprocess --raw-dir /path/to/mimiciii/1.4 --output-dir data/readmission --check-paths
```

检查通过后去掉 `--check-paths` 才会创建生成文件。原预处理划分逻辑保留；正式研究前需另行审计患者是否跨划分。

## 路径约定

默认外部数据为 `data/readmission/`，模型为 `models/readmission/`，输出为 `runs/readmission/<入口名>/`。路径均锚定仓库，CLI 相对输入路径按调用者目录解析。原脚本的相对图表和 checkpoint 写入实验输出目录，不再散落在调用目录。历史 `dead/data40k` 默认仍指向独立缺失数据集，不擅自替换成 discharge。

可在启动前设置 `TACL_READMISSION_DATA_DIR`、`TACL_READMISSION_MODEL_DIR`、`TACL_MIMICIII_RAW_DIR`。显式 CLI 文件参数用于覆盖特定输入。每次实验使用不同 `--output_dir`；BERT 原代码拒绝非空输出目录。

## 数据与模型前提

CSV 列为 `ID,TEXT,Label`，CL 训练数据另需 `difficulty`。训练、验证、测试文件和模型均未包含在此任务中。BERT 模型预期本地 `pytorch_model.bin`、`bert_config.json` 和 tokenizer 词表。Word2Vec 基线还需 `word2vec.model`。MIMIC 下载资格及数据许可由使用者按官方流程准备。

## 已知行为与兼容限制

- `pytorch_pretrained_bert` 属于旧库；源码同时包含较新的 Torch API、旧 `scipy.interp` 和 pandas 用法。帮助及路径检查通过不能证明这些版本已兼容。
- CL 版按验证损失上升推进阶段，不等同论文的 Macro-F1 增长率公式。
- `best_model` 命名与实际最后保存参数并不总一致；纯 `--do_eval` 流程可能仍依赖输出目录中的训练 checkpoint。结构整理保留这些行为，详见根目录实现审计。
- `--readmission_mode` 原默认仍为 `None`，执行训练时请显式填写；未静默修改原保存命名逻辑。
- `dead/`、可视化脚本和 Notebook 为历史材料；没有确认完整的院外死亡预测训练管线。
- `test.py` 是模型下载示例，不是测试套件；仅显式运行该入口时才执行其原下载逻辑。

验证范围为轻量入口、路径和控制流程一致性；未运行完整 GPU 训练。
