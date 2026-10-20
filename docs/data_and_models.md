# 数据和模型准备

| 任务 | 已提供 | 仍需外部准备 |
|---|---|---|
| ICD | `data/icd/splits/<dataset>/` 下 19 个 split/subset Feather | 完整处理语料、聚类难度列、对应语言模型及 tokenizer |
| TCM | `data/tcm/provided/dev.json`、知识和词表；`data/tcm/samples/` 原始格式样例 | 现代入口匹配的训练/验证/测试 JSONL 和预训练权重 |
| Readmission | 源码及数据处理流程 | 有权限的 MIMIC-III、处理后的 CSV、ClinicalBERT/BioBERT/Word2Vec 权重 |
| 院外死亡预测 | 少量历史片段 | 完整管线、MIMIC-IV 任务定义和可核对划分 |

ICD 语料默认位于 `data/icd/datasets/<dataset>/`，split 单独位于 `data/icd/splits/<dataset>/`。原文件名保留；请用 Hydra `data.dir`、`data.split_dir` 指定外部目录。不要把 split 文件误当作完整语料。管道预期 `_id/text/target/num_words/num_targets/difficulty` 及对应编码列，按 `_id` 关联划分。

TCM 原始样例包含 `example_id/lcd_id/lcd_name/syndrome/chief_complaint/medical_history/FDMR/norm_syndrome` 等字段，没有现代训练所需的 `TEXT/LABEL`。这意味着仍需要文本构建与标签编码映射，不能仅改路径后声称可训练。词表和停用词仍保留为任务内代码资源。

Readmission CSV 约定 `ID,TEXT,Label`。CL 文件需匹配脚本的 difficulty 约定；不同评分算法的输出不等价。

研究使用原官方划分作可比较主结果，并审计患者跨划分和文本重复；若发现问题，另报患者互斥的敏感性实验，清楚区分协议。聚类、标准化、标签频率和检索索引只能在训练部分拟合。验证/校准流程固定，测试集在方法和超参数冻结后评估。

新下载的患者数据、模型权重与训练输出默认不进入 Git。遵守原数据及模型的访问和许可条件。
