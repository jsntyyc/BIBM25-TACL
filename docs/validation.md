# 验证记录

验证日期：2026-10-20。上游提交：`67306350a049daf87ef5ac384b367c87d5b349eb`。测试使用标准库或工作区隔离的轻量依赖；未安装任务训练环境，也未执行模型训练。

| 项目 | 结果 |
|---|---|
| Git 历史 | 完整非 shallow 克隆；`git fsck --full` 通过 |
| 仓库结构 | 183 个任务 Python 文件语法通过；465 条来源记录；114 个资产/归档成员校验通过 |
| 公共入口 | 根检查器 34 个模块 `--help` 在 `python -S` 下通过；TCM 检查覆盖 14 个入口，合计 46 个不同入口 |
| 根测试 | 6 项通过：帮助、缺失路径副作用、原参数、历史模板预检、本地模型路径、6 个主要 Readmission 实验类/函数 AST 保持 |
| ICD 测试 | 7 项标准库路径/配置/导入检查通过；101 个 YAML 解析、30 个 Hydra 实验配置组合及默认 split 存在性通过 |
| ICD 算法对照 | 三个 Trainer 的 `fit` 与原 ZIP AST 相同；hard 初始化等既有行为保留 |
| TCM 测试 | 6 组通过：语法、14 个帮助、无训练库导入、JSONL 契约、默认值、预检副作用 |
| TCM 算法对照 | 11 个现代/课程执行体与原提交 AST 对照通过，允许变化限于 CLI、路径、设备和输出 |
| 安装/工作目录 | 根 editable 包构建成功；从外部含空格目录导入三个任务入口成功，未导入 Torch |

## 重复检查

```console
python scripts/check_repository.py --report docs/validation.json
python -m unittest discover -s tests -v
python -m unittest tasks.icd.tests.test_paths -v
python -m tasks.tcm.check_entrypoints
```

Hydra 组合检查使用临时目录中的 `hydra-core`，没有安装 Torch。旧 ICD 指标/工厂测试需要真实训练依赖，部分要求 CUDA；缺失的上游 `prepare_data.utils` 所对应测试显式标为跳过。历史源码含少数无效转义的 SyntaxWarning，未影响 AST 检查，未以格式化改变原算法。

帮助和预检通过仅证明入口契约。完整训练、旧库兼容性、checkpoint 实际恢复及论文指标尚未验证。研究资料的结构/引用检查与训练验证分开记录于 `docs/research/qa_report.json`。
