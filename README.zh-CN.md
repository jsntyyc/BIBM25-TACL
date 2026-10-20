# TACL 论文代码整理版

本仓库对应 BIBM 2025 论文 [TACL](https://arxiv.org/abs/2510.15269v2)，从上游提交 `6730635` 完整克隆并保留历史。当前整理分支为 `refactor/project-organization`。

## 使用顺序

1. 阅读 [论文与实现对照](docs/implementation_audit.md)，区分发布代码行为和论文描述。
2. 按任务阅读 [ICD](tasks/icd/README.md)、[再入院](tasks/readmission/README.md)、[TCM](tasks/tcm/README.md) 指南，建立独立依赖环境。
3. 先运行入口 `--help` 和 `--check-paths`，再准备外部数据及权重。
4. 在仓库根执行 `python scripts/check_repository.py` 与 `python -m unittest discover -s tests -v`，查看结构检查。

默认采用模块方式启动：`python -m tasks.<任务>.<入口>`。从其他目录启动时，先在目标环境执行 `python -m pip install -e . --no-deps`。这只安装仓库命名空间，不自动安装 PyTorch 或下载模型。

## 整理范围

三套代码分别放入 `tasks/icd`、`tasks/readmission`、`tasks/tcm`。数据位于 `data`，外部模型位于 `models`，新输出位于 `runs`。压缩包保存在 `archives`，原说明与许可保存在 `docs/upstream`。每个迁移文件的来源和校验和见 `docs/migration.csv`。

结构整理保留课程阶段、指标、阈值、训练超参数及 checkpoint 顺序。ICD 从 hard 初始化、若干脚本与论文不同的阶段推进条件等既有问题均在对照文档中记录，没有作为结构调整静默修改。旧模板、Notebook 和实验变体继续保留；帮助信息通过不等于完整训练已通过。

## 数据和研究交付

现有 ICD Feather 多为划分/子集文件，不能替代完整语料。TCM 只提供部分数据和原始格式样例，现代入口仍需额外训练/测试文件与模型。再入院任务需按许可自行准备 MIMIC-III 和权重。完整院外死亡预测管线尚未确认。

`docs/research` 收录截至 2026-10-20 的文献比较、方案正文与实验计划。新研究以单卡约 48GB、ICD 为主线、TCM 跨任务验证设计；新方法与 GPU 实验属于下一阶段，方案中的收益均为待验证假设。

可直接阅读 [19 页中文研究方案书 PDF](docs/research/TACL-research-proposal.pdf)，其中包含 22 篇核心文献、数学定义和分阶段实验设计。
