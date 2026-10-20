TACL 后续研究资料（截至2026-10-20）

research_content.json：中文方案书排版源，14节正文、2个附录；数学公式同时提供LaTeX与纯文本。
literature.csv：22篇一手文献，含状态、机制、资源限制、适用点、novelty边界与证据定位。
references.bib / references.json：同一22篇文献的引用格式。
experiment_matrix.csv：30项规格行，含明确InfoBatch同风险适配及可选BLS代理；B4两行复用B2，额外训练预算为0；均未运行训练。
implementation_checklist.csv：29项实现与验证要求；均为planned_not_implemented。
qa_report.json：结构、引用及元数据一致性检查结果；不代表算法或模型测试通过。

本研究资料工作只进行项目/文献只读调研与研究资料撰写；主任务已经实际重构仓库，另有记录。研究资料工作没有实现新课程、运行GPU训练或下载受限医疗数据。
研究中的默认配置、预算、成功阈值与复杂度量级是设计；未提供任何编造的准确率、速度、显存或GPU小时。
ADO按ICLR2025正式页记载，Data Agent按ICML2026/PMLR正式页记载。
InfoBatch按ICLR2024正式页及官方PDF记载；BLS按作者arXiv中CVPR2026 accepted声明记载，未独立核实CVF正式收录。
TACL作者终稿确认BIBM2025 Regular；IEEE页面本次触发验证页；MRR论文代码链接本次返回404。
本资料的PDF由主任务单独排版生成，结构化JSON是正文源。
数学复核已纳入：u=0版本等价r；B5为固定b0/B6为动态债务；κ=1端点二者等价；抽样前定有效批大小保预算；EMA年龄衰减只读；原TACL与同风险adaptation分表；TCM采用argmax。
