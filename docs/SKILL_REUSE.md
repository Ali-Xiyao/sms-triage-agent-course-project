# 外部技能与项目任务技能的对应关系

## 实际采用的外部技能

采用K-Dense Inc. Scientific Agent Skills中的scikit-learn技能1.5，固定提交a25cf9ec00e2745d03f8704acf6f6919694ccd9a。来源、原文SHA256和许可元数据见`reused_skills/scikit-learn/SOURCE.json`。SKILL.md及两份已阅读参考文件原样保留；技能头声明BSD-3-Clause，仓库MIT许可全文另存，二者不混称为统一许可证。没有拷贝上游脚本。

这是补充阶段使用的**开发与实验复核技能**，不声称最初训练就使用了它，也不声称它在运行短信时选工具。

输入：原始公开数据、规范化分组划分、冻结Pipeline、目标指标和已有实验脚本。输出：适配后的离线审计程序、检查结果及对限制的说明。Skill本身是程序性指南，不是预测器，不会自动执行命令。其上游示例需要Python、sklearn及本地文件权限；安装要联网但没有服务凭据。项目只阅读和适配相关指南，不执行未知脚本，不提供短信或密钥给外部站点。

## 阅读与适配

1. 阅读SKILL.md的Pipeline、防止泄漏、独立单元划分、适当指标和持久化说明。
2. 阅读references/pipelines_and_composition.md的按折拟合、训练/推理分离；阅读model_evaluation.md的正类、PR指标、CV标准差及可信模型加载限制。
3. 上游针对sklearn1.9.1，交付模型在1.7.2训练；为保存既有实验及joblib兼容性，没有升级或执行其uv安装命令。采用1.7.2可用的clone、fit、transform及metrics；没有引入表格StandardScaler或不相关聚类功能。
4. 文本任务改用字符TF-IDF＋LR，维持稀疏矩阵；独立单元改为短信规范化键。数据缺少来源/时间，明确近似模板仍可能跨集合。

## 实际行为核验

运行`python src/audit_skill_reuse.py`，`results/skill_reuse_audit.json`记录7项检查：

- 仅训练文本重新拟合的词表与冻结词表一致；IDF也一致。
- 留出标记只做transform，不写入词表。
- 冻结测试分数逐条重放与原CSV一致；Precision、Recall、F1复算一致。
- 规范化分组ID跨集合无重复；受信任模型运行前后字节哈希不变。

所有检查通过。不由此声称不存在语义模板泄漏，或Skill提高了F1。其贡献是把过程要求转成可复核检查，性能贡献未另行归因。

## 与自编运行技能和工具库的区别

| 对象 | 来源 | 发生阶段 | 实际作用 |
|---|---|---|---|
| scikit-learn外部Skill1.5 | K-Dense上游固定提交 | 补充开发复核 | 指导Pipeline、隔离和评价核验 |
| docs/SKILL.md | 本项目自编2.0 | 每次LLM运行 | 输入边界、工具契约、证据和人工确认 |
| sklearn1.7.2实现 | scikit-learn维护者 | 训练及预测 | 特征提取、分类器及指标 |
| API执行器与规则 | 本项目实现 | 运行 | 消息循环、权限、异常与证据校验 |

自编技能不能冒充外部复用；外部指南也不能冒充短信自动规划。组件来源、版本及许可见THIRD_PARTY.md。

引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 。2026-11-08核验当前arXiv记录为v2，未标期刊出版。
