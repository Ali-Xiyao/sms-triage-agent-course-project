# 组件复用与来源

新增LLM模式通过Python标准库实现OpenAI兼容API工具调用协议，API Key和模型ID留空，真实服务尚未运行。未引入第三方Agent框架或复制SDK代码。项目中的短信分流技能是本项目编写且运行时读取的任务契约；实际复用的课程工具是scikit-learn的特征提取和Pipeline。下表区分第三方实现与项目编排。

| 名称 | 版本 | 许可 | 来源 | 用途及适配 |
|---|---|---|---|---|
| SMS Spam Collection | 2011 / SHA256见metrics | CC BY 4.0 | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | 公开英文分类语料；保留原作者归属及readme；规范化去重、排除标签冲突 |
| scikit-learn | 1.7.2 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn/tree/1.7.2 | TfidfVectorizer、Pipeline、NB、LR、评估与分层划分；自行配置参数和拒判工作流 |
| NumPy | 2.3.5 | BSD-3-Clause | https://github.com/numpy/numpy/tree/v2.3.5 | 稀疏贡献计算、bootstrap、指标统计 |
| SciPy | 运行环境版本见metrics | BSD-3-Clause，附属代码许可见安装包 | https://github.com/scipy/scipy | sklearn的稀疏矩阵和优化基础 |
| pandas | 2.3.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas/tree/v2.3.3 | 表格解析与CSV导出 |
| joblib | 1.5.2 | BSD-3-Clause | https://github.com/joblib/joblib/tree/1.5.2 | 本地模型持久化；不接受用户上传模型 |
| Matplotlib | 3.10.7 | Matplotlib license（基于PSF） | https://matplotlib.org/stable/project/license.html | 实验图表；图表脚本为本项目编写 |
| Python | 3.12系列 | PSF | https://www.python.org/psf/license/ | HTTP服务、URL静态解析、输入校验、unittest |
| python-docx | 运行时版本见许可证导出 | MIT | https://github.com/python-openxml/python-docx | 报告构建辅助；非Agent运行依赖 |
| docxcompose | 2.2.0 | MIT | https://github.com/4teamwork/docxcompose | 合并学校封面与报告正文；仅报告构建，不参与Agent推理 |

所有第三方库均通过公开Python包仓库安装，未将其源码复制为本项目原创代码。完整许可文本与分发包元数据保存在 `docs/licenses/`；若不同平台依赖版本变化，应核对实际安装包的许可证。数据作者为Tiago Almeida和José María Gómez Hidalgo；本项目对数据做规范化分组和抽样处理。CC BY 4.0全文：https://creativecommons.org/licenses/by/4.0/legalcode 。数据许可不代表可忽略原文隐私，本项目界面不存储输入，错误展示做规则脱敏。

## 深入分析所复用的文本分类工具

阅读范围为已安装scikit-learn 1.7.2中 `feature_extraction/text.py` 的TfidfVectorizer和 `pipeline.py` 的Pipeline，及官方API说明。输入为一维字符串列表；fit_transform学习训练词表和文档频率，transform产生CSR稀疏矩阵。Pipeline负责按顺序执行变换和分类，但不提供语言识别、隐私、错误恢复或人工确认。

权限：训练只读本地公开语料、写本地模型；推理只读取本地模型，不需网络、云服务或密钥。joblib加载本质涉及反序列化，所以只加载固定路径受信任模型。链接工具调用urllib.parse.urlsplit只拆解字符串，不执行HTTP请求。

适配：比较词级与字符级特征，设置min_df=2限制罕见噪声，sublinear_tf降低重复词支配，LR加入balanced权重和C=8；训练与测试严格分离。项目自行实现双阈值、静态链接风险覆盖、OOV与中文回退、可解释贡献和结构化轨迹。

行为核验：tests验证确定性、训练模型加载、异常输入、工具注入失败、无已知词、中文越界、短链覆盖、数据划分无重复。`results/component_audit.json`记录源码文件、SHA256和对应组件身份。`model_only`→`reject_only`→`full_agent`的消融直接验证模型与编排各自贡献，不将拒判视作预测成功。

局限：TF-IDF不理解语用和上下文；字符片段解释可读性较差；predict_proba未做独立校准，只显示模型分数；静态域名规则不能确认恶意，也会要求复核正常短链接。

## LLM协议与服务边界

协议来源：OpenAI Function calling，https://developers.openai.com/api/docs/guides/function-calling ，核对日期2026-11-07。HTTP适配、工具执行器、ReviewStore与模拟客户端由本项目实现。没有提供大模型权重或第三方服务许可；真正调用时须遵守配置服务商的条款。仅本地分类和demo路径可离线；llm模式需要HTTPS访问配置的服务，发送短信和工具结果。
