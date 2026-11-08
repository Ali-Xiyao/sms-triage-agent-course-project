# 短信分流与人工复核 Agent

机器学习课程项目。**大模型通过工具调用执行任务，本地TF-IDF＋逻辑回归提供分类证据，用户确认或修正后完成任务。** 原离线分类与规则流程保留为对照。已完成DeepSeek Flash真实工具调用实验和网页确认演示；公开配置Key留空，默认模型为低价Flash。

## 运行

建议Python 3.12。模型与公开数据已提供。首次安装需要网络。

```sh
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python src/app.py
```

打开 http://127.0.0.1:8765 。默认选中真实API模式；未配置时返回`configuration_required`。可以明确选择“离线协议演示”，验证工具执行、人工确认和报告导出；该模式的模型响应由脚本模拟，不是大模型输出。仅加载随项目提供的可信joblib文件。

## 配置真实大模型

```sh
cp .env.example .env
```

在`.env`中填写自己的配置，API Key只在本地填写，禁止提交：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_TIMEOUT_SECONDS=30
LLM_MAX_TURNS=6
LLM_MAX_OUTPUT_TOKENS=700
```

`LLM_MODEL`填服务商实际提供、支持Chat Completions函数工具调用的模型ID。`LLM_BASE_URL`为服务商HTTPS根路径（通常包含`/v1`，不含`/chat/completions`）。环境变量优先于.env。配置后重启。DeepSeek标准接口使用非思考模式，关闭其beta strict字段；本地证据与参数检查仍执行。其他兼容服务未全面验证。

真实模式会把短信、任务目标与工具结果发给该API服务；界面已提示。密钥不进入浏览器、任务结果或公开文件。不访问短信中的URL，不发送或删除短信。

```sh
# 有密钥时，真实大模型工具调用
python src/app.py --mode llm --text "Free prize cash now https://bit.ly/example"
# 无密钥时，模拟模型响应，但执行真实本地工具
python src/app.py --mode demo --text "Free prize cash now https://bit.ly/example"
# 原传统工作流对照
python src/app.py --mode local --text "Free prize cash claim now"
```

## Agent如何执行

1. 读取`docs/SKILL.md`作为任务技能，提供用户目标、原始短信和工具schema。
2. 请求模型返回下一步动作，`tool_choice=auto`；工具顺序由模型返回值决定。
3. 校验函数白名单、JSON参数和任务边界，执行本地工具。
4. 以`role=tool`和匹配的`tool_call_id`把实际结果回传模型，进入下一轮。
5. 模型调用`submit_assessment`，执行器检查分类、规则及必要的链接证据是否齐全。风险与意见冲突强制转人工。
6. 进入`pending_review`；用户确认或修正后进入`completed`，可导出JSON任务报告。

最多6轮请求（可配置1至10）、16次工具调用；每次网络请求有超时。普通聊天文本不能绕过工具直接完成任务。没有开放shell、文件路径或任意联网工具。这里实现的是有边界的任务型LLM Agent，不是通用自主助手。

| 工具 | 输入 | 实际能力 |
|---|---|---|
| classify_sms | 无参数，绑定任务原文 | 真实本地分类、分数、特征、英文适用性 |
| inspect_links | 无参数，绑定原文 | 静态域名与短链等标志；不联网 |
| lookup_policy | scope/links/uncertainty/privacy | 读取项目操作规则及可引用ID；不是权威反诈知识库 |
| submit_assessment | action、summary、evidence_ids | 核验依据、提交建议、等待人工确认 |

## 验证和证据边界

```sh
python -m unittest discover -s tests -v
python src/evaluate_llm_contract.py
# 仅在配置密钥且愿意发送config/live_eval_cases.json中的公开构造样例时运行
python src/evaluate_llm_live.py
```

52项软件测试、5类模拟协议任务和7项复用审计通过。真实API证据分四批：20条修复前、同20条修复后、100条新任务、10条各3次重复，共170次批量任务执行、493次实际请求。

- 80条公开短信：local匹配77/80，LLM匹配75/80；LLM额外复核2条原ham，双方保留3条spam漏判。
- 新边界/链接10条符合固定目标；新注入10条满足工具与证据契约，不能称分类全对。
- 10条×3次动作均一致，但仅7条工具名序列相同，摘要均非逐字一致。
- 首轮发现的4条阈值误述，显式返回阈值后在修复组降至0条。没有证明LLM提升分类或用户效率。

按官方较高价格、忽略缓存的保守估算，四批API合计0.3157美元（非账单，另有两次浏览器演示）。公开响应ID、usage、全轨迹、五条自然短信偏差、冻结计划及源文件版本见[真实Agent实验](docs/真实Agent实验.md)与results/llm/。120个不同文本，重复运行不当作新增独立样本。

![DeepSeek真实API工具调用与人工确认](docs/llm/live_completed.png)

## 课程机器学习实验

公开数据原始5574条，规范化分组去重后5137条；训练/验证/测试为3082/1027/1028。仅训练集拟合词表和模型；验证集选出字符LR。独立测试Precision=0.9655、Recall=0.9032、F1=0.9333。词LR F1=0.9212，多数类0，关键词0.5417。模型冻结，没有因为加入LLM而重新选择测试结果。

原`model_only`、`reject_only`、`full_agent`三行是**离线基线流程的消融**，不代表新LLM版本；拒判没有减少测试错误。LLM不会替代课程模型输出未经评估的中文分类概率。真实API功能已实测；是否改善实际用户任务仍需独立数据和用户研究。

```sh
python src/experiment.py             # 重训，会覆盖模型和主结果
python src/plot_results.py
python src/extended_experiments.py  # 55次CV拟合、15次学习曲线拟合等补充分析
```

图表采用暖白底、陶土橙、橄榄绿与沙色，公共样式在`src/figure_style.py`；点形与线型也用于区分数据。图表依赖中文字体；Linux可安装fonts-noto-cjk。补充实验结果保存在`results/extended/`，图在`figures/extended/`；均为冻结主结果后的探索性分析。

## 文档导航

正式报告包含Agent工具循环、数学方法、主要实验与运行演示；逐条误判、完整记录和接口细节在仓库中：

- [实验补充说明](docs/实验补充说明.md)：数据清洗、参数表、55次交叉验证拟合、学习曲线、阈值分析、全部16条误判和8张实验图。
- [真实Agent实验](docs/真实Agent实验.md)：20条任务、local对照、费用、阈值解释错误与前后修复。
- [技能复用](docs/SKILL_REUSE.md)：K-Dense外部技能1.5的来源、适配及7项离线核验。
- [Agent设计](docs/LLM_AGENT.md)：模型选工具、结果回传、证据检查、人工确认及当前验证范围。
- [接口与配置](docs/接口说明.md)：配置字段、HTTP请求、返回状态及导出方法。
- [组件与许可证](docs/THIRD_PARTY.md)：依赖版本、来源、许可和适配说明。
- [答辩提纲](docs/答辩提纲.md)：演示步骤和常见问题。

实现入口为src/llm_agent.py与src/app.py；课程模型在src/core.py。主实验见results/metrics.json，补充实验见results/extended/，软件测试和浏览器记录见results/llm/。

## 提交与查阅

课程要求正式报告提交PDF；Word作为本地编辑备份。报告中的仓库地址及源码链接可点击，PDF附录列出组件版本、来源与许可证。提交时同时提供本仓库及演示材料或可运行原型。报告含个人信息，仅在本地提交包中提供。

## 限制与来源

只验证英文历史spam分类。没有近期中文诈骗数据或真人用户实验；线上LLM为旧公开语料抽样、构造任务和短期重复，无独立近期语料或真人研究。静态标志不能验证网站安全；人工确认也不自动使建议正确。LLM摘要可能出错；规则和工具白名单限制操作权限，但不保证完全消除提示注入影响。复核状态仅驻留内存，最多100条，服务重启后清除；可导出本次报告，不自动训练或永久保存私人短信。

数据：Almeida & Hidalgo (2011), SMS Spam Collection, UCI, DOI https://doi.org/10.24432/C5CC84 ，CC BY4.0。API协议参照[OpenAI官方函数调用文档](https://developers.openai.com/api/docs/guides/function-calling)，采用Python标准库自行适配，没有复制SDK或引入第三方Agent框架。详见`docs/LLM_AGENT.md`。

公开仓库：https://github.com/Ali-Xiyao/sms-triage-agent-course-project 。仓库只发布代码、公开数据与演示材料，正式身份报告仅在本地提交包中提供。

## 生成式AI辅助

Codex参与方案细化、代码与调试、实验脚本、图表和报告起草修改；DeepSeek Flash是运行与评估组件。通过可复算实验、52项测试、哈希及实际API记录核验；保留错误和限制，生成内容仍需提交者最终核对与解释。外部技能及第三方算法、数据分别归属其原作者。
