# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**：这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里，不要另起多个 PRD 文件。
> **更新时机**：每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 用户需求：银行营销数据分析 + 在线预测系统 | 写成用户故事 |
| 缺陷 Bug | 测试 / 线上日志 / 用户反馈 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | 开发 / Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR，等待 CI 和 Review |
| 合并 | Done | PR 合并 main，自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**：分支名带 Issue 号，PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>，
我想要 <能力>，
以便 <价值>。

验收标准：
- AC1: Given <前提>，When <动作>，Then <可验证结果>。
- AC2: <补充标准>

技术备注：
- <可选：约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI · 状态: Backlog

作为 **项目开发者**，
我想要 项目具备基础工程结构、测试、CI，
以便 后续每次开发都能自动检查代码质量与测试结果。

验收标准：
- AC1: 项目根目录包含 `src/`、`tests/`、`requirements.txt`、`requirements-dev.txt`、`Dockerfile`、`.gitignore`。
- AC2: `.github/workflows/ci.yml` 存在，PR 触发 CI，包含 `ruff format --check .`、`ruff check .`、`pytest --cov --cov-fail-under=80`、`docker build`。
- AC3: `ruff format --check . && ruff check .` 本地通过无报错。
- AC4: `pytest` 本地全绿（含覆盖率 >=80%）。
- AC5: CI 全绿后合并 main（由人工合并）。

### US-2 数据分析交互页面 · 状态: Backlog

作为 **银行业务分析人员**，
我想要 在浏览器中交互式地探索银行营销数据，
以便 快速了解客户特征分布、认购率趋势、关键影响因素，辅助营销决策。

验收标准：
- AC1: Given 应用启动后访问首页，When 页面加载，Then 展示数据分析页面。
- AC2: Given 加载数据，When 页面渲染，Then 显示整体认购率（subscribe=yes 占比）的概览卡片（KPI）。
- AC3: Given 数据已加载，When 用户浏览，Then 可看到各数值特征的描述性统计（均值、标准差、分位数等）。
- AC4: Given 数据已加载，When 用户选择某个分类特征，Then 展示该特征各取值下的认购率对比柱状图。
- AC5: Given 数据已加载，When 用户选择某数值特征，Then 展示该特征的分布直方图（按认购与否分层）。
- AC6: Given 数据已加载，When 用户查看相关性，Then 展示数值特征间的相关性热力图。
- AC7: Given 数据已加载，When 用户选择两个特征，Then 展示散点图/箱线图等适合的可视化。

技术备注：
- 使用 Streamlit 的 `st.plotly_chart` 实现交互式图表。
- 数据加载应有缓存（`@st.cache_data`），避免每次交互都重新读取 CSV。

### US-3 模型离线训练 · 状态: Backlog

作为 **数据科学家**，
我想要 基于历史训练数据训练一个认购预测分类模型，并保存模型文件，
以便 后续在线预测系统可以加载已训练好的模型进行实时预测。

验收标准：
- AC1: Given `data/train.csv` 存在，When 运行训练脚本，Then 完成数据预处理（缺失值处理、编码、标准化等）。
- AC2: Given 训练数据已预处理，When 训练完成，Then 模型在 `data/test.csv` 上的 AUC >= 0.70。
- AC3: Given 训练完成，When 保存模型，Then 模型文件（含预处理管道）持久化到 `models/` 目录。
- AC4: Given 模型已保存，When 加载模型并预测，Then 输出包含预测类别（yes/no）和预测概率。

技术备注：
- 使用 scikit-learn 的 `Pipeline` 组合预处理 + 分类器，方便整体序列化/反序列化。
- 候选模型：逻辑回归（基线）、随机森林、LightGBM（如 AUC 不足则升级）。
- 训练脚本可独立运行：`python -m src.model.train`。

### US-4 在线预测系统页面 · 状态: Backlog

作为 **银行客户经理**，
我想要 通过点选式的表单输入客户特征，点击预测按钮后获得该客户是否会认购的预测结果，
以便 在日常工作中快速评估客户意向，优先跟进高意向客户。

验收标准：
- AC1: Given 应用已启动且模型已训练，When 访问预测页面，Then 显示客户特征输入表单（点选/下拉/滑块，覆盖所有模型输入特征）。
- AC2: Given 表单已填写完整，When 点击"预测"按钮，Then 返回预测结果：认购/不认购 + 认购概率。
- AC3: Given 表单未填写完整，When 点击"预测"按钮，Then 显示友好的校验提示，不会报服务器错误。
- AC4: Given 预测结果已展示，When 用户修改输入并再次预测，Then 结果实时更新。
- AC5: Given 模型文件不存在，When 访问预测页面，Then 显示"请先训练模型"的提示，不崩溃。

技术备注：
- 分类特征用 `st.selectbox`/`st.radio`，数值特征用 `st.slider`/`st.number_input`。
- 表单默认值设为各特征的中位数/众数，方便快速试用。
- 预测输入需经过与训练时相同的预处理管道。

### US-5 应用入口与健康检查 · 状态: Backlog

作为 **运维人员**，
我想要 应用有统一的入口、清晰的导航和健康检查端点，
以便 确认服务正常运行，并可快速诊断故障。

验收标准：
- AC1: Given 应用启动，When 访问根路径，Then 显示应用标题、功能导航（数据分析 / 在线预测）。
- AC2: Given 应用运行中，When 访问 `/healthz`，Then 返回 HTTP 200 且 body 含 `{"status": "ok"}`。
- AC3: Given 任何未捕获异常，When 发生错误，Then 页面显示友好错误信息而非堆栈跟踪。
- AC4: Given `docker run -p 8004:8501 bank-marketing-predict:latest`，When 容器启动，Then `curl http://localhost:8004/healthz` 返回 200。

技术备注：
- Streamlit 默认端口为 8501（容器内），宿主机映射到 8004。
- 健康检查端点在 `src/app.py` 中通过 Streamlit 的 `?healthz=` 查询参数或独立 Flask/FastAPI 端点实现。若 Streamlit 原生不支持，可在 Docker 中额外启动一个轻量健康检查 HTTP 服务，或使用 `/` 路径的 HTTP 200 作为健康信号。

---

## 5. 非功能需求

- **安全**：密钥只进 Secrets，不进 Git。本项目无外部 API 密钥需求。
- **可维护**：一需求一小 PR，避免大爆炸式提交。
- **可测试**：核心逻辑必须有单元测试（数据加载、预处理、模型训练、预测逻辑）。
- **可部署**：Docker 容器化，本地 `docker run` 后健康检查通过。
- **性能**：页面首次加载（含模型加载）应在 15 秒内完成；单次预测响应应在 3 秒内完成。
- **数据隐私**：原始数据与模型产物不进 Git（`.gitignore` 排除）。
