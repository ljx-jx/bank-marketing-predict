# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**：这是项目的"身份档案"。AI 接管项目时先读这里，了解项目目标、技术栈、目录、部署取值。
> **更新时机**：架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。

---

## 1. 项目是什么

- **项目名称**：`bank-marketing-predict`
- **一句话目标**：基于银行营销历史数据，提供交互式数据分析和在线预测（客户是否会认购定期存款）的 Web 应用。
- **使用者/受益者**：银行业务人员 / 营销决策者 — 通过数据洞察辅助营销决策，通过预测模型预判客户认购意向。
- **核心功能**：
  - **数据分析交互页面**：上传/加载银行营销数据后，提供描述性统计、特征分布、相关性分析、客户分群等交互式可视化探索能力。
  - **在线预测系统**：基于历史数据离线训练分类模型（预测 `subscribe`），构建点选式输入表单，用户输入客户特征后实时返回认购预测结果及其置信度。
- **输入/数据**（如有）：`data/train.csv`（22500 条）、`data/test.csv`（7500 条）— 银行客户营销通话记录与认购标签。数据含 20 个输入特征 + 1 个目标列 `subscribe`（yes/no）。**数据不进 Git**（`.gitignore` 排除），本地开发时直接读取。

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 课程指定版本，生态成熟，ML/数据分析库丰富 |
| Web/应用框架 | Streamlit | 纯 Python 即可构建数据应用，内置交互组件，适合数据分析和 ML 演示场景 |
| 数据处理 | pandas, numpy | 数据分析标配 |
| 可视化 | plotly / matplotlib | 交互式图表，与 Streamlit 原生集成好 |
| 机器学习 | scikit-learn | 分类模型训练（逻辑回归、随机森林等），模型持久化（joblib/pickle） |
| 测试 | pytest | 课程指定，标准 Python 测试框架 |
| 格式/静态检查 | ruff | 课程指定，统一格式化 + lint |
| 打包/运行 | Docker | 课程指定，本地部署用 |
| CI/CD | GitHub Actions | 课程指定，只做 CI（不做 CD） |

## 3. 目录地图

```text
bank-marketing-predict/
├── standards/                  # AI 项目记忆与通用规范
│   ├── README.md
│   ├── 00-project-context.md
│   ├── 01-requirements.md
│   ├── PROGRESS.md
│   ├── 02-coding-standards.md
│   ├── 03-testing-standards.md
│   ├── 04-git-workflow.md
│   ├── 05-cicd-standards.md
│   ├── 06-ai-collab-protocol.md
│   └── templates/
├── data/                       # 原始数据（不进 Git）
│   ├── train.csv
│   └── test.csv
├── src/                        # 应用源码
│   ├── __init__.py
│   ├── app.py                  # Streamlit 主入口（多页面）
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── 01_data_analysis.py # 数据分析页面
│   │   └── 02_prediction.py    # 在线预测页面
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py            # 离线训练脚本
│   │   └── predict.py          # 预测逻辑
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py      # 数据加载与校验
│       └── preprocessing.py    # 特征工程/预处理
├── models/                     # 训练产物：模型文件（不进 Git）
├── tests/                      # 测试目录
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_predict.py
│   └── test_app.py
├── requirements.txt            # 生产运行依赖
├── requirements-dev.txt        # 本地/CI 检查依赖（pytest, ruff 等）
├── Dockerfile
├── .github/workflows/
│   └── ci.yml                  # CI only（无 CD）
├── .gitignore
└── README.md
```

> 新增目录前先更新本节，避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | `>=80%`（核心业务逻辑必须覆盖） |
| 构建 | `docker build` 成功（仅 CI 执行，本地不强制） |
| 业务/模型指标 | 模型 AUC >= 0.70，预测接口返回概率 + 分类结果 |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**，只进 GitHub Secrets / 环境变量。
- **大文件、数据集（`data/`）、模型产物（`models/`）不进 Git**，在 `.gitignore` 中排除。
- `main` 分支受保护，日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。
- 本项目**只做 CI，不做 CD**；本地 `docker run` 部署验证。

## 6. 部署/CI 占位符取值

> `guides/` 和 workflow 里的通用占位符，在本项目里的真实值只写这里。

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `bank-marketing-predict` | 应用名/镜像名/容器名 |
| `<DEPLOY_DIR>` | 无 | 本项目不做远程 CD，仅本地部署 |
| `<PORT>` | `8004` | 服务端口 |
| `<PYVER>` | `3.11` | Python 版本 |
| `<HEALTHCHECK>` | `/healthz` | 健康检查端点（Streamlit 应用内实现） |
| `<SSH_USER>` | 无 | 本项目不做远程 CD |
| `<SSH_HOST>` | 无 | 本项目不做远程 CD |
