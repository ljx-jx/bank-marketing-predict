# bank-marketing-predict

银行营销数据分析与认购预测系统

基于银行客户营销通话数据，提供交互式数据分析和在线预测（客户是否会认购定期存款）。

## 技术栈

- **Python 3.11** + **Streamlit** (Web 应用)
- **scikit-learn** (机器学习 · RandomForest AUC=0.89)
- **pandas + plotly** (数据分析与可视化)
- **pytest + ruff** (测试与代码质量)
- **Docker** (容器化部署，可选)

---

## 本地部署（推荐）

### 前置条件

- Python 3.11 环境（推荐 conda，也可用系统 venv）
- 项目根目录下 `data/train.csv` 和 `data/test.csv` 已就位

### 步骤 1：创建 Python 环境

**方式 A：conda（推荐）**
```bash
conda create -y -n bank-marketing-predict python=3.11
conda activate bank-marketing-predict
```

**方式 B：系统 venv**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

> 如果 `conda create` 报 `CondaToSNonInteractiveError`，先执行：
> ```bash
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
> ```

### 步骤 2：安装依赖

**国内用户（清华源，更快）：**
```bash
pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**海外用户：**
```bash
pip install -r requirements-dev.txt
```

### 步骤 3：训练模型

```bash
python -m src.model.train
```

输出示例：
```
Loaded 22500 samples, 20 features
Positive class ratio: 13.12%
Train: 18000, Validation: 4500 (stratified)
RF   AUC=0.8905  Accuracy=0.8818
Model saved to models/model_pipeline.pkl
SUCCESS: AUC meets the 0.70 threshold.
```

> 训练完成后，模型保存为 `models/model_pipeline.pkl`（约 50 MB），模型指标保存在 `models/metrics.json`。

### 步骤 4：启动应用

```bash
streamlit run src/app.py --server.port 8004
```

输出：
```
  Local URL: http://localhost:8004
  Network URL: http://192.168.x.x:8004
```

浏览器打开 **http://localhost:8004** 即可使用。

---

## Docker 部署（可选）

确保已安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 并正在运行，然后：

```bash
# 1. 先在本机训练好模型（同上步骤 1-3）

# 2. 构建镜像
docker build -t bank-marketing-predict:latest .

# 3. 启动容器（宿主机 8004 → 容器内 8501）
docker run -d --name bank-pred -p 8004:8501 bank-marketing-predict:latest

# 4. 验证健康检查
curl http://localhost:8002/healthz
# 返回: {"status": "ok"}
```

浏览器打开 **http://localhost:8004**。

### Docker 常用命令
```bash
docker logs bank-pred        # 查看日志
docker stop bank-pred         # 停止
docker start bank-pred        # 启动
docker rm -f bank-pred        # 删除容器
```

---

## 页面功能

| 页面 | 路径 | 功能 |
|---|---|---|
| 🏠 首页 | `/` | 系统介绍 + KPI 概览 |
| 📊 数据分析 | 左侧导航 | 特征分布、认购率对比、相关性热力图、散点/箱线图 |
| 🔮 在线预测 | 左侧导航 | 填写客户特征 → 预测认购概率 |

---

## 本地开发

```bash
# 格式检查 + 静态检查
ruff format --check .
ruff check .

# 运行所有测试
pytest

# 覆盖率报告
pytest --cov=src --cov-fail-under=80 --cov-report=term
```

---

## 项目结构

```text
bank-marketing-predict/
├── data/                           # 原始数据（不进 Git）
│   ├── train.csv                   # 22,500 条训练数据
│   └── test.csv                    # 7,500 条测试数据
├── models/                         # 训练产物（不进 Git）
│   └── model_pipeline.pkl          # 训练后的 sklearn Pipeline
├── src/
│   ├── app.py                      # Streamlit 入口
│   ├── health.py                   # /healthz 健康检查端点
│   ├── pages/
│   │   ├── 01_data_analysis.py     # 数据分析页面
│   │   └── 02_prediction.py        # 在线预测页面
│   ├── model/
│   │   ├── train.py                # 模型训练脚本
│   │   └── predict.py              # 预测逻辑
│   └── utils/
│       ├── data_loader.py          # 数据加载
│       └── preprocessing.py        # sklearn 预处理管道
├── tests/                          # 测试（58 个用例，87% 覆盖率）
├── requirements.txt                # 生产依赖
├── requirements-dev.txt            # 开发/CI 依赖
├── Dockerfile                      # Docker 镜像
├── .github/workflows/ci.yml        # GitHub Actions CI
└── standards/                      # 项目规范文档
```
