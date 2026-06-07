# bank-marketing-predict

银行营销数据分析与认购预测系统

基于银行客户营销通话数据，提供交互式数据分析和在线预测（客户是否会认购定期存款）。

## 技术栈

- **Python 3.11** + **Streamlit** (Web 应用)
- **scikit-learn** (机器学习)
- **pandas + plotly** (数据分析与可视化)
- **pytest + ruff** (测试与代码质量)
- **Docker** (容器化部署)

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt -r requirements-dev.txt

# 2. 训练模型
python -m src.model.train

# 3. 启动应用
streamlit run src/app.py --server.port 8501

# 4. 或使用 Docker
docker build -t bank-marketing-predict .
docker run -p 8004:8501 bank-marketing-predict:latest
```

## 访问

- 应用: http://localhost:8004
- 健康检查: http://localhost:8004/healthz
