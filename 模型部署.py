import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体支持中文
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model_path = "LGBMRegressor.pkl"
model = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="轻量级梯度提升回归模型预测与 SHAP 可视化", page_icon="💕👩‍⚕️🏥")
st.title("💕👩‍⚕️🏥 轻量级梯度提升回归模型预测与 SHAP 可视化")
st.write("通过输入所有变量的值进行单个样本分娩心理创伤的风险预测，可以得到该样本罹患分娩心理创伤的概率，并结合 SHAP 力图分析结果，有助于临床医护人员了解具体的风险因素和保护因素。")

# 特征范围定义
feature_ranges = {
    "年龄": {"type": "numerical", "min": 18, "max": 42, "default": 18},
    "体重": {"type": "numerical", "min": 52, "max": 91, "default": 52},
    "居住地": {"type": "categorical", "options": [1, 2]},
    "婚姻状况": {"type": "categorical", "options": [1, 2]},
    "就业情况": {"type": "categorical", "options": [1, 2]},
    "学历": {"type": "categorical", "options": [1, 2, 3, 4]},
    "医疗费用支付方式": {"type": "categorical", "options": [1, 2, 3]},
    "怀孕次数": {"type": "numerical", "min": 1, "max": 8, "default": 1},
    "分娩次数": {"type": "numerical", "min": 1, "max": 4, "default": 1},
    "分娩方式": {"type": "categorical", "options": [1, 2, 3]},
    "不良孕产史": {"type": "categorical", "options": [1, 2]},
    "终止妊娠经历": {"type": "categorical", "options": [1, 2]},
    "妊娠周数": {"type": "numerical", "min": 29, "max": 44, "default": 29},
    "妊娠合并症": {"type": "categorical", "options": [1, 2]},
    "妊娠并发症": {"type": "categorical", "options": [1, 2]},
    "喂养方式": {"type": "categorical", "options": [1, 2, 3]},
    "新生儿是否有出生缺陷或疾病": {"type": "categorical", "options": [1, 2]},
    "家庭人均月收入": {"type": "categorical", "options": [1, 2]},
    "使用无痛分娩技术": {"type": "categorical", "options": [1, 2]},
    "产时疼痛": {"type": "numerical", "min": 0, "max": 10, "default": 0},
    "产后疼痛": {"type": "numerical", "min": 1, "max": 9, "default": 1},
    "产后照顾婴儿方式": {"type": "categorical", "options": [1, 2, 3, 4, 5]},
    "近1月睡眠质量": {"type": "categorical", "options": [1, 2, 3, 4]},
    "近1月夜间睡眠时长": {"type": "numerical", "min": 3, "max": 11, "default": 3},
    "近1月困倦程度": {"type": "categorical", "options": [1, 2, 3, 4]},
    "孕期体育活动等级": {"type": "categorical", "options": [1, 2, 3, 4]},
    "抑郁": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "焦虑": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "侵入性反刍性沉思": {"type": "numerical", "min": 0, "max": 30, "default": 0},
    "目的性反刍性沉思": {"type": "numerical", "min": 0, "max": 28, "default": 0},
    "心理弹性": {"type": "numerical", "min": 6, "max": 30, "default": 6},
    "家庭支持": {"type": "numerical", "min": 0, "max": 10, "default": 0},
}

# 动态生成输入项
st.sidebar.header("变量输入区域")
st.sidebar.write("请输入变量值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_value = model.predict(features)[0]
    st.write(f"Predicted 分娩心理创伤 score: {predicted_value:.2f}")

    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 获取基础值和第一个样本的 SHAP 值
    base_value = explainer.expected_value
    shap_values_sample = shap_values[0]

    # 定义特征名称和其对应的值
    features_with_values = np.array([
        f"X1={feature_values[0]}",
        f"X2={feature_values[1]}",
        f"X3={feature_values[2]}",
        f"X4={feature_values[3]}",
        f"X5={feature_values[4]}",
        f"X6={feature_values[5]}",
        f"X7={feature_values[6]}",
        f"X8={feature_values[7]}",
        f"X9={feature_values[8]}",
        f"X10={feature_values[9]}",
        f"X11={feature_values[10]}",
        f"X12={feature_values[11]}",
        f"X13={feature_values[12]}",
        f"X14={feature_values[13]}",
        f"X15={feature_values[14]}",
        f"X16={feature_values[15]}",
        f"X17={feature_values[16]}",
        f"X18={feature_values[17]}",
        f"X19={feature_values[18]}",
        f"X20={feature_values[19]}",
        f"X21={feature_values[20]}",
        f"X22={feature_values[21]}",
        f"X23={feature_values[22]}",
        f"X24={feature_values[23]}",
        f"X25={feature_values[24]}",
        f"X26={feature_values[25]}",
        f"X27={feature_values[26]}",
        f"X28={feature_values[27]}",
        f"X29={feature_values[28]}",
        f"X30={feature_values[29]}",
        f"X31={feature_values[30]}",
        f"X32={feature_values[31]}"
    ])

    # SHAP 力图
    st.write("### SHAP 力图")
    force_plot = shap.force_plot(
        base_value,
        shap_values_sample,
        features_with_values,
        feature_names=list(feature_ranges.keys()),
        matplotlib=True,
        show=False
    )
    st.pyplot(force_plot)

# 添加变量含义表格
st.write("### 变量含义")

feature_descriptions = {
    "X1": "年龄",
    "X2": "体重",
    "X3": "居住地",
    "X4": "婚姻状况",
    "X5": "就业情况",
    "X6": "学历",
    "X7": "医疗费用支付方式",
    "X8": "怀孕次数",
    "X9": "分娩次数",
    "X10": "分娩方式",
    "X11": "不良孕产史",
    "X12": "终止妊娠经历",
    "X13": "妊娠周数",
    "X14": "妊娠合并症",
    "X15": "妊娠并发症",
    "X16": "喂养方式",
    "X17": "新生儿是否有出生缺陷或疾病",
    "X18": "家庭人均月收入",
    "X19": "使用无痛分娩技术",
    "X20": "产时疼痛",
    "X21": "产后疼痛",
    "X22": "产后照顾婴儿方式",
    "X23": "近1月睡眠质量",
    "X24": "近1月夜间睡眠时长",
    "X25": "近1月困倦程度",
    "X26": "孕期体育活动等级",
    "X27": "抑郁",
    "X28": "焦虑",
    "X29": "侵入性反刍性沉思",
    "X30": "目的性反刍性沉思",
    "X31": "心理弹性",
    "X32": "家庭支持"
}

# 构建HTML表格
table_html = "<table><tr><th>变量</th><th>含义</th></tr>"
for key, value in feature_descriptions.items():
    table_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
table_html += "</table>"

# 设置表格样式为三线表
table_html = f"""
<style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        padding: 8px;
    }}
    th {{
        border-top: 2px solid #000000;
        border-bottom: 1px solid #000000;
    }}
    td {{
        border-bottom: 1px solid #000000;
    }}
    tr:last-child td {{
        border-bottom: 2px solid #000000;
    }}
</style>
{table_html}
"""

# 显示表格
st.write(table_html, unsafe_allow_html=True)
