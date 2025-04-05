import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib

# 强制使用 PDF 后端（无字体依赖）
matplotlib.use('PDF')  # 必须在其他 matplotlib 导入前设置

# 配置全局字体参数
plt.rcParams.update({
    'font.sans-serif': ['SimSun', 'STSong'],  # 使用 PDF 内置中文字体
    'axes.unicode_minus': False
})

# 加载模型
model = joblib.load("LGBMRegressor.pkl")

# 页面配置
st.set_page_config(layout="wide", page_title="分娩心理创伤预测系统")
st.title("🏥 分娩心理创伤风险预测与解释")

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

# 侧边栏输入
st.sidebar.header("参数输入")
feature_values = []
for feature, props in feature_ranges.items():
    if props["type"] == "numerical":
        val = st.sidebar.number_input(
            f"{feature} ({props['min']}-{props['max']})",
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"])
        )
    else:
        val = st.sidebar.selectbox(
            f"{feature}", 
            options=props["options"]
        )
    feature_values.append(val)

# 预测与可视化
if st.button("开始分析"):
    # 模型预测
    sample = np.array([feature_values])
    pred = model.predict(sample)[0]
    st.success(f"预测风险值：{pred:.2f}%")
    
    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    # 生成中文标签
    labels = [f"{name}={val}" for name, val in zip(feature_ranges.keys(), feature_values)]
    
    # 创建 PDF 图像
    fig = plt.figure(figsize=(20, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        features=labels,
        matplotlib=True,
        show=False,
        text_rotation=15  # 防止文字重叠
    )
    
    # 保存并显示图像
    fig.savefig("shap_output.pdf", bbox_inches='tight', dpi=300)
    
    # 转换为 PNG 显示
    import pdf2image
    images = pdf2image.convert_from_path("shap_output.pdf")
    images[0].save("shap_output.png", "PNG")
    st.image("shap_output.png")

    # 清理临时文件
    import os
    os.remove("shap_output.pdf")
    os.remove("shap_output.png")
