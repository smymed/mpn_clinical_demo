import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="MPN风险智能预测系统", page_icon="🩺", layout="centered")
st.title("🩺MPN风险智能预测模型（临床模型demo）-by smy")
st.info("💡 提示：请输入患者的原始临床数据进行预测（本工具仅供科研辅助，不作为单一临床诊断依据。）")

# ==========================================
# 2. 数据字典映射 
# ==========================================

# 2.1 肺段映射 (Lobe_Segment)
segment_mapping = {
    "左肺上叶尖后段": 1,
    "左肺上叶前段": 2,
    "左肺上叶上舌段": 3,
    "左肺上叶下舌段": 4,
    "左肺下叶背段": 5,
    "左肺下叶前内基底段": 6,
    "左肺下叶外基底段": 7,
    "左肺下叶后基底段": 8,
    "右肺上叶尖段": 9,
    "右肺上叶后段": 10,
    "右肺上叶前段": 11,
    "右肺中叶外侧段": 12,
    "右肺中叶内侧段": 13,
    "右肺下叶背段": 14,
    "右肺下叶内基底段": 15,
    "右肺下叶前基底段": 16,
    "右肺下叶外基底段": 17,
    "右肺下叶后基底段": 18
}

# 2.2 影像学特征映射 (Radiology_Feature)
radiology_mapping = {
    "实性结节": 1,
    "混合磨玻璃结节 (mGGN)": 2,
    "纯磨玻璃结节（pGGN）": 3
}

# ==========================================
# 3. 加载模型与工具
# ==========================================
@st.cache_resource
def load_assets():
    # 检查文件是否存在
    if not os.path.exists('best_model.joblib') or not os.path.exists('scaler.joblib'):
        return None, None
    
    # 加载模型
    model = joblib.load('best_model.joblib')
    
    # 加载标准化器
    scaler_obj = joblib.load('scaler.joblib')
    
    
    if isinstance(scaler_obj, list):
        scaler_obj = scaler_obj[0]
        
    return model, scaler_obj

model, scaler = load_assets()

if model is None:
    st.error("❌ 启动失败：未找到模型文件！请确保当前目录下存在 'best_model.joblib' 和 'scaler.joblib'")
    st.stop()

# ==========================================
# 4. 用户输入界面
# ==========================================
st.markdown("### 📋 患者特征录入")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📍 解剖与影像特征")
    
    # 1. 肺叶/肺段 (下拉选择中文)
    selected_segment_name = st.selectbox(
        "肺叶/肺段 (Lobe Segment)",
        options=list(segment_mapping.keys()),
        index=0
    )
    
    # 2. 影像学特征 (下拉选择中文)
    selected_rad_name = st.selectbox(
        "影像学特征 (Radiology Feature)",
        options=list(radiology_mapping.keys()),
        index=0
    )
    
    st.markdown("##### 📏 结节大小")
    diameter = st.number_input("结节直径 (cm)", 0.0, 3.0, 1.00, step=0.01)

with col2:
    st.markdown("##### 🔢 肺功能指标")
    fev1_abs = st.number_input("FEV1 绝对值 (L)", 0.0, 10.0, 2.50, step=0.01)
    fev1_pred = st.number_input("FEV1 占预计值百分比 (%)", 0.0, 200.0, 80.0, 0.01)
    
    st.markdown("##### 🔢 结节位置计数(仅计算>5mm结节)")
    loc_rml = st.number_input("右中叶 (RML) 结节个数", 0, 50, 0)
    loc_lul = st.number_input("左上叶 (LUL) 结节个数", 0, 50, 0)
    loc_lll = st.number_input("左下叶 (LLL) 结节个数", 0, 50, 0)

# ==========================================
# 5. 预测逻辑
# ==========================================
if st.button("🚀 开始风险预测", type="primary", use_container_width=True):
    try:
        # --- 步骤 A: 准备连续变量进行标准化 ---
        # 必须严格遵守训练时 scaler 的列顺序
        # 顺序: [FEV1_Abs, FEV1_Pred_Pct, Loc_RML, Loc_LUL, Loc_LLL, Nodule_Diameter]
        
        continuous_data = pd.DataFrame([[
            fev1_abs, 
            fev1_pred, 
            loc_rml, 
            loc_lul, 
            loc_lll, 
            diameter
        ]], columns=['FEV1_Abs', 'FEV1_Pred_Pct', 'Loc_RML', 'Loc_LUL', 'Loc_LLL', 'Nodule_Diameter'])
        
        # 检查维度匹配
        if scaler.n_features_in_ != 6:
            st.error(f"维度错误")
            st.stop()
            
        # 执行标准化
        scaled_values = scaler.transform(continuous_data)
        df_scaled = pd.DataFrame(scaled_values, columns=continuous_data.columns)
        
        # --- 步骤 B: 拼接最终输入向量 ---
        # 获取中文对应的数值
        segment_value = segment_mapping[selected_segment_name]
        rad_value = radiology_mapping[selected_rad_name]
        
        final_input = pd.DataFrame()
        final_input['FEV1_Abs'] = df_scaled['FEV1_Abs']
        final_input['FEV1_Pred_Pct'] = df_scaled['FEV1_Pred_Pct']
        final_input['Loc_RML'] = df_scaled['Loc_RML']
        final_input['Loc_LUL'] = df_scaled['Loc_LUL']
        final_input['Loc_LLL'] = df_scaled['Loc_LLL']
        
        # 插入分类变量 (直接使用映射后的数值)
        final_input['Lobe_Segment'] = float(segment_value)     
        final_input['Radiology_Feature'] = float(rad_value)  
        
        final_input['Nodule_Diameter'] = df_scaled['Nodule_Diameter']

        # --- 步骤 C: 模型预测 ---
        # 获取正类概率 (假设 1 为恶性/高风险)
        prob = model.predict_proba(final_input)[0][1]
        prediction = model.predict(final_input)[0]
        
        # --- 步骤 D: 结果展示 ---
        st.divider()
        st.subheader("📊 预测报告")
        
        r1, r2, r3 = st.columns([1, 1, 2])
        
        with r1:
            if prediction == 1:
                st.error("**预测类别：高风险**")
            else:
                st.success("**预测类别：低风险**")
        with r2:
            st.metric("风险概率", f"{prob:.2%}")
        with r3:
            if prob > 0.5:
                st.warning("⚠️ 建议：考虑进一步检查或缩短随访周期。")
            else:
                st.info("✅ 建议：风险较低，可按常规流程随访。")
                
        # 概率条
        st.progress(float(prob), text=f"恶性概率值: {prob:.4f}")

    except Exception as e:
        st.error(f"发生错误: {e}")
        st.write("调试建议：请确认 scaler.joblib ")


