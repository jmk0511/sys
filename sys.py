import time  
import streamlit as st
import pandas as pd
import re
import jieba
import numpy as np
from pypinyin import lazy_pinyin, Style
from datetime import datetime
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import requests
import os
import io
import zipfile
import sqlite3
import bcrypt

# ====================== 用户认证模块 ======================
def init_auth_db():
    """初始化数据库连接（新增历史记录字段）"""
    conn = sqlite3.connect('user_auth.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            history_id TEXT NOT NULL,
            upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_data BLOB,
            cleaned_data BLOB,
            predicted_data BLOB,
            analysis_report BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    return conn

@st.cache_resource
def get_auth_db():
    return init_auth_db()

# ====================== 新增历史记录管理模块 ======================
def create_history_entry(user_id):
    """生成唯一历史记录ID（网页1）"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{user_id}_{timestamp}"

def save_full_process_data(user_id, history_id, raw_df, cleaned_df, predicted_df, analysis_reports):
    """保存完整分析流程数据（网页3）"""
    with sqlite3.connect('user_auth.db', check_same_thread=False) as conn:
        try:
            # 序列化数据
            raw_buffer = io.BytesIO()
            raw_df.to_parquet(raw_buffer)
            cleaned_buffer = io.BytesIO()
            cleaned_df.to_parquet(cleaned_buffer)
            predicted_buffer = io.BytesIO()
            predicted_df.to_parquet(predicted_buffer)
            
            # 压缩分析报告（网页5）
            report_buffer = io.BytesIO()
            with zipfile.ZipFile(report_buffer, 'w') as zip_file:
                for product, report in analysis_reports.items():
                    safe_name = re.sub(r'[\\/*?:"<>|]', "_", product)
                    zip_file.writestr(f"{safe_name}_分析.txt", report)
            
            # 插入数据库（网页3）
            conn.execute('''
                INSERT INTO user_data 
                (user_id, history_id, raw_data, cleaned_data, predicted_data, analysis_report)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, history_id,
                  raw_buffer.getvalue(),
                  cleaned_buffer.getvalue(),
                  predicted_buffer.getvalue(),
                  report_buffer.getvalue()))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"历史保存失败: {str(e)}")
            return False

def load_history_data(user_id, history_id):
    """加载历史记录（网页7）"""
    with sqlite3.connect('user_auth.db', check_same_thread=False) as conn:
        result = conn.execute('''
            SELECT raw_data, cleaned_data, predicted_data, analysis_report 
            FROM user_data 
            WHERE user_id = ? AND history_id = ?
        ''', (user_id, history_id)).fetchone()
        
        if result:
            return {
                'raw': pd.read_parquet(io.BytesIO(result[0])),
                'cleaned': pd.read_parquet(io.BytesIO(result[1])),
                'predicted': pd.read_parquet(io.BytesIO(result[2])),
                'report': result[3]
            }
        return None

def get_user_history(user_id):
    """获取用户历史记录列表（网页2）"""
    with sqlite3.connect('user_auth.db', check_same_thread=False) as conn:
        return conn.execute('''
            SELECT DISTINCT history_id, upload_time 
            FROM user_data 
            WHERE user_id = ?
            ORDER BY upload_time DESC
        ''', (user_id,)).fetchall()

def delete_history(user_id, history_id):
    """删除历史记录（网页4）"""
    with sqlite3.connect('user_auth.db', check_same_thread=False) as conn:
        try:
            conn.execute('''
                DELETE FROM user_data 
                WHERE user_id = ? AND history_id = ?
            ''', (user_id, history_id))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"删除失败: {str(e)}")
            return False

# ====================== 修改后的主界面模块 ======================
def main_interface():
    st.title(f"欢迎回来，{st.session_state.username}！")
    
    # 历史记录侧边栏（网页6）
    with st.sidebar:
        st.subheader("📜 分析历史")
        history_list = get_user_history(st.session_state.user_id)
        
        if history_list:
            selected_history = st.selectbox(
                "选择历史记录",
                options=[h[0] for h in history_list],
                format_func=lambda x: datetime.strptime(x.split('_')[1], "%Y%m%d%H%M%S").strftime('%Y-%m-%d %H:%M')
            )
            
            cols = st.columns([3,1])
            with cols[0]:
                if st.button("🔍 加载历史"):
                    history_data = load_history_data(st.session_state.user_id, selected_history)
                    if history_data:
                        st.session_state.update({
                            'raw_df': history_data['raw'],
                            'cleaned_df': history_data['cleaned'],
                            'predicted_df': history_data['predicted'],
                            'analysis_reports': io.BytesIO(history_data['report'])
                        })
                        st.rerun()
            with cols[1]:
                if st.button("🗑 删除", type="secondary"):
                    if delete_history(st.session_state.user_id, selected_history):
                        st.success("删除成功")
                        st.rerun()
        else:
            st.caption("暂无历史记录")
            
    # 文件上传模块
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], 
                                    help="支持UTF-8编码文件，最大100MB")
    
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            history_id = create_history_entry(st.session_state.user_id)
            conn = get_auth_db()
            conn.execute('''
                INSERT INTO user_data 
                (user_id, history_id, raw_data)
                VALUES (?, ?, ?)
            ''', (st.session_state.user_id, history_id, uploaded_file.getvalue()))
            conn.commit()
            st.session_state.raw_df = raw_df
        except Exception as e:
            st.error(f"文件读取失败: {str(e)}")

    # 数据展示模块
    if st.session_state.raw_df is not None:
        with st.expander("📂 原始数据详情", expanded=False):
            st.write(f"记录数：{len(st.session_state.raw_df)}")
            st.dataframe(st.session_state.raw_df, use_container_width=True, height=300)
            if st.button("🗑️ 清除当前数据"):
                st.session_state.raw_df = None
                st.session_state.cleaned_df = None
                st.session_state.predicted_df = None
                st.rerun()

    # 数据清洗模块
    if st.session_state.raw_df is not None:
        st.divider()
        st.subheader("数据清洗模块")
        
        if st.button("🚀 开始清洗", help="点击开始独立清洗流程", use_container_width=True):
            with st.spinner('正在处理数据...'):
                start_time = time.time()
                cleaned_df = cleaning(st.session_state.raw_df.copy())
                if save_user_data(st.session_state.user_id, 'cleaned_data', cleaned_df):
                    st.session_state.cleaned_df = cleaned_df
                st.session_state.processing_time = time.time() - start_time

        if st.session_state.cleaned_df is not None:
            with st.expander("✨ 清洗后数据详情", expanded=False):
                st.write(f"唯一产品列表：{st.session_state.cleaned_df['产品'].unique().tolist()}")
                st.dataframe(
                    st.session_state.cleaned_df[['昵称','日期','地区','产品', '评分','评论']],
                    use_container_width=True,
                    height=400
                )

    # 预测分析模块
    if st.session_state.cleaned_df is not None:
        st.divider()
        st.subheader("预测模块")
    
        if st.button("🔮 预测推荐指数", use_container_width=True):
            if 'model' not in st.session_state:
                st.error("模型未加载，无法进行预测！")
                return
        
            cleaned_df = st.session_state.cleaned_df.copy()
        
            with st.status("🧠 正在生成预测...", expanded=True) as status:
                try:
                    # ...（原有预测处理代码不变）
                    
                    # 预测完成后保存完整记录
                    history_id = create_history_entry(st.session_state.user_id)
                    if save_full_process_data(
                        st.session_state.user_id,
                        history_id,
                        st.session_state.raw_df,
                        st.session_state.cleaned_df,
                        cleaned_df[['产品', '评论', '系统推荐指数']],
                        st.session_state.analysis_reports
                    ):
                        st.session_state.predicted_df = cleaned_df[['产品', '评论', '系统推荐指数']]
                        status.update(label="✅ 预测完成！", state="complete")
                    
                except Exception as e:
                    status.update(label="❌ 预测出错！", state="error")
                    st.error(f"错误详情：{str(e)}")
                    st.stop()

    # 深度分析模块
    if st.session_state.predicted_df is not None:
        st.divider()
        st.subheader("深度分析模块")
    
        if st.button("📊 生成产品分析报告", type="primary"):
            analysis_results = analyze_products(st.session_state.predicted_df)
            st.session_state.analysis_reports = analysis_results
            
            # 自动保存分析报告
            history_id = create_history_entry(st.session_state.user_id)
            save_full_process_data(
                st.session_state.user_id,
                history_id,
                st.session_state.raw_df,
                st.session_state.cleaned_df,
                st.session_state.predicted_df,
                analysis_results
            )




# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 初始化模型和数据库
    if 'model' not in st.session_state:
        try:
            # 加载机器学习模型
            st.session_state.model = joblib.load('model.joblib')
            st.session_state.tfidf = joblib.load('tfidf_vectorizer.joblib')
            category_mappings = joblib.load('category_mappings.joblib')
            st.session_state.region_mapping = category_mappings['region']
            st.session_state.product_mapping = category_mappings['product']
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")

    # 初始化会话状态
    session_keys = ['logged_in', 'username', 'user_id', 'raw_df', 'cleaned_df', 'predicted_df']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # 页面配置
    st.set_page_config(
        page_title="电商用户购买决策AI辅助支持系统",
        layout="wide",
        page_icon="🛒",
        initial_sidebar_state="expanded"
    )
    
    # 流程控制
    if not st.session_state.logged_in:
        auth_gate()
    else:
        main_interface()