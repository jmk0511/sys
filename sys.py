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
    """初始化数据库连接"""
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
    """获取数据库连接"""
    return init_auth_db()

# 用户认证功能实现（必须先于auth_gate定义）
def register_user(username, password):
    """注册新用户"""
    conn = get_auth_db()
    try:
        cursor = conn.cursor()
        # 检查用户名是否存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "用户名已存在"
        
        # 生成密码哈希
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # 插入新用户
        cursor.execute('''
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
        ''', (username, password_hash))
        conn.commit()
        return True, "注册成功"
    except Exception as e:
        conn.rollback()
        return False, f"注册失败: {str(e)}"
    finally:
        conn.close()

def verify_login(username, password):
    """验证用户登录"""
    conn = get_auth_db()
    try:
        cursor = conn.cursor()
        # 获取用户信息
        cursor.execute('''
            SELECT id, password_hash FROM users 
            WHERE username = ?
        ''', (username,))
        user = cursor.fetchone()
        
        if not user:
            return False, "用户不存在", None
        
        user_id, stored_hash = user
        # 转换字节类型（SQLite存储时可能转为字符串）
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode('utf-8')
        
        # 验证密码
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True, "登录成功", user_id
        return False, "密码错误", None
    except Exception as e:
        return False, f"登录异常: {str(e)}", None
    finally:
        conn.close()

# 数据持久化相关函数
def save_user_data(user_id, data_type, df):
    """保存用户数据到数据库"""
    try:
        conn = get_auth_db()
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        conn.execute(f'''
            UPDATE user_data 
            SET {data_type} = ?
            WHERE user_id = ?
            ORDER BY upload_time DESC
            LIMIT 1
        ''', (buffer.getvalue(), user_id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"数据保存失败: {str(e)}")
        return False

def load_user_data(user_id, data_type):
    """从数据库加载用户数据"""
    try:
        conn = get_auth_db()
        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT {data_type} FROM user_data
            WHERE user_id = ?
            ORDER BY upload_time DESC
            LIMIT 1
        ''', (user_id,))
        data = cursor.fetchone()
        if data and data[0]:
            return pd.read_parquet(io.BytesIO(data[0]))
        return None
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def create_history_entry(user_id):
    """创建新的历史记录条目"""
    history_id = f"history_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn = get_auth_db()
    conn.execute('''
        INSERT INTO user_data 
        (user_id, history_id)
        VALUES (?, ?)
    ''', (user_id, history_id))
    conn.commit()
    return history_id

def get_user_history(user_id):
    """获取用户历史记录列表"""
    conn = get_auth_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT history_id, upload_time FROM user_data
        WHERE user_id = ?
        ORDER BY upload_time DESC
    ''', (user_id,))
    return cursor.fetchall()

def load_history_data(user_id, history_id):
    """加载指定历史记录数据"""
    try:
        conn = get_auth_db()
        cursor = conn.cursor()
        
        # 获取原始数据
        cursor.execute('''
            SELECT raw_data, cleaned_data, predicted_data, analysis_report
            FROM user_data
            WHERE user_id = ? AND history_id = ?
        ''', (user_id, history_id))
        data = cursor.fetchone()
        
        return {
            'raw': pd.read_parquet(io.BytesIO(data[0])) if data[0] else None,
            'cleaned': pd.read_parquet(io.BytesIO(data[1])) if data[1] else None,
            'predicted': pd.read_parquet(io.BytesIO(data[2])) if data[2] else None,
            'report': data[3] if data[3] else None
        }
    except Exception as e:
        st.error(f"历史记录加载失败: {str(e)}")
        return None

def delete_history(user_id, history_id):
    """删除指定历史记录"""
    try:
        conn = get_auth_db()
        conn.execute('''
            DELETE FROM user_data
            WHERE user_id = ? AND history_id = ?
        ''', (user_id, history_id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"删除失败: {str(e)}")
        return False

# ====================== 认证入口页面 ======================
def auth_gate():
    """认证入口页面"""
    st.title("电商决策支持系统")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔑 用户登录", expanded=True):
            login_username = st.text_input("用户名", key="login_user")
            login_password = st.text_input("密码", type="password", key="login_pw")
            if st.button("登录"):
                if not login_username or not login_password:
                    st.error("请输入用户名和密码")
                else:
                    success, msg, user_id = verify_login(login_username, login_password)
                    if success:
                        st.session_state.update({
                            'logged_in': True,
                            'username': login_username,
                            'user_id': user_id,
                            'raw_df': load_user_data(user_id, 'raw_data'),
                            'cleaned_df': load_user_data(user_id, 'cleaned_data'),
                            'predicted_df': load_user_data(user_id, 'predicted_data')
                        })
                        st.rerun()
                    else:
                        st.error(msg)

    with col2:
        with st.expander("📝 新用户注册", expanded=True):
            reg_username = st.text_input("注册用户名", key="reg_user")
            reg_password = st.text_input("注册密码", type="password", key="reg_pw")
            if st.button("立即注册"):
                if len(reg_password) < 6:
                    st.error("密码至少需要6位")
                elif not reg_username:
                    st.error("请输入用户名")
                else:
                    success, msg = register_user(reg_username, reg_password)
                    if success:
                        st.success(msg + "，请返回登录")
                    else:
                        st.error(msg)

# ====================== 主界面模块 ====================== 
def main_interface():
    st.title(f"欢迎回来，{st.session_state.username}！")
    
    # 历史记录侧边栏
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
                    # 特征工程
                    tfidf_features = st.session_state.tfidf.transform(cleaned_df['评论'])
                    
                    # 类别特征转换
                    region_encoded = cleaned_df['地区'].map(st.session_state.region_mapping).fillna(-1)
                    product_encoded = cleaned_df['产品'].map(st.session_state.product_mapping).fillna(-1)
                    
                    # 组合特征
                    combined_features = hstack([
                        tfidf_features,
                        np.array(region_encoded)[:, None],
                        np.array(product_encoded)[:, None]
                    ])
                    
                    # 进行预测
                    predictions = st.session_state.model.predict(combined_features)
                    cleaned_df['系统推荐指数'] = predictions
                    
                    # 保存预测结果
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

# 辅助函数（需要根据实际业务实现）
def cleaning(raw_df):
    """数据清洗函数示例"""
    # 去除HTML标签
    raw_df['评论'] = raw_df['评论'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))
    
    # 中文分词
    raw_df['分词结果'] = raw_df['评论'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    
    # 拼音转换
    raw_df['拼音'] = raw_df['产品'].apply(lambda x: ' '.join(lazy_pinyin(x, style=Style.TONE3)))
    
    # 情感分析
    raw_df['情感得分'] = raw_df['评论'].apply(lambda x: SnowNLP(str(x)).sentiments)
    
    return raw_df

def analyze_products(predicted_df):
    """生成分析报告示例"""
    report = io.BytesIO()
    
    # 生成各产品分析
    product_analysis = predicted_df.groupby('产品').agg({
        '系统推荐指数': ['mean', 'count']
    }).reset_index()
    
    # 生成报告图表
    with pd.ExcelWriter(report, engine='xlsxwriter') as writer:
        product_analysis.to_excel(writer, sheet_name='产品分析', index=False)
        
    report.seek(0)
    return report.getvalue()

def save_full_process_data(user_id, history_id, raw_df, cleaned_df, predicted_df, report):
    """完整流程数据保存"""
    try:
        conn = get_auth_db()
        
        # 转换数据为字节流
        raw_bytes = raw_df.to_parquet(index=False)
        cleaned_bytes = cleaned_df.to_parquet(index=False) 
        predicted_bytes = predicted_df.to_parquet(index=False)
        
        conn.execute('''
            UPDATE user_data SET
                raw_data = ?,
                cleaned_data = ?,
                predicted_data = ?,
                analysis_report = ?
            WHERE history_id = ?
        ''', (raw_bytes, cleaned_bytes, predicted_bytes, report, history_id))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"完整流程保存失败: {str(e)}")
        return False