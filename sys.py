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
from pathlib import Path

# ====================== 用户认证模块 ======================
def init_auth_db():
    """初始化数据库连接（新增预测记录表和评论表）[6,8](@ref)"""
    conn = sqlite3.connect('user_auth.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 主数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_data BLOB,
            cleaned_data BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # 新增预测记录表[6](@ref)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT DEFAULT 'v1.2',
            result_data BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # 新增评论关联表[8](@ref)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            content TEXT,
            score INTEGER,
            FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
        )
    ''')
    
    conn.commit()
    return conn

@st.cache_resource
def get_auth_db():
    """获取持久化数据库连接（启用WAL模式提升性能）[6,8](@ref)"""
    conn = init_auth_db()
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def register_user(username, password):
    """用户注册逻辑"""
    conn = get_auth_db()
    try:
        if conn.execute('SELECT username FROM users WHERE username = ?', (username,)).fetchone():
            return False, "用户名已存在"
        
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                    (username, hashed_pw.decode('utf-8')))
        conn.commit()
        return True, "注册成功"
    except Exception as e:
        return False, str(e)

def verify_login(username, password):
    """用户登录验证"""
    conn = get_auth_db()
    user = conn.execute('''
        SELECT id, password_hash FROM users WHERE username = ?
    ''', (username,)).fetchone()
    
    if not user:
        return False, "用户不存在", None
    
    if bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
        return True, "登录成功", user[0]
    else:
        return False, "密码错误", None

# ====================== 数据管理模块 ======================
def save_user_data(user_id, df):
    conn = get_auth_db()
    try:
        # 插入主记录
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_data 
            (user_id, raw_data, history_id)  -- 显式包含字段
            VALUES (?, ?, ?)  -- 新增占位符
        ''', (user_id, df.to_json(), 0))  # 给history_id赋默认值
        
        # 获取最新data_id
        data_id = cursor.lastrowid
        
        # 批量插入评论数据
        comments = df[['评论']].to_dict('records')
        cursor.executemany('''
            INSERT INTO comments (data_id, content)
            VALUES (?, ?)
        ''', [(data_id, c['评论']) for c in comments])
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"数据保存失败: {str(e)}")
        return False

def load_user_data(user_id, data_type):
    """从数据库加载用户数据"""
    with sqlite3.connect('user_auth.db', check_same_thread=False) as conn:
        try:
            query = f'''
                SELECT {data_type} 
                FROM user_data 
                WHERE user_id = ?
                ORDER BY data_id DESC 
                LIMIT 1
            '''
            result = conn.execute(query, (user_id,)).fetchone()
            if result and result[0]:
                return pd.read_parquet(io.BytesIO(result[0]))
            return None
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            return None

def save_prediction_data(user_id, df):
    """保存完整预测记录（新增关联存储）[6,8](@ref)"""
    conn = get_auth_db()
    try:
        # 序列化预测结果
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # 插入预测记录
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (user_id, result_data)
            VALUES (?, ?)
        ''', (user_id, buffer.getvalue()))
        
        # 获取最新预测ID
        prediction_id = cursor.lastrowid
        
        # 插入评论数据
        comments = df[['评论', '系统推荐指数']].to_dict('records')
        cursor.executemany('''
            INSERT INTO comments (prediction_id, content, score)
            VALUES (?, ?, ?)
        ''', [(prediction_id, c['评论'], c['系统推荐指数']) for c in comments])
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"预测数据保存失败: {str(e)}")
        return False

@st.cache_data(ttl=3600)
def load_history_data(user_id):
    """缓存历史记录查询[7,8](@ref)"""
    return pd.read_sql('''
        SELECT prediction_id, prediction_time, model_version 
        FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_time DESC
    ''', get_auth_db(), params=(user_id,))

#-------------历史记录查看-----------
def show_history(user_id):
    conn = get_auth_db()
    try:
        # 获取用户所有预测记录
        history = pd.read_sql('''
            SELECT prediction_id, prediction_time, model_version 
            FROM predictions
            WHERE user_id = ?
            ORDER BY prediction_time DESC
        ''', conn, params=(user_id,))
        
        # 显示历史记录
        selected = st.selectbox("选择历史记录", history['prediction_id'])
        
        # 加载详细数据
        detail = pd.read_sql('''
            SELECT c.content, p.result_data->>'系统推荐指数' as score
            FROM predictions p
            JOIN comments c ON p.data_id = c.data_id
            WHERE p.prediction_id = ?
        ''', conn, params=(selected,))
        
        st.dataframe(detail)
    except Exception as e:
        st.error(f"历史记录加载失败: {str(e)}")

# ====================== 核心业务模块 ======================
def load_rebate_keywords():
    default_keywords = ['好评返现', '晒图奖励', '评价有礼', '五星好评', '返现红包']
    file_path = 'rebate_keywords.txt'
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                valid_keywords = [kw for kw in lines if re.match(r'^[\u4e00-\u9fa5]+$', kw)]
                return valid_keywords if valid_keywords else default_keywords
        return default_keywords
    except Exception as e:
        st.error(f"⚠️ 关键词文件读取失败，已启用默认规则: {str(e)}")
        return default_keywords

def cleaning(df):
    progress = st.progress(0)
    status = st.status("🚀 正在处理数据...")
    
    try:
        status.write("1. 过滤汉字少于5个的评论...")
        df['汉字数'] = df['评论'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
        df = df[df['汉字数'] > 5].drop(columns=['汉字数'])
        progress.progress(16)

        status.write("2. 删除产品信息缺失的评论...")
        original_count = len(df)
        df = df.dropna(subset=['产品'])
        removed_count = original_count - len(df)
        status.write(f"已清除{removed_count}条无产品信息的记录")
        progress.progress(32)

        status.write("2.5 标准化产品名称格式...")
        df['产品'] = df['产品'].str.replace(r'[^\w\s\u4e00-\u9fa5]', '', regex=True)
        df['产品'] = df['产品'].str.strip().str.upper()
        progress.progress(40)

        status.write("3. 检测重复评论...")
        df = df[~df.duplicated(subset=['评论'], keep='first')]
        progress.progress(48)

        status.write("4. 检测好评返现...")
        rebate_pattern = build_rebate_pattern()
        df = df[~df['评论'].str.contains(rebate_pattern, na=False)]
        progress.progress(64)

        status.write("5. 检测可疑水军...")
        df = filter_spam_comments(df)
        progress.progress(80)

        df = df.reset_index(drop=True)
        progress.progress(100)
        status.update(label="✅ 清洗完成！", state="complete")
        return df
    except Exception as e:
        status.update(label="❌ 处理出错！", state="error")
        st.error(f"错误详情：{str(e)}")
        return df

def build_rebate_pattern():
    patterns = []
    base_keywords = load_rebate_keywords()
    
    for kw in base_keywords:
        patterns.append(re.escape(kw))
        patterns.append(re.sub(r'([\u4e00-\u9fa5])', r'\1\\s*', kw))
        full_pinyin = ''.join(lazy_pinyin(kw, style=Style.NORMAL))
        patterns.append(re.sub(r'([a-z])\d?', r'\1\\d*', full_pinyin))
        initials = ''.join([p[0] for p in lazy_pinyin(kw, style=Style.INITIALS) if p])
        if initials:
            patterns.append(re.sub(r'(.)', r'\1\\W*', initials))

    base_patterns = [
        r'返\s*现', r'评.{0,3}返', 
        r'加\s*[微Vv]', r'领\s*红\s*包',
        r'\d+\s*元\s*奖', r'[Qq扣]\\s*裙'
    ]
    
    final_pattern = '|'.join(patterns + base_patterns)
    return re.compile(final_pattern, flags=re.IGNORECASE)

def filter_spam_comments(df):
    try:
        df['日期'] = pd.to_datetime(df['日期'])
        df_sorted = df.sort_values(['昵称', '地区', '日期'])
        grouped = df_sorted.groupby(['昵称', '地区'])
        df_sorted['time_diff'] = grouped['日期'].diff().dt.total_seconds().abs()
        df_sorted['is_spam'] = (df_sorted['time_diff'] <= 300) & (df_sorted['time_diff'] > 0)
        return df_sorted[~df_sorted['is_spam']].drop(columns=['time_diff', 'is_spam'])
    except KeyError:
        return df

def extract_keywords(text, n=5):
    words = [word for word in jieba.cut(str(text)) if len(word) > 1]
    return ' '.join(words[:n])

def calculate_scores(row):
    try:
        text = str(row['评论'])
        sentiment = SnowNLP(text).sentiments
        authenticity = min(len(text)/100, 1)
        relevance = len(str(row.get('关键词', '')).split())/10
        return pd.Series([sentiment, authenticity, relevance])
    except:
        return pd.Series([0.5, 0.5, 0.5])

def generate_analysis_prompt(product_name, comments, scores):
    return f"""请根据电商评论数据生成产品分析报告，要求：
1. 产品名称：{product_name}
2. 基于以下{len(comments)}条真实评论（评分分布：{scores}）：
{comments[:5]}...（显示前5条示例）
3. 输出结构：
【产品总结】用50字概括整体评价
【推荐指数】根据评分分布给出1-10分
【主要优点】列出3-5个核心优势，带具体例子
【主要缺点】列出3-5个关键不足，带具体例子
【购买建议】给出是否推荐的结论及原因
请用markdown格式输出，避免专业术语，保持口语化"""

def call_deepseek_api(prompt):
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 5000
    }
    
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                                headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"API错误：{response.text}"
    except Exception as e:
        return f"请求失败：{str(e)}"

def analyze_products(df):
    analysis_results = {}
    start_time = time.time()
    
    with st.status("🔍 深度分析进行中...", expanded=True) as status:
        for product, group in df.groupby('产品'):
            status.write(f"正在分析：{product}...")
            
            comments = group['评论'].tolist()
            scores = group['系统推荐指数'].value_counts().to_dict()
            
            prompt = generate_analysis_prompt(
                product_name=product,
                comments=comments,
                scores=scores
            )
            
            analysis_result = call_deepseek_api(prompt)
            analysis_results[product] = analysis_result
            
            time.sleep(0.5)
            
        duration = time.time() - start_time
        status.update(label=f"✅ 分析完成！总耗时 {duration:.2f} 秒", state="complete")
    
    return analysis_results

# ====================== 界面控制模块 ======================
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

def main_interface():
    """主业务界面"""
    st.title(f"欢迎回来，{st.session_state.username}！")
    
    # ========= 新增侧边栏历史记录 =========
    with st.sidebar.expander("📜 历史记录", expanded=True):
        try:
            history = load_history_data(st.session_state.user_id)
            
            if not history.empty:
                selected = st.selectbox(
                    "选择历史记录", 
                    options=history['prediction_id'].astype(str) + " | " + history['prediction_time'].astype(str),
                    format_func=lambda x: f"版本 {x.split('|')[1].strip()}"
                )
                selected_id = int(selected.split('|')[0])
                
                # 加载详细数据
                detail = pd.read_sql('''
                    SELECT c.content, c.score 
                    FROM predictions p
                    JOIN comments c ON p.prediction_id = c.prediction_id
                    WHERE p.prediction_id = ?
                ''', get_auth_db(), params=(selected_id,))
                
                st.dataframe(detail, 
                           column_config={
                               "content": "评论内容",
                               "score": st.column_config.ProgressColumn(
                                   "推荐度",
                                   min_value=1,
                                   max_value=10
                               )
                           },
                           height=200)
            else:
                st.info("暂无历史记录")
        except Exception as e:
            st.error(f"历史记录加载失败: {str(e)}")    
    
    # 文件上传模块
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], 
                                    help="支持UTF-8编码文件，最大100MB")
    
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            conn = get_auth_db()
            conn.execute('''
                INSERT INTO user_data (user_id, raw_data)
                VALUES (?, ?)
            ''', (st.session_state.user_id, uploaded_file.getvalue()))
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

    # ====================== 预测分析模块 ======================
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
                # 特征提取流程
                    status.write("1. 提取关键词...")
                    cleaned_df['关键词'] = cleaned_df['评论'].apply(lambda x: extract_keywords(x, n=5))
                
                    status.write("2. 计算情感特征...")
                    scores = cleaned_df.apply(calculate_scores, axis=1)
                    cleaned_df[['情感度', '真实性', '参考度']] = scores
                
                    status.write("3. 文本特征转换...")
                    keywords_tfidf = st.session_state.tfidf.transform(cleaned_df['关键词'])
                
                    status.write("4. 合并特征...")
                    numeric_features = cleaned_df[['情感度', '真实性', '参考度']].values
                    features = hstack([keywords_tfidf, numeric_features])
                
                    status.write("5. 处理分类特征...")
                    cleaned_df['地区_编码'] = pd.Categorical(
                        cleaned_df['地区'], 
                        categories=st.session_state.region_mapping
                    ).codes
                    cleaned_df['产品_编码'] = pd.Categorical(
                        cleaned_df['产品'],
                        categories=st.session_state.product_mapping
                    ).codes
                    final_features = hstack([features, cleaned_df[['地区_编码', '产品_编码']].values])
                
                    status.write("6. 进行模型预测...")
                    predicted_scores = st.session_state.model.predict(final_features)
                    cleaned_df['系统推荐指数'] = np.round(predicted_scores).clip(1, 10).astype(int)
                
                    if save_prediction_data(st.session_state.user_id, cleaned_df):
                        st.session_state.predicted_df = cleaned_df[['产品', '评论', '系统推荐指数']]
                    
                except Exception as e:
                    status.update(label="❌ 预测出错！", state="error")
                    st.error(f"错误详情：{str(e)}")
                    st.stop()

            # 预测结果展示
            if st.session_state.predicted_df is not None:
                st.success("预测结果：")
                st.dataframe(
                    st.session_state.predicted_df,
                    use_container_width=True,
                    height=600,
                    hide_index=True,
                    column_config={
                        "产品": "商品名称",
                        "评论": st.column_config.TextColumn(width="large"),
                        "系统推荐指数": st.column_config.ProgressColumn(
                            "推荐度",
                            help="AI推荐指数(1-10)",
                            format="%d",
                            min_value=1,
                            max_value=10
                        )
                    }
            )
            st.caption(f"总记录数：{len(st.session_state.predicted_df)} 条")
    
            # 数据导出功能
            csv = st.session_state.predicted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ 下载预测结果",
                data=csv,
                file_name='predicted_scores.csv',
                mime='text/csv',
                key='prediction_download'
            )

    # ====================== 深度分析模块 ======================
    if st.session_state.predicted_df is not None:
        st.divider()
        st.subheader("深度分析模块")
    
        if st.button("📊 生成产品分析报告", type="primary"):
            analysis_results = analyze_products(st.session_state.predicted_df)
            st.session_state.analysis_reports = analysis_results
        
            # 报告展示组件
            for product, report in analysis_results.items():
                with st.expander(f"​**​{product}​**​ 完整分析报告", expanded=False):
                    st.markdown(report)

            # 批量导出功能
            if analysis_results:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for product, report in analysis_results.items():
                        safe_name = re.sub(r'[\\/*?:"<>|]', "_", product)
                        zip_file.writestr(f"{safe_name}_分析.txt", report)
                zip_buffer.seek(0)
            
                st.download_button(
                    label="⬇️ 下载全部分析报告",
                    data=zip_buffer,
                    file_name="产品分析报告.zip",
                    mime="application/zip",
                    key='full_report_download'
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