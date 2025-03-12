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

# 初始化session状态
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None

# 加载模型和预处理对象
if 'model' not in st.session_state:
    try:
        st.session_state.model = joblib.load('model.joblib')
        st.session_state.tfidf = joblib.load('tfidf_vectorizer.joblib')
        category_mappings = joblib.load('category_mappings.joblib')
        st.session_state.region_mapping = category_mappings['region']
        st.session_state.product_mapping = category_mappings['product']
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")

# 页面配置
st.set_page_config(page_title="CSV数据清洗工具", layout="wide")
st.title("自动化数据清洗与推荐预测系统")

# ---------------------- 数据清洗函数 ----------------------
def cleaning(df):
    """核心清洗逻辑"""
    progress = st.progress(0)
    status = st.status("🚀 正在处理数据...")
    
    try:
        # 步骤1：基础过滤
        status.write("1. 过滤汉字少于5个的评论...")
        df['汉字数'] = df['评论'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
        df = df[df['汉字数'] > 5].drop(columns=['汉字数'])
        progress.progress(20)

        # 步骤2：重复评论过滤
        status.write("2. 检测重复评论...")
        df = df[~df.duplicated(subset=['评论'], keep='first')]
        progress.progress(40)

        # 步骤3：返现检测
        status.write("3. 检测好评返现...")
        rebate_pattern = build_rebate_pattern()
        df = df[~df['评论'].str.contains(rebate_pattern, na=False)]
        progress.progress(60)

        # 步骤4：水军检测
        status.write("4. 检测可疑水军...")
        df = filter_spam_comments(df)
        progress.progress(80)

        # 最终处理
        df = df.reset_index(drop=True)
        progress.progress(100)
        status.update(label="✅ 清洗完成！", state="complete")
        return df
    except Exception as e:
        status.update(label="❌ 处理出错！", state="error")
        st.error(f"错误详情：{str(e)}")
        return df

def build_rebate_pattern():
    """构建返现检测正则"""
    base_keywords = ['好评返现', '晒图奖励', '评价有礼']
    patterns = []
    for kw in base_keywords:
        full_pinyin = ''.join(lazy_pinyin(kw, style=Style.NORMAL))
        patterns.append(re.escape(full_pinyin))
        initials = ''.join([p[0] for p in lazy_pinyin(kw, style=Style.INITIALS) if p])
        if initials:
            patterns.append(re.escape(initials))
    patterns += [
        r'返\s*现', r'评.{0,3}返', 
        r'加\s*微', r'领\s*红\s*包',
        r'\d+\s*元\s*奖'
    ]
    return re.compile('|'.join(patterns), flags=re.IGNORECASE)

def filter_spam_comments(df):
    """水军检测算法"""
    try:
        df['日期'] = pd.to_datetime(df['日期'])
        df_sorted = df.sort_values(['昵称', '地区', '日期'])
        grouped = df_sorted.groupby(['昵称', '地区'])
        df_sorted['time_diff'] = grouped['日期'].diff().dt.total_seconds().abs()
        df_sorted['is_spam'] = (df_sorted['time_diff'] <= 300) & (df_sorted['time_diff'] > 0)
        return df_sorted[~df_sorted['is_spam']].drop(columns=['time_diff', 'is_spam'])
    except KeyError:
        return df

# ---------------------- 预测相关函数 ----------------------
def extract_keywords(text, n=5):
    """提取关键词"""
    words = [word for word in jieba.cut(str(text)) if len(word) > 1]
    return ' '.join(words[:n])

def calculate_scores(row):
    """计算特征分数"""
    try:
        text = str(row['评论'])
        sentiment = SnowNLP(text).sentiments
        authenticity = min(len(text)/100, 1)
        relevance = len(str(row.get('关键词', '')).split())/10
        return pd.Series([sentiment, authenticity, relevance])
    except:
        return pd.Series([0.5, 0.5, 0.5])

# ---------------------- 界面布局 ----------------------
# 文件上传模块
uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], 
                               help="支持UTF-8编码文件，最大100MB")

if uploaded_file and st.session_state.raw_df is None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)

# 显示原始数据
if st.session_state.raw_df is not None:
    with st.expander("📂 永久查看原始数据", expanded=True):
        st.write(f"原始记录数：{len(st.session_state.raw_df)}")
        st.dataframe(st.session_state.raw_df.head(3), use_container_width=True)
        
# 数据清洗模块
if st.session_state.raw_df is not None:
    st.divider()
    st.subheader("数据清洗模块")
    
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🚀 开始清洗", help="点击开始独立清洗流程", use_container_width=True):
            with st.spinner('正在处理数据...'):
                start_time = time.time()
                st.session_state.cleaned_df = cleaning(st.session_state.raw_df.copy())
                st.session_state.processing_time = time.time() - start_time

    if st.session_state.cleaned_df is not None:
        with col2:
            if st.button("🔍 查看清洗结果", help="独立查看清洗数据", use_container_width=True):
                with st.expander("✨ 清洗后数据详情", expanded=True):
                    st.dataframe(
                        st.session_state.cleaned_df[['昵称','日期','地区','产品', '评分','评论']],
                        use_container_width=True,
                        height=400
                    )
                    csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ 下载清洗数据",
                        data=csv,
                        file_name='cleaned_data.csv',
                        mime='text/csv'
                    )

# 预测模块
if st.session_state.cleaned_df is not None:
    st.divider()
    st.subheader("预测模块")
    
    if st.button("🔮 预测推荐指数", help="点击进行推荐指数预测", use_container_width=True):
        if 'model' not in st.session_state:
            st.error("模型未加载，无法进行预测！")
            st.stop()
        
        cleaned_df = st.session_state.cleaned_df.copy()
        
        with st.status("🧠 正在生成预测...", expanded=True) as status:
            try:
                # 特征工程
                status.write("1. 提取关键词...")
                cleaned_df['关键词'] = cleaned_df['评论'].apply(lambda x: extract_keywords(x, n=5))
                
                status.write("2. 计算情感特征...")
                scores = cleaned_df.apply(calculate_scores, axis=1)
                cleaned_df[['情感度', '真实性', '参考度']] = scores
                
                # TF-IDF转换
                status.write("3. 文本特征转换...")
                keywords_tfidf = st.session_state.tfidf.transform(cleaned_df['关键词'])
                
                # 构建特征矩阵
                status.write("4. 合并特征...")
                numeric_features = cleaned_df[['情感度', '真实性', '参考度']].values
                features = hstack([keywords_tfidf, numeric_features])
                
                # 分类编码
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
                
                # 预测
                status.write("6. 进行模型预测...")
                predicted_scores = st.session_state.model.predict(final_features)
                cleaned_df['系统推荐指数'] = np.round(predicted_scores).clip(1, 10).astype(int)
                
                # 保存结果
                st.session_state.predicted_df = cleaned_df[['产品', '评论', '系统推荐指数']]
                status.update(label="✅ 预测完成！", state="complete")
                
            except Exception as e:
                status.update(label="❌ 预测出错！", state="error")
                st.error(f"错误详情：{str(e)}")
                st.stop()

        # 显示预测结果
        if st.session_state.predicted_df is not None:
            st.success("预测结果预览：")
            st.dataframe(st.session_state.predicted_df.head(10), use_container_width=True)
            
            # 下载预测结果
            csv = st.session_state.predicted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ 下载预测结果",
                data=csv,
                file_name='predicted_scores.csv',
                mime='text/csv',
                key='prediction_download'
            )