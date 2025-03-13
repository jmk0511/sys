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
import zipfile  # 新增ZIP压缩库

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

st.set_page_config(page_title="CSV数据清洗工具", layout="wide")
st.title("自动化数据清洗与推荐预测系统")

# ---------------------- 数据清洗函数（已集成产品名称标准化）----------------------
def cleaning(df):
    progress = st.progress(0)
    status = st.status("🚀 正在处理数据...")
    
    try:
        # 步骤1：基础过滤
        status.write("1. 过滤汉字少于5个的评论...")
        df['汉字数'] = df['评论'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
        df = df[df['汉字数'] > 5].drop(columns=['汉字数'])
        progress.progress(16)

        # 步骤2：删除产品为空的数据
        status.write("2. 删除产品信息缺失的评论...")
        original_count = len(df)
        df = df.dropna(subset=['产品'])
        removed_count = original_count - len(df)
        status.write(f"已清除{removed_count}条无产品信息的记录")
        progress.progress(32)

        # 步骤2.5：标准化产品名称（新增功能）
        status.write("2.5 标准化产品名称格式...")
        df['产品'] = df['产品'].str.replace(r'[^\w\s\u4e00-\u9fa5]', '', regex=True)
        df['产品'] = df['产品'].str.strip().str.upper()
        progress.progress(40)

        # 步骤3：检测重复评论
        status.write("3. 检测重复评论...")
        df = df[~df.duplicated(subset=['评论'], keep='first')]
        progress.progress(48)

        # 步骤4：检测好评返现
        status.write("4. 检测好评返现...")
        rebate_pattern = build_rebate_pattern()
        df = df[~df['评论'].str.contains(rebate_pattern, na=False)]
        progress.progress(64)

        # 步骤5：检测可疑水军
        status.write("5. 检测可疑水军...")
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

# ---------------------- 预测相关函数 ----------------------
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
    

# ----------------------  DeepSeek 分析模块 ----------------------
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
        "max_tokens": 50000
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

# ---------------------- 界面布局 ----------------------
uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], help="支持UTF-8编码文件，最大100MB")

if uploaded_file and st.session_state.raw_df is None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)

if st.session_state.raw_df is not None:
    with st.expander("📂 永久查看原始数据", expanded=True):
        st.write(f"原始记录数：{len(st.session_state.raw_df)}")
        st.dataframe(st.session_state.raw_df, use_container_width=True, height=500)

# 数据清洗模块
if st.session_state.raw_df is not None:
    st.divider()
    st.subheader("数据清洗模块")
    
    if st.button("🚀 开始清洗", help="点击开始独立清洗流程", use_container_width=True):
        with st.spinner('正在处理数据...'):
            start_time = time.time()
            st.session_state.cleaned_df = cleaning(st.session_state.raw_df.copy())
            st.session_state.processing_time = time.time() - start_time

    # 直接展示清洗结果（移除查看按钮）
    if st.session_state.cleaned_df is not None:
        with st.expander("✨ 清洗后数据详情", expanded=True):
            st.write(f"唯一产品列表：{st.session_state.cleaned_df['产品'].unique().tolist()}")
            st.dataframe(
                st.session_state.cleaned_df[['昵称','日期','地区','产品', '评分','评论']],
                use_container_width=True,
                height=500
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
                
                st.session_state.predicted_df = cleaned_df[['产品', '评论', '系统推荐指数']]
                status.update(label="✅ 预测完成！", state="complete")
                
            except Exception as e:
                status.update(label="❌ 预测出错！", state="error")
                st.error(f"错误详情：{str(e)}")
                st.stop()

        if st.session_state.predicted_df is not None:
            st.success("预测结果：")
            st.dataframe(st.session_state.predicted_df, use_container_width=True, height=600, hide_index=True)
            st.caption(f"总记录数：{len(st.session_state.predicted_df)} 条")
    
            csv = st.session_state.predicted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ 下载预测结果",
                data=csv,
                file_name='predicted_scores.csv',
                mime='text/csv',
                key='prediction_download'
            )
            
# ---------------------- 分析模块 ----------------------
if st.session_state.predicted_df is not None:
    st.divider()
    st.subheader("深度分析模块")
    
    if st.button("📊 生成产品分析报告", type="primary"):
        analysis_results = analyze_products(st.session_state.predicted_df)
        st.session_state.analysis_reports = analysis_results  # 存储报告到session
        
        # 展示所有报告
        for product, report in analysis_results.items():
            with st.expander(f"**{product}** 完整分析报告", expanded=False):
                st.markdown(report)

        # 添加统一下载按钮
        if analysis_results:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for product, report in analysis_results.items():
                    # 处理特殊字符文件名
                    safe_name = re.sub(r'[\\/*?:"<>|]', "_", product)
                    zip_file.writestr(f"{safe_name}_analysis.md", report)
            zip_buffer.seek(0)
            
            st.download_button(
                label="⬇️ 下载全部分析报告",
                data=zip_buffer,
                file_name="产品分析报告.zip",
                mime="application/zip",
                key='full_report_download'
            )