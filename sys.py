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
        progress.progress(16)

        # 新增步骤：删除产品为空的数据
        status.write("2. 删除产品信息缺失的评论...")
        original_count = len(df)
        df = df.dropna(subset=['产品'])  # 关键修改点[1,5](@ref)
        removed_count = original_count - len(df)
        status.write(f"已清除{removed_count}条无产品信息的记录")
        progress.progress(32)

        # 原步骤调整为后续步骤
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
    

# ----------------------  DeepSeek 分析模块 ----------------------
def generate_analysis_prompt(product_name, comments, scores):
    """构建分析提示词模板"""
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
    """调用DeepSeek API"""
    api_key = st.secrets["DEEPSEEK_API_KEY"]  # 正式使用请改用 secrets
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2000
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
    """执行产品分析主逻辑"""
    analysis_results = {}
    start_time = time.time()  # 记录开始时间
    
    with st.status("🔍 深度分析进行中...", expanded=True) as status:
        # 按产品分组分析
        for product, group in df.groupby('产品'):
            status.write(f"正在分析：{product}...")
            
            # 准备数据
            comments = group['评论'].tolist()
            scores = group['系统推荐指数'].value_counts().to_dict()
            
            # 生成提示词
            prompt = generate_analysis_prompt(
                product_name=product,
                comments=comments,
                scores=scores
            )
            
            # 调用API
            analysis_result = call_deepseek_api(prompt)
            analysis_results[product] = analysis_result
            
            time.sleep(0.08)
            
            
        duration = time.time() - start_time  # 计算耗时
        status.update(label=f"✅ 分析完成！总耗时 {duration:.2f} 秒", state="complete")  # 显示耗时
    
    return analysis_results



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
        st.dataframe(
            st.session_state.raw_df,
            use_container_width=True,
            height=500  # 设置固定高度启用滚动条
        )
        
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
                    st.write(f"清洗后记录数：{len(st.session_state.cleaned_df)}")  # 新增数量显示
                    st.dataframe(
                        st.session_state.cleaned_df[['昵称','日期','地区','产品', '评分','评论']],
                        use_container_width=True,
                        height=500  # 设置滚动条
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
            st.success("预测结果：")
            st.dataframe(
                st.session_state.predicted_df,
                use_container_width=True,
                height=600,  # 设置固定高度启用滚动条
                hide_index=True  # 可选：隐藏默认索引
            )
    
            # 添加数据统计信息
            st.caption(f"总记录数：{len(st.session_state.predicted_df)} 条")
    
            # 下载预测结果（保持原代码不变）
            csv = st.session_state.predicted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ 下载预测结果",
                data=csv,
                file_name='predicted_scores.csv',
                mime='text/csv',
                key='prediction_download'
            )
            
# ---------------------- 在预测模块后添加分析模块 ----------------------
if st.session_state.predicted_df is not None:
    st.divider()
    st.subheader("深度分析模块")
    
    if st.button("📊 生成产品分析报告", type="primary"):
        # 执行分析
        analysis_results = analyze_products(st.session_state.predicted_df)
        
        # 展示结果
        for product, report in analysis_results.items():
            with st.expander(f"**{product}** 完整分析报告", expanded=False):
                st.markdown(report)
                
            # 添加下载按钮
            st.download_button(
                label=f"⬇️ 下载 {product} 报告",
                data=report,
                file_name=f"{product}_analysis.md",
                mime="text/markdown"
            )
