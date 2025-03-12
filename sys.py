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

# ---------------------- 初始化全局状态 ----------------------
if 'sys' not in st.session_state:
    st.session_state.sys = {
        # 数据存储
        'raw_df': None,
        'cleaned_df': None,
        'predicted_df': None,
        'analysis': {},
        
        # 界面状态
        'show_raw': True,
        'show_cleaned': False,
        'show_predicted': False,
        'show_analysis': {},
        
        # 模型相关
        'model': None,
        'tfidf': None,
        'region_map': None,
        'product_map': None
    }

# ---------------------- 核心功能函数 ----------------------
def cleaning(df):
    """数据清洗核心逻辑"""
    progress = st.progress(0)
    status = st.status("🚀 正在处理数据...")
    
    try:
        # 步骤1：基础过滤
        status.write("1. 过滤汉字少于5个的评论...")
        df['汉字数'] = df['评论'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
        df = df[df['汉字数'] > 5].drop(columns=['汉字数'])
        progress.progress(16)

        # 步骤2：删除产品为空数据
        status.write("2. 删除产品信息缺失的评论...")
        original_count = len(df)
        df = df.dropna(subset=['产品'])
        removed_count = original_count - len(df)
        status.write(f"已清除{removed_count}条无产品信息的记录")
        progress.progress(32)

        # 步骤3：重复评论过滤
        status.write("3. 检测重复评论...")
        df = df[~df.duplicated(subset=['评论'], keep='first')]
        progress.progress(48)

        # 步骤4：返现检测
        status.write("4. 检测好评返现...")
        rebate_pattern = build_rebate_pattern()
        df = df[~df['评论'].str.contains(rebate_pattern, na=False)]
        progress.progress(64)

        # 步骤5：水军检测
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
    """构建返现正则表达式"""
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

def predict_recommendation():
    """执行预测流程"""
    with st.status("🧠 正在生成预测...", expanded=True) as status:
        try:
            cleaned_df = st.session_state.sys['cleaned_df'].copy()
            
            # 特征工程
            status.write("1. 提取关键词...")
            cleaned_df['关键词'] = cleaned_df['评论'].apply(lambda x: extract_keywords(x, n=5))
            
            status.write("2. 计算情感特征...")
            scores = cleaned_df.apply(calculate_scores, axis=1)
            cleaned_df[['情感度', '真实性', '参考度']] = scores
            
            # TF-IDF转换
            status.write("3. 文本特征转换...")
            keywords_tfidf = st.session_state.sys['tfidf'].transform(cleaned_df['关键词'])
            
            # 构建特征矩阵
            status.write("4. 合并特征...")
            numeric_features = cleaned_df[['情感度', '真实性', '参考度']].values
            features = hstack([keywords_tfidf, numeric_features])
            
            # 分类编码
            status.write("5. 处理分类特征...")
            cleaned_df['地区_编码'] = pd.Categorical(
                cleaned_df['地区'], 
                categories=st.session_state.sys['region_map']
            ).codes
            cleaned_df['产品_编码'] = pd.Categorical(
                cleaned_df['产品'],
                categories=st.session_state.sys['product_map']
            ).codes
            final_features = hstack([features, cleaned_df[['地区_编码', '产品_编码']].values])
            
            # 预测
            status.write("6. 进行模型预测...")
            predicted_scores = st.session_state.sys['model'].predict(final_features)
            cleaned_df['系统推荐指数'] = np.round(predicted_scores).clip(1, 10).astype(int)
            
            # 保存结果
            st.session_state.sys['predicted_df'] = cleaned_df[['产品', '评论', '系统推荐指数']]
            status.update(label="✅ 预测完成！", state="complete")
            st.session_state.sys['show_predicted'] = True
            
        except Exception as e:
            status.update(label="❌ 预测出错！", state="error")
            st.error(f"错误详情：{str(e)}")
            st.stop()

def analyze_products():
    """执行深度分析"""
    analysis_results = {}
    start_time = time.time()
    
    with st.status("🔍 深度分析进行中...", expanded=True) as status:
        df = st.session_state.sys['predicted_df']
        for product, group in df.groupby('产品'):
            status.write(f"正在分析：{product}...")
            
            comments = group['评论'].tolist()
            scores = group['系统推荐指数'].value_counts().to_dict()
            
            prompt = f"""请根据电商评论数据生成分析报告，要求：
1. 产品名称：{product}
2. 基于{len(comments)}条评论（评分分布：{scores}）
3. 输出结构：
【总结】50字概括
【推荐指数】1-10分
【优点】3-5个带例子
【缺点】3-5个带例子
【建议】是否推荐及原因
用markdown格式，口语化"""
            
            result = call_deepseek_api(prompt)
            analysis_results[product] = result
            time.sleep(0.1)
            
        duration = time.time() - start_time
        status.update(label=f"✅ 分析完成！耗时 {duration:.1f}秒", state="complete")
    
    st.session_state.sys['analysis'] = analysis_results
    for product in analysis_results:
        st.session_state.sys['show_analysis'][product] = True

def call_deepseek_api(prompt):
    """调用DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                               headers=headers, json=data, timeout=30)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"API调用失败：{str(e)}"

# ---------------------- 界面组件 ----------------------
@st.experimental_fragment
def file_uploader():
    """文件上传组件"""
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], 
                                   help="支持UTF-8编码，最大100MB")
    if uploaded_file and not st.session_state.sys['raw_df']:
        st.session_state.sys['raw_df'] = pd.read_csv(uploaded_file)
        st.session_state.sys['show_raw'] = True

@st.experimental_fragment
def raw_data_viewer():
    """原始数据查看器"""
    if st.session_state.sys['show_raw'] and st.session_state.sys['raw_df'] is not None:
        with st.expander("📂 原始数据", expanded=True):
            st.write(f"总记录数：{len(st.session_state.sys['raw_df'])}")
            st.dataframe(
                st.session_state.sys['raw_df'],
                height=500,
                use_container_width=True
            )
            if st.button("❌ 关闭原始数据", key="close_raw"):
                st.session_state.sys['show_raw'] = False

@st.experimental_fragment
def cleaning_controller():
    """数据清洗控制器"""
    if st.session_state.sys['raw_df'] is not None:
        st.divider()
        st.subheader("数据清洗")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🚀 开始清洗", help="启动清洗流程", use_container_width=True):
                with st.spinner('处理中...'):
                    start = time.time()
                    st.session_state.sys['cleaned_df'] = cleaning(
                        st.session_state.sys['raw_df'].copy()
                    )
                    st.session_state.sys['show_cleaned'] = True
                    st.toast(f"清洗完成，耗时{time.time()-start:.1f}秒")

        with col2:
            if st.session_state.sys['show_cleaned']:
                with st.expander("✨ 清洗结果", expanded=True):
                    st.write(f"有效记录：{len(st.session_state.sys['cleaned_df'])}条")
                    st.dataframe(
                        st.session_state.sys['cleaned_df'][
                            ['昵称','日期','地区','产品','评分','评论']
                        ],
                        height=500,
                        use_container_width=True
                    )
                    if st.button("❌ 关闭清洗结果", key="close_clean"):
                        st.session_state.sys['show_cleaned'] = False

@st.experimental_fragment
def prediction_viewer():
    """预测结果查看器"""
    if st.session_state.sys['cleaned_df'] is not None:
        st.divider()
        st.subheader("预测分析")
        
        if st.button("🔮 生成推荐指数", help="启动预测模型", use_container_width=True):
            predict_recommendation()
        
        if st.session_state.sys['show_predicted']:
            st.success("预测结果")
            st.dataframe(
                st.session_state.sys['predicted_df'],
                height=600,
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"总记录数：{len(st.session_state.sys['predicted_df'])}条")
            
            csv_data = st.session_state.sys['predicted_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ 下载预测结果",
                data=csv_data,
                file_name='predictions.csv',
                mime='text/csv',
                key='dl_pred'
            )
            if st.button("❌ 关闭预测结果", key="close_pred"):
                st.session_state.sys['show_predicted'] = False

@st.experimental_fragment
def analysis_reporter():
    """分析报告生成器"""
    if st.session_state.sys['predicted_df'] is not None:
        st.divider()
        st.subheader("深度分析")
        
        if st.button("📊 生成分析报告", type="primary"):
            analyze_products()
        
        for product in st.session_state.sys['analysis']:
            if st.session_state.sys['show_analysis'].get(product, False):
                with st.expander(f"📈 {product} 分析报告", expanded=True):
                    st.markdown(st.session_state.sys['analysis'][product])
                    st.download_button(
                        label=f"⬇️ 下载{product}报告",
                        data=st.session_state.sys['analysis'][product],
                        file_name=f"{product}_analysis.md",
                        mime="text/markdown",
                        key=f"dl_{product}"
                    )
                    if st.button(f"❌ 关闭{product}报告", key=f"close_{product}"):
                        st.session_state.sys['show_analysis'][product] = False

# ---------------------- 主程序入口 ----------------------
def main():
    # 初始化模型
    if not st.session_state.sys['model']:
        try:
            st.session_state.sys.update({
                'model': joblib.load('model.joblib'),
                'tfidf': joblib.load('tfidf_vectorizer.joblib'),
                'region_map': joblib.load('category_mappings.joblib')['region'],
                'product_map': joblib.load('category_mappings.joblib')['product']
            })
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            return

    # 页面配置
    st.set_page_config(page_title="智能数据工厂", layout="wide")
    st.title("📊 电商评论分析系统")
    
    # 功能组件
    file_uploader()
    raw_data_viewer()
    cleaning_controller()
    prediction_viewer()
    analysis_reporter()
    
    # 自动刷新
    if any([st.session_state.sys['show_cleaned'], 
           st.session_state.sys['show_predicted'],
           any(st.session_state.sys['show_analysis'].values())]):
        st.rerun()

if __name__ == "__main__":
    main()