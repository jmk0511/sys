import time  
import streamlit as st
import pandas as pd
import re
from pypinyin import lazy_pinyin, Style
from datetime import datetime
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# 初始化session状态（关键数据结构隔离）
if 'raw_df' not in st.session_state: # 原始数据
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state: # 清洗后数据
    st.session_state.cleaned_df = None
if 'predicted_df' not in st.session_state: # 预测结果
    st.session_state.predicted_df = None

# 加载模型和预处理对象
if 'model' not in st.session_state:
    try:
        # 加载模型
        st.session_state.model = joblib.load('model.joblib')
        # 加载TF-IDF向量器
        st.session_state.tfidf = joblib.load('tfidf_vectorizer.joblib')
        # 加载分类映射
        category_mappings = joblib.load('category_mappings.joblib')
        st.session_state.region_mapping = category_mappings['region']
        st.session_state.product_mapping = category_mappings['product']
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        
        
# 页面配置
st.set_page_config(page_title="CSV数据清洗工具", layout="wide")
st.title("自动化数据清洗工具")

def cleaning(df):
    """核心清洗逻辑（参考网页5、网页6的清洗流程）"""
    progress = st.progress(0)
    status = st.status("🚀 正在处理数据...")
    
    try:
        # 步骤1：基础过滤
        status.write("1. 过滤汉字少于5个的评论...")
        # 计算每个评论中的汉字数量
        df['汉字数'] = df['评论'].apply(lambda x: len(re.findall(r'[\u4e00-\u9fff]', str(x))))
        # 保留汉字数超过5的行
        df = df[df['汉字数'] > 5].drop(columns=['汉字数'])  # 
        progress.progress(20)

        # 步骤2：重复评论过滤（参考网页6重复值处理）
        status.write("2. 检测重复评论...")
        df = df[~df.duplicated(subset=['评论'], keep='first')]
        progress.progress(40)

        # 步骤3：返现检测（参考网页1文本清洗）
        status.write("3. 检测好评返现...")
        rebate_pattern = build_rebate_pattern()
        df = df[~df['评论'].str.contains(rebate_pattern, na=False)]
        progress.progress(60)

        # 步骤4：水军检测（参考网页8时间序列处理）
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
    """构建返现检测正则（参考网页3文本处理技巧）"""
    base_keywords = ['好评返现', '晒图奖励', '评价有礼']
    patterns = []
    
    # 生成拼音变体（参考网页1文本清洗）
    for kw in base_keywords:
        full_pinyin = ''.join(lazy_pinyin(kw, style=Style.NORMAL))
        patterns.append(re.escape(full_pinyin))
        
        initials = ''.join([p[0] for p in lazy_pinyin(kw, style=Style.INITIALS) if p])
        if initials:
            patterns.append(re.escape(initials))
    
    # 通用模式（参考网页6异常值处理）
    patterns += [
        r'返\s*现', r'评.{0,3}返', 
        r'加\s*微', r'领\s*红\s*包',
        r'\d+\s*元\s*奖'
    ]
    return re.compile('|'.join(patterns), flags=re.IGNORECASE)

def filter_spam_comments(df):
    """水军检测算法（参考网页8时间序列处理）"""
    try:
        # 转换时间格式
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 按昵称和地区分组排序
        df_sorted = df.sort_values(['昵称', '地区', '日期'])
        grouped = df_sorted.groupby(['昵称', '地区'])
        
        # 计算时间差（参考网页5数据范围约束）
        df_sorted['time_diff'] = grouped['日期'].diff().dt.total_seconds().abs()
        
        # 标记可疑记录（5分钟内多次评价）
        df_sorted['is_spam'] = (df_sorted['time_diff'] <= 300) & (df_sorted['time_diff'] > 0)
        
        return df_sorted[~df_sorted['is_spam']].drop(columns=['time_diff', 'is_spam'])
    except KeyError:
        return df

# 新增预测函数（参考model.ipynb的特征工程）
def predict_recommendation(df):
    """执行完整预测流程"""
    progress = st.progress(0)
    status = st.status("🔮 正在生成预测...")
    
    try:
        # 特征工程
        status.write("1. 提取关键词...")
        df['关键词'] = df['评论'].apply(lambda x: ' '.join([word for word in jieba.cut(str(x)) if len(word) > 1][:5]))
        progress.progress(30)

        # 情感计算（参考model.ipynb的评分逻辑）
        status.write("2. 计算情感指标...")
        df[['情感度', '真实性', '参考度']] = df.apply(lambda row: pd.Series([
            SnowNLP(str(row['评论'])).sentiments,
            min(len(str(row['评论']))/100, 1),
            len(row['关键词'].split())/10
        ]), axis=1)
        progress.progress(60)

        # 特征矩阵构建（复用训练时的TF-IDF）
        status.write("3. 构建特征矩阵...")
        keywords_tfidf = st.session_state.tfidf.transform(df['关键词'])
        num_features = df[['情感度', '真实性', '参考度']].values
        categorical_features = df[['地区', '产品']].apply(lambda col: col.map(
            st.session_state.region_mapping if col.name == '地区' 
            else st.session_state.product_mapping
        )).values
        final_features = hstack([keywords_tfidf, num_features, categorical_features])
        progress.progress(80)

        # 执行预测
        status.write("4. 生成推荐指数...")
        df['推荐指数'] = st.session_state.model.predict(final_features).round().clip(1, 10).astype(int)
        progress.progress(100)
        status.update(label="✅ 预测完成！", state="complete")
        return df
    except Exception as e:
        status.update(label="❌ 预测失败", state="error")
        st.error(f"预测错误：{str(e)}")
        return df


# 文件上传模块（始终显示）
uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], 
                               help="支持UTF-8编码文件，最大100MB")

if uploaded_file and st.session_state.raw_df is None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)

# 显示原始数据（上传后永久可查看）
if st.session_state.raw_df is not None:
    with st.expander("📂 永久查看原始数据", expanded=True):
        st.write(f"原始记录数：{len(st.session_state.raw_df)}")
        st.dataframe(st.session_state.raw_df.head(3), use_container_width=True)
        
   
# 清洗模块（独立按钮组）
if st.session_state.raw_df is not None:
    st.divider()
    st.subheader("数据清洗模块")
    
    col1, col2 = st.columns([1,3])
    with col1:
        # 触发清洗
        if st.button("🚀 开始清洗", help="点击开始独立清洗流程", 
                   use_container_width=True):
            with st.spinner('正在处理数据...'):
                start_time = time.time()
                st.session_state.cleaned_df = cleaning(st.session_state.raw_df.copy())
                st.session_state.processing_time = time.time() - start_time

    # 显示清洗结果                
    if st.session_state.cleaned_df is not None:
        with col2:
            if st.button("🔍 查看清洗结果", help="独立查看清洗数据", 
                       use_container_width=True):
                with st.expander("✨ 清洗后数据详情", expanded=True):
                    st.dataframe(
                        st.session_state.cleaned_df[['产品', '评论']],
                        use_container_width=True,
                        height=400
                    )
                    # 下载清洗结果
                    csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ 下载清洗数据",
                        data=csv,
                        file_name='cleaned_data.csv',
                        mime='text/csv'
                    )
                    
# 在现有清洗模块后添加
if st.session_state.cleaned_df is not None:
    st.divider()
    st.subheader("智能预测模块")
    
    # 双栏布局
    col_pred, col_display = st.columns([1,3])
    
    with col_pred:
        if st.button("✨ 执行预测", help="基于清洗后的数据生成推荐指数", 
                    use_container_width=True):
            with st.spinner('预测中...'):
                st.session_state.predicted_df = predict_recommendation(
                    st.session_state.cleaned_df.copy()
                )
                
    if st.session_state.predicted_df is not None:
        with col_display:
            with st.expander("📊 预测结果分析", expanded=True):
                # 结果可视化
                st.write("推荐指数分布:")
                hist_data = pd.cut(st.session_state.predicted_df['推荐指数'], 
                                 bins=[0,5,8,10], 
                                 labels=['差评', '中评', '好评'])
                st.bar_chart(hist_data.value_counts())
                
                # 抽样展示
                st.write("抽样结果（含预测值）:")
                sample_data = st.session_state.predicted_df.sample(3)[[
                    '产品', '评论', '推荐指数'
                ]]
                st.dataframe(sample_data.style.applymap(
                    lambda x: "background-color: #e6ffe6" if x>=8 else 
                    ("#fff3e6" if x>=5 else "#ffe6e6"), 
                    subset=['推荐指数']
                ))
                
                # 下载功能
                csv = st.session_state.predicted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ 下载完整预测数据",
                    data=csv,
                    file_name='predicted_data.csv',
                    mime='text/csv'
                )                    
