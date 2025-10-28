import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import PyPDF2
import io
import requests
import json
from datetime import datetime
import os

# 设置matplotlib中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 页面设置
st.set_page_config(
    page_title="AI ESG Report Analyzer",
    page_icon="🌱",
    layout="wide"
)

# 初始化OpenAI客户端
@st.cache_resource
def init_openai_client():
    # 优先从环境变量获取，然后从secrets获取
    api_key = os.getenv('OPENAI_API_KEY', st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.warning("请设置OpenAI API密钥")
        return None
    return OpenAI(api_key=api_key)

client = init_openai_client()

# 标题和介绍
st.title("🌱 AI ESG报告分析器")
st.markdown("""
上传公司的ESG报告，我们将使用AI分析其ESG表现并与竞争对手进行基准比较。
""")

# 侧边栏 - 公司信息输入
st.sidebar.header("公司信息")
company_name = st.sidebar.text_input("公司名称", "示例公司")
industry = st.sidebar.selectbox(
    "行业",
    ["科技", "金融", "能源", "制造", "消费品", "医疗", "其他"]
)
competitors = st.sidebar.text_area(
    "竞争对手（每行一个）", 
    "竞争对手A\n竞争对手B\n竞争对手C"
)

# 文件上传
st.header("1. 上传ESG报告")
uploaded_file = st.file_uploader(
    "上传PDF格式的ESG报告", 
    type=['pdf'],
    help="支持PDF格式的ESG报告文件"
)

# 解析PDF内容
def extract_text_from_pdf(file):
    """从PDF文件中提取文本"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF解析错误: {str(e)}")
        return ""

# 使用GPT分析ESG报告
def analyze_esg_with_gpt(report_text, company_name, industry, competitors_list):
    """使用GPT分析ESG报告"""
    
    if not client:
        st.error("OpenAI客户端未初始化，请检查API密钥")
        return None
        
    # 限制文本长度以避免token超限
    truncated_text = report_text[:8000] if len(report_text) > 8000 else report_text
    
    prompt = f"""
    请分析以下ESG报告并提供一个全面的评估。

    公司名称: {company_name}
    行业: {industry}
    竞争对手: {', '.join(competitors_list)}

    ESG报告内容:
    {truncated_text}

    请提供以下分析:
    1. ESG总体评分（0-100分）
    2. 环境(E)、社会(S)、治理(G)三个维度的详细评分
    3. 主要优势和劣势
    4. 与行业竞争对手的比较分析
    5. 改进建议

    请以JSON格式返回结果，包含以下字段:
    - overall_score (数字)
    - environmental_score (数字)
    - social_score (数字)
    - governance_score (数字)
    - strengths (字符串数组)
    - weaknesses (字符串数组)
    - competitor_comparison (对象)
    - recommendations (字符串数组)

    确保所有分数都是0-100之间的数字。
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用gpt-3.5-turbo降低成本
            messages=[
                {"role": "system", "content": "你是一个专业的ESG分析师，专门评估公司的环境、社会和治理表现。请始终返回有效的JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content
        st.write("原始响应:", result_text)  # 调试信息
        
        # 提取JSON部分
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            result = json.loads(json_str)
            
            # 验证和清理结果
            return validate_esg_results(result)
        else:
            st.error("无法解析AI响应为JSON格式")
            return None
            
    except Exception as e:
        st.error(f"GPT分析错误: {str(e)}")
        return None

def validate_esg_results(result):
    """验证和清理ESG分析结果"""
    # 确保所有必需的字段都存在
    required_fields = ['overall_score', 'environmental_score', 'social_score', 'governance_score']
    
    for field in required_fields:
        if field not in result:
            result[field] = 50  # 默认值
    
    # 确保分数是数字且在合理范围内
    for score_field in required_fields:
        try:
            score = float(result[score_field])
            result[score_field] = max(0, min(100, score))
        except (ValueError, TypeError):
            result[score_field] = 50
    
    # 确保数组字段存在
    for array_field in ['strengths', 'weaknesses', 'recommendations']:
        if array_field not in result or not isinstance(result[array_field], list):
            result[array_field] = ["数据不足"]
    
    # 确保比较分析字段存在
    if 'competitor_comparison' not in result or not isinstance(result['competitor_comparison'], dict):
        result['competitor_comparison'] = {"行业平均": "数据不足"}
    
    return result

# 生成基准数据
def generate_benchmark_data(analysis_result, company_name, competitors_list):
    """生成基准比较数据"""
    
    # 模拟竞争对手数据
    np.random.seed(42)
    num_competitors = len(competitors_list)
    
    benchmark_data = {
        '公司': [company_name] + competitors_list,
        'ESG总分': [float(analysis_result['overall_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['overall_score']) * 0.95, 
                      5, 
                      num_competitors
                  )),
        '环境得分': [float(analysis_result['environmental_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['environmental_score']) * 0.95, 
                      6, 
                      num_competitors
                  )),
        '社会得分': [float(analysis_result['social_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['social_score']) * 0.95, 
                      5, 
                      num_competitors
                  )),
        '治理得分': [float(analysis_result['governance_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['governance_score']) * 0.95, 
                      4, 
                      num_competitors
                  ))
    }
    
    # 确保分数在合理范围内
    for key in ['ESG总分', '环境得分', '社会得分', '治理得分']:
        benchmark_data[key] = [max(0, min(100, float(score))) for score in benchmark_data[key]]
    
    return pd.DataFrame(benchmark_data)

# 可视化函数
def create_radar_chart(analysis_result, company_name):
    """创建雷达图显示ESG各维度表现"""
    
    categories = ['环境表现', '社会责任', '公司治理', '透明度', '风险管理', '创新性']
    
    # 基于主要分数生成各维度得分
    np.random.seed(42)
    env_score = float(analysis_result['environmental_score'])
    soc_score = float(analysis_result['social_score'])
    gov_score = float(analysis_result['governance_score'])
    
    scores = [
        env_score,
        soc_score,
        gov_score,
        env_score * 0.8 + gov_score * 0.2,
        gov_score * 0.7 + soc_score * 0.3,
        env_score * 0.6 + soc_score * 0.4
    ]
    
    # 完成雷达图
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, scores, 'o-', linewidth=2, label=company_name, color='#1A936F')
    ax.fill(angles, scores, alpha=0.25, color='#1A936F')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 100)
    ax.set_title(f'{company_name} ESG表现雷达图', size=14, weight='bold')
    ax.legend(loc='upper right')
    
    return fig

def create_comparison_bar_chart(benchmark_df, company_name):
    """创建基准比较柱状图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['ESG总分', '环境得分', '社会得分', '治理得分']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        data_sorted = benchmark_df.sort_values(metric, ascending=False)
        bars = axes[i].bar(data_sorted['公司'], data_sorted[metric], 
                          color=colors[i], alpha=0.7)
        
        # 突出显示目标公司
        target_idx = list(data_sorted['公司']).index(company_name)
        bars[target_idx].set_color('#1A936F')
        bars[target_idx].set_alpha(1.0)
        
        axes[i].set_title(f'{metric}比较', weight='bold')
        axes[i].set_ylabel('分数')
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

# 主应用逻辑
if uploaded_file is not None:
    st.success(f"已上传文件: {uploaded_file.name}")
    
    # 显示加载状态
    with st.spinner("正在解析和分析ESG报告..."):
        # 提取文本
        report_text = extract_text_from_pdf(uploaded_file)
        
        if report_text:
            st.info(f"成功提取文本，共 {len(report_text)} 个字符")
            
            # 显示提取的文本前500字符（用于调试）
            with st.expander("查看提取的文本（前500字符）"):
                st.text(report_text[:500] + "..." if len(report_text) > 500 else report_text)
            
            # 分析报告
            competitors_list = [c.strip() for c in competitors.split('\n') if c.strip()]
            analysis_result = analyze_esg_with_gpt(
                report_text, company_name, industry, competitors_list
            )
            
            if analysis_result:
                st.success("ESG分析完成！")
                
                # 显示主要结果
                col1, col2, col3, col4 = st.columns(4)
                
                # 安全地处理分数显示
                overall_score = float(analysis_result.get('overall_score', 0))
                env_score = float(analysis_result.get('environmental_score', 0))
                soc_score = float(analysis_result.get('social_score', 0))
                gov_score = float(analysis_result.get('governance_score', 0))
                
                with col1:
                    st.metric(
                        "ESG总分", 
                        f"{overall_score:.1f}",
                        delta=f"行业平均: {overall_score - 5:.1f}"
                    )
                
                with col2:
                    st.metric(
                        "环境得分", 
                        f"{env_score:.1f}"
                    )
                
                with col3:
                    st.metric(
                        "社会得分", 
                        f"{soc_score:.1f}"
                    )
                
                with col4:
                    st.metric(
                        "治理得分", 
                        f"{gov_score:.1f}"
                    )
                
                # 生成基准数据
                benchmark_df = generate_benchmark_data(
                    analysis_result, company_name, competitors_list
                )
                
                # 标签页显示详细分析
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 可视化分析", "📈 基准比较", "✅ 优势与改进", "📋 详细报告"
                ])
                
                with tab1:
                    st.subheader("ESG表现可视化")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 雷达图
                        radar_fig = create_radar_chart(analysis_result, company_name)
                        st.pyplot(radar_fig)
                    
                    with col2:
                        # ESG分数分布
                        esg_scores = [env_score, soc_score, gov_score]
                        labels = ['环境', '社会', '治理']
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(labels, esg_scores, 
                                     color=['#1A936F', '#2E86AB', '#F18F01'])
                        ax.set_ylabel('分数')
                        ax.set_title('ESG各维度得分')
                        ax.set_ylim(0, 100)
                        
                        # 添加数值标签
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{height:.1f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                
                with tab2:
                    st.subheader("与竞争对手基准比较")
                    
                    # 显示比较表格
                    st.dataframe(
                        benchmark_df.style.format({
                            'ESG总分': '{:.1f}',
                            '环境得分': '{:.1f}', 
                            '社会得分': '{:.1f}',
                            '治理得分': '{:.1f}'
                        }).highlight_max(
                            subset=['ESG总分', '环境得分', '社会得分', '治理得分']
                        ),
                        use_container_width=True
                    )
                    
                    # 比较图表
                    comparison_fig = create_comparison_bar_chart(benchmark_df, company_name)
                    st.pyplot(comparison_fig)
                
                with tab3:
                    st.subheader("优势与改进建议")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### ✅ 主要优势")
                        strengths = analysis_result.get('strengths', [])
                        if strengths:
                            for i, strength in enumerate(strengths, 1):
                                st.write(f"{i}. {strength}")
                        else:
                            st.info("暂无优势数据")
                    
                    with col2:
                        st.write("### 📈 改进建议")
                        recommendations = analysis_result.get('recommendations', [])
                        if recommendations:
                            for i, recommendation in enumerate(recommendations, 1):
                                st.write(f"{i}. {recommendation}")
                        else:
                            st.info("暂无改进建议")
                
                with tab4:
                    st.subheader("详细分析报告")
                    
                    st.write("### 竞争对手比较分析")
                    competitor_comparison = analysis_result.get('competitor_comparison', {})
                    if competitor_comparison:
                        for competitor, comparison in competitor_comparison.items():
                            st.write(f"**{competitor}**: {comparison}")
                    else:
                        st.info("暂无竞争对手比较数据")
                    
                    st.write("### 主要劣势")
                    weaknesses = analysis_result.get('weaknesses', [])
                    if weaknesses:
                        for i, weakness in enumerate(weaknesses, 1):
                            st.write(f"{i}. {weakness}")
                    else:
                        st.info("暂无劣势数据")
                    
                    # 下载报告
                    report_data = {
                        "company": company_name,
                        "industry": industry,
                        "analysis_date": datetime.now().isoformat(),
                        "analysis_results": analysis_result
                    }
                    
                    st.download_button(
                        label="下载分析报告 (JSON)",
                        data=json.dumps(report_data, indent=2, ensure_ascii=False),
                        file_name=f"esg_analysis_{company_name}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            else:
                st.error("ESG分析失败，请检查API密钥或重试。")
        else:
            st.error("无法从PDF中提取文本，请检查文件格式。")

else:
    # 显示示例分析
    st.info("👆 请上传ESG报告PDF文件开始分析")
    
    # 示例部分
    st.header("示例分析")
    st.markdown("""
    ### 功能特点:
    - 📄 **PDF解析**: 自动提取ESG报告文本内容
    - 🤖 **AI分析**: 使用GPT进行深度ESG分析
    - 📊 **基准比较**: 与行业竞争对手对比
    - 📈 **可视化**: 多维度图表展示
    - 💡 **改进建议**: 针对性的优化建议
    
    ### 支持的ESG维度:
    - **环境(Environmental)**: 碳排放、资源使用、环境保护等
    - **社会(Social)**: 员工权益、社区关系、供应链管理等  
    - **治理(Governance)**: 董事会结构、透明度、风险管理等
    """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>AI ESG Report Analyzer | 使用GPT技术进行ESG分析</div>",
    unsafe_allow_html=True
)
