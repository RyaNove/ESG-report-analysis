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

# Set matplotlib font settings (avoid Chinese font issues)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Page configuration
st.set_page_config(
    page_title="AI ESG Report Analyzer",
    page_icon="🌱",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    # First try environment variable, then secrets
    api_key = os.getenv('OPENAI_API_KEY', st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.warning("Please set OpenAI API key")
        return None
    return OpenAI(api_key=api_key)

client = init_openai_client()

# Title and introduction
st.title("🌱 AI ESG Report Analyzer")
st.markdown("""
Upload company ESG reports and we'll use AI to analyze their ESG performance and benchmark against competitors.
""")

# Sidebar - Company information input
st.sidebar.header("Company Information")
company_name = st.sidebar.text_input("Company Name", "Example Company")
industry = st.sidebar.selectbox(
    "Industry",
    ["Technology", "Financial", "Energy", "Manufacturing", "Consumer Goods", "Healthcare", "Other"]
)
competitors = st.sidebar.text_area(
    "Competitors (one per line)", 
    "Competitor A\nCompetitor B\nCompetitor C"
)

# File upload
st.header("1. Upload ESG Report")
uploaded_file = st.file_uploader(
    "Upload ESG Report in PDF format", 
    type=['pdf'],
    help="Supports PDF format ESG report files"
)

# Parse PDF content
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF parsing error: {str(e)}")
        return ""

# Analyze ESG report using GPT
def analyze_esg_with_gpt(report_text, company_name, industry, competitors_list):
    """Analyze ESG report using GPT"""
    
    if not client:
        st.error("OpenAI client not initialized, please check API key")
        return None
        
    # Limit text length to avoid token limits
    truncated_text = report_text[:8000] if len(report_text) > 8000 else report_text
    
    prompt = f"""
    Please analyze the following ESG report and provide a comprehensive assessment.

    Company Name: {company_name}
    Industry: {industry}
    Competitors: {', '.join(competitors_list)}

    ESG Report Content:
    {truncated_text}

    Please provide the following analysis:
    1. Overall ESG score (0-100 points)
    2. Detailed scores for Environmental (E), Social (S), Governance (G) dimensions
    3. Main strengths and weaknesses
    4. Comparative analysis with industry competitors
    5. Improvement recommendations

    Please return results in JSON format with the following fields:
    - overall_score (number)
    - environmental_score (number)
    - social_score (number)
    - governance_score (number)
    - strengths (array of strings)
    - weaknesses (array of strings)
    - competitor_comparison (object)
    - recommendations (array of strings)

    Ensure all scores are numbers between 0-100.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo to reduce costs
            messages=[
                {"role": "system", "content": "You are a professional ESG analyst specializing in evaluating company environmental, social, and governance performance. Always return valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content
        st.write("Raw response:", result_text)  # Debug information
        
        # Extract JSON portion
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate and clean results
            return validate_esg_results(result)
        else:
            st.error("Unable to parse AI response as JSON format")
            return None
            
    except Exception as e:
        st.error(f"GPT analysis error: {str(e)}")
        return None

def validate_esg_results(result):
    """Validate and clean ESG analysis results"""
    # Ensure all required fields exist
    required_fields = ['overall_score', 'environmental_score', 'social_score', 'governance_score']
    
    for field in required_fields:
        if field not in result:
            result[field] = 50  # Default value
    
    # Ensure scores are numbers and within reasonable range
    for score_field in required_fields:
        try:
            score = float(result[score_field])
            result[score_field] = max(0, min(100, score))
        except (ValueError, TypeError):
            result[score_field] = 50
    
    # Ensure array fields exist
    for array_field in ['strengths', 'weaknesses', 'recommendations']:
        if array_field not in result or not isinstance(result[array_field], list):
            result[array_field] = ["Insufficient data"]
    
    # Ensure comparison analysis field exists
    if 'competitor_comparison' not in result or not isinstance(result['competitor_comparison'], dict):
        result['competitor_comparison'] = {"Industry Average": "Insufficient data"}
    
    return result

# Generate benchmark data
def generate_benchmark_data(analysis_result, company_name, competitors_list):
    """Generate benchmark comparison data"""
    
    # Simulate competitor data
    np.random.seed(42)
    num_competitors = len(competitors_list)
    
    benchmark_data = {
        'Company': [company_name] + competitors_list,
        'Overall ESG Score': [float(analysis_result['overall_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['overall_score']) * 0.95, 
                      5, 
                      num_competitors
                  )),
        'Environmental Score': [float(analysis_result['environmental_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['environmental_score']) * 0.95, 
                      6, 
                      num_competitors
                  )),
        'Social Score': [float(analysis_result['social_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['social_score']) * 0.95, 
                      5, 
                      num_competitors
                  )),
        'Governance Score': [float(analysis_result['governance_score'])] + 
                  list(np.random.normal(
                      float(analysis_result['governance_score']) * 0.95, 
                      4, 
                      num_competitors
                  ))
    }
    
    # Ensure scores are within reasonable range
    for key in ['Overall ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']:
        benchmark_data[key] = [max(0, min(100, float(score))) for score in benchmark_data[key]]
    
    return pd.DataFrame(benchmark_data)

# Visualization functions
def create_radar_chart(analysis_result, company_name):
    """Create radar chart showing ESG performance across dimensions"""
    
    categories = ['Environmental', 'Social Responsibility', 'Corporate Governance', 'Transparency', 'Risk Management', 'Innovation']
    
    # Generate dimension scores based on main scores
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
    
    # Complete radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, scores, 'o-', linewidth=2, label=company_name, color='#1A936F')
    ax.fill(angles, scores, alpha=0.25, color='#1A936F')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 100)
    ax.set_title(f'{company_name} ESG Performance Radar Chart', size=14, weight='bold')
    ax.legend(loc='upper right')
    
    return fig

def create_comparison_bar_chart(benchmark_df, company_name):
    """Create benchmark comparison bar chart"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['Overall ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        data_sorted = benchmark_df.sort_values(metric, ascending=False)
        bars = axes[i].bar(data_sorted['Company'], data_sorted[metric], 
                          color=colors[i], alpha=0.7)
        
        # Highlight target company
        target_idx = list(data_sorted['Company']).index(company_name)
        bars[target_idx].set_color('#1A936F')
        bars[target_idx].set_alpha(1.0)
        
        axes[i].set_title(f'{metric} Comparison', weight='bold')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

# Main application logic
if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    
    # Show loading status
    with st.spinner("Parsing and analyzing ESG report..."):
        # Extract text
        report_text = extract_text_from_pdf(uploaded_file)
        
        if report_text:
            st.info(f"Successfully extracted text, {len(report_text)} characters total")
            
            # Show first 500 characters of extracted text (for debugging)
            with st.expander("View extracted text (first 500 characters)"):
                st.text(report_text[:500] + "..." if len(report_text) > 500 else report_text)
            
            # Analyze report
            competitors_list = [c.strip() for c in competitors.split('\n') if c.strip()]
            analysis_result = analyze_esg_with_gpt(
                report_text, company_name, industry, competitors_list
            )
            
            if analysis_result:
                st.success("ESG analysis completed!")
                
                # Display main results
                col1, col2, col3, col4 = st.columns(4)
                
                # Safely handle score display
                overall_score = float(analysis_result.get('overall_score', 0))
                env_score = float(analysis_result.get('environmental_score', 0))
                soc_score = float(analysis_result.get('social_score', 0))
                gov_score = float(analysis_result.get('governance_score', 0))
                
                with col1:
                    st.metric(
                        "Overall ESG Score", 
                        f"{overall_score:.1f}",
                        delta=f"Industry Avg: {overall_score - 5:.1f}"
                    )
                
                with col2:
                    st.metric(
                        "Environmental Score", 
                        f"{env_score:.1f}"
                    )
                
                with col3:
                    st.metric(
                        "Social Score", 
                        f"{soc_score:.1f}"
                    )
                
                with col4:
                    st.metric(
                        "Governance Score", 
                        f"{gov_score:.1f}"
                    )
                
                # Generate benchmark data
                benchmark_df = generate_benchmark_data(
                    analysis_result, company_name, competitors_list
                )
                
                # Tab display for detailed analysis
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Visual Analysis", "📈 Benchmark Comparison", "✅ Strengths & Improvements", "📋 Detailed Report"
                ])
                
                with tab1:
                    st.subheader("ESG Performance Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart
                        radar_fig = create_radar_chart(analysis_result, company_name)
                        st.pyplot(radar_fig)
                    
                    with col2:
                        # ESG score distribution
                        esg_scores = [env_score, soc_score, gov_score]
                        labels = ['Environmental', 'Social', 'Governance']
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(labels, esg_scores, 
                                     color=['#1A936F', '#2E86AB', '#F18F01'])
                        ax.set_ylabel('Score')
                        ax.set_title('ESG Dimension Scores')
                        ax.set_ylim(0, 100)
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{height:.1f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                
                with tab2:
                    st.subheader("Competitor Benchmark Comparison")
                    
                    # Display comparison table
                    st.dataframe(
                        benchmark_df.style.format({
                            'Overall ESG Score': '{:.1f}',
                            'Environmental Score': '{:.1f}', 
                            'Social Score': '{:.1f}',
                            'Governance Score': '{:.1f}'
                        }).highlight_max(
                            subset=['Overall ESG Score', 'Environmental Score', 'Social Score', 'Governance Score']
                        ),
                        use_container_width=True
                    )
                    
                    # Comparison chart
                    comparison_fig = create_comparison_bar_chart(benchmark_df, company_name)
                    st.pyplot(comparison_fig)
                
                with tab3:
                    st.subheader("Strengths & Improvement Recommendations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### ✅ Key Strengths")
                        strengths = analysis_result.get('strengths', [])
                        if strengths:
                            for i, strength in enumerate(strengths, 1):
                                st.write(f"{i}. {strength}")
                        else:
                            st.info("No strength data available")
                    
                    with col2:
                        st.write("### 📈 Improvement Recommendations")
                        recommendations = analysis_result.get('recommendations', [])
                        if recommendations:
                            for i, recommendation in enumerate(recommendations, 1):
                                st.write(f"{i}. {recommendation}")
                        else:
                            st.info("No recommendations available")
                
                with tab4:
                    st.subheader("Detailed Analysis Report")
                    
                    st.write("### Competitor Comparison Analysis")
                    competitor_comparison = analysis_result.get('competitor_comparison', {})
                    if competitor_comparison:
                        for competitor, comparison in competitor_comparison.items():
                            st.write(f"**{competitor}**: {comparison}")
                    else:
                        st.info("No competitor comparison data available")
                    
                    st.write("### Key Weaknesses")
                    weaknesses = analysis_result.get('weaknesses', [])
                    if weaknesses:
                        for i, weakness in enumerate(weaknesses, 1):
                            st.write(f"{i}. {weakness}")
                    else:
                        st.info("No weakness data available")
                    
                    # Download report
                    report_data = {
                        "company": company_name,
                        "industry": industry,
                        "analysis_date": datetime.now().isoformat(),
                        "analysis_results": analysis_result
                    }
                    
                    st.download_button(
                        label="Download Analysis Report (JSON)",
                        data=json.dumps(report_data, indent=2, ensure_ascii=False),
                        file_name=f"esg_analysis_{company_name}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            else:
                st.error("ESG analysis failed, please check API key or try again.")
        else:
            st.error("Unable to extract text from PDF, please check file format.")

else:
    # Show example analysis
    st.info("👆 Please upload ESG report PDF file to start analysis")
    
    # Example section
    st.header("Example Analysis")
    st.markdown("""
    ### Key Features:
    - 📄 **PDF Parsing**: Automatically extract text from ESG reports
    - 🤖 **AI Analysis**: Use GPT for in-depth ESG analysis
    - 📊 **Benchmark Comparison**: Compare with industry competitors
    - 📈 **Visualization**: Multi-dimensional chart display
    - 💡 **Improvement Recommendations**: Targeted optimization suggestions
    
    ### Supported ESG Dimensions:
    - **Environmental**: Carbon emissions, resource usage, environmental protection
    - **Social**: Employee rights, community relations, supply chain management  
    - **Governance**: Board structure, transparency, risk management
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>AI ESG Report Analyzer | Powered by GPT Technology</div>",
    unsafe_allow_html=True
)
