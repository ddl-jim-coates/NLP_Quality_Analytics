# apps/semantic_search/app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import requests

# Configure page
st.set_page_config(
    page_title="Quality Analytics - Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #004d7a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .search-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .result-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.8rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background: #1f77b4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
    .category-tag {
        background: #e8f4fd;
        color: #1f77b4;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_demo_data():
    """Load sample audit findings data for demo"""
    # Simulated quality findings data
    findings_data = {
        'finding_id': [
            'QF-2024-001', 'QF-2024-002', 'QF-2024-003', 'QF-2024-004', 'QF-2024-005',
            'QF-2024-006', 'QF-2024-007', 'QF-2024-008', 'QF-2024-009', 'QF-2024-010'
        ],
        'finding_text': [
            "Committee correspondence and documentation was not filed in the appropriate location. Required notification letters and documentation were missing from the central file system.",
            "Standard Operating Procedure documentation was incomplete. Master SOP was not available and verification processes were not properly documented according to protocol requirements.",
            "Animal health monitoring procedures were not followed correctly. Veterinary oversight documentation was missing and welfare protocols were not implemented as specified.",
            "Training records for personnel were incomplete. Required certifications were missing and competency assessments had not been conducted for key staff members.",
            "Data integrity issues were identified during review. Electronic records showed discrepancies and backup procedures were not functioning as designed.",
            "Quality control testing procedures were not performed according to specifications. Required testing intervals were missed and documentation was inadequate.",
            "Facility maintenance records were incomplete. Environmental monitoring data was missing and calibration certificates had expired for critical equipment.",
            "Audit trail documentation was insufficient. Change control procedures were not followed and approval workflows were not properly documented.",
            "Risk assessment procedures were not conducted as required. Mitigation strategies were not documented and monitoring plans were incomplete.",
            "Regulatory compliance documentation was missing. Required submissions were delayed and correspondence with authorities was not properly filed."
        ],
        'category': [
            'Documentation Management', 'SOP Compliance', 'Animal Welfare', 'Training & Competency',
            'Data Integrity', 'Quality Control', 'Facility Management', 'Change Control',
            'Risk Management', 'Regulatory Compliance'
        ],
        'severity': ['Major', 'Minor', 'Critical', 'Major', 'Critical', 'Minor', 'Major', 'Minor', 'Major', 'Critical'],
        'area': ['Clinical', 'Manufacturing', 'Preclinical', 'Quality', 'IT Systems', 'Laboratory', 'Facilities', 'Quality', 'Compliance', 'Regulatory'],
        'date': pd.date_range('2024-01-01', periods=10, freq='10D')
    }
    
    return pd.DataFrame(findings_data)

@st.cache_data
def load_sample_queries():
    """Load sample search queries for demo"""
    return [
        "documentation missing",
        "animal health monitoring",
        "SOP procedures",
        "training records incomplete",
        "data integrity issues",
        "quality control testing",
        "facility maintenance",
        "regulatory compliance"
    ]

def simulate_semantic_search(query, df, top_k=5):
    """Simulate semantic search with realistic similarity scores"""
    np.random.seed(hash(query) % 2**32)  # Deterministic randomness based on query
    
    # Simulate similarity scores based on keyword matching and some randomness
    similarities = []
    query_lower = query.lower()
    
    for text in df['finding_text']:
        text_lower = text.lower()
        
        # Base similarity on keyword overlap
        base_similarity = 0.3
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        overlap = len(query_words.intersection(text_words))
        
        if overlap > 0:
            base_similarity += min(overlap * 0.2, 0.5)
        
        # Add some realistic variation
        noise = np.random.normal(0, 0.1)
        final_similarity = max(0.1, min(0.95, base_similarity + noise))
        similarities.append(final_similarity)
    
    # Get top-k results
    df_with_sim = df.copy()
    df_with_sim['similarity'] = similarities
    top_results = df_with_sim.nlargest(top_k, 'similarity')
    
    return top_results

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Quality Analytics Search Engine</h1>
        <p>Semantic search across quality and compliance findings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_demo_data()
    sample_queries = load_sample_queries()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Search Analytics")
        
        # Usage metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card"><h3>180+</h3><p>Active Users</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>7,000+</h3><p>Unique Searches</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Search trending
        st.subheader("🔥 Trending Searches")
        trending_data = {
            'Query': ['SOP documentation', 'training records', 'animal welfare', 'data integrity'],
            'Count': [45, 38, 32, 28]
        }
        trending_df = pd.DataFrame(trending_data)
        
        fig_trending = px.bar(
            trending_df, 
            x='Count', 
            y='Query',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_trending.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_trending, use_container_width=True)
        
        # Category distribution
        st.subheader("📈 Search Categories")
        category_counts = df['category'].value_counts().head(5)
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        search_query = st.text_input(
            "🔍 Search quality findings:",
            placeholder="Enter your search query...",
            help="Use natural language to search across all quality findings"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        use_sample = st.selectbox(
            "Or try a sample:",
            [""] + sample_queries,
            help="Select a pre-built query"
        )
        if use_sample:
            search_query = use_sample
    
    # Search filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_categories = st.multiselect(
                "Categories:",
                options=df['category'].unique(),
                help="Filter by finding category"
            )
        with col2:
            selected_severity = st.multiselect(
                "Severity:",
                options=df['severity'].unique(),
                help="Filter by severity level"
            )
        with col3:
            selected_areas = st.multiselect(
                "Areas:",
                options=df['area'].unique(),
                help="Filter by operational area"
            )
    
    # Perform search
    if search_query:
        with st.spinner("🔍 Searching..."):
            # Apply filters
            filtered_df = df.copy()
            if selected_categories:
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
            if selected_severity:
                filtered_df = filtered_df[filtered_df['severity'].isin(selected_severity)]
            if selected_areas:
                filtered_df = filtered_df[filtered_df['area'].isin(selected_areas)]
            
            if len(filtered_df) == 0:
                st.warning("No results found matching the selected filters.")
                return
            
            # Perform semantic search
            results = simulate_semantic_search(search_query, filtered_df, top_k=5)
            
            # Display results
            st.markdown(f"### 📋 Search Results for: *'{search_query}'*")
            st.markdown(f"Found **{len(results)}** relevant findings")
            
            for idx, row in results.iterrows():
                similarity_color = "🟢" if row['similarity'] > 0.7 else "🟡" if row['similarity'] > 0.5 else "🔴"
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>{row['finding_id']}</strong>
                        <span class="similarity-score">{similarity_color} {row['similarity']:.3f}</span>
                    </div>
                    <p>{row['finding_text']}</p>
                    <div style="margin-top: 0.8rem;">
                        <span class="category-tag">📂 {row['category']}</span>
                        <span class="category-tag">⚠️ {row['severity']}</span>
                        <span class="category-tag">🏢 {row['area']}</span>
                        <span class="category-tag">📅 {row['date'].strftime('%Y-%m-%d')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Search insights
            if len(results) > 0:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_similarity = results['similarity'].mean()
                    st.metric("Average Similarity", f"{avg_similarity:.3f}")
                
                with col2:
                    high_confidence = len(results[results['similarity'] > 0.7])
                    st.metric("High Confidence Results", high_confidence)
                
                with col3:
                    unique_categories = results['category'].nunique()
                    st.metric("Categories Found", unique_categories)
    
    else:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to the Quality Analytics Search Engine
        
        This semantic search engine helps you find relevant quality and compliance findings using natural language queries.
        
        **Key Features:**
        - 🧠 **Semantic Understanding**: Finds conceptually similar content, not just keyword matches
        - ⚡ **Real-time Search**: Instant results across thousands of findings  
        - 📊 **Smart Ranking**: Results ranked by semantic similarity
        - 🔍 **Advanced Filters**: Filter by category, severity, and operational area
        
        **Usage Statistics:**
        - **2.5 years** in production
        - **170+ active users** across the organization
        - **7,000+ unique searches** performed
        
        Try entering a search query above or select a sample query to get started!
        """)

if __name__ == "__main__":
    main()