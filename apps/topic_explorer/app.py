# apps/topic_explorer/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import json

# Configure page
st.set_page_config(
    page_title="Quality Analytics - Topic Explorer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for proper contrast and readability
st.markdown("""
<style>
    /* Override Streamlit's default styling */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #17a2b8 0%, #138496 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
    }
    
    .main-header p {
        color: white !important;
        margin: 0.5rem 0 0 0;
    }
    
    /* Topic cards */
    .topic-card {
        background: #ffffff;
        color: #333333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .topic-card strong {
        color: #17a2b8;
        font-weight: bold;
    }
    
    .topic-card p {
        color: #333333;
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    
    /* Keyword tags */
    .keyword-tag {
        background: #e8f7f9;
        color: #17a2b8 !important;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #17a2b8;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: white;
        color: #333333;
        border: 1px solid #cccccc;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background-color: white;
        color: #333333;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background-color: white;
        color: #333333;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #17a2b8;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #138496;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #333333;
    }
    
    /* Fix any white text issues */
    .stMarkdown, .stText, p, div, span {
        color: #333333 !important;
    }
    
    /* Ensure readability in all elements */
    .element-container, .stMarkdown > div {
        color: #333333 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: white;
        color: #333333;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e8f7f9;
        color: #333333;
        border: 1px solid #17a2b8;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fff3cd;
        color: #333333;
        border: 1px solid #ffc107;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        color: #333333;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_topic_data():
    """Load sample topic modeling data"""
    topics_data = {
        'topic_id': [0, 1, 2, 3, 4, 5, 6, 7],
        'topic_name': [
            'Documentation & Filing',
            'Training & Competency',
            'Equipment & Maintenance',
            'Data Integrity',
            'Protocol Compliance',
            'Regulatory Submissions',
            'Quality Control Testing',
            'Risk Management'
        ],
        'keywords': [
            ['documentation', 'filing', 'records', 'correspondence', 'missing'],
            ['training', 'competency', 'certification', 'personnel', 'qualified'],
            ['equipment', 'maintenance', 'calibration', 'facility', 'monitoring'],
            ['data', 'integrity', 'electronic', 'backup', 'validation'],
            ['protocol', 'procedure', 'compliance', 'deviation', 'sop'],
            ['regulatory', 'submission', 'authority', 'approval', 'filing'],
            ['testing', 'quality', 'control', 'specification', 'results'],
            ['risk', 'assessment', 'mitigation', 'monitoring', 'management']
        ],
        'document_count': [89, 67, 54, 72, 83, 45, 61, 38],
        'avg_similarity': [0.78, 0.82, 0.75, 0.88, 0.79, 0.73, 0.81, 0.76]
    }
    
    return pd.DataFrame(topics_data)

@st.cache_data
def generate_topic_evolution_data():
    """Generate time series data for topic evolution"""
    dates = pd.date_range('2023-01-01', '2024-06-01', freq='M')
    topics = ['Documentation & Filing', 'Training & Competency', 'Equipment & Maintenance', 'Data Integrity']
    
    evolution_data = []
    for topic in topics:
        base_value = np.random.randint(5, 15)
        for date in dates:
            # Add some seasonal variation and trend
            trend = np.random.normal(0, 2)
            seasonal = 3 * np.sin(2 * np.pi * date.month / 12)
            value = max(0, base_value + trend + seasonal + np.random.normal(0, 1))
            
            evolution_data.append({
                'date': date,
                'topic': topic,
                'document_count': int(value)
            })
    
    return pd.DataFrame(evolution_data)

def create_topic_network():
    """Create a network visualization of topic relationships"""
    # Create a simple network graph
    G = nx.Graph()
    
    # Add nodes (topics)
    topics = [
        'Documentation', 'Training', 'Equipment', 'Data Integrity',
        'Protocol', 'Regulatory', 'Quality Control', 'Risk Mgmt'
    ]
    
    for topic in topics:
        G.add_node(topic)
    
    # Add edges (relationships between topics)
    relationships = [
        ('Documentation', 'Protocol', 0.7),
        ('Training', 'Protocol', 0.6),
        ('Equipment', 'Quality Control', 0.8),
        ('Data Integrity', 'Quality Control', 0.9),
        ('Regulatory', 'Protocol', 0.5),
        ('Risk Mgmt', 'Equipment', 0.6),
        ('Risk Mgmt', 'Data Integrity', 0.7)
    ]
    
    for source, target, weight in relationships:
        G.add_edge(source, target, weight=weight)
    
    return G

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Topic Explorer</h1>
        <p>Machine learning-powered topic discovery and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    topics_df = load_topic_data()
    evolution_df = generate_topic_evolution_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Analysis Controls")
        
        # Topic filtering
        selected_topics = st.multiselect(
            "Select Topics:",
            options=topics_df['topic_name'].tolist(),
            default=topics_df['topic_name'].tolist()[:4],
            help="Choose topics to analyze"
        )
        
        # Time range
        st.subheader("📅 Time Range")
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2024, 6, 1))
        
        # Analysis options
        st.subheader("📊 Visualization Options")
        show_keywords = st.checkbox("Show Keywords", value=True)
        show_evolution = st.checkbox("Show Topic Evolution", value=True)
        show_network = st.checkbox("Show Topic Network", value=False)
        
        # Model info
        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        st.markdown("""
        - **Algorithm**: Latent Dirichlet Allocation (LDA)
        - **Topics**: 8 discovered topics
        - **Documents**: 500+ quality findings
        - **Coherence Score**: 0.84
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Topic Overview", "📈 Topic Evolution", "🔗 Topic Network", "🔍 Topic Details"])
    
    with tab1:
        st.markdown("### 📊 Topic Distribution and Performance")
        
        # Filter data
        filtered_topics = topics_df[topics_df['topic_name'].isin(selected_topics)]
        
        if len(filtered_topics) == 0:
            st.warning("Please select at least one topic to analyze.")
            return
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Topic distribution pie chart
            fig_pie = px.pie(
                filtered_topics,
                values='document_count',
                names='topic_name',
                title='Document Distribution by Topic',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Average similarity by topic
            fig_bar = px.bar(
                filtered_topics,
                x='topic_name',
                y='avg_similarity',
                title='Average Topic Coherence',
                color='avg_similarity',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(
                height=400,
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Topic summary table
        st.markdown("### 📋 Topic Summary")
        summary_df = filtered_topics[['topic_name', 'document_count', 'avg_similarity']].copy()
        summary_df.columns = ['Topic', 'Documents', 'Coherence Score']
        summary_df['Documents'] = summary_df['Documents'].astype(int)
        summary_df['Coherence Score'] = summary_df['Coherence Score'].round(3)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with tab2:
        if show_evolution:
            st.markdown("### 📈 Topic Evolution Over Time")
            
            # Filter evolution data
            filtered_evolution = evolution_df[
                (evolution_df['topic'].isin(selected_topics)) &
                (evolution_df['date'] >= pd.to_datetime(start_date)) &
                (evolution_df['date'] <= pd.to_datetime(end_date))
            ]
            
            if len(filtered_evolution) > 0:
                # Time series plot
                fig_evolution = px.line(
                    filtered_evolution,
                    x='date',
                    y='document_count',
                    color='topic',
                    title='Topic Prevalence Over Time',
                    markers=True
                )
                fig_evolution.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Number of Documents',
                    legend_title='Topic'
                )
                st.plotly_chart(fig_evolution, use_container_width=True)
                
                # Trend analysis
                st.markdown("### 📊 Trend Analysis")
                trend_summary = filtered_evolution.groupby('topic').agg({
                    'document_count': ['mean', 'std', 'min', 'max']
                }).round(2)
                trend_summary.columns = ['Average', 'Std Dev', 'Minimum', 'Maximum']
                st.dataframe(trend_summary, use_container_width=True)
            else:
                st.warning("No data available for selected time range and topics.")
        else:
            st.info("Enable 'Show Topic Evolution' in the sidebar to view temporal analysis.")
    
    with tab3:
        if show_network:
            st.markdown("### 🔗 Topic Relationship Network")
            
            # Create network graph
            G = create_topic_network()
            
            # Get node positions
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Create edge traces
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = G[edge[0]][edge[1]]['weight']
                edge_info.append(f"{edge[0]} ↔ {edge[1]}: {weight:.2f}")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                marker=dict(
                    size=30,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            )
            
            # Create figure
            fig_network = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Topic Relationship Network',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Node size represents topic importance",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#888", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            st.markdown("""
            **Network Interpretation:**
            - **Nodes**: Individual topics discovered by the model
            - **Edges**: Semantic relationships between topics
            - **Edge thickness**: Strength of relationship (correlation)
            """)
        else:
            st.info("Enable 'Show Topic Network' in the sidebar to view topic relationships.")
    
    with tab4:
        st.markdown("### 🔍 Detailed Topic Analysis")
        
        # Topic selection for detailed view
        selected_topic = st.selectbox(
            "Select topic for detailed analysis:",
            options=topics_df['topic_name'].tolist()
        )
        
        if selected_topic:
            topic_data = topics_df[topics_df['topic_name'] == selected_topic].iloc[0]
            
            # Topic details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", topic_data['document_count'])
            with col2:
                st.metric("Coherence Score", f"{topic_data['avg_similarity']:.3f}")
            with col3:
                coverage = topic_data['document_count'] / topics_df['document_count'].sum()
                st.metric("Coverage", f"{coverage:.1%}")
            
            # Keywords
            if show_keywords:
                st.markdown("#### 🏷️ Top Keywords")
                keywords = topic_data['keywords']
                keyword_html = " ".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                st.markdown(keyword_html, unsafe_allow_html=True)
            
            # Sample documents (simulated)
            st.markdown("#### 📄 Sample Documents")
            sample_docs = [
                f"Sample document 1 for {selected_topic.lower()} showing relevant content...",
                f"Another example document discussing {selected_topic.lower()} with specific details...",
                f"Third sample showing {selected_topic.lower()} related findings and observations..."
            ]
            
            for i, doc in enumerate(sample_docs, 1):
                st.markdown(f"""
                <div class="topic-card">
                    <strong>Document {i}</strong><br>
                    {doc}
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()