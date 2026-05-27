import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Cortex - Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0a0e27;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1a1f3a;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 CORTEX // Market Intelligence Engine")
st.markdown("**Autonomous Financial Intelligence Powered by 22-Agent Swarm**")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    asset_type = st.selectbox(
        "Asset Class",
        ["Equities", "Crypto", "Commodities", "Forex"]
    )
    
    time_range = st.selectbox(
        "Time Range",
        ["1H", "4H", "1D", "1W", "1M"]
    )
    
    risk_level = st.slider(
        "Risk Tolerance (%)",
        0, 100, 50
    )

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Active Signals", "24", "+3", delta_color="normal")

with col2:
    st.metric("Win Rate", "78.3%", "+2.1%", delta_color="normal")

with col3:
    st.metric("Avg Return", "12.5%", "-0.8%", delta_color="inverse")

with col4:
    st.metric("Portfolio VaR", "3.2%", "–0.1%", delta_color="normal")

st.divider()

# Data generation for demo
@st.cache_data
def generate_mock_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Price': np.cumsum(np.random.randn(100)) + 100,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'Signal_Score': np.random.uniform(0, 100, 100),
        'VaR_95': np.random.uniform(2, 5, 100)
    })
    return data

data = generate_mock_data()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Price Action & Signals")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Price'],
        mode='lines',
        name='Price',
        line=dict(color='#00d4ff', width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Signal_Score'],
        mode='markers',
        name='Signal Score',
        marker=dict(size=8, color=data['Signal_Score'], 
                   colorscale='RdYlGn', colorbar=dict(title="Score"))
    ))
    fig_price.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("📊 Value at Risk (VaR)")
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=data['Date'],
        y=data['VaR_95'],
        fill='tozeroy',
        name='VaR 95%',
        line=dict(color='#ff6b6b', width=2)
    ))
    fig_var.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_var, use_container_width=True)

st.divider()

# Agent swarm status
st.subheader("🐝 22-Agent Swarm Status")

agent_data = {
    'Agent': [
        'FilingWatcher', 'ExecWatcher', 'SocialRadar', 'MacroTracker',
        'DarkPoolMonitor', 'NewsParser', 'SentimentAnalyzer', 'TechnicalAnalyst',
        'FundamentalAnalyst', 'RiskAssessor', 'CorrelationEngine', 'AnomalyDetector',
        'NewsScrapers', 'FinancialParser', 'MarketMaker', 'VolatilityTracker',
        'EarningsPredictor', 'InsiderTracker', 'ReportGenerator', 'StrategyOptimizer',
        'ExecRecorder', 'ConversationAgent'
    ],
    'Status': np.random.choice(['Active', 'Processing', 'Idle'], 22),
    'Uptime %': np.random.uniform(95, 100, 22)
}

agent_df = pd.DataFrame(agent_data)

col1, col2 = st.columns(2)

with col1:
    status_counts = agent_df['Status'].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        color_discrete_map={'Active': '#00d4ff', 'Processing': '#ffaa00', 'Idle': '#555555'}
    )
    fig_status.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_status, use_container_width=True)

with col2:
    fig_uptime = px.bar(
        agent_df.sort_values('Uptime %', ascending=False).head(10),
        x='Uptime %',
        y='Agent',
        orientation='h',
        color='Uptime %',
        color_continuous_scale='Viridis'
    )
    fig_uptime.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_uptime, use_container_width=True)

st.divider()

# Recent intelligence notes
st.subheader("📝 Recent Intelligence Notes")

notes = [
    {
        'time': '09:45 AM',
        'ticker': 'NVDA',
        'signal': 'BULLISH',
        'reason': 'Executive reshuffle detected - CTO transition signals AI investment pivot',
        'confidence': '92%'
    },
    {
        'time': '08:30 AM',
        'ticker': 'TSLA',
        'signal': 'BEARISH',
        'reason': 'Dark pool anomaly: 2.3M shares detected at 15% discount to market',
        'confidence': '78%'
    },
    {
        'time': '07:15 AM',
        'ticker': 'META',
        'signal': 'NEUTRAL',
        'reason': 'Regulatory filing shows standard quarterly capital allocation',
        'confidence': '65%'
    }
]

for note in notes:
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        st.text(note['time'])
    with col2:
        st.text(f"**{note['ticker']}**")
    with col3:
        signal_color = "🟢" if note['signal'] == "BULLISH" else "🔴" if note['signal'] == "BEARISH" else "⚪"
        st.text(f"{signal_color} {note['reason']}")
    with col4:
        st.text(note['confidence'])
    st.divider()

# Footer
st.markdown("""
---
**CORTEX** | Market Alpha via Autonomous Intelligence  
Powered by: Bright Data • DeepSeek • Gemini • ChromaDB  
*Delivering real-time mathematical investment intelligence 6 hours before legacy networks update*
""")
