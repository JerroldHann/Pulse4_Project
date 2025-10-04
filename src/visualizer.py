import streamlit as st
import plotly.graph_objects as go

def risk_radar():
    categories = ['Credit', 'Frequency', 'Amount', 'Diversity', 'Geo']
    values = [0.8, 0.6, 0.9, 0.7, 0.4]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Risk Profile'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False,
        template='plotly_dark',
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)
