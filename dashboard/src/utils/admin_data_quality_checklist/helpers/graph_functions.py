import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def plot_pie_chart(labels, values, title):
    # Sort values and labels in descending order
    sorted_data = sorted(zip(values, labels), reverse=True)
    sorted_values, sorted_labels = zip(*sorted_data)
    formatted_values = [format((v),',d') for v in sorted_values]
    color_map = {
        label: "#3b8e51" if "Unique" in label else "#9e2f17"
        for label in sorted_labels
    }
    fig = px.pie(
        names=sorted_labels, 
        values=sorted_values, 
        title=title,
        color=sorted_labels,
        color_discrete_map=color_map)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        showlegend= True
    )
    fig.update_traces(
        customdata=formatted_values,
        texttemplate="%{label}<br>%{customdata} (%{percent})",
        textinfo='none')
    return fig

@st.cache_data
def plot_100_stacked_bar_chart(data: pd.DataFrame, x: str, y: str, color: str, title: str, x_label: str, y_label: str):
    fig = px.bar(data, x=x, y=y, color=color, title=title, 
                labels={x: x_label, 'value': y_label},
                barmode='relative')  # This makes it a 100% stacked bar chart
    
    fig.update_layout(legend_title_text='')
    return fig
