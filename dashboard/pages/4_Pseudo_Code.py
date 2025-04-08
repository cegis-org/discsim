import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from src.utils.utility_functions import set_page_config,setFooter,setheader
from src.utils.pseudo_code.helpers.file_upload import handle_file_upload
set_page_config()

API_BASE_URL = os.getenv("API_BASE_URL")
PSEUDO_CODE_ENDPOINT = f"{API_BASE_URL}/pseudo_code"
def pseudo_code_analysis():
    st.sidebar.header("Insights on Discrepancies in Growth Monitoring")
    # File selection
    file_option = st.sidebar.radio(
        "Choose an option:",
        ("Upload a new file", "Select a previously uploaded file"),
    )
    uploaded_file = handle_file_upload(
        file_option, category="pseudo_code_analysis"
    )
    # Upload CSV file
    #uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        uploaded_file.seek(0)
        files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
        with st.spinner('Analyzing... Please wait.'):
            response = requests.post(PSEUDO_CODE_ENDPOINT,files=files)

        if response.status_code == 200:
            st.markdown("<h3 style='text-align:center'>Nested Supervision :: Nutrition", unsafe_allow_html=True)
            status,message,data = response.json()
            if 'summary' in data:
                a,c,d,e = st.columns(4)
                a.metric("Children remeasured", format(data['summary']['totalSampleSize'],',d'), border=True)
                c.metric("Projects Covered", format(data['summary']['projects'],',d'), border=True)
                d.metric("Sectors Covered", format(data['summary']['sectors'],',d'), border=True)
                e.metric("AWCs Visited", format(data['summary']['AWC'],',d'), border=True)

            district, project, sector,awc = st.tabs(["District Level", "Project Level", "Sector Level", "AWC Level"])

            with district:
                st.markdown("<h4 style='text-align:center'>District-level insights on Nested Supervision intervention", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if 'districtLevelInsights' in data:
                        container = st.container(border=True)
                        sameHeightWeight = pd.DataFrame(data['districtLevelInsights']['sameHeightWeight'])
                        container.markdown("<h6 style='text-align:center'>Cases with Exact Same Height and Weight Measurements", unsafe_allow_html=True)
                        fig_same_values = px.bar(
                            sameHeightWeight, 
                            x=sameHeightWeight["Metric"].astype(str) + " %",
                            y="Percentage (%)", 
                            color="Metric", 
                            text=sameHeightWeight["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'Exact same height': '#4285f4', 'Exact same weight': '#34a853'}
                        )
                        fig_same_values.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title="% Remeasurements",showgrid=False,showticklabels=False)
                        )
                        container.plotly_chart(fig_same_values)
                        with container.expander("Show Data"):
                            sameHeightWeight['Percentage (%)'] = sameHeightWeight['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(sameHeightWeight,hide_index=True,use_container_width=True)
                with col2:
                    if 'childrenCategory' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        childrenCategory = pd.DataFrame(data['districtLevelInsights']['childrenCategory'])
                        container.markdown("<h6 style='text-align:center'>Average Difference in Height & Weight Measurement", unsafe_allow_html=True)
                        fig_combined = px.bar(
                            childrenCategory.melt(
                                id_vars=['Category'], 
                                value_vars=['Average Height Difference (cms)', 'Average Weight Difference (kgs)'], 
                                var_name='Metric', 
                                value_name='Value'
                            ),
                            x='Category', 
                            y='Value', 
                            color='Metric', 
                            text="Value", 
                            barmode='group',
                            color_discrete_map = {'Average Height Difference (cms)': '#4285f4', 'Average Weight Difference (kgs)': '#34a853'}
                        )
                        fig_combined.update_traces(width=0.1)
                        fig_combined.update_layout(
                            barcornerradius=5,
                            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5,title=None),
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title="Difference (in cms & kgs)",showgrid=False,showticklabels=False),
                            bargroupgap=0.1,
                        )
                        container.plotly_chart(fig_combined)
                        with container.expander("Show Data"):
                            st.dataframe(childrenCategory,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px;padding:0.3rem'>Wasting [Weight-For-Height]", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if 'wastingLevels' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        wastingLevel = pd.DataFrame(data['districtLevelInsights']['wastingLevels'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Wasting Level between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>Note: AWT SAM and AWT Wasting figures are for the re-measurement<br> subset only and not for the entire universe of AWC children", unsafe_allow_html=True)
                        fig_wasting_metrics = px.bar(
                            wastingLevel, 
                            x="Metric", y="Percentage (%)", 
                            color="Metric", 
                            text=wastingLevel["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT SAM': '#1a73e8', 'Supervisor SAM': '#0b5394', 'AWT Wasting': '#e06666', 'Supervisor Wasting': '#cc0000'}
                        )
                        fig_wasting_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title="% Remeasurements",showgrid=False,showticklabels=False)
                        )
                        container.plotly_chart(fig_wasting_metrics)
                        with container.expander("Show Data"):
                            wastingLevel['Percentage (%)'] = wastingLevel['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(wastingLevel,hide_index=True,use_container_width=True)

                with col2:
                    if 'wastingClassification' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        wastingClassification = pd.DataFrame(data['districtLevelInsights']['wastingClassification'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Wasting Classification between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>SAM - Severely acutely malnourished [>3 SD (Standard Deviation)],<br> MAM = Moderately acutely malnourished [2-3 SD]", unsafe_allow_html=True)
                        fig_wasting_metrics = px.bar(
                            wastingClassification, 
                            y="Metric", x="Percentage (%)", 
                            color="Metric", 
                            text=wastingClassification["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT Normal; Supervisor SAM': '#990000', 'AWT Normal; Supervisor MAM': '#cc0000', 'AWT MAM; Supervisor SAM': '#e06666', 'Other Misclassifications': '#fbbc04', 'Same Classification': '#34a853'},
                            orientation='h'
                        )
                        fig_wasting_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True),
                            bargap=0.5
                        )
                        container.plotly_chart(fig_wasting_metrics)
                        with container.expander("Show Data"):
                            wastingClassification['Percentage (%)'] = wastingClassification['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(wastingClassification,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px;padding:0.3rem'>Underweight [Weight-For-Age]", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if 'underweightLevels' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        underweightLevels = pd.DataFrame(data['districtLevelInsights']['underweightLevels'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Underweight Level between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>Note: AWT SUW and AWT Underweight figures are for the re-measurement<br> subset only and not for the entire universe of AWC children", unsafe_allow_html=True)
                        fig_underweight_metrics = px.bar(
                            underweightLevels, 
                            x="Metric", y="Percentage (%)", 
                            color="Metric", 
                            text=underweightLevels["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT SUW': '#1a73e8', 'Supervisor SUW': '#0b5394', 'AWT UW': '#e06666', 'Supervisor UW': '#cc0000'}
                        )
                        fig_underweight_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title="% Remeasurements",showgrid=False,showticklabels=False)
                        )
                        container.plotly_chart(fig_underweight_metrics)
                        with container.expander("Show Data"):
                            underweightLevels['Percentage (%)'] = underweightLevels['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(underweightLevels,hide_index=True,use_container_width=True)

                with col2:
                    if 'underweightClassification' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        underweightClassification = pd.DataFrame(data['districtLevelInsights']['underweightClassification'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Underweight Classification between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>SUW - Severely underweight [>3 SD],<br> MUW = Moderately Underweight [2-3 SD]", unsafe_allow_html=True)
                        fig_underweight_metrics = px.bar(
                            underweightClassification, 
                            y="Metric", x="Percentage (%)", 
                            color="Metric", 
                            text=underweightClassification["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT Normal; Supervisor SUW': '#990000', 'AWT Normal; Supervisor MUW': '#e06666', 'Other Misclassifications': '#fbbc04', 'Same Classification': '#34a853'},
                            orientation='h'
                        )
                        fig_underweight_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True),
                            bargap=0.5
                        )
                        container.plotly_chart(fig_underweight_metrics)
                        with container.expander("Show Data"):
                            underweightClassification['Percentage (%)'] = underweightClassification['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(underweightClassification,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px;padding:0.3rem'>Stunting [Height For Age]", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if 'stuntingLevels' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        stuntingLevels = pd.DataFrame(data['districtLevelInsights']['stuntingLevels'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Stunting Level between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>Note: AWT SS and AWT Stunting figures are for the re-measurement<br> subset only and not for the entire universe of AWC children", unsafe_allow_html=True)
                        fig_underweight_metrics = px.bar(
                            stuntingLevels, 
                            x="Metric", y="Percentage (%)", 
                            color="Metric", 
                            text=stuntingLevels["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT SS': '#1a73e8', 'Supervisor SS': '#0b5394', 'AWT Stunting': '#e06666', 'Supervisor Stunting': '#cc0000'}
                        )
                        fig_underweight_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title="% Remeasurements",showgrid=False,showticklabels=False)
                        )
                        container.plotly_chart(fig_underweight_metrics)
                        with container.expander("Show Data"):
                            stuntingLevels['Percentage (%)'] = stuntingLevels['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(stuntingLevels,hide_index=True,use_container_width=True)

                with col2:
                    if 'stuntingClassification' in data['districtLevelInsights']:
                        container = st.container(border=True)
                        stuntingClassification = pd.DataFrame(data['districtLevelInsights']['stuntingClassification'])
                        container.markdown("<h6 style='text-align:center;padding:0'>Difference in Stunting classification between AWTs and Supervisors", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px;margin:1px'>SS - Severely stunted [>3 SD],<br> MS = Moderately stunted [2-3 SD]", unsafe_allow_html=True)
                        fig_underweight_metrics = px.bar(
                            stuntingClassification, 
                            y="Metric", x="Percentage (%)", 
                            color="Metric", 
                            text=stuntingClassification["Percentage (%)"].astype(str) + " %",
                            color_discrete_map = {'AWT Normal; Supervisor SS': '#990000', 'AWT Normal; Supervisor MS': '#cc0000', 'AWT MS; Supervisor SS': '#e06666', 'Other Misclassifications': '#fbbc04', 'Same Classifications': '#34a853'},
                            orientation='h'
                        )
                        fig_underweight_metrics.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            margin=dict(t=0, b=0),
                            height=300,
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True),
                            bargap=0.5
                        )
                        container.plotly_chart(fig_underweight_metrics)
                        with container.expander("Show Data"):
                            stuntingClassification['Percentage (%)'] = stuntingClassification['Percentage (%)'].apply(lambda x: f'{x} %')
                            st.dataframe(stuntingClassification,hide_index=True,use_container_width=True)


            with project:
                col1,col2 = st.columns(2)
                with col1:
                    if 'sameHeight' in data['projectLevelInsights']:
                        container = st.container(border=True)
                        projectSameHeight = pd.DataFrame(data['projectLevelInsights']['sameHeight'])
                        container.markdown("<h6 style='text-align:center;padding-bottom:0'>Remeasurements with Exact Same AWT and Supervisor Height Measurements", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px'>Top 10 Projects", unsafe_allow_html=True)
                        top_12_projectSameHeight = projectSameHeight.nlargest(10, 'Exact_Same_Height_%')
                        top_12_projectSameHeight = top_12_projectSameHeight.sort_values(by='Exact_Same_Height_%', ascending=True)
                        fig_top_12_projectSameHeight = px.bar(
                            top_12_projectSameHeight,
                            x='Exact_Same_Height_%', 
                            y='Proj_Name',
                            orientation='h',
                            text=top_12_projectSameHeight["Exact_Same_Height_%"].astype(str) + " %",
                        )
                        fig_top_12_projectSameHeight.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            height=300,
                            margin=dict(t=0, b=0,l=0,r=0),
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True)
                        )
                        container.plotly_chart(fig_top_12_projectSameHeight)
                        with container.expander("Show Data"):
                            st.dataframe(projectSameHeight,hide_index=True,use_container_width=True)
                    
                with col2:
                    if 'sameWeight' in data['projectLevelInsights']:
                        container = st.container(border=True)
                        projectSameWeight = pd.DataFrame(data['projectLevelInsights']['sameWeight'])
                        container.markdown("<h6 style='text-align:center;padding-bottom:0'>Remeasurements with Exact Same AWT and Supervisor Weight Measurements", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px'>Top 10 Projects", unsafe_allow_html=True)
                        top_12_projectSameWeight = projectSameWeight.nlargest(10, 'Exact_Same_Weight_%')
                        top_12_projectSameWeight = top_12_projectSameWeight.sort_values(by='Exact_Same_Weight_%', ascending=True)
                        fig_top_12_projectSameWeight = px.bar(
                            top_12_projectSameWeight,
                            x='Exact_Same_Weight_%', 
                            y='Proj_Name',
                            orientation='h',
                            text=top_12_projectSameWeight["Exact_Same_Weight_%"].astype(str) + " %",
                        )
                        fig_top_12_projectSameWeight.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            height=300,
                            margin=dict(t=0, b=0,l=0,r=0),
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True)
                        )
                        container.plotly_chart(fig_top_12_projectSameWeight)
                        with container.expander("Show Data"):
                            st.dataframe(projectSameWeight,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px'>WASTING [WEIGHT-FOR-HEIGHT]", unsafe_allow_html=True)

                if 'wastingLevels' in data['projectLevelInsights']:
                    container = st.container(border=True)
                    projectWastingLevels = pd.DataFrame(data['projectLevelInsights']['wastingLevels'])
                    container.markdown("<h6 style='text-align:center;'>Difference in Wasting levels between AWTs and Supervisors", unsafe_allow_html=True)
                    categories = ['AWT_SAM_%', 'AWT_Wasting_%','Supervisor_SAM_%', 'Supervisor_Wasting_%']
                    colors = ['#4285f4', '#0b5394', '#e06666', '#cc0000']
                    wasting_table_melted = projectWastingLevels.melt(id_vars=['Proj_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_wasting_levels = px.bar(
                        wasting_table_melted,
                        x='Proj_Name',
                        y='Percentage',
                        color='Category',
                        text=wasting_table_melted['Percentage'].astype(str) + '%',
                        barmode='group',
                        color_discrete_sequence=colors,
                    )
                    fig_wasting_levels.update_traces(
                        textposition='outside',
                        textangle=0,
                        textfont=dict(
                            color='black',
                            size=14,
                        )
                    )
                    fig_wasting_levels.update_layout(
                        barcornerradius=2,
                        margin=dict(t=0, b=0),
                        xaxis=dict(title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=False),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_wasting_levels)
                    with container.expander("Show Data"):
                        st.dataframe(projectWastingLevels,hide_index=True,use_container_width=True)


                if 'wastingClassification' in data['projectLevelInsights']:
                    container = st.container(border=True)
                    projectWastingClassification = pd.DataFrame(data['projectLevelInsights']['wastingClassification'])
                    container.markdown("<h6 style='text-align:center;'>Difference between AWT and Supervisor in Wasting Classification", unsafe_allow_html=True)
                    categories = ['AWT_Normal_Sup_SAM_%', 'AWT_Normal_Sup_MAM_%', 'AWT_MAM_Sup_SAM_%', 'Other_Misclassifications_%', 'Same_Classifications_%']
                    colors = ['darkred', 'red', 'lightcoral', 'gold', 'green']
                    project_analysis_melted = projectWastingClassification.melt(id_vars=['Proj_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_projectWastingClassification = px.bar(
                        project_analysis_melted,
                        x='Percentage',
                        y='Proj_Name',
                        color='Category',
                        text=project_analysis_melted['Percentage'].astype(str) + '%',
                        color_discrete_sequence=colors,
                    )
                    fig_projectWastingClassification.update_layout(
                        barcornerradius=5,
                        margin=dict(t=0, b=0),
                        height=400,
                        xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=True),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_projectWastingClassification)
                    with container.expander("Show Data"):
                        st.dataframe(projectWastingClassification,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px'>UNDERWEIGHT [WEIGHT-FOR-AGE]", unsafe_allow_html=True)

                if 'underweightLevels' in data['projectLevelInsights']:
                    container = st.container(border=True)
                    projectUnderweightLevels = pd.DataFrame(data['projectLevelInsights']['underweightLevels'])
                    container.markdown("<h6 style='text-align:center;'>Difference in Underweight levels between AWTs and Supervisor", unsafe_allow_html=True)
                    categories = ['AWT_SUW_%', 'Sup_SUW_%', 'AWT_Underweight_%', 'Sup_Underweight_%']
                    colors = ['#4285f4', '#0b5394', '#e06666', '#cc0000']
                    project_analysis_melted = projectUnderweightLevels.melt(id_vars=['Proj_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_uw_levels = px.bar(
                        project_analysis_melted,
                        x='Proj_Name',
                        y='Percentage',
                        color='Category',
                        text=project_analysis_melted['Percentage'].astype(str) + '%',
                        barmode='group',
                        color_discrete_sequence=colors,
                    )
                    fig_uw_levels.update_traces(
                        textposition='outside',
                        textangle=0,
                        textfont=dict(
                            color='black',
                            size=14,
                        )
                    )
                    fig_uw_levels.update_layout(
                        barcornerradius=2,
                        margin=dict(t=0, b=0),
                        xaxis=dict(title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=False),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_uw_levels)
                    with container.expander("Show Data"):
                        st.dataframe(projectUnderweightLevels,hide_index=True,use_container_width=True)
                            
                if 'underweightClassification' in data['projectLevelInsights']:
                    container = st.container(border=True)
                    projectUnderweightClassification = pd.DataFrame(data['projectLevelInsights']['underweightClassification'])
                    container.markdown("<h6 style='text-align:center;'>Difference between AWT and Supervisor in Underweight Classification", unsafe_allow_html=True)
                    categories = ['AWT_Normal_Sup_SUW_%', 'AWT_Normal_Sup_MUW_%', 'Other_Misclassifications_%', 'Same_Classifications_%']
                    colors = ['darkred', 'red', 'gold', 'green']
                    project_analysis_melted = projectUnderweightClassification.melt(id_vars=['Proj_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_projectUnderweightClassification = px.bar(
                        project_analysis_melted,
                        x='Percentage',
                        y='Proj_Name',
                        color='Category',
                        orientation='h',
                        text=project_analysis_melted['Percentage'].astype(str) + '%',
                        color_discrete_sequence=colors,
                    )
                    fig_projectUnderweightClassification.update_layout(
                        barcornerradius=5,
                        margin=dict(t=0, b=0),
                        height=400,
                        xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=True),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_projectUnderweightClassification)
                    with container.expander("Show Data"):
                        st.dataframe(projectUnderweightClassification,hide_index=True,use_container_width=True)

                if 'discrepancy' in data['projectLevelInsights']:
                    container = st.container(border=True)
                    container.markdown("<h6 style='text-align:center;padding-bottom:0'>Discrepancy Zoning Based on Percentile", unsafe_allow_html=True)
                    projectDisc = pd.DataFrame(data['projectLevelInsights']['discrepancy'])
                    fig_treemap = px.treemap(
                        projectDisc, 
                        path=['Proj_Name'], 
                        values='Discrepancy Rate (%)', 
                        color='Zone',
                        color_discrete_map={'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'},
                        hover_data={
                            'Total_Remeasurements': True,
                            'Discrepancy Rate (%)': True,
                            'Percentile_Rank (%)': True,
                            'Zone': True
                        }
                    )
                    fig_treemap.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                    fig_treemap.update_traces(
                        marker=dict(
                            cornerradius=5,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=(
                            "<b>%{label}</b><br>"
                            "Total Measurements: %{customdata[0]}<br>"
                            "Discp. Rate: %{customdata[1]:.1f}%<br>"
                            "Percentile Rank: %{customdata[2]:.1f}%"
                        )
                    )
                    container.plotly_chart(fig_treemap)
                    with container.expander("Show Data"):
                            st.dataframe(projectDisc,hide_index=True,use_container_width=True)
                    
            with sector:
                col1,col2 = st.columns(2)
                with col1:
                    if 'sameHeight' in data['sectorLevelInsights']:
                        container = st.container(border=True)
                        sectorSameHeight = pd.DataFrame(data['sectorLevelInsights']['sameHeight'])
                        container.markdown("<h6 style='text-align:center;padding-bottom:0'>Remeasurements with Exact Same AWT and Supervisor Height Measurements", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px'>Top 10 Sectors", unsafe_allow_html=True)
                        top_12_sectorSameHeight = sectorSameHeight.nlargest(10, 'Exact_Same_Height_%')
                        top_12_sectorSameHeight = top_12_sectorSameHeight.sort_values(by='Exact_Same_Height_%', ascending=True)
                        fig_top_12_sectorSameHeight = px.bar(
                            top_12_sectorSameHeight,
                            x='Exact_Same_Height_%', 
                            y='Sec_Name',
                            orientation='h',
                            text=top_12_sectorSameHeight["Exact_Same_Height_%"].astype(str) + " %",
                        )
                        fig_top_12_sectorSameHeight.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            height=300,
                            margin=dict(t=0, b=0,l=0,r=0),
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True)
                        )
                        container.plotly_chart(fig_top_12_sectorSameHeight)
                        with container.expander("Show Data"):
                            st.dataframe(sectorSameHeight,hide_index=True,use_container_width=True)
                with col2:
                    if 'sameWeight' in data['sectorLevelInsights']:
                        container = st.container(border=True)
                        sectorSameWeight = pd.DataFrame(data['sectorLevelInsights']['sameWeight'])
                        container.markdown("<h6 style='text-align:center;padding-bottom:0'>Remeasurements with Exact Same AWT and Supervisor Weight Measurements", unsafe_allow_html=True)
                        container.markdown("<p style='text-align:center;color:grey;font-size:12px'>Top 10 Sectors", unsafe_allow_html=True)
                        top_12_sectorSameWeight = sectorSameWeight.nlargest(10, 'Exact_Same_Weight_%')
                        top_12_sectorSameWeight = top_12_sectorSameWeight.sort_values(by='Exact_Same_Weight_%', ascending=True)
                        fig_top_12_sectorSameWeight = px.bar(
                            top_12_sectorSameWeight,
                            x='Exact_Same_Weight_%', 
                            y='Sec_Name',
                            orientation='h',
                            text=top_12_sectorSameWeight["Exact_Same_Weight_%"].astype(str) + " %",
                        )
                        fig_top_12_sectorSameWeight.update_layout(
                            barcornerradius=5,
                            showlegend=False,
                            height=300,
                            margin=dict(t=0, b=0,l=0,r=0),
                            xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=False),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True)
                        )
                        container.plotly_chart(fig_top_12_sectorSameWeight)
                        with container.expander("Show Data"):
                            st.dataframe(sectorSameWeight,hide_index=True,use_container_width=True)

                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px'>WASTING [WEIGHT-FOR-HEIGHT]", unsafe_allow_html=True)

                if 'wastingLevels' in data['sectorLevelInsights']:
                    container = st.container(border=True)
                    sectorWastingLevels = pd.DataFrame(data['sectorLevelInsights']['wastingLevels'])
                    container.markdown("<h6 style='text-align:center;'>Difference in Wasting levels between AWTs and Supervisors", unsafe_allow_html=True)
                    categories = ['AWT_SAM_%', 'AWT_Wasting_%','Supervisor_SAM_%', 'Supervisor_Wasting_%']
                    colors = ['#4285f4', '#0b5394', '#e06666', '#cc0000']
                    wasting_table_melted = sectorWastingLevels.melt(id_vars=['Sec_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_wasting_levels = px.bar(
                        wasting_table_melted,
                        x='Sec_Name',
                        y='Percentage',
                        color='Category',
                        text=wasting_table_melted['Percentage'].astype(str) + '%',
                        barmode='group',
                        color_discrete_sequence=colors,
                    )
                    fig_wasting_levels.update_layout(
                        barcornerradius=2,
                        margin=dict(t=0, b=0),
                        xaxis=dict(title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=False),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_wasting_levels)
                    with container.expander("Show Data"):
                        st.dataframe(sectorWastingLevels,hide_index=True,use_container_width=True)

                if 'wastingClassification' in data['sectorLevelInsights']:
                    container = st.container(border=True)
                    sectorWastingClassification = pd.DataFrame(data['sectorLevelInsights']['wastingClassification'])
                    container.markdown("<h6 style='text-align:center;'>Difference between AWT and Supervisor in Wasting Classification", unsafe_allow_html=True)
                    categories = ['AWT_Normal_Sup_SAM_%', 'AWT_Normal_Sup_MAM_%', 'AWT_MAM_Sup_SAM_%', 'Other_Misclassifications_%', 'Same_Classifications_%']
                    colors = ['darkred', 'red', 'lightcoral', 'gold', 'green']
                    sector_analysis_melted = sectorWastingClassification.melt(id_vars=['Sec_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_sectorWastingClassification = px.bar(
                        sector_analysis_melted,
                        x='Percentage',
                        y='Sec_Name',
                        color='Category',
                        text=sector_analysis_melted['Percentage'].astype(str) + '%',
                        color_discrete_sequence=colors,
                    )
                    fig_sectorWastingClassification.update_layout(
                        barcornerradius=5,
                        margin=dict(t=0, b=0),
                        height=400,
                        xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=True),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_sectorWastingClassification)
                    with container.expander("Show Data"):
                        st.dataframe(sectorWastingClassification,hide_index=True,use_container_width=True)

                
                st.markdown("<h4 style='text-align:center;background-color:#34a853;color:white;margin-bottom:10px;border-radius:10px'>UNDERWEIGHT [WEIGHT-FOR-AGE]", unsafe_allow_html=True)

                if 'underweightLevels' in data['sectorLevelInsights']:
                    container = st.container(border=True)
                    sectorUnderweightLevels = pd.DataFrame(data['sectorLevelInsights']['underweightLevels'])
                    container.markdown("<h6 style='text-align:center;'>Difference in Underweight levels between AWTs and Supervisor", unsafe_allow_html=True)
                    categories = ['AWT_SUW_%', 'Sup_SUW_%', 'AWT_Underweight_%', 'Sup_Underweight_%']
                    colors = ['#4285f4', '#0b5394', '#e06666', '#cc0000']
                    sector_analysis_melted = sectorUnderweightLevels.melt(id_vars=['Sec_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    #sector_analysis_melted = sector_analysis_melted.nlargest(10, 'Percentage')
                    fig_uw_levels = px.bar(
                        sector_analysis_melted,
                        x='Sec_Name',
                        y='Percentage',
                        color='Category',
                        text=sector_analysis_melted['Percentage'].astype(str) + '%',
                        barmode='group',
                        color_discrete_sequence=colors,
                    )
                    fig_uw_levels.update_layout(
                        barcornerradius=2,
                        margin=dict(t=0, b=0),
                        xaxis=dict(title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=False),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_uw_levels)
                    with container.expander("Show Data"):
                        st.dataframe(sectorUnderweightLevels,hide_index=True,use_container_width=True)
                            
                if 'underweightClassification' in data['sectorLevelInsights']:
                    container = st.container(border=True)
                    sectorUnderweightClassification = pd.DataFrame(data['sectorLevelInsights']['underweightClassification'])
                    container.markdown("<h6 style='text-align:center;'>Difference between AWT and Supervisor in Underweight Classification", unsafe_allow_html=True)
                    categories = ['AWT_Normal_Sup_SUW_%', 'AWT_Normal_Sup_MUW_%', 'Other_Misclassifications_%', 'Same_Classifications_%']
                    colors = ['darkred', 'red', 'gold', 'green']
                    project_analysis_melted = sectorUnderweightClassification.melt(id_vars=['Sec_Name'], value_vars=categories, var_name='Category', value_name='Percentage')
                    fig_sectorUnderweightClassification = px.bar(
                        project_analysis_melted,
                        x='Percentage',
                        y='Sec_Name',
                        color='Category',
                        orientation='h',
                        text=project_analysis_melted['Percentage'].astype(str) + '%',
                        color_discrete_sequence=colors,
                    )
                    fig_sectorUnderweightClassification.update_layout(
                        barcornerradius=5,
                        margin=dict(t=0, b=0),
                        height=400,
                        xaxis=dict(range=[0,100],title=None,showgrid=False,showticklabels=True),
                        yaxis=dict(title=None,showgrid=False,showticklabels=True),
                        legend=dict(title=None,orientation='h'),
                        bargap=0.2
                    )
                    container.plotly_chart(fig_sectorUnderweightClassification)
                    with container.expander("Show Data"):
                        st.dataframe(sectorUnderweightClassification,hide_index=True,use_container_width=True)

                if 'discrepancy' in data['sectorLevelInsights']:
                    container = st.container(border=True)
                    container.markdown("<h6 style='text-align:center;padding-bottom:0'>Discrepancy Zoning Based on Percentile", unsafe_allow_html=True)
                    sectorDisc = pd.DataFrame(data['sectorLevelInsights']['discrepancy'])
                    fig_treemap = px.treemap(
                        sectorDisc, 
                        path=['Sec_Name'], 
                        values='Discrepancy Rate (%)', 
                        color='Zone',
                        color_discrete_map={'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'},
                        hover_data={
                            'Total_Remeasurements': True,
                            'Discrepancy Rate (%)': True,
                            'Percentile_Rank (%)': True,
                            'Zone': True
                        }
                    )
                    fig_treemap.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                    fig_treemap.update_traces(
                        marker=dict(
                            cornerradius=5,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=(
                            "<b>%{label}</b><br>"
                            "Total Measurements: %{customdata[0]}<br>"
                            "Discp. Rate: %{customdata[1]:.1f}%<br>"
                            "Percentile Rank: %{customdata[2]:.1f}%"
                        )
                    )
                    container.plotly_chart(fig_treemap)
                    with container.expander("Show Data"):
                            st.dataframe(sectorDisc,hide_index=True,use_container_width=True)    

            with awc:
                if 'discrepancy' in data['awcLevelInsights']:
                    container = st.container(border=True)
                    container.markdown("<h6 style='text-align:center;padding-bottom:0'>Discrepancy Zoning Based on Percentile", unsafe_allow_html=True)
                    awcDisc = pd.DataFrame(data['awcLevelInsights']['discrepancy'])
                    fig_treemap = px.treemap(
                        awcDisc, 
                        path=['AWC_Name'], 
                        values='Discrepancy Rate (%)', 
                        color='Zone',
                        color_discrete_map={'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'},
                        hover_data={
                            'Total_Remeasurements': True,
                            'Discrepancy Rate (%)': True,
                            'Percentile_Rank (%)': True,
                            'Zone': True
                        }
                    )
                    fig_treemap.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                    fig_treemap.update_traces(
                        marker=dict(
                            cornerradius=5,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=(
                            "<b>%{label}</b><br>"
                            "Total Measurements: %{customdata[0]}<br>"
                            "Discp. Rate: %{customdata[1]:.1f}%<br>"
                            "Percentile Rank: %{customdata[2]:.1f}%"
                        )
                    )
                    container.plotly_chart(fig_treemap)
                    with container.expander("Show Data"):
                            st.dataframe(awcDisc,hide_index=True,use_container_width=True)    
                        
        else:
            st.error(f"Error: {response.json()['detail']}")


    else:
        st.info("Please upload a CSV file (UTF-8 encoded) to begin.")

if __name__ == "__main__":
    selectedNav = setheader("Post Survey Nutrition")
    if selectedNav == "Pre Survey":
          st.switch_page("pages/1_Pre_Survey.py")
    if selectedNav == "Admin Data Quality":
          st.switch_page("pages/2_Admin_Data_Quality_Checklist.py")
    pseudo_code_analysis()

    setFooter()
