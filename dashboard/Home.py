import streamlit as st
import os
from src.utils.utility_functions import set_page_config,setheader,setFooter
set_page_config("collapsed")



if __name__ == "__main__":
      selectedNav = setheader("Home")
      if selectedNav == "Intervention Design":
            st.switch_page("pages/1_Intervention_Design.py")
      if selectedNav == "Admin Data Diagnostic":
            st.switch_page("pages/2_Admin_Data_Quality_Checklist.py")
      if selectedNav == "Intervention Analytics":
            st.switch_page("pages/3_Intervention_Analytics.py")
      setFooter()

      with st.container():
            addOncss = """
            <style>
            div[data-testid="stElementToolbar"]{
                  visibility: hidden !important;
                  opacity: 0 !important;
            }
            div.st-key-mainContent img{
                  margin-top:8px
            }
            div[data-testid="stAppViewContainer"] {
                  pointer-events: auto !important;
            }
            </style>
            """
            st.markdown(addOncss, unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center;color:#136a9a;font-weight:800;padding:0;margin:0'>Welcome to VALIData", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align:center;color:#34a853;margin-bottom:-50px;font-weight:400'>CEGIS’ Digital Public Good offering", unsafe_allow_html=True)
      

      with st.container(key="mainContent"):
            left, middle, right = st.columns([1,3,1])
            with middle:
                  st.write('----')
                  st.markdown("<div style=text-align:justify;margin-bottom:20px> <span style='color:#34a853;font-weight:700;font-size:22px'>VALIData</span> is a Digital Public Good offering by the <a href='https://cegis.org/about' target='_blank'>Centre for Effective Governance of Indian States</a>. It is intended to be a one stop platform for all things data analytics geared towards enhancing the state governments’ ability for data-driven decision-making, especially in the sectors of Health, Education, and Nutrition.</div>",unsafe_allow_html=True)
                  
                  st.markdown("<div style=text-align:justify;margin-bottom:20px><span style='color:#34a853;font-weight:600;font-size:20px'>VALIData Features</span>: VALIData contains a plethora of analytical packages that shall help in different stages of data systems strengthening</div>",unsafe_allow_html=True)
                  
                  script_dir = os.path.dirname(os.path.abspath(__file__))

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/add_module.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Admin Data Diagnostic</span> module consists of features that shall help users assess the quality of admin data and identify various issues present in it. For example, one of the features this module/package contains is calculating the percentage of data points that are valid, accurate, and reliable, for any specific indicator<div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/intervention_design.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Intervention Design</span> module consists of features that shall help users design any intervention to improve Health/Education/Nutrition data quality or strengthen their data systems. For example, one of the features this module/package contains is a sample size calculator that takes in different contextual specifications and constraints from the user and throws out an optimal sample size for the intervention</div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/intervention_analytics.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Intervention Analytics</span> module consists of features that shall help users view analytical insights from the intervention. For example, one of the packages in this module is the nested supervision analytics package which shall help in generating insights from a nested supervision intervention, at different levels to the corresponding layers of staff</div>",unsafe_allow_html=True)

                  st.markdown("<div style=text-align:justify;margin-bottom:20px><span style='color:#34a853;font-weight:600;font-size:20px'>Three sectoral VALIData tools</span>: We are building three parallel VALIData tools which shall contain analytical packages under all the three modules mentioned above.</div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/validata_health.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>VALIData Health</span> shall consist of analytical packages under Admin Data Diagnostic, Intervention Design, Intervention Analytics modules, with some common cross-sectoral features and other features specifically developed for Health sector data</div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/validata_education.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>VALIData Education</span> shall consist of analytical packages under Admin Data Diagnostic, Intervention Design, Intervention Analytics modules, with some common cross-sectoral features and other features specifically developed for analysing Education sector data<div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/validata_nutrition.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>VALIData Nutrition</span> shall consist of analytical packages under Admin Data Diagnostic, Intervention Design, Intervention Analytics modules, with some common cross-sectoral features and other features specifically developed for analysing Nutrition sector data<div>",unsafe_allow_html=True)
            
                  st.markdown("<div style=text-align:justify;margin-bottom:20px><span style='color:#34a853;font-weight:600;font-size:20px'>Target audience for VALIData</span>: We are building a plethora of data analytics packages on VALIData for:<div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/cegis_project.jpg"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Project delivery teams in CEGIS</span> who work with the Health, Education, and Nutrition departments in different states</div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/partner_registeration.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Partner organisations</span> working with the government in Health, Education, and Nutrition space to improve state capacity, service delivery, admin data quality, data-driven governance, etc. through their own interventions<div>",unsafe_allow_html=True)

                  image, desc = st.columns([1,12])
                  image.image(os.path.join(script_dir, "images/gov_stakeholders.png"))
                  desc.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700'>Government stakeholders</span> who can benefit from VALIData’s automated analytical templates to improve data quality, service delivery performance, and enhance data-driven decision-making within the government<div>",unsafe_allow_html=True)

                  st.write("")
                  st.write("")
                  st.markdown("<div style=text-align:justify;margin-bottom:20px><span style='font-weight:700;color:#34a853'>VALIData</span> is being built brick-by-brick, with new and useful features, thanks to the useful feedback we are receiving from all these stakeholders. This is intended to be a public good for anyone interested in enhancing the state capacity and data-driven decision-making ability of India’s state governments. Please try out the tool, and reach out to us with your ideas on what more we can build. We would be extremely happy to hear from you! <div>",unsafe_allow_html=True)
                  st.write("")
                  st.write("")
