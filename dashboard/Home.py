import streamlit as st
import os
from src.utils.utility_functions import set_page_config,setheader,setFooter
set_page_config("collapsed")



if __name__ == "__main__":
    selectedNav = setheader()
    if selectedNav == "Pre Survey":
          st.switch_page("pages/1_Pre_Survey.py")
    if selectedNav == "Admin Data Quality":
          st.switch_page("pages/2_Admin_Data_Quality_Checklist.py")
    if selectedNav == "Post Survey":
          st.switch_page("pages/3_Post_Survey.py")
    setFooter()

    with st.container():
      st.markdown("<h1 style='text-align: center;'>DiscSim | A CEGIS Project", unsafe_allow_html=True)

      with st.container():
            left, middle, right = st.columns([1,3,1])
            with middle:
                  st.write("DiscSim is a simulation tool developed for the Center for Effective Governance of Indian States (CEGIS), an organization dedicated to assisting state governments in India to achieve better development outcomes.")
                  
                  st.subheader("Overview")
                  st.write("An important goal of CEGIS is to improve the quality of administrative data collected by state governments. One approach is to re-sample a subset of the data and measure deviations from the original samples collected. These deviations are quantified as **discrepancy scores**, and significant scores are flagged for third-party intervention.")
                  st.write("Often, it's unclear which re-sampling strategy yields the most accurate and reliable discrepancy scores. The goal of this project is to create a simulator that predicts discrepancy scores and assesses their statistical accuracy across different re-sampling strategies.")
                  st.write("DiscSim comprises a backend API built with FastAPI and a frontend interface developed using Streamlit. The project utilizes PostgreSQL for database management and is containerized with Docker for easy deployment.")

                  st.subheader("About CEGIS")
                  desc, image = st.columns([3,1])
                  desc.write("The Centre for Effective Governance of Indian States (CEGIS Foundation) is dedicated to enabling transformative improvement in the functioning of the Indian state.")
                  script_dir = os.path.dirname(os.path.abspath(__file__))
                  image.image(os.path.join(script_dir, "logo_page.png"))
                  st.write("Our work is informed by research, evidence and a practical orientation towards implementable ideas. We partner with state governments to improve the tools they need to design effective governance reforms, while also offering strategic implementation support. Our focus is on strengthening governance by improving four core functions of the government - outcome measurement, personnel management, strategic financial management and leveraging markets for improved public service delivery.")

                  tab1, tab2, tab3 = st.tabs(["Our Mission", "Our Approach", "Our Values"])
                  with tab1:
                        st.write('''
                              We aim to improve lives by helping state governments deliver better development outcomes.
                                 
                              CEGIS believes that strengthening the capacity of state governments is the most cost-effective way of improving development outcomes at scale.''')
                  with tab2:
                        st.write('''
                        At CEGIS, we believe that governments should function as high-performing organisations with the following four features: the ability to measure developmental outcomes precisely, to manage personnel effectively, to employ strategic public finance in order to maximise return on investment and welfare, and finally to be able to manage market interfaces for optimal public service delivery. 
                        ''')

                        with st.expander("Outcome Measurement"):
                              st.write('''
                                    State governments in India often focus on implementing programs based on inputs like budgetary allocations and workforce utilisation rather than expected outcomes such as the experience of beneficiaries of programs. One of the reasons for this is the limited availability of precise, consistent and comprehensive data regarding outcomes. Our Outcome Measurement team is working to address this gap by assisting governments in evaluating the quantity, quality and usability of administrative data. Our work in this area is divided into four key streams:

                                    1. Setting systems for conducting annual citizen-level Key Performance Indicators (KPI) surveys for slow-moving outcome indicators
                                    2. Setting systems for High Frequency Measurement (HFM) of citizen and official-level  fast-moving output indicators of quality of public services
                                    3. Strengthening government’s own administrative data quality (ADQ)
                                    4. Strengthening data use strategy

                                    Together, these initiatives will build the capacity of  governments to gather precise, consistent, and comprehensive data to evaluate the effectiveness of services and make informed decisions, thus leading to better public service delivery and positive experiences for end beneficiaries.
                              ''')
                        
                        with st.expander("Personnel Measurement"):
                              st.write('''
                                    In high-performing organisations, frontline staff and managers are held accountable for achieving desired outcomes but are also granted autonomy in determining how they carry out their responsibilities. The opposite is often true in government settings, which leads to rule-based administration rather than role-based administration, affecting service delivery according to  the stated objectives. 

                                    Our Personnel Management team addresses this gap by employing a competency-driven approach to managing human resources in government. The team works with central, state and local governments to actualise their human capital by improving human resource management practices and organisational effectiveness. We believe that supporting governments in undertaking these reforms will result in improved execution capacity and a fulfilling career journey for government officials. Our work in this area is divided into three key streams: 

                                    1. Competency-based recruitment
                                    2. Competency-based capacity building (learning and development)
                                    3. Performance measurement and management
                                    
                                    Together, these initiatives will bring more clarity to the work done by government officials, enhance workplace effectiveness and enable the officials to demonstrate accountability for outcomes, leading to empowered, equipped and future ready government officials that is a key to achieving citizen-centric service delivery.
                              ''')

                        with st.expander("Strategic Public Finance"):
                              st.write('''
                                    Governments in India often face challenges in each of the four stages of the annual cycle of public financial management - planning and allocation, release and expenditure, accounting and audit, and utilisation of current year data as input for planning for the subsequent year. The planning stage is hampered by the lack of outcome data, which restricts the effective allocation of funds to areas with the greatest need. Allocations and releases are also restricted by liquidity constraints, caused by suboptimal revenue collections. These limitations, compounded by inefficient fund flow channels, often result in funds not reaching the right destination at the right time impacting expenditure pertaining to complex activities associated with public service delivery. In addition, the dense and inconsistent accounting systems in government make it difficult to leverage it for analysis and planning. Finally, there is limited possibility of triangulating expenditure data with outcomes in order to plan correctly for the next year. 

                                    Our Strategic Public Finance team supports the governments in strengthening public financial management by adopting a budgetary life cycle approach, addressing issues from planning and allocation to auditing by solving key problems in the budgeting process, improving sufficiency and efficiency, and ensuring smooth funds flow. Our work in this area is divided in four key streams:

                                    1. Strengthening revenue systems
                                    2. Fiscal exchange systems
                                    3. Quality of expenditure
                                    4. Macro fiscal and economic policy
                                    
                                    These initiatives will ensure efficient fund allocation and flow, thereby improving financial efficiency and public service delivery. 
                              ''')

                        with st.expander("States and Markets"):
                              st.write('''
                                    Governments have three distinct roles in the ecosystem for service delivery to its citizens  – direct provision, managing interface with market through regulation, and policy. Traditionally, government focus has been more on the “direct provision” function. However, a combination of fiscal constraints, market-based efficiencies and incentive mismatches makes it necessary for governments to judiciously engage with non-state actors (both for-profit and non-profit) to ensure access to affordable and high-quality services.

                                    To create effective “service ecosystems”, the State and Markets team supports governments in measuring cost-effectiveness of delivery across different forms of provision, and in defining and implementing mechanisms to better coordinate actions across public and private sectors. Our approach includes identifying the right combination of these three levers - provision, regulation and policy - while considering market and government (in)efficiencies, building government and market capacities, defining and implementing incentives to drive performance, and instituting high-quality monitoring to complete the feedback loop.

                                    By facilitating the engagement between states and non-state actors, the State and Markets team strives to overcome traditional constraints, enhance market efficiencies and ensure access to high-quality, affordable services. Our mission is to create low-friction service ecosystems that enhance government and market efficiency and, ultimately, the quality of life of common citizens.
                              ''')

                  with tab3:
                        st.video("https://cegis.org/sites/default/files/2024-05/CEGIS%20Shared%20Values%20Video%20_%20Short%20Cut%20_%20Aparna%20VO.mp4")

                  st.subheader("Modules")
                  presurvey, admindata, postsurvey = st.tabs(["Pre Survey", "Admin Data Quality Checks", "Post Survey"])

                  with presurvey:
                        st.page_link("pages/1_Pre_Survey.py", label="Go to Module Page",icon=":material/display_external_input:")
                        with st.expander("Third Party Sampling Strategy"):
                              st.write('''
                                    demo
                                          ''')
                        
                  with admindata:
                        st.page_link("pages/2_Admin_Data_Quality_Checklist.py", label="Go to Module Page",icon=":material/display_external_input:")
                        with st.expander("Unique Identifier"):
                              st.write('''
                                    demo
                                    ''')
                        with st.expander("Remove Duplicates"):
                              st.write('''
                                    demo
                                    ''')
                  
                  with postsurvey:
                        st.page_link("pages/3_Post_Survey.py", label="Go to Module Page",icon=":material/display_external_input:")
                        with st.expander("ECD Nested Supervision"):
                              st.write('''
                                    demo
                                    ''')
      
      