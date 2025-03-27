import streamlit as st 
import pandas as pd
from io import StringIO


st.set_page_config(page_title="KLTN-FRM Analysis", layout="wide")
st.title('FRM Analysis')

tab1, tab2, tab3 = st.tabs(['Table','Analysis','Prediction'])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Upload your dataset here**')
        with st.expander('Upload Dataset'):
            uploaded_file = st.file_uploader("Choose a file")
    with col2:
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border = True):
            with st.expander('Correlation Analysis'):
                st.write('Plot 1')
                st.write('Plot 4')
                st.write('Plot 7')
    with col2:
        with st.container(border = True):
            with st.expander('Distance'):
                st.write('Plot 2')
                st.write('Plot 5')
                st.write('Plot 8')        
    with col3:
        with st.container(border = True):
            with st.expander('Distance'):
                st.write('Plot 3')
                st.write('Plot 6')
                st.write('Plot 9')

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border = True):
            with st.expander('Months'):
                st.write('Plot 1')
                st.write('Plot 4')
    with col2:
        with st.container(border = True):
            with st.expander('Quarter'):
                st.write('Plot 2')
                st.write('Plot 5')
    
    with col3:
        with st.container(border = True):
            with st.expander('Year'):
                st.write('Plot 3')
                st.write('Plot 6')        