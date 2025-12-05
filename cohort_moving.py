import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('Прогнозирвоание методом передвижки возрастов')


#________________________Получение_данных_____________________________#

@st.cache_data
def load_data():
    co_df = pd.read_excel(
        'content/population/cohort_data.xlsx',
        header = 1,
        index_col = 0)

    return co_df

cohort_df_2023 = load_data()
#____________________________Передвижка_______________________________#

def move_cohorts():
    st.write(1)
    
    new_pop_cohrt_male, new_pop_cohrt_female = [0]* len(cohort_df_2023.index), [0]* len(cohort_df_2023.index)
    
    df_new = pd.DataFrame({
        'Возрастные группы': cohort_df_2023.index,
        'Численность мужчин': new_pop_cohrt_male,
        'Численность женщин': new_pop_cohrt_female})
    df_new.set_index('Возрастные группы', inplace = True)
    st.write(df_new)
    #return df_new


#____________________________Интерфейс_______________________________#



#st.dataframe(cohort_df_2023)

#year = st.slider('Передвижка', 2023, 2043, step = 5, on_change = move_cohorts())


st.image("content/population_pyramid.gif") 


