import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecast_expsm import auto_forecast_exp_smooth
import altair as alt
st.set_page_config(layout="wide")
st.title('Прогнозирвоание методом передвижки возрастов')


#________________________Получение_данных_____________________________#

@st.cache_data
def load_data():
    co_df = pd.read_excel('content/population/cohort_data(1).xlsx', header = 1, index_col = 0)

    mm_df = pd.read_csv('content/migration/by_sex_age_sep/men.csv', sep=';')
    mm_df.set_index('Возраст', inplace=True)
    mm_df.index = mm_df.index.astype(str)
    mm_df = mm_df.T
    mm_df.index.name = 'Годы'
    mm_df.index = mm_df.index.astype(str)

    mf_df = pd.read_csv('content/migration/by_sex_age_sep/women.csv', sep=';')
    mf_df.set_index('Возраст', inplace=True)
    mf_df.index = mf_df.index.astype(str)
    mf_df = mf_df.T
    mf_df.index.name = 'Годы'
    mf_df.index = mf_df.index.astype(str)

    b_c_df = pd.read_csv('content/coeff/births_coeff.csv', sep=';')
    b_c_df.set_index('Годы', inplace=True)
    b_c_df.index = b_c_df.index.astype(str)

    d_c_df_m = pd.read_csv('content/coeff/deaths_coeff/male_deaths.csv', sep=';')
    d_c_df_f = pd.read_csv('content/coeff/deaths_coeff/fem_deaths.csv', sep=';')

    
    return co_df, mm_df, mf_df, b_c_df


cohort_df_2023, mig_men, mig_women, births_coeff = load_data()

@st.cache_data
def get_forcasts():
    new_years = [str(int(mig_men.index[-1]) + i) for i in range(1, 16)]
    
    future_menmig, future_womenmig = dict(), dict()
    for i in range(len(mig_men.columns)):
        models_mmig = st.session_state.get(f'auto.{mig_men.columns[i]}', {})
        models_fmig = st.session_state.get(f'auto.{mig_women.columns[i]}', {})
        if models_mmig == {}:
            models_mmig = auto_forecast_exp_smooth(mig_men, mig_men.columns[i], False)
            models_key = f'auto.миграционного движения мужчин.{mig_men.columns[i]}'
            st.session_state[models_key] = models_mmig
            models_fmig = auto_forecast_exp_smooth(mig_women, mig_women.columns[i], False)
            models_key = f'auto.миграционного движения женщин.{mig_women.columns[i]}'
            st.session_state[models_key] = models_fmig

        future_menmig[mig_men.columns[i]] = models_mmig[2][4].forecast(16) / 1000
        
        future_womenmig[mig_men.columns[i]] = models_fmig[2][4].forecast(16) / 1000

    
    future_men = pd.DataFrame(
        future_menmig,
        index = new_years
    )
    future_men.index.name = 'Годы'
    future_men.index = future_men.index.astype(int)

    future_women = pd.DataFrame(
        future_womenmig,
        index = new_years
    )
    future_women.index.name = 'Годы'
    future_women.index = future_women.index.astype(int)
    
    future_brcf = dict()
    for i in range(len(births_coeff.columns)):
        models_birthscf = st.session_state.get(f'auto.{births_coeff.columns[i]}', {})
        if models_birthscf == {}:
            models_birthscf = auto_forecast_exp_smooth(births_coeff, births_coeff.columns[i], False)
            models_key = f'auto.коэффициента рождаемости.{births_coeff.columns[i]}'
            st.session_state[models_key] = models_birthscf
        
        max_mean = max([models_birthscf[2][4].forecast(15).mean(),
                        models_birthscf[1][4].forecast(15).mean(),
                        models_birthscf[0][4].forecast(15).mean()])
        
        ind = 0
        for j in range(3):
            if models_birthscf[j][4].forecast(15).mean() == max_mean:
                ind = j
                break
            
        future_brcf[births_coeff.columns[i]] = models_birthscf[ind][4].forecast(15)
    future_births_coeff = pd.DataFrame(future_brcf)
    future_births_coeff['Годы'] = new_years
    future_births_coeff.set_index('Годы', inplace = True)
    future_births_coeff.index = future_births_coeff.index.astype(int)

    sums_for5y_b, sums_for5y_mm, sums_for5y_mf = {}, {}, {}
    for start_y in range(2023, 2034, 5):
        end_y = start_y + 5
        sums_for5y_b[end_y] = future_births_coeff.loc[start_y+1:end_y].mean(axis=0)
        sums_for5y_mm[end_y] = future_men.loc[start_y+1:end_y].sum(axis=0)
        sums_for5y_mf[end_y] = future_women.loc[start_y+1:end_y].sum(axis=0)
        
    df_birthscoeff = pd.DataFrame(sums_for5y_b).T
    df_menmig = pd.DataFrame(sums_for5y_mm).T
    df_womenmig = pd.DataFrame(sums_for5y_mf).T
    
    return df_menmig, df_womenmig, df_birthscoeff

def load_step5():
    return (
        pd.read_csv("content/migr_men_step5.csv", index_col=0),
        pd.read_csv("content/migr_women_step5.csv", index_col=0),
        pd.read_csv("content/births_coeff_step5.csv", index_col=0),
    )
        
migr_men_step5, migr_women_step5, births_coeff_step5 = load_step5()


men_prop = 0.514569
wom_prop = 0.485431

df_all_forc = {2023 : pd.DataFrame({
            'Численность, мужчин': cohort_df_2023['Численность, мужчин'],
            'Численность, женщин': cohort_df_2023['Численность, женщин']}),
               2028: None, 2033: None, 2038: None}
#____________________________Передвижка_______________________________#

def get_births(year, prev_coh):
    global births_coeff_step5, men_prop, wom_prop   
        
    births = []
    for age in births_coeff_step5.columns[:-2]:
        births.append(prev_coh.loc[age,'Численность, женщин'] * births_coeff_step5.loc[year, age] / 1000)
    
    summ = sum(births)*5
    
    return summ * men_prop, summ * wom_prop


def move_cohorts(year):
    global df_all_forc, cohort_df_2023, migr_men_step5, migr_women_step5

    if df_all_forc[year] is not None:
        df_new = df_all_forc[year]

    
    else:
        df_prev = df_all_forc[year-5]
        new_male, new_female = [0]* len(cohort_df_2023.index), [0]* len(cohort_df_2023.index)
        
        for i in range(1, len(cohort_df_2023.index)-1):
            new_male[i] = df_prev.iloc[i-1]['Численность, мужчин'] * cohort_df_2023.iloc[i-1]['Коэффициент дожития, мужчин'] + migr_men_step5.loc[year, cohort_df_2023.index[i]] 
            new_female[i] = df_prev.iloc[i-1]['Численность, женщин'] * cohort_df_2023.iloc[i-1]['Коэффициент дожития, женщин'] + migr_women_step5.loc[year, cohort_df_2023.index[i]]  


        new_male[0], new_female[0] = get_births(year, df_prev)
        new_male[-1] = df_prev.iloc[-2]['Численность, мужчин'] * cohort_df_2023.iloc[-2]['Коэффициент дожития, мужчин'] + migr_men_step5.loc[year, cohort_df_2023.index[-2]] + df_prev.iloc[-1]['Численность, мужчин'] * cohort_df_2023.iloc[-1]['Коэффициент дожития, мужчин']
        new_female[-1] = df_prev.iloc[-2]['Численность, женщин'] * cohort_df_2023.iloc[-2]['Коэффициент дожития, женщин'] + migr_women_step5.loc[year, cohort_df_2023.index[-2]] + df_prev.iloc[-1]['Численность, женщин'] * cohort_df_2023.iloc[-1]['Коэффициент дожития, женщин'] 
        
        df_new = pd.DataFrame({
            'Возрастные группы': cohort_df_2023.index,
            'Численность, мужчин': new_male,
            'Численность, женщин': new_female})
        df_new.set_index('Возрастные группы', inplace = True)

        df_all_forc[year] = df_new
        

    df_men = pd.DataFrame({
            'Возрастные группы': cohort_df_2023.index,
            'Пол': 'Мужчины',
            'Численность': df_all_forc[year]['Численность, мужчин'] * (-1)})
    df_women = pd.DataFrame({
            'Возрастные группы': cohort_df_2023.index,
            'Пол': 'Женщины',
            'Численность': df_all_forc[year]['Численность, женщин']})
    
    df_for_plot = pd.concat([df_men, df_women])
    
    max_val = max(max(df_all_forc[year]['Численность, мужчин']), max(df_all_forc[year]['Численность, женщин']))
    chart = (
        alt.Chart(df_for_plot)
        .mark_bar()
        .encode(
            y=alt.Y(
                'Возрастные группы:N',
                sort=list(df_for_plot.index.astype(str)),
                title='Возрастные группы'
            ),
            x=alt.X(
                'Численность:Q',
                scale=alt.Scale(domain=[-max_val * 1.1, max_val * 1.1]),
                title='Численность'
            ),
            color=alt.Color(
                'Пол:N',
                scale=alt.Scale(
                    domain=['Мужчины', 'Женщины']
                ),
                legend=alt.Legend(title='Пол')
            ),
            tooltip=[
                alt.Tooltip('Пол:N'),
                alt.Tooltip('Возрастные группы:N'),
                alt.Tooltip('Численность:Q', format=',.0f')
            ]
        )
        .properties(
            width='container',
            height=600,
            title=f'Половозрастная пирамида, {year} год'
        )
    )

    st.write(f"Год: {year}, Сумма мужчин: {round(sum(df_new['Численность, мужчин']), 2)}, Сумма женщин: {round(sum(df_new['Численность, женщин']), 2)}")
    
    
    return chart, df_new


#____________________________Интерфейс_______________________________#


params_menu = st.tabs(['2023', '2028', '2033', '2038'])
years = [2023, 2028, 2033, 2038]

for tab, year in zip(params_menu, years):
    with tab:
        chrt2023, table2023 = move_cohorts(year)
        st.altair_chart(chrt2023)
        st.write(table2023)
    

