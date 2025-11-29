
#________________________________Прогнозирование_демографической_ситуации_в_РФ______________________________________________#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
import altair as alt
#_______________________________Получение_данных_и_настройка_параметров_прогноза____________________________________________#

@st.cache_data
def load_data():
    bd_df = pd.read_csv('content/births_deaths.csv', delimiter=';')
    bd_df.set_index('Годы', inplace=True)
    
    m_df = pd.read_csv('content/migration/overall.csv', delimiter=';')
    m_df.set_index('Годы', inplace=True)
    
    p_df = pd.read_csv('content/population/population1897_2025.csv', sep=';')
    p_df.set_index('Годы', inplace=True)
    
    md_df = pd.read_csv('content/marryiages/marryiages_divorces.csv', sep=';')
    md_df.set_index('Годы', inplace=True)

    return bd_df, m_df, p_df, md_df
    

births_deaths_df, migration_df, population_df, marriages_divorces_df = load_data()


def display_sidebar(object_name = '', tab_key = '', params_vis = None, trend_status = None):
    t_ratio, f_len, tr, seas, s_period = None, None, None, None, None

    with st.sidebar.expander(f'**Настройка параметров прогнозной модели {object_name}:**'): 
        if params_vis:
            f_len = st.slider(
                "Горизонт прогноза (лет)",
                1, 20, 5,
                key = f'length_{tab_key}')

            t_ratio = st.slider(
                "Доля тестовой выборки",
                0.1, 0.4, 0.2,
                step=0.05,
                key = f'test_{tab_key}'
            )

            if trend_status:
                tr = st.radio(
                    "Тренд: ",
                    options = [("Аддитивный", "add"), ("Мультипликативный", 'mul')],
                    format_func= (lambda x: x[0]),
                    key = f'trend_{tab_key}'
                )

            seas = st.radio(
                "Сезонность: ",
                [('Нет', None), ("Аддитивный", "add"), ("Мультипликативный", 'mul')],
                format_func= (lambda x: x[0]),
                key = f'seasonal_{tab_key}'
            )

            s_period = None
            if seas[1]:
                s_period = st.slider(
                    "Период сезонности",
                    2, 11, 2,
                    key = f'period_{tab_key}'
                )
            
    return t_ratio, f_len, tr[1] if tr else tr, seas[1] if seas else seas, s_period



def define_error_result(rmse, mape):
    message = f'На основе выбранных параметров была сформирована прогнозная модель с ошибкой:\n **RMSE: {rmse:.2f}**,\n**MAPE: {mape:.2f}%**.'
    if mape < 10:
        st.success(f'**Высокое** качество прогноза.\n '+message)
    elif mape < 15:
        st.warning(f'**Среднее** качество прогноза. '+message)
    else:
        st.error(f'**Низкое** качество прогноза.\n '+message)
        


#_______________________________Реализация_математической_части_прогноза____________________________________________________#


# Реализация метода Фостера-Стюарта
def IsTrend(series, p_level=0.95):
    m, l = 1, 1 
    D = 0
    for cur_ind in range(1, len(series)): 
        m, l = 1, 1
        for prev_ind in range(cur_ind):
            if series[cur_ind] <= series[prev_ind]:
                m *= 0
            elif series[cur_ind] >= series[prev_ind]:
                l *= 0
        D += m-l
    print('D: ', D)
    D = abs(D)
    
    # Сравнение с табличными данными
    delta = (2*math.log(len(series))-0.8456)**0.5
    t_table = scipy.stats.t.ppf((1+p_level)/2 , len(series)-1)
    print(D/delta, t_table)
    
    return True if D/delta > t_table else None


#Скользящее среднее
def smoothing_one_window(segment, p):
    return sum(segment)/(2*p+1)

def smoothing(series, window=3):
    return series.rolling(window=window, center=True).mean()

#Экспоненциальное сглаживание (Хольта и Хольта-Винтерса) 
def forecast_exp_smooth(train, test, column_name, forecast_len = 5, trend = None, season = None, period = None):
    
    model = ExponentialSmoothing(train[column_name], trend=trend, seasonal=season, seasonal_periods=period).fit(optimized=True)
    forecast = model.forecast(len(test))

    rmse = np.sqrt(mean_squared_error(test[column_name], forecast))
    mape = mean_absolute_percentage_error(test[column_name], forecast) * 100

    prediction = model.forecast(forecast_len)
    
    return rmse, mape, prediction, model

# Подбор периода с минимальной ошибкой
def find__best_season_period(test, train, column_name, trend, season):
    model = ExponentialSmoothing(train[column_name], trend=trend, seasonal=season, seasonal_periods=3).fit(optimized=True)
    opt_period = 0
    min_rmse = float('inf')
    for i in range (2, len(train)//2+1):
        model = ExponentialSmoothing(train[column_name], trend=trend, seasonal=season, seasonal_periods=i).fit()
        forecast = model.forecast(len(test))
        rmse = np.sqrt(mean_squared_error(list(test[column_name]), forecast))
        
        if (rmse < min_rmse):
            opt_period = i
            min_rmse = rmse
    return opt_period 

# Основная функция
def split_data(data, test_ratio = 0.1):
    test_size = int(len(data)*test_ratio)
    train = data[:-test_size]
    test = data[-test_size:]
    return train, test 


def plot_chart(data, pred_column, demo_param, test_ratio, forecast_len, trend = None, season = None, period = None): 
    train, test = split_data(data, test_ratio)
    
    rmse, mape, future, model = forecast_exp_smooth(train, test, pred_column, forecast_len, trend = trend, season = season, period = period)
    define_error_result(rmse, mape)

    new_years = [data.index[-1] + i for i in range(1, forecast_len+1)]
    
    
    forecast_df = pd.DataFrame({
        'Годы': new_years,
        'Прогноз ' + pred_column.lower() : future.values
    })
    
    history_df = data.reset_index()[['Годы', pred_column]]

    merged_df = pd.concat([history_df, forecast_df])
    merged_df.set_index('Годы', inplace=True)
    merged_df.loc[data.index[-1], 'Прогноз ' + pred_column.lower()] = merged_df.loc[data.index[-1], pred_column]
    merged_df.index = merged_df.index.astype(str)

    
    st.subheader(f'Прогноз {pred_column} на период {new_years[1]} - {new_years[-1]}')
    st.line_chart(merged_df, x_label = 'Годы', y_label = 'Человек')

    
    with st.expander("Параметры модели"):
        for k, v in model.params.items():
            st.write(f"**{k}**: {v}")
    return merged_df


#_______________________________Интерфейсная_часть_приложения_______________________________________________________________#



#____Выбор_объекта_прогнозирования_____

selectbox_type = st.sidebar.selectbox(
    "Прогнозирвоание экспоненциальным сглаживанием: ",
    ("Демографических процессов",
     "Демографических показателей")
)


if selectbox_type == "Демографических процессов":
    st.header('Прогнозирвоание демографических процессов методом экспоненциального сглаживания')

    selectbox_proc = st.sidebar.selectbox(
        "Демографический процесс: ",
        ("Рождаемость и смертность",
         "Миграция",
         "Браки и разводы")
    )


else:
    st.header('Прогнозирование демографических показателей методом экспоненциального сглаживания')
    selectbox_proc = st.sidebar.selectbox(
        "Демографический показатель: ",
        ("Численнность населения",
         "Коэффициент рождаемости",
         "Коэффициент смертности",
         "Ожидаемая продолжительность жизни")
    )



#____Построение_прогнозов_на основе_выбора____

        
if selectbox_proc == "Рождаемость и смертность":
    
    st.subheader('Прогноз динамики рождаемости и смертности')
    params_menu = ['Анализ текущих данных', 'Прогноз рождаемости', 'Прогноз смертности', 'Прогноз естественного прироста']
    analisys_tab, birth_tab, death_tab, diff_tab = st.tabs(params_menu)
    
    
    births_with_pred, deaths_with_pred = None, None
    column_names = ['Рождений', 'Смертей', 'Естественный прирост']
    

    with analisys_tab:
        st.subheader('Исторические данные о естественном движении населения')
        df = births_deaths_df.copy()
        df[column_names[1]] *= -1
        st.dataframe(births_deaths_df)

        st.subheader('Диаграмма естественного движения населения')
        
        st.line_chart(df[column_names])
        
    with birth_tab:
        st.subheader('Прогноз динамики рождаемости')
        
        trend_birth = IsTrend(list(births_deaths_df[column_names[0]]))
        trend_ru = 'есть' if trend_birth else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('рождаемости', 'births', True, trend_birth)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        births_with_pred = plot_chart(births_deaths_df, column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        
    with death_tab:
        st.subheader('Прогноз динамики смертности')
        
        trend_death = IsTrend(list(births_deaths_df[column_names[1]]))
        
        trend_ru = 'есть' if trend_death else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('смертности', 'deaths', True, trend_death)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
        
        deaths_with_pred = plot_chart(births_deaths_df, column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        
    with diff_tab:
        st.subheader('Прогноз естественного движения')

        if not(births_with_pred.empty) and not(deaths_with_pred.empty):
            merged_b_d = pd.concat([births_with_pred, deaths_with_pred], axis = 1)
            merged_b_d[column_names[1]] *= -1
            merged_b_d["Прогноз смертей"] *= -1
            merged_b_d["Естественный прирост"] = merged_b_d[column_names[0]] + merged_b_d[column_names[1]]
            merged_b_d["Прогноз естественного прироста"] =  merged_b_d["Прогноз рождений"] + merged_b_d["Прогноз смертей"]

            st.dataframe(merged_b_d[(~merged_b_d["Прогноз естественного прироста"].isna())
                                    & (merged_b_d[column_names[0]].isna())][[col for col in merged_b_d.columns if 'Прогноз' in col]])
            
            st.subheader('Прогноз естественного прироста населения')
            st.line_chart(merged_b_d)
        else:
            st.error('Сначала получите прогнозы рождаемости и смертности')
   

elif selectbox_proc == "Миграция":
    st.subheader('Прогноз миграционного движения')
    
    params_menu = ['Анализ текущих данных', 'Прогноз числа прибывших', 'Прогноз числа выбывших', 'Прогноз миграционного прироста']
    analisys_tab, in_tab, out_tab, diff_tab = st.tabs(params_menu)
    
    
    in_with_pred, out_with_pred = None, None
    column_names = migration_df.columns[:3]
    

    with analisys_tab:
        st.subheader('Исторические данные о миграционном движении населения')
        df = migration_df.copy()
        df[column_names[1]] *= -1
        st.dataframe(migration_df)

        st.subheader('Диаграмма миграционного движения населения')
        st.line_chart(df[column_names])

        
        df[migration_df.columns[4]] *= -1
        st.subheader('Диаграмма международного обмена населения')
        st.line_chart(df[migration_df.columns[3:]])

        
        st.subheader('Соотношение общего миграционного прироста населения к международному')
        st.bar_chart(df[migration_df.columns[2::3]], stack = False)
        
        
    with in_tab:
        st.subheader('Прогноз иммиграции населения РФ')

        trend_imig = IsTrend(list(migration_df[column_names[0]]))
        trend_ru = 'есть' if trend_imig else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('иммиграции', 'in', True, trend_imig)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        in_with_pred = plot_chart(migration_df, column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        

    with out_tab:
        st.subheader('Прогноз эмиграции население в РФ')

        trend_mig = IsTrend(list(migration_df[column_names[1]]))
        trend_ru = 'есть' if trend_mig else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('эмиграции', 'out', True, trend_mig)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        out_with_pred = plot_chart(migration_df, column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


    with diff_tab:
        st.subheader('Прогноз миграционного движения')

        if not(in_with_pred.empty) and not(out_with_pred.empty):
            merged = pd.concat([in_with_pred, out_with_pred], axis = 1)
            merged[column_names[1]] *= -1
            merged["Прогноз "+column_names[1].lower()] *= -1
            merged["Миграционный прирост"] = merged[column_names[0]] + merged[column_names[1]]
            merged["Прогноз миграционного прироста"] =  merged["Прогноз "+column_names[0].lower()] + merged["Прогноз "+column_names[1].lower()]

            st.dataframe(merged[(~merged["Прогноз миграционного прироста"].isna())
                                    & (merged[column_names[0]].isna())][[col for col in merged.columns if 'Прогноз' in col]])
            
            st.subheader('Прогноз миграционного прироста населения')
            st.line_chart(merged)
        else:
            st.error('Сначала получите прогнозы иммиграции и эммиграции')
   

        
        

elif selectbox_proc == "Браки и разводы":
    st.subheader('Прогоноз динамики брачности и разводимости')

    params_menu = ['Анализ текущих данных', 'Прогноз брачности', 'Прогноз разводимости', 'Прогноз их соотношения']
    analisys_tab, mar_tab, div_tab, diff_tab = st.tabs(params_menu)
    
    
    mar_with_pred, div_with_pred = None, None
    marriages_divorces_df['Соотношение браков и разводов'] = marriages_divorces_df[marriages_divorces_df.columns[0]]-marriages_divorces_df[marriages_divorces_df.columns[2]]
    marriages_divorces_df['Соотношение браков и разводов на 1000 чел.'] = marriages_divorces_df[marriages_divorces_df.columns[1]]-marriages_divorces_df[marriages_divorces_df.columns[3]]

    
    column_names = marriages_divorces_df.columns[::2]
    column_names_1000 = marriages_divorces_df.columns[1::2]

    with analisys_tab:
        st.subheader('Исторические данные о заключении и расторжении браков в РФ')
        df = marriages_divorces_df.copy()
        df[column_names[1]] *= -1
        st.dataframe(marriages_divorces_df)

        st.subheader('Диаграмма брачности и разводимости населения')
        st.line_chart(df[column_names])

        df[column_names_1000[1]] *= -1
        st.subheader('Диаграмма брачности и разводимости населения на 1000 человек')
        st.line_chart(df[column_names_1000])

    with mar_tab:
        st.subheader('Прогноз брачности населения РФ')

        trend_mar = IsTrend(list(marriages_divorces_df[column_names[0]]))
        trend_ru = 'есть' if trend_mar else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('брачности', 'mar', True, trend_mar)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        mar_with_pred = plot_chart(marriages_divorces_df, column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        

    with div_tab:
        st.subheader('Прогноз разводимости население в РФ')

        trend_div = IsTrend(list(marriages_divorces_df[column_names[1]]))
        trend_ru = 'есть' if trend_div else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('разводимости', 'div', True, trend_div)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        div_with_pred = plot_chart(marriages_divorces_df, column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


    with diff_tab:
        st.subheader('Прогноз соотношения брачности и разводимости')

        if not(mar_with_pred.empty) and not(div_with_pred.empty):
            merged = pd.concat([mar_with_pred, div_with_pred], axis = 1)
            merged[column_names[1]] *= -1
            merged["Прогноз "+column_names[1].lower()] *= -1
            merged["Соотношение брачности и разводимости"] = merged[column_names[0]] + merged[column_names[1]]
            merged["Прогноз соотношения брачности и разводимости"] =  merged["Прогноз "+column_names[0].lower()] + merged["Прогноз "+column_names[1].lower()]

            st.dataframe(merged[(~merged["Прогноз соотношения брачности и разводимости"].isna())
                                    & (merged[column_names[0]].isna())][[col for col in merged.columns if 'Прогноз' in col]])
            
            st.subheader('Прогноз соотношения брачности и разводимости')
            st.line_chart(merged)
        else:
            st.error('Сначала получите прогнозы брачности и разводимости')
   
   
        
        
elif selectbox_proc == "Численнность населения":
    st.subheader('Прогноз динамики численнности населения')
    st.dataframe(population_df)

elif selectbox_proc == "Коэффициент рождаемости":
    st.subheader('Прогноз динамики показателя коэффициента рождаемости')

elif selectbox_proc == "Коэффициент смертности":
    st.subheader('Прогноз динамики показателя коэффициента смертности')

elif selectbox_proc == "Ожидаемая продолжительность жизни":
    st.subheader('Прогоноз динамики брачности и разводимости')
