

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
import os


#_______________________________Получение_данных_и_настройка_параметров_прогноза____________________________________________#

@st.cache_data
def load_data():
    bd_df = pd.read_csv('content/births_deaths.csv', sep=';')
    bd_df.set_index('Годы', inplace=True)
    bd_df.index = bd_df.index.astype(str)
    
    m_df = pd.read_csv('content/migration/overall.csv', sep=';')
    m_df.set_index('Годы', inplace=True)
    m_df.index = m_df.index.astype(str)

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
    
    md_df = pd.read_csv('content/marryiages/marryiages_divorces.csv', sep=';')
    md_df.set_index('Годы', inplace=True)
    md_df.index = md_df.index.astype(str)
    
    p_df = pd.read_csv('content/population/population1897_2025.csv', sep=';')
    p_df.set_index('Годы', inplace=True)
    p_df.index = p_df.index.astype(str)

    b_c_df = pd.read_csv('content/coeff/births_coeff.csv', sep=';')
    b_c_df.set_index('Годы', inplace=True)
    b_c_df.index = b_c_df.index.astype(str)

    d_c_df_m = pd.read_csv('content/coeff/deaths_coeff/male_deaths.csv', sep=';')
    d_c_df_f = pd.read_csv('content/coeff/deaths_coeff/fem_deaths.csv', sep=';')
    d_c_df_m.set_index('Годы', inplace=True)
    d_c_df_f.set_index('Годы', inplace=True)
    d_c_df_m.index = d_c_df_m.index.astype(str)
    d_c_df_f.index = d_c_df_f.index.astype(str)

    le_df = pd.read_csv('content/coeff/life_expectancy.csv', sep=';')
    le_df.set_index('Годы', inplace=True)
    le_df.index = le_df.index.astype(str)

    return bd_df, m_df, mm_df, mf_df, md_df, p_df, b_c_df, d_c_df_m, d_c_df_f, le_df

#@st.cache_data
#def load_districts_geo_data():
#    federal_dist = 'content/Federal_districts'

#    geo_dist = []
#    for file_name in os.listdir(federal_dist):
#        if '.geojson' in file_name:
#            geo_dist.append(gpd.read_file(os.path.join(federal_dist, file_name)))

#    return geo_dist

    

births_deaths_df, migration_df, mig_men_df, mig_women_df, marriages_divorces_df, population_df, births_coeff_df, deaths_m_coeff_df, deaths_f_coeff_df, life_expec = load_data()
#geo_coord = load_districts_geo_data()
#st.write(geo_coord)

def display_sidebar(object_name = '', tab_key = '', size_df = 0, params_vis = None, trend_status = None):
    global select_params
    t_ratio, f_len, tr, seas, s_period = None, None, None, None, None

    if size_df < 5:
        st.sidebar.warning('Недостаточно данных для настройки модели')
        return t_ratio, f_len, tr, seas, s_period

    min_test_ratio = 0.1 if size_df >= 10 else round(1/size_df, 2)

    min_train_size = 6
    max_test_ratio = min(max(0.1, round(1 - min_train_size / size_df, 2)), 0.4)
    
    if max_test_ratio < min_test_ratio:
        min_test_ratio, max_test_ratio = max_test_ratio, min_test_ratio
        
    if select_params:
        with st.sidebar.expander(f'**Настройка параметров прогнозной модели {object_name}:**'): 
            if params_vis:
                f_len = st.slider(
                    "Горизонт прогноза (лет)",
                    1, 20, 5,
                    key = f'length_{tab_key}')

                if max_test_ratio == min_test_ratio:
                    t_ratio = max_test_ratio
                    st.info(f"Доля тестовой выборки: {t_ratio}")
                else:
                    t_ratio = st.slider(
                        "Доля тестовой выборки",
                        min_value = float(min_test_ratio),
                        max_value = float(max_test_ratio),
                        value = float(min(0.2, max_test_ratio)),
                        step=0.05,
                        key = f'test_{tab_key}'
                    )
                train_size = (1-t_ratio)*size_df

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

                max_period = max(2, train_size//2)
                s_period = None
                if seas[1]:
                    s_period = st.slider(
                        "Период сезонности",
                        min_value = 2,
                        max_value = min(11, max_period),
                        value = 2,
                        key = f'period_{tab_key}'
                    )
    else:
        f_len = st.slider(
            "Горизонт прогноза (лет)",
            1, 20, 5,
            key = f'length_{tab_key}')

            
    return t_ratio, f_len, tr[1] if tr else tr, seas[1] if seas else seas, s_period



def define_error_result(rmse, mape):
    message = f'''На основе выбранных параметров была сформирована прогнозная модель с ошибкой:\n   
**RMSE**: {rmse:.2f},\n**MAPE**: {mape:.2f}%.'''
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

@st.cache_data(show_spinner = True)
def auto_forecast_exp_smooth(data, column_name, info_print = True):

    min_train_size = 6
    min_tr, max_tr, step = 0.1 if len(data) >= 10 else round(1/len(data), 2), min(max(0.1, round(1 - min_train_size / len(data), 2)), 0.4), 0.05
    if min_tr > max_tr:
        min_tr, max_tr = max_tr, min_tr
    
    if len(data) < 6:
        train, test = split_data(data, max_tr)
        model = ExponentialSmoothing(train[column_name]).fit(optimized=True)
        forecast = model.forecast(len(test))

        rmse = np.sqrt(mean_squared_error(test[column_name], forecast))
        mape = mean_absolute_percentage_error(test[column_name], forecast) * 100
        if info_print:
            st.info(f"""Данных недостаточно, поэтому был построен один прогноз при помощи простого экспоненциального сглаживания""")
        
        return [(0.0, None, None, None, model, rmse, mape)]
    
    test_ratio_range = np.arange(min_tr, max_tr+step, step).round(2).tolist()

    var_tr_list = ['add', 'mul'] if IsTrend(data[column_name]) else [None]
    
    seas_l = [None, 'add'] + (['mul'] if (data[column_name] > 0).all() else [])
    
    top3 = [float('inf')]*3
    models = []
    for test_rat in test_ratio_range:
        train, test = split_data(data, test_rat)
        if len(train) < min_train_size or len(test) == 0:
            continue
        train_size = len(train)
        for tr in var_tr_list:
            for seas in seas_l:
                period = None
                
                if seas:
                    max_period = max(2, train_size//2)
                    if min(max_period, train_size // 2) < 2:
                        continue
                    for period in range(2, max_period+1):
                        try:
                            model = ExponentialSmoothing(train[column_name], trend=tr, seasonal=seas, seasonal_periods=period).fit(optimized=True)
                            forecast = model.forecast(len(test))

                            rmse = np.sqrt(mean_squared_error(test[column_name], forecast))
                            mape = mean_absolute_percentage_error(test[column_name], forecast) * 100
                            
                            models.append((test_rat, tr, seas, period, model, mape, rmse))
                        except Exception:
                            continue
                else:
                    model = ExponentialSmoothing(train[column_name], trend=tr, seasonal=None).fit(optimized=True)
                    forecast = model.forecast(len(test))

                    rmse = np.sqrt(mean_squared_error(test[column_name], forecast))
                    mape = mean_absolute_percentage_error(test[column_name], forecast) * 100
                
                    models.append((test_rat, tr, seas, period, model, mape, rmse))

    top3 = sorted(models, key=lambda x: x[-1])
    
    if len(top3) > 3:
        top3 = models[:3]
    
    if info_print and top3:
        rmses = "  \n".join([
            f"{m + 1}. RMSE = {top3[m][-1]:.2f}; MAPE = {top3[m][-2]:.2f}%"
            for m in range(len(top3))
        ])

        st.info(
            "Были подобраны модели с наименьшими ошибками:  \n"
            f"{rmses}  \n\n"
            "RMSE — среднеквадратичная ошибка, показывающая средний размер отклонения прогноза "
            "от фактических значений.  \n"
            "MAPE — средняя абсолютная процентная ошибка, показывающая среднюю ошибку прогноза в процентах."
        )

                        
    return top3
                                            
    
    
#Экспоненциальное сглаживание (Хольта и Хольта-Винтерса)
@st.cache_data(show_spinner = True)
def forecast_exp_smooth(train, test, column_name, trend = None, season = None, period = None):
    
    model = ExponentialSmoothing(train[column_name], trend=trend, seasonal=season, seasonal_periods=period).fit(optimized=True)
    forecast = model.forecast(len(test))

    rmse = np.sqrt(mean_squared_error(test[column_name], forecast))
    mape = mean_absolute_percentage_error(test[column_name], forecast) * 100
    
    return rmse, mape, model


# Основная функция
def split_data(data, test_ratio = 0.1):
    
    test_size = max(1, int(len(data)*test_ratio))
    if len(data) - test_size < 2:
        test_size = len(data) - 2
    
    train = data[:-test_size]
    test = data[-test_size:]
    return train, test 


def show_model_params(model):
    params = model.params

    param_descriptions = {
        "smoothing_level": (
            "Коэффициент сглаживания уровня ряда (α)",
            "Определяет, насколько сильно модель учитывает последние наблюдения."
            "Чем значение ближе к 1, тем быстрее прогноз реагирует на новые изменения."
        ),
        "smoothing_trend": (
            "Коэффициент сглаживания тренда (β)",
            "Задает скорость обновления оценки тренда (роста или снижения показателя)."
        ),
        "smoothing_seasonal": (
            "Коэффициент сглаживания сезонности (γ)",
            "Показывает, насколько сильно учитываются сезонные колебания в данных."
        ),
        "damping_trend": (
            "Коэффициент затухания тренда (φ)",
            "Позволяет постепенно ослаблять влияние тренда в долгосрочном прогнозе, предотвращая чрезмерный рост или снижение показателя."
            "Значение ближе к 1 означает более устойчивый тренд."
        ),
        "initial_level": (
            "Начальная оценка уровня временного ряда",
            "Базовое значение временного ряда, с которого начинается построение модели."
        ),
        "initial_trend": (
            "Начальное значение тренда",
            "Начальная оценка первоначальной скорости изменения показателя во времени."
        ),
    }

    important_params = [
        "smoothing_level",
        "smoothing_trend",
        "smoothing_seasonal",
        "damping_trend",
        "initial_level",
        "initial_trend",
    ]

    rows = []

    for key in important_params:
        value = params.get(key)

        if value is not None and pd.notna(value):
            title, description = param_descriptions[key]
            rows.append({
                "Параметр": title,
                "Значение": round(value, 2) if isinstance(value, float) else value,
                "Пояснение": description
            })

    if rows:
        for row in rows:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin-bottom: 10px;
                    background-color: #FAFAFA;
                ">
                    <b>{row["Параметр"]}</b><br>
                    <span style="color: #555;">Значение:</span>
                    <code>{row["Значение"]}</code><br><br>
                    <span>{row["Пояснение"]}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Для выбранной модели дополнительные параметры сглаживания не были рассчитаны.")


def plot_chart(data, name, pred_column, demo_param, test_ratio, forecast_len, trend = None, season = None, period = None):
    global select_params

    new_years = [str(int(data.index[-1]) + i) for i in range(forecast_len+1)]
    history_df = data.reset_index()[['Годы', pred_column]]
    history_df.index = history_df.index.astype(str)
    history_df = history_df.set_index('Годы')
    
    if select_params:
        model_key = f'{name}.{pred_column}.{test_ratio}.{trend}.{season}.{period}'
        if model_key not in st.session_state:
            train, test = split_data(data, test_ratio)
            rmse, mape, model = forecast_exp_smooth(train, test, pred_column, trend = trend, season = season, period = period)
            st.session_state[model_key] = {'model': model, 'rmse': rmse, 'mape': mape}

        state_model = st.session_state.get(model_key, {})
        model, rmse, mape = state_model.get('model'), state_model.get('rmse'), state_model.get('mape')
        
        future = model.forecast(forecast_len)

        define_error_result(rmse, mape)
        future = pd.concat([data[data.index == data.index[-1]][pred_column], future]).reset_index(drop = True)
        
        forecast_df = pd.DataFrame({
            'Годы': new_years,
            'Прогноз ' + name.lower() : future.values
        })
        forecast_df = forecast_df.set_index('Годы')
        
        merged_df = pd.concat([history_df, forecast_df], axis = 1)
    
        st.subheader(f'Прогноз {name} на период {new_years[1]} - {new_years[-1]}')
        st.line_chart(merged_df, x_label = 'Годы', y_label = 'Человек')

        with st.expander("Параметры модели"):
            show_model_params(model)
    else:
        models_key = f'auto.{name}.{pred_column}'
        if models_key not in st.session_state:
            best_models = auto_forecast_exp_smooth(data, pred_column)
            st.session_state[models_key] = best_models

        best_models = st.session_state.get(models_key, [])

        list_of_df = {'Годы': new_years}
        for i in range(len(best_models)):
            future = best_models[i][4].forecast(forecast_len)
            list_of_df[f'Прогноз {name.lower()} {i+1}'] = pd.concat([data[data.index == data.index[-1]][pred_column], future]).reset_index(drop = True)

        forecast_df = pd.DataFrame(list_of_df)
    
        forecast_df = forecast_df.set_index('Годы')
    
    
        merged_df = pd.concat([history_df, forecast_df], axis = 1)
        
        st.subheader(f'Прогноз {name} на период {new_years[1]} - {new_years[-1]}')
        st.line_chart(merged_df, x_label = 'Годы', y_label = 'Человек')

        with st.expander("Параметры модели"):
            for j in range(len(best_models)):
                st.write(f'Модель {j+1}:')
                show_model_params(best_models[j][4])
                
    return merged_df


def make_diff(table_1, table_2, hist_diff_col, forc_diff_name, param1_name, param2_name):
    merged_df = pd.concat([table_1, table_2, hist_diff_col], axis = 1)
    
    if select_params:
        merged_df[f"Прогноз {forc_diff_name}"] =  merged_df[f"Прогноз {param1_name}"] + merged_df[f"Прогноз {param2_name}"]
        
    else:
        for i in range(1,4):
            merged_df[f"Прогноз {forc_diff_name} {i}"] =  merged_df[f"Прогноз {param1_name} {i}"] + merged_df[f"Прогноз {param2_name} {i}"]

    
    return merged_df


#def plot_sum_chart():
    # births_with_pred = births_with_pred.reset_index()
    #            births_with_pred = births_with_pred.melt(
    #                id_vars = 'Годы',
    #                value_vars = births_with_pred.columns[1:],
    #                var_name = 'Тип',
    #                value_name = 'Человек'
    #            )
    #            births_with_pred = births_with_pred.dropna()
    #            st.dataframe(births_with_pred)
    #birth_bar = alt.Chart(births_with_pred).mark_bar().encode(
    #            x = 'Годы:N',
    #            xOffset = 'Тип',
    #            y = 'Человек:Q',
    #            color = 'Тип:N'
    #        )
            
    #st.altair_chart(birth_bar)
    
            
    

#_______________________________Интерфейсная_часть_приложения_______________________________________________________________#



#____Выбор_объекта_прогнозирования_____

selectbox_type = st.sidebar.selectbox(
    "Прогнозирование экспоненциальным сглаживанием: ",
    ("Демографических процессов",
     "Демографических показателей")
)


if selectbox_type == "Демографических процессов":
    st.header('Прогнозирование демографических процессов методом экспоненциального сглаживания')

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

st.sidebar.markdown("---")

#____Построение_прогнозов_на основе_выбора____
select_params = st.sidebar.checkbox("Подобрать параметры вручную", value=True, key='auto')
        
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
        
        st.line_chart(df[column_names], x_label = 'Годы', y_label = 'Человек')
        
    with birth_tab:
        st.subheader('Прогноз динамики рождаемости')
        
        trend_birth = IsTrend(list(births_deaths_df[column_names[0]]))
        trend_ru = 'есть' if trend_birth else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('рождаемости', 'births', len(births_deaths_df), True, trend_birth)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        births_with_pred = plot_chart(births_deaths_df, 'рождаемости', column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        
    with death_tab:
        st.subheader('Прогноз динамики смертности')
        
        trend_death = IsTrend(list(births_deaths_df[column_names[1]]))
        
        trend_ru = 'есть' if trend_death else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('смертности', 'deaths', len(births_deaths_df), True, trend_death)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
        
        deaths_with_pred = plot_chart(births_deaths_df, 'смертности', column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        
    with diff_tab:
        st.subheader('Прогноз естественного движения')

        if not(births_with_pred.empty) and not(deaths_with_pred.empty):
            merged_b_d = make_diff(births_with_pred, deaths_with_pred * (-1), df[column_names[2]], 'естественного прироста', 'рождаемости', 'смертности')
            
            st.line_chart(merged_b_d, x_label = 'Годы', y_label = 'Человек')
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

        chart_type_menu = ['Общий поток миграции', 'Миграция в рамках международного обмена', 'Миграция по полу и возрасту']
        gen_tab, abroad_tab, as_tab = st.tabs(chart_type_menu)

        with gen_tab:
            st.subheader('Диаграмма миграционного движения населения')
            st.line_chart(df[column_names])

        with abroad_tab:
            df[migration_df.columns[4]] *= -1
            st.subheader('Диаграмма международного обмена населения')
            st.line_chart(df[migration_df.columns[3:]], x_label = 'Годы', y_label = 'Человек')

            st.subheader('Соотношение общего миграционного прироста населения к международному')
            st.bar_chart(df[migration_df.columns[2::3]], x_label = 'Годы', y_label = 'Человек', stack = False)
        with as_tab:
            st.subheader('Показатели международной миграции мужчин разных возрастов')
            st.write(mig_men_df)
            st.line_chart(mig_men_df)
            st.subheader('Показатели международной миграции женщин разных возрастов')
            st.line_chart(mig_women_df)
            
        
        
    with in_tab:
        st.subheader('Прогноз иммиграции населения РФ')

        trend_imig = IsTrend(list(migration_df[column_names[0]]))
        trend_ru = 'есть' if trend_imig else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('иммиграции', 'in', len(migration_df), True, trend_imig)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        in_with_pred = plot_chart(migration_df, 'иммиграции', column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        

    with out_tab:
        st.subheader('Прогноз эмиграции население в РФ')

        trend_mig = IsTrend(list(migration_df[column_names[1]]))
        trend_ru = 'есть' if trend_mig else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('эмиграции', 'out', len(migration_df), True, trend_mig)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        out_with_pred = plot_chart(migration_df, 'эмиграции', column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


    with diff_tab:
        st.subheader('Прогноз миграционного движения')

        chart_forc_menu = ['Общий поток миграции', 'Прогноз по полу и возрасту']
        gen_forc_tab, as_forc_tab = st.tabs(chart_forc_menu)

        with gen_forc_tab:
            if not(in_with_pred.empty) and not(out_with_pred.empty):
                
                merged_i_e = make_diff(in_with_pred, out_with_pred * (-1), df[column_names[2]], 'миграционного прироста', 'иммиграции', 'эмиграции')

                st.line_chart(merged_i_e, x_label = 'Годы', y_label = 'Человек')
            else:
                st.error('Сначала получите прогнозы иммиграции и эммиграции')
        with as_forc_tab:
            trend_list_m = [IsTrend(list(mig_men_df[i])) for i in mig_men_df.columns]
            trend_list_f = [IsTrend(list(mig_women_df[i])) for i in mig_women_df.columns]
            forecast_list_m, forecast_list_f = {}, {}

            test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('миграционного движения для мужчин и женщин', 'mig_mf', len(mig_men_df), True, True)
            
            tabs = st.tabs([f"{col}" for col in mig_women_df.columns])

            for i in range(len(mig_men_df.columns)):
                with tabs[i]:
                    st.write(f'Прогноз для {mig_women_df.columns[i]}')
                    trend_ru = 'есть' if trend_list_m[i] else 'отсутствует'                
                    st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
                    
                    forecast_list_m[i] = plot_chart(mig_men_df, 'миграционного движения мужчин', mig_men_df.columns[i], selectbox_proc, test_ratio,
                                                    forecast_len, trend_type if trend_list_m[i] else None, seasonal_type, seasonal_periods)


                    trend_ru = 'есть' if trend_list_f[i] else 'отсутствует'                
                    st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
                    
                    forecast_list_f[i] = plot_chart(mig_women_df, 'миграционного движения женщин', mig_women_df.columns[i], selectbox_proc, test_ratio,
                                                    forecast_len, trend_type if trend_list_f[i] else None, seasonal_type, seasonal_periods)

    

        
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

        chart_type_menu = ['Общая', 'На 1000 человек']

        gen_tab, thous_tab = st.tabs(chart_type_menu)

        with gen_tab:
            st.subheader('Диаграмма брачности и разводимости населения')
            st.line_chart(df[column_names], x_label = 'Годы', y_label = 'Человек')

        with thous_tab:
            df[column_names_1000[1]] *= -1
            st.subheader('Диаграмма брачности и разводимости населения на 1000 человек')
            st.line_chart(df[column_names_1000], x_label = 'Годы', y_label = 'Человек')

    with mar_tab:
        st.subheader('Прогноз брачности населения РФ')

        trend_mar = IsTrend(list(marriages_divorces_df[column_names[0]]))
        trend_ru = 'есть' if trend_mar else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('брачности', 'mar', len(marriages_divorces_df), True, trend_mar)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        mar_with_pred = plot_chart(marriages_divorces_df, 'брачности', column_names[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)
        

    with div_tab:
        st.subheader('Прогноз разводимости население в РФ')

        trend_div = IsTrend(list(marriages_divorces_df[column_names[1]]))
        trend_ru = 'есть' if trend_div else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('разводимости', 'div', len(marriages_divorces_df), True, trend_div)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        div_with_pred = plot_chart(marriages_divorces_df, 'разводимости', column_names[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


    with diff_tab:
        st.subheader('Прогноз соотношения брачности и разводимости')

        if not(mar_with_pred.empty) and not(div_with_pred.empty):
            merged_m_d = make_diff(mar_with_pred, div_with_pred * (-1), df[column_names[2]], 'соотношения брачности и разводимости', 'брачности', 'разводимости')

            st.line_chart(merged_m_d, x_label = 'Годы', y_label = 'Количество')
            
        else:
            st.error('Сначала получите прогнозы брачности и разводимости')
   
   
        
        
elif selectbox_proc == "Численнность населения":
    st.subheader('Прогноз динамики численнности населения')

    params_menu = ['Анализ текущих данных', 'Прогноз численности населения']
    analisys_tab, forecast_tab = st.tabs(params_menu)
    
    with analisys_tab:
        st.subheader('Исторические данные о численности населения РФ')
        st.dataframe(population_df)
        st.line_chart(population_df,x_label = 'Годы', y_label = 'Человек')

    with forecast_tab:
        st.subheader('Прогноз численности населения РФ')

        trend_pop = IsTrend(list(population_df[population_df.columns[0]]))
        trend_ru = 'есть' if trend_pop else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('численность', 'popul', len(population_df), True, trend_pop)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        popul_with_pred = plot_chart(population_df, 'численнности', population_df.columns[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


#births_coeff_df, deaths_m_coeff_df, deaths_f_coeff_df, life_expec
elif selectbox_proc == "Коэффициент рождаемости":
    st.subheader('Коэффициент рождаемости')

    params_menu = ['Анализ текущих данных', 'Прогноз динамики показателя']
    analisys_tab, forecast_tab = st.tabs(params_menu)
    
    with analisys_tab:
        st.subheader('Исторические данные о коэффициенте рождаемости')
        st.dataframe(births_coeff_df)
        
        data_menu = ['Суммарный коэффициент рождаемости', 'Возрастные коэфффициенты рождаемости']
        sum_tab, ages_tab = st.tabs(data_menu)
        with sum_tab:
            st.line_chart(births_coeff_df[births_coeff_df.columns[-1]], x_label = 'Годы', y_label = 'Среднее число детей на 1 женщину')
        with ages_tab:
            data_menu = ['Линейная диаграмма', 'Столбчатая диаграмма']
            line_tab, bar_tab = st.tabs(data_menu)
            with line_tab:
                st.line_chart(births_coeff_df[births_coeff_df.columns[:-1]], x_label = 'Годы', y_label = 'Рождений на 1000 женщин')
            with bar_tab:
                st.bar_chart(births_coeff_df[births_coeff_df.columns[:-1]], x_label = 'Годы', y_label = 'Рождений на 1000 женщин')

    with forecast_tab:
        st.subheader('Прогноз динамики показателя коэффициента рождаемости')

        trend_list = [IsTrend(list(births_coeff_df[i])) for i in births_coeff_df.columns]
        forecast_list = {}

        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('коэффициентов рождаемости', 'bc', len(births_coeff_df), True, True)
        tabs = st.tabs([f"{col}" for col in births_coeff_df.columns])
        for i in range(len(births_coeff_df.columns)):
            with tabs[i]:
                st.write(f'Прогноз для {births_coeff_df.columns[i]}')
                trend_ru = 'есть' if trend_list[i] else 'отсутствует'                
                st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
                
                forecast_list[i] = plot_chart(births_coeff_df, 'коэффициента рождаемости', births_coeff_df.columns[i], selectbox_proc, test_ratio, forecast_len, trend_type if trend_list[i] else None, seasonal_type, seasonal_periods)

elif selectbox_proc == "Коэффициент смертности":
    st.subheader('Прогноз динамики показателя коэффициента смертности')

    params_menu = ['Анализ текущих данных', 'Прогноз динамики показателя']
    analisys_tab, forecast_tab = st.tabs(params_menu)
    
    with analisys_tab:
        st.subheader('Исторические данные о коэффициенте смертности')
        st.dataframe(deaths_m_coeff_df)
        
        data_menu = ['Линейная диаграмма', 'Столбчатая диаграмма']
        line_tab, bar_tab = st.tabs(data_menu)
        with line_tab:
            st.line_chart(deaths_m_coeff_df[deaths_m_coeff_df.columns[:-1]], x_label = 'Годы', y_label = 'Смертей на 1000 чловек')
        with bar_tab:
            st.bar_chart(deaths_m_coeff_df[deaths_m_coeff_df.columns[:-1]], x_label = 'Годы', y_label = 'Смертей на 1000 чловек')

    with forecast_tab:
        st.subheader('Прогноз динамики показателя коэффициента смертности')

        trend_list = [IsTrend(list(deaths_m_coeff_df[i])) for i in deaths_m_coeff_df.columns]
        forecast_list = {}
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('коэффициентов смертности', 'dc', len(deaths_m_coeff_df), True, True)
        tabs = st.tabs([f"{col}" for col in deaths_m_coeff_df.columns])
        for i in range(len(deaths_m_coeff_df.columns)):
            with tabs[i]:
                st.write(f'Прогноз для {deaths_m_coeff_df.columns[i]}')
                trend_ru = 'есть' if trend_list[i] else 'отсутствует'                
                st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')
                
                forecast_list[i] = plot_chart(deaths_m_coeff_df, 'коэффициента смертности', deaths_m_coeff_df.columns[i], selectbox_proc, test_ratio, forecast_len, trend_type if trend_list[i] else None, seasonal_type, seasonal_periods)

    

elif selectbox_proc == "Ожидаемая продолжительность жизни":
    st.subheader('Прогоноз динамики показателя ожидаемой продолжительности жизни')
      
    params_menu = ['Анализ текущих данных', 'Прогноз общей ожидаемой продолжительности жизни',
                   'Прогноз ожидаемой продолжительности жизни мужчин', 'Прогноз ожидаемой продолжительности жизни женщин']
    analisys_tab, gen_tab, mal_tab, fem_tab = st.tabs(params_menu)
    
    
    g_with_pred, m_with_pred, f_with_pred = None, None, None
    

    with analisys_tab:
        st.subheader('Исторические данные об ожидаемой продолжительности жизни населения')
        st.dataframe(life_expec)

        menu = ['Линейная диаграмма', 'Столбчатая диаграмма']
        line_tab, bar_tab = st.tabs(menu)
        st.subheader('Соотношение показателей ожидаемой продолжительности жизни')
        with line_tab:
            st.line_chart(life_expec)

        with bar_tab:
            st.bar_chart(life_expec, x_label = 'Годы', y_label = 'Возраст', stack = False)
        
        
    with gen_tab:
        st.subheader('Прогноз показателя ожидаемой продолжительности жизни')

        trend_gen = IsTrend(list(life_expec[life_expec.columns[0]]))
        trend_ru = 'есть' if trend_gen else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('показателя ожидаемой продолжительности жизни', 'life', len(life_expec), True, trend_gen)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        g_with_pred = plot_chart(life_expec, 'показателя ожидаемой продолжительности жизни', life_expec.columns[0], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)

    with mal_tab:
        st.subheader('Прогноз показателя ожидаемой продолжительности жизни мужчин')

        trend_male = IsTrend(list(life_expec[life_expec.columns[1]]))
        trend_ru = 'есть' if trend_male else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('показателя ожидаемой продолжительности жизни мужчин', 'm_life', len(life_expec), True, trend_male)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        m_with_pred = plot_chart(life_expec, 'показателя ожидаемой продолжительности жизни мужчин', life_expec.columns[1], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)

    with fem_tab:
        st.subheader('Прогноз показателя ожидаемой продолжительности жизни женщин')

        trend_female = IsTrend(list(life_expec[life_expec.columns[2]]))
        trend_ru = 'есть' if trend_female else 'отсутствует'
        
        test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods = display_sidebar('показателя ожидаемой продолжительности жизни женщин', 'f_life', len(life_expec), True, trend_female)
        
        st.info(f'Методом Фостера-Стюарта определено, что тренд в данных {trend_ru}')

        f_with_pred = plot_chart(life_expec, 'показателя ожидаемой продолжительности жизни женщин', life_expec.columns[2], selectbox_proc, test_ratio, forecast_len, trend_type, seasonal_type, seasonal_periods)


        
        

