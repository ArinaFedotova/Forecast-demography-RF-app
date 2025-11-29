
import streamlit as st

st.set_page_config(
    page_title = 'Прогнозирование демографической ситуации Российской Федерации',
    page_icon = "📈",
    layout = 'wide'
)



pages = [
    st.Page("home.py", title = 'Главная'),
    
    st.Page("info.py", title="О методе прогнозирования"),
    
    st.Page("forecast_expsm.py", title="Прогноз методом экспоненциального сглаживания"),

    st.Page("cohort_moving.py", title="Прогноз методом передвижки возрастов")
]



pg = st.navigation(pages)

pg.run()
