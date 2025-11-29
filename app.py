import streamlit as st

st.title('Прогнозирование демографической ситуации Российской Федерации')


pages = [
    st.Page("info.py", title="О методе прогнозирования"),
    
    st.Page("forecast_expsm.py", title="Прогноз методом экспоненциального сглаживания"),

    st.Page("cohort_moving.py", title="Прогноз методом передвижки возрастов")
]



pg = st.navigation(pages)

pg.run()
