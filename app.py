import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Прогноз цены автомобиля",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("Car Sales - car_data.csv")

model = load_model()
df = load_data()

companies = sorted(df["Company"].dropna().unique())
transmissions = sorted(df["Transmission"].dropna().unique())
colors = sorted(df["Color"].dropna().unique())
body_styles = sorted(df["Body Style"].dropna().unique())
regions = sorted(df["Dealer_Region"].dropna().unique())

st.title("🚗 Прогноз цены автомобиля")
st.markdown("""
Выберите характеристики автомобиля, и модель предскажет его стоимость.  
**Модель:** Random Forest Regressor
""")
st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Основные параметры")
    company = st.selectbox("Марка", companies)

    filtered_models = sorted(
        df[df["Company"] == company]["Model"].dropna().unique()
    )
    model_name = st.selectbox("Модель", filtered_models)

with col_right:
    st.subheader("⚙️ Характеристики авто")
    transmission = st.selectbox("Коробка передач", transmissions)
    color = st.selectbox("Цвет", colors)
    body_style = st.selectbox("Тип кузова", body_styles)
    dealer_region = st.selectbox("Регион дилера", regions)

st.divider()

if st.button("🔍 Рассчитать цену", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Company": company,
        "Model": model_name,
        "Transmission": transmission,
        "Color": color,
        "Body Style": body_style,
        "Dealer_Region": dealer_region
    }])

    predicted_price = model.predict(input_data)[0]

    st.success(f"💰 Прогнозируемая цена автомобиля: **${predicted_price:,.2f}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Марка", company)
    m2.metric("Модель", model_name)
    m3.metric("Коробка", transmission)

    with st.expander("📋 Введённые данные"):
        display_df = input_data.T.reset_index()
        display_df.columns = ["Параметр", "Значение"]
        display_df["Значение"] = display_df["Значение"].astype(str)
        st.dataframe(display_df, use_container_width=True)

st.caption("🚗 Прогноз цены автомобиля")
