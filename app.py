
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Факторный анализ", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Меню навигации")
page = st.sidebar.radio("Перейти к разделу:", [
    "Загрузка данных", 
    "Корреляционный анализ", 
    "Факторный анализ (PCA)", 
    "Визуализация", 
    "Экспорт результатов", 
    "О приложении"
])

# --- UPLOADER AND DATA STORAGE ---
if "df" not in st.session_state:
    st.session_state.df = None

if page == "Загрузка данных":
    st.title("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите файл Excel или CSV", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df.dropna()
            st.success("Файл успешно загружен и обработан!")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

elif page == "Корреляционный анализ":
    st.title("📊 Корреляционный анализ")
    if st.session_state.df is not None:
        numeric_df = st.session_state.df.select_dtypes(include=["float64", "int64"])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Сначала загрузите данные.")

elif page == "Факторный анализ (PCA)":
    st.title("🧠 Факторный анализ (Principal Component Analysis)")
    if st.session_state.df is not None:
        df = st.session_state.df.select_dtypes(include=["float64", "int64"])
        selected_features = st.multiselect("Выберите признаки для анализа", df.columns.tolist(), default=df.columns.tolist())
        if selected_features:
            data = df[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_

            st.subheader("🔢 Объяснённая дисперсия")
            st.bar_chart(explained_var)

            st.subheader("📎 Компонентные нагрузки")
            loadings = pd.DataFrame(pca.components_.T, index=selected_features, columns=[f"PC{i+1}" for i in range(len(selected_features))])
            st.dataframe(loadings.round(3))
            st.session_state.loadings = loadings
        else:
            st.info("Выберите хотя бы один признак.")
    else:
        st.warning("Сначала загрузите данные.")

elif page == "Визуализация":
    st.title("📈 Визуализация Scree Plot")
    if st.session_state.df is not None:
        df = st.session_state.df.select_dtypes(include=["float64", "int64"])
        data = df
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        pca = PCA()
        pca.fit(X_scaled)
        exp_var = pca.explained_variance_ratio_

        fig, ax = plt.subplots()
        ax.plot(range(1, len(exp_var)+1), exp_var, 'o-', color='b')
        ax.set_title("Scree Plot")
        ax.set_xlabel("Номер компоненты")
        ax.set_ylabel("Объяснённая дисперсия")
        st.pyplot(fig)
    else:
        st.warning("Сначала загрузите данные.")

elif page == "Экспорт результатов":
    st.title("💾 Экспорт компонентных нагрузок")
    if "loadings" in st.session_state:
        csv = st.session_state.loadings.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Скачать как CSV",
            data=csv,
            file_name="factor_loadings.csv",
            mime="text/csv"
        )
    else:
        st.info("Сначала выполните факторный анализ.")

elif page == "О приложении":
    st.title("ℹ️ О приложении")
    st.markdown("""
Это учебное программное приложение разработано в рамках учебной практики по теме:  
**«Разработка программного приложения для реализации кластерного анализа данных»**  
на примере **Банка Синара**.

Приложение позволяет:
- загружать таблицы Excel и CSV;
- выполнять корреляционный анализ;
- проводить факторный анализ (PCA);
- визуализировать Scree Plot;
- экспортировать факторные нагрузки для отчёта.

Разработано с использованием Python, Streamlit, sklearn, seaborn и pandas.


**Автор**: студент РЭУ им. Г.В. Плеханова, 2025 г.
    """)

