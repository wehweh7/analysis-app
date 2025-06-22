
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Факторный анализ", layout="wide")

st.title("📊 Программное приложение для факторного анализа данных")

uploaded_file = st.file_uploader("Загрузите файл Excel или CSV", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📄 Загруженные данные:")
        st.dataframe(df.head())

        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        if numeric_df.empty:
            st.warning("Нет числовых данных для анализа.")
        else:
            st.subheader("📈 Предобработка")
            st.write("🔍 Обнаружение и удаление пропущенных значений...")
            numeric_df.dropna(inplace=True)

            st.write("✅ Нормализация данных...")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            st.subheader("📊 Корреляционная матрица")
            corr = pd.DataFrame(scaled_data, columns=numeric_df.columns).corr()
            fig_corr, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig_corr)

            st.subheader("📉 PCA (Факторный анализ)")
            pca = PCA()
            components = pca.fit_transform(scaled_data)

            exp_var = pca.explained_variance_ratio_
            st.write("🔢 Доля объяснённой дисперсии:")
            st.bar_chart(exp_var)

            st.write("📌 Scree Plot")
            fig_scree, ax2 = plt.subplots()
            ax2.plot(range(1, len(exp_var)+1), exp_var, 'o-', color='b')
            ax2.set_title('Scree Plot')
            ax2.set_xlabel('Номер компоненты')
            ax2.set_ylabel('Объяснённая дисперсия')
            st.pyplot(fig_scree)

            st.write("📎 Компонентные нагрузки:")
            loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(len(pca.components_))], index=numeric_df.columns)
            st.dataframe(loadings.round(3))

    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
