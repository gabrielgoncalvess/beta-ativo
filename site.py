import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

st.title("Beta de um ativo")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

ativo = st.text_input("Escolha o ativo (Ex: ITUB3, HGLG11, IVVB11): ")
data_inicial = st.text_input("Data inicial no formato DD/MM/AAAA: ")
data_final = st.text_input("Data final no formato DD/MM/AAAA: ")

if st.button('Calcular Beta'):
    try:
        data_inicial = f"{data_inicial[3:5]}/{data_inicial[:2]}/{data_inicial[6:]}"
        data_final = f"{data_final[3:5]}/{data_final[:2]}/{data_final[6:]}"

        df_ativo = web.get_data_yahoo(ativo + '.SA', data_inicial, data_final)['Adj Close']
        df_ibov = web.get_data_yahoo('^BVSP', data_inicial, data_final)['Adj Close']

        retorno_ativo = (df_ativo/df_ativo.shift(1)) - 1
        retorno_ibovespa = (df_ibov/df_ibov.shift(1)) - 1

        df_retornos = pd.DataFrame()
        df_retornos[ativo] = retorno_ativo
        df_retornos["IBOV"] = retorno_ibovespa

        df_retornos = df_retornos.dropna()

        # Criar arrays para as variáveis x e y no modelo de regressão
        x = np.array(df_retornos['IBOV']).reshape((-1,1))
        y = np.array(df_retornos[ativo])

        # Usar modelo de regressão
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x, y)

        st.write(f'Beta: {model.coef_[0]:.4f}')

        fig, ax = plt.subplots()
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_retornos['IBOV'],df_retornos[ativo])
        line = slope*x+intercept
        plt.plot(x, line, 'r', label='y={:.4f}x+({:.4f})'.format(slope,intercept))
        plt.scatter(x, y)
        plt.legend(fontsize=10)
        plt.grid()
        plt.axhline(linewidth=1.5, color='grey')
        plt.axvline(linewidth=1.5, color='grey')
        plt.xlabel('IBOVESPA')
        plt.ylabel(ativo)
        plt.show()

        st.pyplot(fig)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_retornos)

        st.download_button(
        label="Baixar planilha",
        data=csv,
        file_name='retornos.csv',
        mime='text/csv',
        )
    except:
        st.write("Erro na execução, tente novamente")