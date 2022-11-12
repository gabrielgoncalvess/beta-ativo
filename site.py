import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="Beta ativo",
)

st.title("Beta de um ativo")

with st.sidebar:
    texto_sidebar = st.markdown(
        """Este site possui o objetivo de calcular o beta de um ativo a partir de variáveis como a data inicial, final e o
        nome do ativo com dados obtidos do Yahoo Finance.
        \nApós fornecer essas informações, as tabelas são extraídas como no exemplo:
        \nPETR3
        """
    )

    df_petr = web.get_data_yahoo("PETR3" + '.SA', "10/10/2022")
    df_petr.index = df_petr.index.strftime("%d/%m/%Y")
    df = st.dataframe(df_petr.head())

    texto_sidebar2 = st.markdown("""
    IBOVESPA    
    """)

    df_ibov_ex = web.get_data_yahoo("^BVSP", "10/10/2022")
    df_ibov_ex.index = df_ibov_ex.index.strftime("%d/%m/%Y")
    df = st.dataframe(df_ibov_ex.head())

    texto_sidebar3 = st.markdown("""
    Com esses dados, é calculado o retorno discreto diário de cada tabela utilizando o "Adj Close" e, com isso,
     é feita uma regressão linear para obter o beta e um gráfico de dispersão com a reta destacada.
    \n
    Depois de usar o botão "Calcular Beta", aparecerá uma opção "Baixar planilha" no final da página, sendo um
    csv com a data e retornos diários do ativo escolhido e do Ibovespa.
    \n \n \n
    """)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

ativo = st.text_input("Escolha o ativo (Ex: ITUB3, PETR3, CPLE11): ").strip().upper()
data_inicial = st.text_input("Data inicial no formato DD/MM/AAAA: ").strip()
data_final = st.text_input("Data final no formato DD/MM/AAAA: ").strip()

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