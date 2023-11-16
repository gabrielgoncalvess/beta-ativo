import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yfin
import datetime

#yfin.pdr_override()

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

    dict_petr = {'High': {'11/11/2022': 31.16,
  '10/11/2022': 30.24,
  '09/11/2022': 31.18,
  '08/11/2022': 30.9,
  '07/11/2022': 31.9},
 'Low': {'11/11/2022': 29.28,
  '10/11/2022': 28.92,
  '09/11/2022': 30.0,
  '08/11/2022': 29.99,
  '07/11/2022': 30.45},
 'Open': {'11/11/2022': 29.61,
  '10/11/2022': 29.31,
  '09/11/2022': 30.52,
  '08/11/2022': 30.38,
  '07/11/2022': 31.39},
 'Close': {'11/11/2022': 30.7,
  '10/11/2022': 29.69,
  '09/11/2022': 30.06,
  '08/11/2022': 30.63,
  '07/11/2022': 30.63},
 'Volume': {'11/11/2022': 34866200,
  '10/11/2022': 39818600,
  '09/11/2022': 15928500,
  '08/11/2022': 19140100,
  '07/11/2022': 27442700},
 'Adj Close': {'11/11/2022': 30.7,
  '10/11/2022': 29.69,
  '09/11/2022': 30.06,
  '08/11/2022': 30.63,
  '07/11/2022': 30.63}}
 

    #df_petr = web.get_data_yahoo("PETR3" + '.SA', "10/10/2022")
    #df_petr.index = df_petr.index.strftime("%d/%m/%Y")
    df = st.dataframe(pd.DataFrame(dict_petr).head().style.format({"High":"{:,.2f}","Low":"{:,.2f}","Open":"{:,.2f}","Close":"{:,.2f}","Volume":"{:,}","Adj Close":"{:,.2f}"}, decimal=',', thousands='.'))

    texto_sidebar2 = st.markdown("""
    IBOVESPA    
    """)

    dict_ibov = {'High': {'11/11/2022': 113009.62,
  '10/11/2022': 113579.0,
  '09/11/2022': 116183.0,
  '08/11/2022': 117072.0,
  '07/11/2022': 118240.0},
 'Low': {'11/11/2022': 109408.1,
  '10/11/2022': 108516.0,
  '09/11/2022': 113110.0,
  '08/11/2022': 114688.0,
  '07/11/2022': 115266.0},
 'Open': {'11/11/2022': 109775.46,
  '10/11/2022': 113579.0,
  '09/11/2022': 116153.0,
  '08/11/2022': 115340.0,
  '07/11/2022': 118148.0},
 'Close': {'11/11/2022': 112253.49,
  '10/11/2022': 109775.0,
  '09/11/2022': 113580.0,
  '08/11/2022': 116160.0,
  '07/11/2022': 115342.0},
 'Volume': {'11/11/2022': 25038700,
  '10/11/2022': 26029300,
  '09/11/2022': 20531600,
  '08/11/2022': 14239800,
  '07/11/2022': 15221900},
 'Adj Close': {'11/11/2022': 112253.49,
  '10/11/2022': 109775.0,
  '09/11/2022': 113580.0,
  '08/11/2022': 116160.0,
  '07/11/2022': 115342.0}}

    #df_ibov_ex = web.get_data_yahoo("^BVSP", "10/10/2022")
    #df_ibov_ex.index = df_ibov_ex.index.strftime("%d/%m/%Y")
    df = st.dataframe(pd.DataFrame(dict_ibov).head().style.format({"High":"{:,.2f}","Low":"{:,.2f}","Open":"{:,.2f}","Close":"{:,.2f}","Volume":"{:,}","Adj Close":"{:,.2f}"}, decimal=',', thousands='.'))

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

col1, col2 = st.columns(2)

with col1:
    tipo_retorno = st.radio(
            "Escolha um tipo de retorno",
            ["Discreto", "Logaritmo"],
        )
    tipo_ativo = st.radio(
            "Escolha um tipo de ativo",
            ["Nacional", "Internacional"],
        )

with col2:
    indexador = st.radio(
            "Escolha um indexador",
            ["IBOVESPA", "S&P 500", "Nasdaq", "Dow Jones"],
        )

dict_indexador = {
    "IBOVESPA": "^BVSP",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI"
}

ativo = st.text_input("Escolha o ativo (Ex: ITUB3, PETR3, CPLE11): ").strip().upper()
data_inicial = st.text_input("Data inicial no formato DD/MM/AAAA: ").strip()
data_final = st.text_input("Data final no formato DD/MM/AAAA: ").strip()

if st.button('Calcular Beta'):
    try:
        # data_inicial = f"{data_inicial[3:5]}/{data_inicial[:2]}/{data_inicial[6:]}"
        # data_final = f"{data_final[3:5]}/{data_final[:2]}/{data_final[6:]}"
        
        # data_inicial = f"{data_inicial[6:]}-{data_inicial[3:5]}-{data_inicial[:2]}"
        # data_final = f"{data_final[6:]}-{data_final[3:5]}-{data_final[:2]}"

        data_inicial = datetime.datetime(int(f'{data_inicial[6:]}'), int(f'{data_inicial[3:5]}'), int(f'{data_inicial[:2]}'))
        data_final = datetime.datetime(int(f'{data_final[6:]}'), int(f'{data_final[3:5]}'), int(f'{data_final[:2]}'))

        label_ativo = ativo

        if tipo_ativo == "Nacional":
            ativo+=".SA"

        # df_ativo = web.DataReader(ativo, start=data_inicial, end=data_final)['Adj Close']
        # df_ibov = web.DataReader(dict_indexador[indexador], start=data_inicial, end=data_final)['Adj Close']

        df_ativo_base = yfin.Ticker(ativo)
        df_ativo_nome = yfin.Ticker(ativo).info["longName"]
        df_ativo = df_ativo_base.history(start=data_inicial,end=data_final)['Close']
        df_ibov = yfin.Ticker(dict_indexador[indexador]).history(start=data_inicial,end=data_final)['Close']

        df_ativo.index = df_ativo.index.tz_localize(None)
        df_ibov.index = df_ibov.index.tz_localize(None)

        if tipo_retorno == "Logaritmo":
            retorno_ativo = np.log(df_ativo/df_ativo.shift(1)) 
            retorno_ibovespa = np.log(df_ibov/df_ibov.shift(1)) 
        elif tipo_retorno == "Discreto":
            retorno_ativo = (df_ativo/df_ativo.shift(1)) - 1
            retorno_ibovespa = (df_ibov/df_ibov.shift(1)) - 1

        df_retornos = pd.DataFrame()
        df_retornos[label_ativo] = retorno_ativo
        df_retornos[indexador] = retorno_ibovespa

        df_retornos = df_retornos.dropna()

        # Criar arrays para as variáveis x e y no modelo de regressão
        x = np.array(df_retornos[indexador]).reshape((-1,1))
        y = np.array(df_retornos[label_ativo])

        # Usar modelo de regressão
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x, y)

        st.write(f'{df_ativo_nome}')
        st.write(f'Beta: {model.coef_[0]:.4f}')

        fig, ax = plt.subplots()
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_retornos[indexador],df_retornos[label_ativo])
        line = slope*x+intercept
        plt.plot(x, line, 'r', label='y={:.4f}x+({:.4f})'.format(slope,intercept))
        plt.scatter(x, y)
        plt.legend(fontsize=10)
        plt.grid()
        plt.axhline(linewidth=1.5, color='grey')
        plt.axvline(linewidth=1.5, color='grey')
        plt.xlabel(indexador)
        plt.ylabel(label_ativo)
        plt.show()

        st.pyplot(fig)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_retornos.round(6).iloc[::-1])

        st.download_button(
        label="Baixar planilha",
        data=csv,
        file_name='retornos.csv',
        mime='text/csv',
        )
    except Exception as e:
        st.write("Erro na execução, tente novamente")
        # st.error(f"An error occurred: {e}")
