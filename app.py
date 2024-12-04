import streamlit as st
import gdown
import pickle
import pandas as pd
from prophet import Prophet
import requests

# Função para carregar o modelo Prophet
@st.cache_resource
def carregar_modelo():
    # URL do modelo Prophet salvo no Google Drive
    #https://drive.google.com/file/d/11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD/view?usp=sharing
    url = 'https://drive.google.com/uc?id=11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD'
    
    # Fazer o download do modelo
    gdown.download(url, 'prophet_model.pkl', quiet=False)
    
    # Carregar o modelo Prophet com pickle
    with open('prophet_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    return modelo

def dados_xls():
    url = 'https://github.com/marialuisamartins/tech_fase4/blob/main/ipeadata.csv'
    arquivo_local = 'ipeadata.csv'
    response = requests.get(url)
    with open(arquivo_local, 'wb') as file:
        file.write(response.content)

    return arquivo_local

# Função principal do aplicativo
def main():
    # Configuração da página
    st.set_page_config(page_title='MVP para análise temporal de petróleo',
                       page_icon='🛢️')
    
    st.write('# MVP para análise de preço do petróleo Brent')

    # Carregar o modelo Prophet
    modelo = carregar_modelo()
    st.success("Modelo carregado com sucesso!")

    dados_xls()

    # Ler o arquivo Excel
    ipeadata = pd.read_csv('ipeadata.csv')


    # Definir a coluna 'data' como índice e filtrar para a partir de 2021
    ipeadata['data'] = pd.to_datetime(ipeadata['data'])
    ipeadata.set_index('data', inplace=True)
    ipeadata_filtered = ipeadata[ipeadata.index >= '2021-01-01']

    # Adicionar cabeçalho superior estilizado com HTML
    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 36px;
                color: #2C3E50;
                font-weight: bold;
                margin-top: 20px;
            }
            .header-bar {
                background-color: #2C3E50;
                padding: 10px;
                color: white;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }
        </style>
        <div class="header-bar">Dashboard Interativo de Previsão e Análise do Preço do Petróleo</div>
        <div class="title">Exploração de Dados e Previsões</div>
    """, unsafe_allow_html=True)

    # Barra lateral com logo da FIAP
    st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/5/54/Logo_FIAP.png', use_container_width=True)
    st.sidebar.header("Exploração do Preço do Petróleo")
    st.sidebar.write("Análise detalhada de preços e previsões para insights de mercado.")
    
    # Informações estatísticas importantes
    st.markdown("### Estatísticas e Insights Importantes")
    st.write(f"**Valor máximo no período**: ${ipeadata_filtered['preco'].max():,.2f}")
    st.write(f"**Valor mínimo no período**: ${ipeadata_filtered['preco'].min():,.2f}")
    st.write(f"**Média do período**: ${ipeadata_filtered['preco'].mean():,.2f}")
    st.write(f"**Desvio Padrão do período**: ${ipeadata_filtered['preco'].std():,.2f}")

   # Previsões iniciais
    df_forecast = pd.DataFrame({'ds': ipeadata_filtered.index, 'y': ipeadata_filtered['preco']})
    
    # Configuração do sidebar para entradas do usuário
    st.sidebar.header("Configurações da Previsão")
    periods = st.sidebar.number_input("Quantos dias você quer prever?", min_value=1, value=30)
    
    # Botão para gerar previsão
    if st.button("Gerar Previsão"):
        # Criar DataFrame futuro com o número total de períodos necessários
        future = modelo.make_future_dataframe(periods=periods)
    
        # Fazer a previsão
        forecast = modelo.predict(future)
    
        # Filtrar previsões começando do dia seguinte e limitar ao número de dias escolhidos
        hoje = pd.Timestamp.today().normalize()  # Normaliza para ignorar horas
        forecast_filtered = forecast[forecast['ds'] >= hoje].head(periods)
    
        # Mostrar os resultados da previsão
        #st.subheader(f"Resultados da Previsão (a partir de {hoje.date()})")
        #st.write(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        #st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        st.write(ipeadata)
    
        # Plotar o gráfico da previsão
        st.subheader("Gráfico da Previsão")
        st.line_chart(forecast_filtered[['ds', 'yhat']].set_index('ds'))

    # Seções adicionais como placeholders
    st.write("## Dashboard")
    st.write("## Insights")
    st.write("## Modelo")

# Rodar o aplicativo
if __name__ == "__main__":
    main()
