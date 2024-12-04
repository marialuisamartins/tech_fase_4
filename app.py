import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import datetime
import requests
import numpy as np

# Função para carregar os dados e ajustar o modelo ARIMA
def modelo(data, steps=1):
    modelo = ARIMA(data['preco'], order=(2, 1, 2))
    modelo_fit = modelo.fit()
    forecast = modelo_fit.forecast(steps=steps)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast_series = pd.Series(forecast, index=forecast_index)
    return forecast_series

# Baixar o arquivo Excel
url = 'https://raw.githubusercontent.com/marialuisamartins/tech_fase4/6ad3e07bc901fd984eedb3030510b2816aaf7383/ipeadata%5B03-11-2024-01-09%5D.xlsx'
arquivo_local = 'ipeadata.xlsx'

# Fazer o download do arquivo e salvar localmente
response = requests.get(url)
with open(arquivo_local, 'wb') as file:
    file.write(response.content)

# Ler o arquivo Excel
ipeadata = pd.read_excel('ipeadata.xlsx', engine='openpyxl')

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

# Mostrar a previsão para hoje e para amanhã
previsao_hoje = modelo(ipeadata_filtered, steps=1)
previsao_amanha = modelo(ipeadata_filtered, steps=2).iloc[1]

# Calcular a tendência (subindo ou descendo)
tendencia = "subindo" if previsao_amanha > previsao_hoje.iloc[0] else "descendo"
icone_tendencia = "🔼" if tendencia == "subindo" else "🔽"

# Informações sobre a previsão para hoje e amanhã
st.markdown("### Previsão de Preço do Petróleo")
st.write(f"**Previsão para hoje**: ${previsao_hoje.iloc[0]:,.2f}")
st.write(f"**Previsão para amanhã**: ${previsao_amanha:,.2f} {icone_tendencia} ({tendencia})")

# Gráfico comparativo de preços de diferentes anos
st.markdown("### Comparação de Preços de Diferentes Anos")
ipeadata_filtered['year'] = ipeadata_filtered.index.year
fig1, ax1 = plt.subplots(figsize=(14, 6))
sns.lineplot(data=ipeadata_filtered, x=ipeadata_filtered.index, y='preco', hue='year', marker='o', palette='tab10')
ax1.set_title("Comparação de Preços do Petróleo por Ano (2021-2024)")
ax1.set_xlabel("Data")
ax1.set_ylabel("Preço (USD)")
ax1.legend(title='Ano')
st.pyplot(fig1)
st.write("""
    Neste gráfico, podemos observar como os preços do petróleo variam de ano para ano, destacando tendências e possíveis sazonalidades. 
    Fatores como conflitos geopolíticos, mudanças na produção de petróleo e crises econômicas podem influenciar esses movimentos. 
    Por exemplo, picos em certos anos podem ter sido causados por tensões no Oriente Médio ou decisões da OPEP.
""")

# Análise de picos sazonais (detecção de picos mensais)
st.markdown("### Análise de Picos Sazonais")
ipeadata_filtered['month'] = ipeadata_filtered.index.month
monthly_max = ipeadata_filtered.groupby('month')['preco'].max()

# Gráfico de picos mensais
fig2, ax2 = plt.subplots(figsize=(14, 6))
monthly_max.plot(kind='bar', color='orange', ax=ax2)
ax2.set_title("Picos Mensais de Preço do Petróleo (2021-2024)")
ax2.set_xlabel("Mês")
ax2.set_ylabel("Preço (USD)")
st.pyplot(fig2)
st.write("""
    Este gráfico mostra os picos mensais de preços do petróleo. Analisando essas informações, podemos identificar meses específicos em que os preços tendem a atingir seus valores mais altos, 
    o que pode indicar padrões sazonais no mercado. Por exemplo, aumentos em certos meses podem estar relacionados a aumentos na demanda durante o inverno ou à especulação antes de eventos políticos importantes.
""")

# Gráfico dos últimos 15 dias
st.markdown("### Gráfico dos Últimos 15 Dias")
ultimos_15_dias = ipeadata_filtered.tail(15)
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(ultimos_15_dias.index, ultimos_15_dias['preco'], color='blue', marker='o', linestyle='-', label='Preço Diário')
ax3.set_title("Preço do Petróleo - Últimos 15 Dias")
ax3.set_xlabel("Data")
ax3.set_ylabel("Preço (USD)")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)
st.write("""
    Este gráfico mostra os preços do petróleo nos últimos 15 dias. Analisando os preços recentes, podemos identificar tendências de curto prazo e correlacionar com eventos de mercado recentes, como relatórios de produção, mudanças na demanda e anúncios econômicos importantes.
""")

# Gráfico de variação diária de preços
st.markdown("### Variação Diária de Preços")
ipeadata_filtered['variacao_diaria'] = ipeadata_filtered['preco'].pct_change() * 100  # Variação percentual diária
fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(ipeadata_filtered.index, ipeadata_filtered['variacao_diaria'], color='red', marker='x', linestyle='--')
ax4.set_title("Variação Diária Percentual do Preço do Petróleo")
ax4.set_xlabel("Data")
ax4.set_ylabel("Variação (%)")
ax4.grid(True)
st.pyplot(fig4)
st.write("""
    A variação diária dos preços mostra a volatilidade do mercado de petróleo. Picos de variação podem indicar eventos de alto impacto, como notícias de conflitos geopolíticos, mudanças na política de produção da OPEP, ou anúncios econômicos importantes que afetam a confiança dos investidores.
""")

# Análise adicionais (média móvel e distribuição)
st.markdown("### Análises Adicionais")
analise_opcao = st.selectbox(
    "Escolha uma análise para visualizar:",
    ["Média Móvel de 7 dias", "Média Móvel de 30 dias", "Distribuição dos Preços"]
)

if analise_opcao == "Média Móvel de 7 dias":
    st.markdown("#### Média Móvel de 7 dias")
    ipeadata_filtered['media_movel_7'] = ipeadata_filtered['preco'].rolling(window=7).mean()
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    ax5.plot(ipeadata_filtered.index, ipeadata_filtered['preco'], label='Preço Diário', color='blue', alpha=0.5)
    ax5.plot(ipeadata_filtered.index, ipeadata_filtered['media_movel_7'], label='Média Móvel de 7 dias', color='orange')
    ax5.set_title('Preço do Petróleo com Média Móvel de 7 dias')
    ax5.set_xlabel('Data')
    ax5.set_ylabel('Preço (USD)')
    ax5.legend()
    st.pyplot(fig5)

elif analise_opcao == "Média Móvel de 30 dias":
    st.markdown("#### Média Móvel de 30 dias")
    ipeadata_filtered['media_movel_30'] = ipeadata_filtered['preco'].rolling(window=30).mean()
    fig6, ax6 = plt.subplots(figsize=(14, 6))
    ax6.plot(ipeadata_filtered.index, ipeadata_filtered['preco'], label='Preço Diário', color='blue', alpha=0.5)
    ax6.plot(ipeadata_filtered.index, ipeadata_filtered['media_movel_30'], label='Média Móvel de 30 dias', color='green')
    ax6.set_title('Preço do Petróleo com Média Móvel de 30 dias')
    ax6.set_xlabel('Data')
    ax6.set_ylabel('Preço (USD)')
    ax6.legend()
    st.pyplot(fig6)

elif analise_opcao == "Distribuição dos Preços":
    st.markdown("#### Distribuição dos Preços")
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    sns.histplot(ipeadata_filtered['preco'], bins=30, kde=True, color='skyblue')
    ax7.set_title('Distribuição dos Preços do Petróleo')
    ax7.set_xlabel('Preço (USD)')
    ax7.set_ylabel('Frequência')
    st.pyplot(fig7)
