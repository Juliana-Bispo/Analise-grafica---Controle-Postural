import numpy as np
import pandas as pd
import plotly.express as px

def carregar_dados(caminho_arquivo, skip_rows=15, sep="\t", decimal=','):
    try:
        return pd.read_csv(caminho_arquivo, skiprows=skip_rows, sep=sep, decimal=decimal)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None

def calcular_velocidade(aceleracao, delta_t):
    velocidade = np.diff(aceleracao) * delta_t
    velocidade = np.insert(velocidade, 0, 0)  # Adiciona um zero no início para manter o mesmo tamanho
    return velocidade

def calcular_posicao(velocidade, delta_t):
    posicao = np.diff(velocidade) * delta_t
    posicao = np.insert(posicao, 0, 0)  # Adiciona um zero no início para manter o mesmo tamanho
    return posicao

def plotar_trajetoria_3d(df):
    fig = px.line_3d(df, x='x', y='y', z='z', title="Trajetória 3D")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(scene=dict(xaxis=dict(range=[min(df['x']), max(df['x'])]), 
                                 yaxis=dict(range=[min(df['y']), max(df['y'])]),
                                 zaxis=dict(range=[min(df['z']), max(df['z'])])))
    fig.show()

def plotar_grafico_temporal(tempo, eixo, nome_eixo):
    fig = px.line(x=tempo, y=eixo, labels={'x': 'Tempo (s)', 'y': nome_eixo}, title=f"{nome_eixo} vs Tempo")
    fig.update_yaxes(range=[min(eixo), max(eixo)])  # Ajuste manual do range do eixo Y
    fig.show()

# Parâmetros
delta_t = 0.010  # Intervalo de tempo em segundos

# Carregar os dados
dados = carregar_dados(r"C:\Users\julia\controle-postural\BOS_0605_CP_CV_01.txt")

if dados is not None:
    # Extraindo as colunas
    tempo = dados.iloc[:, 0].values
    ax = dados.iloc[:, 1].values
    ay = dados.iloc[:, 2].values
    az = dados.iloc[:, 3].values

    # Calculando a velocidade
    vx = calcular_velocidade(ax, delta_t)
    vy = calcular_velocidade(ay, delta_t)
    vz = calcular_velocidade(az, delta_t)

    # Calculando a posição
    x = calcular_posicao(vx, delta_t)
    y = calcular_posicao(vy, delta_t)
    z = calcular_posicao(vz, delta_t)

    # Criando o dataframe para plotar
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # Plotando o gráfico 3D
    plotar_trajetoria_3d(df)

    # Plotando os gráficos temporais
    plotar_grafico_temporal(tempo, x, 'Posição X')
    plotar_grafico_temporal(tempo, y, 'Posição Y')
    plotar_grafico_temporal(tempo, z, 'Posição Z')

    # Impressões para verificação
    print(ax[:10], ay[:10], az[:10])  # Mostra os primeiros 10 valores
    print(vx[:10], vy[:10], vz[:10])
    print(x[:10], y[:10], z[:10])
    print(df.head(10))  # Verifica os primeiros valores no DataFrame
