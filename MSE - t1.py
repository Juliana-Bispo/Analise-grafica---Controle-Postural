import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função para carregar os dados
def carregar_dados(caminho_arquivo, skip_rows=15, sep="\t", decimal=','):
    try:
        return pd.read_csv(caminho_arquivo, skiprows=skip_rows, sep=sep, decimal=decimal)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None

# Função para calcular a velocidade
def calcular_velocidade(aceleracao, delta_t):
    velocidade = np.diff(aceleracao) * delta_t
    velocidade = np.insert(velocidade, 0, 0)
    return velocidade

# Função para calcular a posição
def calcular_posicao(velocidade, delta_t):
    posicao = np.diff(velocidade) * delta_t
    posicao = np.insert(posicao, 0, 0)
    return posicao

# Função para calcular a Sample Entropy (SampEn)
def sample_entropy(time_series, m, r):
    n = len(time_series)
    def _phi(m):
        x = np.array([time_series[i:i + m] for i in range(n - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (n - m + 1) / (n - m)
    
    return -np.log(_phi(m + 1) / _phi(m))

# Função para calcular a MSE
def multiscale_entropy(data, scale_max, m=2, r=0.2):
    def coarse_graining(data, scale):
        n = len(data)
        b = n // scale
        return np.mean(data[:b*scale].reshape((b, scale)), axis=1)
    
    entropies = []
    for scale in range(1, scale_max + 1):
        coarse_data = coarse_graining(data, scale)
        sampen = sample_entropy(coarse_data, m=m, r=r * np.std(coarse_data))
        entropies.append(sampen)
    return entropies

# Função para calcular estatísticas de SampEn XY
def calcular_estatisticas_sampen(mse_x, mse_y):
    sampen_xy = np.array(mse_x) + np.array(mse_y)

    # Mediana
    mediana_sampen_xy = np.median(sampen_xy)

    # Erro padrão
    erro_padrao_sampen_xy = np.std(sampen_xy, ddof=1) / np.sqrt(len(sampen_xy))

    # Variância
    variancia_sampen_xy = np.var(sampen_xy)

    # Erro normalizado
    erro_normalizado_sampen_xy = erro_padrao_sampen_xy / mediana_sampen_xy

    return (mediana_sampen_xy, erro_padrao_sampen_xy, variancia_sampen_xy, erro_normalizado_sampen_xy)

# Função para plotar as variáveis
def plotar_estatisticas_sampen(r_vals, mediana, erro_padrao, variancia, erro_normalizado):
    plt.figure(figsize=(12, 8))

    # Gráfico 1: Mediana
    plt.subplot(2, 2, 1)
    plt.plot(r_vals, mediana, marker='*')
    plt.title('Mediana SampEn XY')
    plt.xlabel('r')
    plt.ylabel('Mediana SampEn')

    # Gráfico 2: Erro Padrão
    plt.subplot(2, 2, 2)
    plt.plot(r_vals, erro_padrao, marker='*', color='r')
    plt.title('Erro Padrão SampEn XY')
    plt.xlabel('r')
    plt.ylabel('Erro Padrão')

    # Gráfico 3: Variância
    plt.subplot(2, 2, 3)
    plt.plot(r_vals, variancia, marker='*', color='g')
    plt.title('Variância SampEn XY')
    plt.xlabel('r')
    plt.ylabel('Variância')

    # Gráfico 4: Erro Normalizado
    plt.subplot(2, 2, 4)
    plt.plot(r_vals, erro_normalizado, marker='*', color='m')
    plt.title('Erro Normalizado SampEn XY')
    plt.xlabel('r')
    plt.ylabel('Erro Normalizado')

    plt.tight_layout()
    plt.show()

# Parâmetros
delta_t = 0.010  # Intervalo de tempo em segundos
r_vals = np.linspace(0.1, 1.0, 5)  # Diferentes valores de r para análise

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

    # Calculando a posição
    x = calcular_posicao(vx, delta_t)
    y = calcular_posicao(vy, delta_t)

    # Calculando a MSE para diferentes valores de r
    mediana_vals, erro_padrao_vals, variancia_vals, erro_normalizado_vals = [], [], [], []
    
    for r in r_vals:
        mse_x = multiscale_entropy(x, scale_max=20, m=2, r=r)  # Passando m e r
        mse_y = multiscale_entropy(y, scale_max=20, m=2, r=r)  # Passando m e r

        # Calculando estatísticas
        mediana, erro_padrao, variancia, erro_normalizado = calcular_estatisticas_sampen(mse_x, mse_y)
        
        mediana_vals.append(mediana)
        erro_padrao_vals.append(erro_padrao)
        variancia_vals.append(variancia)
        erro_normalizado_vals.append(erro_normalizado)

    # Plotando as estatísticas
    plotar_estatisticas_sampen(r_vals, mediana_vals, erro_padrao_vals, variancia_vals, erro_normalizado_vals)
