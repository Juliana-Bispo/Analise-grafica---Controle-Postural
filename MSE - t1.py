import numpy as np  # Biblioteca para operações matemáticas e arrays
import pandas as pd  # Biblioteca para manipulação de dados em DataFrames
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
import os  # Biblioteca para manipulação de caminhos de arquivos
import tkinter as tk  # Biblioteca para criar interfaces gráficas
from tkinter import scrolledtext  # Para criar uma caixa de texto rolável
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Para embutir gráficos no Tkinter

# Função para carregar os dados de um arquivo
def carregar_dados(caminho_arquivo, skip_rows=15, sep="\t", decimal=','):
    try:
        return pd.read_csv(caminho_arquivo, skiprows=skip_rows, sep=sep, decimal=decimal)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None

# Função para calcular a entropia amostral (SampEn)
def sample_entropy(time_series, m, r):
    n = len(time_series)

    def _phi(m):
        x = np.array([time_series[i:i + m] for i in range(n - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (n - m + 1) / (n - m)

    return -np.log(_phi(m + 1) / _phi(m))

# Função para calcular a entropia multiescala (MSE)
def multiscale_entropy(data, scale_max, m=2, r=0.2):
    def coarse_graining(data, scale):
        n = len(data)
        b = n // scale
        return np.mean(data[:b * scale].reshape((b, scale)), axis=1)

    entropies = []
    for scale in range(1, scale_max + 1):
        coarse_data = coarse_graining(data, scale)
        sampen = sample_entropy(coarse_data, m=m, r=r * np.std(coarse_data))
        entropies.append(sampen)

    return entropies

# Função para calcular estatísticas das entropias SampEn XY
def calcular_estatisticas_sampen(mse_x, mse_y):
    sampen_xy = np.array(mse_x) + np.array(mse_y)

    mediana_sampen_xy = np.median(sampen_xy)
    erro_padrao_sampen_xy = np.std(sampen_xy, ddof=1) / np.sqrt(len(sampen_xy))
    variancia_sampen_xy = np.var(sampen_xy)
    erro_normalizado_sampen_xy = erro_padrao_sampen_xy / mediana_sampen_xy

    return (mediana_sampen_xy, erro_padrao_sampen_xy, variancia_sampen_xy, erro_normalizado_sampen_xy)

# Função para plotar as estatísticas para os três eixos no mesmo gráfico
def plotar_estatisticas_combinadas(r_vals, nome_arquivo, estatisticas):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Usar o nome do arquivo como título do gráfico
    titulo_plot = f'Estatísticas SampEn - {os.path.basename(nome_arquivo)}'
    fig.suptitle(titulo_plot, fontsize=16)

    cores = {'X': 'b', 'Y': 'g', 'Z': 'r'}  # Cores para os eixos

    # Iterar sobre os gráficos e plotar para cada eixo (X, Y e Z)
    for eixo, (mediana_vals, erro_padrao_vals, variancia_vals, erro_normalizado_vals) in estatisticas.items():
        axes[0, 0].plot(r_vals, mediana_vals, marker='*', color=cores[eixo], label=f'Acelerômetro - {eixo}')
        axes[0, 1].plot(r_vals, erro_padrao_vals, marker='o', color=cores[eixo], label=f'Acelerômetro - {eixo}')
        axes[1, 0].plot(r_vals, variancia_vals, marker='s', color=cores[eixo], label=f'Acelerômetro - {eixo}')
        axes[1, 1].plot(r_vals, erro_normalizado_vals, marker='d', color=cores[eixo], label=f'Acelerômetro - {eixo}')

    # Configurar títulos, legendas e limites dos eixos Y
    axes[0, 0].set_title('Mediana')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('Mediana')
    axes[0, 0].set_ylim(0, 5)
    axes[0, 0].legend()

    axes[0, 1].set_title('Erro Padrão')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('Erro Padrão')
    axes[0, 1].set_ylim(0, 0.1)
    axes[0, 1].legend()

    axes[1, 0].set_title('Variância')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('Variância')
    axes[1, 0].set_ylim(0, 0.1)
    axes[1, 0].legend()

    axes[1, 1].set_title('Erro Normalizado')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('Erro Normalizado')
    axes[1, 1].set_ylim(0, 0.05)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig

# Função para exibir a explicação e o gráfico na interface Tkinter
def abrir_interface():
    # Criar a janela principal
    root = tk.Tk()
    root.title("Análise de Entropia Amostral")

    # Criar um layout de grid com duas colunas
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Criar o widget de texto rolável para a explicação
    explicacao_texto = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
    explicacao_texto.grid(row=0, column=1, padx=10, pady=10)

    # Adicionar a explicação no widget de texto
    explicacao_texto.insert(tk.INSERT, """
    Análise dos Gráficos:
    1. **Mediana SampEn**:
       Este gráfico mostra a mediana da entropia amostral (SampEn) para os dados.
       O valor de 'r' varia entre 0,1 e 1, representando diferentes tolerâncias.
       A mediana indica a complexidade global do padrão de movimento.
       Valores mais altos de mediana sugerem maior variabilidade nos dados.

    2. **Erro Padrão SampEn**:
       O erro padrão da mediana indica a precisão dos cálculos de SampEn.
       Quanto menor o erro, mais confiável é a estimativa da mediana.

    3. **Variância SampEn**:
       A variância mede a dispersão dos valores de SampEn.
       Uma maior variância indica maior irregularidade ou imprevisibilidade nos dados.

    4. **Erro Normalizado SampEn**:
       O erro normalizado ajusta a precisão da mediana levando em conta sua magnitude.
       Um erro menor significa que a mediana é bem estimada e confiável.
    """)

    # Carregar os dados e calcular as estatísticas
    dados = carregar_dados(r"C:\Users\julia\Desktop\PIBIC\controle-postural\controle-postural\BOS_0605_CP_CV_01.txt")
    r_vals = np.linspace(0.1, 1.0, 5)  # Valores de r para análise

    if dados is not None:
        ax = dados.iloc[:, 1].values
        ay = dados.iloc[:, 2].values
        az = dados.iloc[:, 3].values

        # Inicializar dicionário para armazenar as estatísticas
        estatisticas = {'X': [], 'Y': [], 'Z': []}

        for r in r_vals:
            mse_x = multiscale_entropy(ax, scale_max=20, m=2, r=r)
            mse_y = multiscale_entropy(ay, scale_max=20, m=2, r=r)
            mse_z = multiscale_entropy(az, scale_max=20, m=2, r=r)

            estatisticas['X'].append(calcular_estatisticas_sampen(mse_x, mse_y))
            estatisticas['Y'].append(calcular_estatisticas_sampen(mse_y, mse_z))
            estatisticas['Z'].append(calcular_estatisticas_sampen(mse_x, mse_z))

        # Converter listas para tuplas para facilitar a descompactação na função de plotagem
        estatisticas = {eixo: list(zip(*valores)) for eixo, valores in estatisticas.items()}

        # Gerar o gráfico
        fig = plotar_estatisticas_combinadas(r_vals, "BOS_0605_CP_CV_01.txt", estatisticas)

        # Embutir o gráfico na interface
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        canvas.draw()

    root.mainloop()

# Iniciar a interface gráfica
abrir_interface()
