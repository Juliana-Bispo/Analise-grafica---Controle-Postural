# Análise de Controle Postural

## Introdução
Este readme tem o objetido de explicar prte a parte o código escrito em Python, o código explicado esta presente no  arquivo [controle e MSE](https://github.com/Juliana-Bispo/Analise-grafica---Controle-Postural/blob/main/controle%20e%20MSE) Este código de Python é projetado para análise avançada de dados de controle postural, focando no cálculo de entropia amostral (SampEn) e entropia multiescala (MSE) em diferentes eixos de movimento.

## Bibliotecas Principais e Importações
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.express as px
from pathlib import Path
```
O trecho importa bibliotecas essenciais para:
- Computações numéricas (NumPy)
- Manipulação de dados (Pandas)
- Plotagem de gráficos (Matplotlib, Plotly)
- Operações de arquivo (OS)
- Interface gráfica do usuário (Tkinter)

## Função de Carregamento de Dados
```python
def carregar_dados(caminho_arquivo, skip_rows=15, sep="\t", decimal=','):
    try:
        return pd.read_csv(caminho_arquivo, skiprows=skip_rows, sep=sep, decimal=decimal)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None
```
- Carrega dados de um arquivo CSV
- Permite personalização de:
  - Pular linhas iniciais
  - Tipo de separador
  - Separador decimal
- Trata erros de arquivo não encontrado

## Cálculo da Sample Entropy (SampEn)
```python
def sample_entropy(time_series, m, r):
    n = len(time_series)

    def _phi(m):
        # Cálculo complexo de similaridade de padrões
        x = np.array([time_series[i:i + m] for i in range(n - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (n - m + 1) / (n - m)

    phi_m_plus_1 = _phi(m + 1)
    phi_m = _phi(m)

    return -np.log(phi_m_plus_1 / phi_m)
```
- Calcula a Entropia Amostral (SampEn)
- Mede a complexidade e previsibilidade de uma série temporal
- Parâmetros:
  - `m`: Dimensão de incorporação
  - `r`: Tolerância para correspondência de padrões

## Função de Multiscale Entropy
```python
def multiscale_entropy(data, scale_max, m=2, r=0.2):
    def coarse_graining(data, scale):
        # Reduz a resolução dos dados pela média
        n = len(data)
        b = n // scale
        return np.mean(data[:b * scale].reshape((b, scale)), axis=1)

    entropies = []
    for scale in range(1, scale_max + 1):
        coarse_data = coarse_graining(data, scale)
        sampen = sample_entropy(coarse_data, m=m, r=r * np.std(coarse_data))
        entropies.append(sampen)

    return entropies
```
- Computa a Entropia Multiescala (MSE)
- Analisa a complexidade dos dados em diferentes escalas de tempo
- Ajuda a entender a dinâmica do sistema em múltiplas resoluções

## Cálculos Estatísticos
```python
def calcular_estatisticas_sampen(mse_x, mse_y):
    sampen_xy = np.array(mse_x) + np.array(mse_y)

    mediana_sampen_xy = np.median(sampen_xy)
    erro_padrao_sampen_xy = np.std(sampen_xy, ddof=1) / np.sqrt(len(sampen_xy))
    variancia_sampen_xy = np.var(sampen_xy)
    erro_normalizado_sampen_xy = erro_padrao_sampen_xy / mediana_sampen_xy

    return (mediana_sampen_xy, erro_padrao_sampen_xy, variancia_sampen_xy, erro_normalizado_sampen_xy)
```
- Calcula métricas estatísticas para Entropia Amostral
- Computa:
  - Mediana
  - Erro Padrão
  - Variância
  - Erro Normalizado

## Funções de Visualização
O script inclui duas funções principais de visualização:
1. `plotar_estatisticas_separadas()`: Cria gráficos estatísticos para eixos {X, Y, Z} separadamente;
2. `plotar_matriz_interativa()`: Gera gráficos interativos HTML usando Plotly

## Interface Tkinter
```python
def abrir_interface():
    root = tk.Tk()
    root.title("Análise de Entropia Amostral")
    
    # Cria uma interface gráfica com:
    # - Texto explicativo rolável
    # - Carregamento de dados
    # - Cálculos de entropia
    # - Plotagem estatística
```
- Fornece uma interface gráfica do usuário
- Exibe texto explicativo
- Carrega e processa dados de controle postural
- Gera e exibe gráficos de análise de entropia

## Execução Principal
```python
abrir_interface()
```
- Inicia a interface Tkinter para começar a análise

## Propósito
Este script é projetado para análise científica de dados de controle postural, especificamente:
- Calculando entropia amostral e multiescala
- Visualizando a complexidade dos dados em diferentes eixos
- Fornecendo visualização interativa e explicação

**Nota**: O script é adaptado para um contexto de pesquisa específico, provavelmente em biomecânica ou ciências do movimento.
