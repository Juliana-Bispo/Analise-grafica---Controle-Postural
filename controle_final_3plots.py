import pandas as pd
import plotly.express as px
import numpy as np
import os
from pathlib import Path

# Função para plotar a matriz em gráficos diferentes com zoom interativo e salvar os gráficos
def plotar_matriz_interativa(tabela, output_dir, nome_arquivo):
    # Define as colunas para cada gráfico
    colunas = [
        [1, 2, 3],  # Gráfico 1: Colunas 1, 2 e 3
        [4, 5, 6],  # Gráfico 2: Colunas 4, 5 e 6
        [7, 8, 9]   # Gráfico 3: Colunas 7, 8 e 9
    ]
    
    nomes_linhas = [
        ['X(m/s²)', 'Y(m/s²)', 'Z(m/s²)'],  # Nomes para o gráfico 1
        ['X(°/s)', 'Y(°/s)', 'Z(°/s)'],     # Nomes para o gráfico 2
        ['Roll(°)', 'Pitch(°)', 'Yaw(°)']   # Nomes para o gráfico 3
    ]
    
    y_labels = [
        'Aceleração (m/s²)',         # Nome do eixo y para o gráfico 1
        'Velocidade Angular (°/s)',  # Nome do eixo y para o gráfico 2
        'Rotação (°)'                # Nome do eixo y para o gráfico 3
    ]    
    
    nome_arquivo_sem_extensao = os.path.splitext(os.path.basename(nome_arquivo))[0]

    # Itera sobre cada conjunto de colunas e nomes de linhas para criar os gráficos
    for i, (cols, nomes) in enumerate(zip(colunas, nomes_linhas)):
        # Verifica se a coluna existe no DataFrame
        if all(tabela.columns[col] in tabela.columns for col in cols):
            # Cria um gráfico de linha para as colunas especificadas
            fig = px.line(tabela, x=tabela.columns[0], y=[tabela.columns[col] for col in cols], 
                          labels={'x': 'Tempo(s)', 'value': y_labels[i]}, 
                          title=f'Gráfico {i + 1}, {nome_arquivo_sem_extensao}, Frequência: 100Hz')
            
            # Define o nome da linha no gráfico
            for j, nome in enumerate(nomes):
                fig.data[j].name = nome  # Define o nome da linha no gráfico
            
            # Atualiza o layout do gráfico para garantir que o eixo y fique normal sem transformações
            fig.update_layout(
                xaxis_title='Tempo (s)', 
                yaxis_title=y_labels[i],
                yaxis=dict(type='linear'),  # Certifica-se de que o eixo y é linear
                width=1200, 
                height=600
            )
            
            # Salva o gráfico na pasta de saída
            output_file = os.path.join(output_dir, f'Grafico_{i + 1}.html')
            fig.write_html(output_file)
            print(f'Gráfico {i + 1} salvo em {output_file}')
        else:
            print(f"Colunas {cols} não encontradas no DataFrame.")

# Função principal para ler todos os arquivos e gerar os gráficos
def processar_arquivos_na_pasta(pasta):
    # Obtém a lista de todos os arquivos .txt na pasta
    arquivos = [f for f in os.listdir(pasta) if f.endswith('.txt')]
    
    for arquivo in arquivos:
        caminho_arquivo = os.path.join(pasta, arquivo)
        # Lê o arquivo .txt, usa o tab como quebra, ignora as 14 linhas de cabeçalho e identifica a ',' como separaçao de decimais
        tabela = pd.read_csv(caminho_arquivo, sep="\t", header=13, decimal=',')
        
        # Cria uma nova pasta para salvar os gráficos, com o nome do arquivo (sem extensão)
        nome_pasta_saida = os.path.join(pasta, Path(arquivo).stem)
        os.makedirs(nome_pasta_saida, exist_ok=True)
        
        # Plota e salva os gráficos
        plotar_matriz_interativa(tabela, nome_pasta_saida, arquivo)

# Define o caminho da pasta onde estão os arquivos
pasta_dos_arquivos = r"C:\Users\julia\controle-postural"

# Chama a função para processar todos os arquivos na pasta
processar_arquivos_na_pasta(pasta_dos_arquivos)
