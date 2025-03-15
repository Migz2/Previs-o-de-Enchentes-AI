# Previsão de Enchentes com Inteligência Artificial

Sistema de previsão do nível de rios utilizando aprendizado de máquina para auxiliar na prevenção de enchentes.

## Sobre o Projeto

Este projeto utiliza técnicas de aprendizado de máquina para prever o nível de rios com base em dados meteorológicos e históricos. O sistema analisa variáveis como:

- Taxa de chuva (mm/h)
- Chuva acumulada no dia (mm)
- Temperatura (ºC)
- Umidade do ar
- Velocidade do vento

O modelo implementado é baseado em Random Forest Regressor e inclui duas abordagens:
1. Modelo padrão (main.py)
2. Modelo com séries temporais (main_ts.py)

## Requisitos

- Python 3.8+
- Bibliotecas Python listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/Previsao-de-Enchentes-AI.git
cd Previsao-de-Enchentes-AI
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
# No Windows
.venv\Scripts\activate
# No macOS/Linux
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Executando o modelo padrão
```bash
python main.py
```

### Executando o modelo com séries temporais
```bash
python main_ts.py
```

## Estrutura do Projeto

- `main.py`: Implementação do modelo de previsão usando Random Forest
- `main_ts.py`: Implementação do modelo usando séries temporais
- `preparar_dados.py`: Script para preparação e limpeza dos dados
- `dados_atualizados.csv`: Conjunto de dados principal
- `dados_atualizados.xlsx`: Versão Excel do conjunto de dados
- `api/`: Diretório contendo arquivos de dados da API meteorológica
- `pdf/`: Documentação e relatórios

## Funcionamento

O sistema funciona em três etapas principais:

1. **Preparação dos dados**: Carregamento, limpeza e formatação dos dados históricos
2. **Treinamento do modelo**: Utilização de algoritmos de aprendizado de máquina para identificar padrões
3. **Previsão**: Geração de previsões do nível do rio com base em novos dados meteorológicos

## Métricas de Avaliação

O desempenho do modelo é avaliado usando:
- MAE (Erro Médio Absoluto)
- MSE (Erro Quadrático Médio)
- RMSE (Raiz do Erro Quadrático Médio)

## Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request
