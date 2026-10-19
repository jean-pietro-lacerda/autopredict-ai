# AutoPredict AI — CAN Bus, Telemetria Industrial e Manutenção Preditiva

Projeto de integração entre **MATLAB/Simulink**, **CAN Bus (Controller Area Network)** e **Machine Learning em Python**, utilizando as bibliotecas **scikit-learn, Pandas, NumPy e Joblib** para monitoramento de condições operacionais, estimativa de falhas e detecção de anomalias.

O sistema utiliza dados do **AI4I 2020 Predictive Maintenance Dataset** para treinar dois modelos de inteligência artificial: Random Forest e Isolation Forest.

Posteriormente, os modelos são utilizados para analisar dados de telemetria gerados por uma simulação simplificada de um motor elétrico no MATLAB/Simulink.

O objetivo principal é demonstrar a aplicação prática de técnicas de **aprendizado supervisionado e não supervisionado**, integrando preparação de dados, engenharia de atributos, treinamento, avaliação, persistência dos modelos e inferência sobre dados externos.

> **Simulink gera os sinais → CAN organiza a telemetria → Python processa os dados → scikit-learn realiza as previsões → sistema apresenta os resultados e avisos.**

---

# 1. Visão Geral

O **AutoPredict AI** foi desenvolvido como uma demonstração técnica de um pipeline de *Condition Monitoring* e *Predictive Maintenance*.

O projeto integra três componentes principais:

1. **MATLAB/Simulink:** geração de sinais sintéticos que representam condições operacionais de um motor elétrico.
2. **CAN Bus:** organização dos sinais em mensagens de telemetria.
3. **Python + scikit-learn:** preparação dos dados, treinamento dos modelos, previsão de falhas e detecção de anomalias.

O funcionamento é dividido em duas etapas:

- **Treinamento offline:** utilização do dataset AI4I 2020 para treinamento dos modelos.
- **Inferência:** análise dos sinais provenientes do Simulink utilizando modelos previamente treinados.

## Fluxo Geral

**Simulação:**

`MATLAB/Simulink → Sensores → CAN Bus → Arquivo .mat`

**Processamento:**

`Arquivo .mat → Python → Validação → Engenharia de Atributos`

**Inteligência Artificial:**

`scikit-learn → Random Forest + Isolation Forest → Resultados`

**Observação:** o Simulink implementa uma aproximação simplificada de sinais operacionais, e não uma simulação eletromecânica completa de um motor real.

---

# 2. Arquitetura do Sistema

## 2.1. Treinamento Offline

Os modelos de Machine Learning são treinados utilizando o **AI4I 2020 Predictive Maintenance Dataset**.

O pipeline segue as etapas:

1. Carregamento do CSV original.
2. Validação e preparação dos dados.
3. Padronização das colunas.
4. Engenharia de atributos.
5. Separação em treinamento e teste.
6. Treinamento do Random Forest e Isolation Forest.
7. Avaliação dos modelos.
8. Persistência dos modelos treinados.

Os modelos são armazenados no diretório `models/`, com gerenciamento realizado pelo módulo `src/model_store.py`.

O sistema mantém compatibilidade com os arquivos:

- `random_forest_model.pkl`
- `isolation_forest_model.pkl`

A persistência foi aprimorada para publicar os modelos como um conjunto consistente, reduzindo o risco de utilizar arquivos de versões incompatíveis.

## 2.2. Inferência sobre Telemetria Simulada

Depois do treinamento, os modelos podem ser reutilizados sem necessidade de executar um novo treinamento.

O Simulink gera os sinais dos sensores, organiza a telemetria em uma mensagem CAN e exporta os dados para arquivos `.mat`.

O módulo `src/simulink_bridge.py` realiza a leitura, validação e processamento das amostras.

Quando os arquivos MAT estão no formato v7.3, utiliza-se a biblioteca `h5py` para leitura da estrutura HDF5.

O sistema realiza três análises diferentes:

- **Random Forest:** estimativa da probabilidade de falha e classificação binária.
- **Isolation Forest:** identificação de observações estatisticamente anômalas.
- **Validação complementar:** identificação de sensores fora das faixas observadas no treinamento.

Essas três informações são apresentadas separadamente.

> O arquivo `.mat` não representa o protocolo CAN. Ele é utilizado para armazenar a telemetria exportada pelo Simulink.

---

# 3. Stack Tecnológica

| Tecnologia | Utilização |
|---|---|
| Python | Linguagem principal |
| MATLAB / Simulink | Simulação e geração de telemetria |
| Vehicle Network Toolbox | Blocos de comunicação CAN |
| scikit-learn | Treinamento e inferência dos modelos |
| Pandas | Manipulação e preparação dos dados |
| NumPy | Operações numéricas |
| Joblib | Serialização dos modelos |
| h5py | Leitura de arquivos MAT baseados em HDF5 |
| unittest | Testes automatizados |
| Git | Controle de versão |
| AI4I 2020 | Dataset de treinamento |

As versões das dependências Python são controladas pelo arquivo `requirements.txt`.

---

# 4. Dataset — AI4I 2020

O projeto utiliza o **AI4I 2020 Predictive Maintenance Dataset**, disponibilizado pelo UCI Machine Learning Repository.

O dataset contém **10.000 registros sintéticos**, representando condições operacionais de equipamentos industriais de usinagem.

As principais variáveis utilizadas pelos modelos são:

| Variável | Descrição | Unidade |
|---|---|---|
| Rotational speed | Velocidade de rotação | RPM |
| Process temperature | Temperatura do processo | K |
| Air temperature | Temperatura ambiente | K |
| Torque | Torque aplicado | Nm |
| Tool wear | Desgaste da ferramenta | min |

O dataset também contém a variável-alvo `machine_failure`, utilizada no treinamento supervisionado do Random Forest.

## 4.1. Distribuição das Classes

| Classe | Quantidade |
|---|---:|
| Sem falha | 9.661 |
| Com falha | 339 |
| **Total** | **10.000** |

O dataset é desbalanceado, com aproximadamente **3,39% dos registros apresentando falha**.

Isso torna importante avaliar métricas como precisão, recall e ROC-AUC, sem depender exclusivamente da acurácia.

## 4.2. Divisão dos Dados

O projeto utiliza uma divisão estratificada:

- **80% para treinamento:** 8.000 registros.
- **20% para teste:** 2.000 registros.

A estratificação ajuda a preservar a proporção das classes entre os conjuntos.

**Fonte:** [AI4I 2020 — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

---

# 5. Pré-processamento

O módulo responsável pelo carregamento e preparação dos dados é `src/data_loader.py`.

Suas principais responsabilidades incluem:

1. Carregar o dataset original.
2. Padronizar os nomes das colunas.
3. Validar a presença e os tipos dos dados necessários.
4. Preparar a estrutura utilizada pelos modelos.
5. Permitir a geração do dataset processado.

Os identificadores `UDI` e `Product ID` não são utilizados como atributos preditivos.

O dataset processado é armazenado em `data/processed/ai4i2020_processed.csv`.

## 5.1. Validação dos Dados

O projeto possui verificações para impedir que valores inválidos sejam utilizados indevidamente.

Entre as verificações implementadas estão:

- Presença das colunas obrigatórias.
- Tipos de dados compatíveis.
- Valores numéricos finitos.
- Rejeição de valores ausentes em campos obrigatórios.
- Estruturas incompatíveis.
- Validação dos rótulos utilizados no treinamento.

Isso reduz o risco de dados incorretos produzirem previsões aparentemente válidas.

---

# 6. Engenharia de Atributos

A engenharia de atributos é implementada no módulo `src/features.py`.

Além dos cinco sensores originais, o projeto calcula quatro atributos derivados, totalizando **nove atributos utilizados na inferência**.

## 6.1. Diferença de Temperatura

Calcula a diferença entre a temperatura do processo e a temperatura ambiente.

**Fórmula:**

`ΔT = T_processo − T_ambiente`

Esse atributo permite avaliar a relação térmica entre o processo e o ambiente.

## 6.2. Torque × Rotação

**Fórmula:**

`A1 = Torque × RPM`

Essa combinação relaciona o torque com a velocidade de rotação.

Ela é proporcional à potência mecânica, mas **não representa diretamente potência em watts**, pois ainda não inclui a conversão de RPM para velocidade angular.

A potência mecânica em watts é calculada por:

**Fórmula:**

`P = Torque × (2 × π × RPM) / 60`

Onde:

- **P:** potência mecânica, em watts (W).
- **Torque:** torque, em newton-metro (Nm).
- **RPM:** velocidade de rotação, em rotações por minuto.
- **π:** constante matemática pi, aproximadamente 3,14159.

## 6.3. Torque / Rotação

**Fórmula:**

`A2 = Torque / RPM`

Esse atributo representa uma relação entre o torque aplicado e a velocidade de rotação.

## 6.4. Desgaste × Torque × Rotação

**Fórmula:**

`A3 = ToolWear × Torque × RPM`

Combina o desgaste acumulado com as condições mecânicas de operação.

## 6.5. Resumo dos Atributos

| Tipo | Atributos |
|---|---|
| Originais | RPM, ProcessTemp, AirTemp, Torque e ToolWear |
| Derivados | Diferença térmica, Torque × RPM, Torque / RPM e ToolWear × Torque × RPM |

As mesmas transformações são aplicadas aos dados de treinamento e aos dados recebidos do Simulink.

---

# 7. Modelos de Inteligência Artificial

O projeto utiliza dois algoritmos do scikit-learn.

## 7.1. Random Forest Classifier

O **Random Forest** é um algoritmo supervisionado composto por múltiplas árvores de decisão.

Seu objetivo é aprender relações entre os atributos operacionais e a ocorrência de falhas presentes no dataset.

O modelo retorna:

- Probabilidade estimada de falha.
- Classificação binária de falha.

A classificação utiliza o limiar de **50%**:

| Probabilidade estimada | Classificação |
|---|---|
| Menor que 0,50 | Sem falha prevista (`0`) |
| Maior ou igual a 0,50 | Falha prevista (`1`) |

O modelo não identifica diretamente qual subtipo de falha ocorreu. Sua previsão representa a classe geral `machine_failure`.

### Resultados no Conjunto de Teste

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0,9703** |
| Average Precision | 0,8198 |
| Precisão da classe falha | 85,94% |
| Recall da classe falha | 80,88% |
| Verdadeiros positivos | 55 |
| Falsos negativos | 13 |
| Falsos positivos | 9 |
| Verdadeiros negativos | 1.923 |

A ROC-AUC indica boa capacidade discriminativa no conjunto de teste utilizado.

Entretanto, esses resultados não garantem o mesmo desempenho em equipamentos físicos reais ou em condições diferentes das observadas durante o treinamento.

## 7.2. Isolation Forest

O **Isolation Forest** é um algoritmo não supervisionado utilizado para detecção de anomalias.

Diferentemente do Random Forest, não utiliza os rótulos de falha para aprender diretamente uma classificação supervisionada.

Ele identifica observações que apresentam características estatisticamente diferentes das observadas no treinamento.

No projeto, as saídas incluem:

- Indicador de decisão do Isolation Forest.
- Classificação de anomalia.

Um indicador de decisão negativo representa uma observação classificada como anômala pelo limiar configurado.

### Diferença entre os Modelos

| Random Forest | Isolation Forest |
|---|---|
| Supervisionado | Não supervisionado |
| Aprende com rótulos de falha | Aprende padrões estatísticos |
| Estima probabilidade de falha | Identifica observações anômalas |
| Classificação binária de falha | Classificação de anomalia |

**Importante:** uma amostra pode ser classificada como falha pelo Random Forest sem ser considerada anômala pelo Isolation Forest.

Essa situação foi observada nos testes com Simulink.

---

# 8. Simulação no MATLAB/Simulink

A simulação utiliza uma representação simplificada de variáveis operacionais de um motor elétrico.

O objetivo não é reproduzir toda a dinâmica eletromecânica de uma máquina real, mas gerar sinais com relações interpretáveis para testar a aplicação de Machine Learning.

Os cinco sensores gerados são:

| Sensor | Significado |
|---|---|
| RPM | Velocidade de rotação |
| ProcessTemp | Temperatura do processo |
| Torque | Torque mecânico |
| AirTemp | Temperatura ambiente |
| ToolWear | Desgaste acumulado |

Os dados são enviados para os blocos responsáveis pela organização da telemetria CAN e posteriormente exportados para arquivos MAT.

---

# 9. Cenário Normal — Configuração Final

O cenário normal utiliza as seguintes configurações.

## 9.1. Rotação do Motor

A rotação é gerada por um bloco **Repeating Sequence**.

Configuração:

| Parâmetro | Valor |
|---|---|
| Time values | `[0 10 20 30 40]` |
| Output values | `[1300 1600 1450 1750 1350]` |

O sinal varia continuamente entre os valores configurados e repete sua sequência.

A rotação representa uma condição operacional variável do motor.

## 9.2. Temperatura do Processo

A temperatura do processo é calculada a partir de uma constante e de uma componente dinâmica dependente da rotação.

| Parâmetro | Valor |
|---|---|
| Constant | `308` |
| Gain | `0.001` |
| Transfer Function | `1/(10s+1)` |

**Expressão simplificada:**

`T_processo(t) = 308 + y(t)`

Onde `y(t)` representa a resposta do filtro de primeira ordem ao sinal de RPM multiplicado pelo ganho `0.001`.

Na telemetria final, a temperatura do processo variou aproximadamente entre **308 K e 310 K**.

## 9.3. Torque

O torque é calculado considerando uma aproximação de potência mecânica constante.

A relação utilizada é:

`P = Torque × (2 × π × RPM) / 60`

Isolando o torque:

`Torque = (P × 60) / (2 × π × RPM)`

Para uma potência experimental de aproximadamente 7.000 W:

`Torque ≈ 66845 / RPM`

No Simulink, essa relação foi implementada utilizando um bloco `Fcn` com a expressão:

`66845/u`

A configuração produz uma relação inversa entre torque e rotação.

Conforme a rotação aumenta, o torque diminui.

A potência mecânica estimada permaneceu próxima de **7 kW** na simulação testada.

**Essa potência foi adotada como hipótese experimental**, não como característica nominal validada de um motor real.

## 9.4. Temperatura Ambiente

A temperatura ambiente permanece constante:

`AirTemp = 300 K`

## 9.5. Desgaste da Ferramenta

O desgaste é representado por um bloco `Ramp`.

| Parâmetro | Valor |
|---|---|
| Slope | `1` |
| Start time | `0` |
| Initial output | `0` |

O sinal aumenta progressivamente ao longo da simulação.

Na execução de 100 segundos, os valores exportados variaram de **0 a 100**.

No AI4I, o desgaste é expresso em minutos. Na simulação, o crescimento foi acelerado artificialmente, e essa correspondência de escala deve ser interpretada como uma simplificação experimental.

---

# 10. Cenário com Perturbação Térmica

O cenário com perturbação foi construído a partir do cenário normal com torque inversamente proporcional à rotação.

A diferença está no sinal de temperatura do processo.

Foi adicionado um bloco `Step` para produzir um aumento controlado de temperatura.

## 10.1. Configuração do Step

| Parâmetro | Valor |
|---|---|
| Step time | `50` |
| Initial value | `0` |
| Final value | `15` |
| Sample time | `0` |

A saída do Step é somada à temperatura normal do processo.

### Comportamento da temperatura

| Intervalo da simulação | Temperatura |
|---|---|
| Antes de 50 segundos | Temperatura normal |
| A partir de 50 segundos | Temperatura normal + 15 K |

Assim:

- Antes de 50 segundos, os sensores representam o cenário normal.
- A partir de 50 segundos, a temperatura do processo sofre um aumento de 15 K.
- Os demais sensores continuam seguindo a configuração original.

A temperatura passou aproximadamente de **309 K para 324 K** no instante da transição.

Essa alteração representa uma **perturbação sintética controlada**, utilizada para avaliar a resposta dos modelos de inteligência artificial.

Não representa necessariamente uma falha física real do motor.

---

# 11. Mapeamento da Mensagem CAN

A telemetria é organizada em uma mensagem CAN padrão.

| Parâmetro | Valor |
|---|---|
| Standard ID | `0x1A0` |
| Decimal ID | `416` |
| Payload | `8 bytes` |
| Nome da mensagem | `Engine data` |

## 11.1. Estrutura do Frame

| Sinal | Bit inicial | Tamanho | Tipo | Unidade |
|---|---:|---:|---|---|
| RPM | 0 | 16 bits | unsigned | RPM |
| ProcessTemp | 16 | 12 bits | unsigned | K |
| Torque | 28 | 12 bits | unsigned | Nm |
| AirTemp | 40 | 12 bits | unsigned | K |
| ToolWear | 52 | 12 bits | unsigned | min |

A distribuição utiliza os 64 bits disponíveis no payload de 8 bytes.

A representação inteira dos sinais pode provocar arredondamentos durante a exportação.

A ordem dos sensores utilizada na telemetria é:

1. RPM
2. ProcessTemp
3. Torque
4. AirTemp
5. ToolWear

O arquivo MAT exportado contém também a informação temporal associada às amostras.

---

# 12. Arquivos de Telemetria

Após os ajustes realizados no Simulink, foram definidos dois arquivos principais de avaliação.

## 12.1. Cenário Normal

**Arquivo:** `telemetry_normal_torque_teste.mat`

Contém a simulação normal com temperatura corrigida e torque inversamente proporcional à rotação.

## 12.2. Cenário com Perturbação

**Arquivo:** `telemetry_fault_corrigido.mat`

Contém a mesma configuração normal, com aumento de 15 K na temperatura de processo a partir de 50 segundos.

Os arquivos são processados pelo módulo `src/simulink_bridge.py`.

A leitura verifica a estrutura, as dimensões, os tipos dos sinais e sua sequência temporal.

---

# 13. Resultados da Integração Simulink–Python

Os dois cenários foram analisados utilizando os mesmos modelos Random Forest e Isolation Forest.

Nenhum retreinamento foi realizado para produzir os resultados desta comparação.

## 13.1. Resultados Gerais

| Indicador | Normal | Perturbação térmica |
|---|---:|---:|
| Amostras analisadas | 58 | 60 |
| Falhas previstas pelo Random Forest | 9 | 9 |
| Anomalias detectadas pelo Isolation Forest | 0 | 0 |
| Probabilidade média de falha | 8,62% | 9,18% |
| Probabilidade máxima de falha | 54% | 54% |
| Avisos de sensores fora do treinamento | 0 | 27 |

## 13.2. Previsões de Falha na Partida

Nos dois cenários, o Random Forest identificou nove amostras como falha no início da simulação.

As classificações ocorreram entre **t = 0 s e t ≈ 2,604 s**.

Nesse período, os sensores apresentavam aproximadamente:

| Sensor | Valor |
|---|---|
| RPM | 1.300–1.378 |
| ProcessTemp | 308 K |
| AirTemp | 300 K |
| Torque | 49–51 Nm |

A diferença térmica era:

`ΔT = 308 − 300 = 8 K`

Essa combinação está relacionada a uma condição sintética de falha por dissipação de calor presente no dataset AI4I.

A regra descrita no dataset considera duas condições simultâneas:

- Diferença térmica menor que **8,6 K**.
- Rotação menor que **1.380 RPM**.

O Random Forest aprendeu associações entre essa condição e os rótulos de falha do treinamento.

As nove classificações representam amostras sinalizadas, não nove eventos mecânicos independentes.

O Isolation Forest não classificou essas observações como anômalas.

## 13.3. Resposta à Perturbação de Temperatura

Aos 50 segundos, o Simulink introduziu um aumento de 15 K na temperatura do processo.

A mudança foi recebida corretamente pelo Python e incorporada aos atributos utilizados pelos modelos.

### Comparação no Instante da Alteração

| Indicador | Antes | Depois |
|---|---:|---:|
| Temperatura do processo | 309 K | 324 K |
| Temperatura ambiente | 300 K | 300 K |
| Diferença térmica | 9 K | 24 K |
| Probabilidade de falha RF | 0% | 2% |
| Indicador IF | +0,13584 | +0,05247 |

O Random Forest aumentou a probabilidade estimada, porém permaneceu abaixo do limiar de classificação de 50%.

O Isolation Forest também apresentou mudança em seu indicador, mas permaneceu acima do limiar utilizado para identificar anomalias.

**Nenhum dos dois modelos produziu novos alertas após a perturbação térmica.**

Esse resultado evidencia que valores fora das faixas do treinamento não são necessariamente classificados como falhas ou anomalias.

---

# 14. Verificação Complementar dos Sensores

Após a avaliação dos modelos, foi implementada uma verificação adicional para identificar sensores cujos valores estejam fora das faixas observadas no treinamento.

Essa funcionalidade utiliza os **8.000 registros do conjunto de treinamento** como referência.

Ela é independente dos modelos de Machine Learning.

## 14.1. Exemplo de Aviso

**Amostra 33**

| Campo | Valor |
|---|---|
| Sensor | `process_temperature_k` |
| Valor recebido | 324 K |
| Limite inferior no treinamento | 305,7 K |
| Limite superior no treinamento | 313,8 K |

No cenário com perturbação térmica, foram identificadas **27 amostras com temperatura do processo fora da faixa do treinamento**.

Nos cenários normais corrigidos, não foram encontrados avisos desse tipo.

## 14.2. Diferença entre os Três Resultados

| Resultado | Significado |
|---|---|
| Falha prevista pelo Random Forest | Classificação supervisionada de falha |
| Anomalia do Isolation Forest | Classificação de comportamento estatisticamente anômalo |
| Sensor fora da faixa de treinamento | Aviso de que o valor recebido ultrapassa o intervalo observado nos dados de treinamento |

O aviso de faixa não significa necessariamente que o equipamento esteja apresentando uma falha física.

Ele informa que o modelo está recebendo valores diferentes daqueles observados durante o treinamento, o que pode afetar a confiabilidade de sua aplicação.

---

# 15. Estrutura do Repositório

A organização principal do projeto é:

| Diretório / Arquivo | Responsabilidade |
|---|---|
| `README.md` | Documentação |
| `main.py` | Pipeline principal |
| `requirements.txt` | Dependências |
| `telemetry_normal_torque_teste.mat` | Telemetria normal |
| `telemetry_fault_corrigido.mat` | Telemetria com perturbação |
| `data/raw/` | Dataset original |
| `data/processed/` | Dataset processado |
| `models/` | Modelos treinados |
| `src/` | Código-fonte da aplicação |
| `tests/` | Testes automatizados |

## 15.1. Principais Módulos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestra o pipeline principal |
| `src/data_loader.py` | Carregamento e preparação dos dados |
| `src/features.py` | Engenharia de atributos |
| `src/train.py` | Treinamento e avaliação dos modelos |
| `src/predict.py` | Predição individual e em lote |
| `src/simulink_bridge.py` | Integração dos arquivos MAT com os modelos |
| `src/validation.py` | Validação dos dados e contratos de entrada |
| `src/model_store.py` | Persistência e carregamento seguro dos modelos |
| `tests/` | Testes automatizados |

A pasta `models/` contém os artefatos dos modelos treinados, incluindo suporte aos arquivos `.pkl` utilizados anteriormente.

---

# 16. Como Executar

## 16.1. Pré-requisitos

Para executar o pipeline Python:

- Python compatível com as versões declaradas em `requirements.txt`.
- Ambiente virtual Python.
- Dependências instaladas.
- Dataset AI4I disponível na estrutura do projeto.

Para gerar novos arquivos de telemetria:

- MATLAB/Simulink.
- Bibliotecas necessárias para os blocos CAN utilizados no modelo.

O treinamento pode ser executado independentemente do Simulink.

## 16.2. Criar o Ambiente Virtual

No terminal, dentro da pasta do projeto, execute:

`python3 -m venv .venv`

Ative o ambiente:

`source .venv/bin/activate`

## 16.3. Instalar as Dependências

`python3 -m pip install -r requirements.txt`

## 16.4. Executar o Pipeline de Treinamento

O arquivo `main.py` executa o processamento dos dados e o treinamento dos modelos.

**Comando:**

`python3 main.py`

O pipeline realiza:

1. Carregamento dos dados.
2. Validação do dataset.
3. Engenharia de atributos.
4. Treinamento dos modelos.
5. Avaliação.
6. Persistência dos modelos.

Na execução validada, o Random Forest apresentou **ROC-AUC = 0,9703**.

**Importante:** executar `main.py` realiza um novo treinamento e pode atualizar os artefatos publicados.

Para analisar arquivos MAT com modelos já treinados, não é necessário executar novamente esse comando.

## 16.5. Executar o Simulink

Abra o modelo Simulink correspondente ao cenário desejado.

Para o cenário normal, execute o modelo com as configurações de rotação, temperatura e torque descritas anteriormente.

Para o cenário com perturbação, execute a versão que contém o bloco Step de aumento de temperatura.

Após executar a simulação, verifique se o arquivo `.mat` foi exportado corretamente.

## 16.6. Executar a Inferência

Com os modelos treinados e os arquivos de telemetria disponíveis, execute:

`python3 -m src.simulink_bridge`

O script realiza:

1. Carregamento dos arquivos MAT configurados.
2. Validação da estrutura dos dados.
3. Extração dos cinco sensores.
4. Criação dos atributos derivados.
5. Carregamento dos modelos previamente treinados.
6. Inferência em lote.
7. Contagem das previsões de falha.
8. Detecção de anomalias.
9. Identificação de sensores fora das faixas do treinamento.
10. Exibição dos resultados.

Os resultados são apresentados no terminal.

O Simulink e o Python não executam, nesta configuração, um processo de inferência online automático.

A simulação gera o arquivo MAT, que é posteriormente analisado pelo Python.

---

# 17. Testes Automatizados

O projeto utiliza testes automatizados com a biblioteca `unittest`.

Para executar os testes:

`python3 -m unittest discover -s tests -v`

Na validação final, foram executados **20 testes automatizados, todos aprovados**.

Os testes cobrem funcionalidades como:

- Imports e execução dos módulos.
- Validação dos sensores.
- Tratamento de entradas inválidas.
- Engenharia de atributos.
- Consistência da classificação no limiar de 0,5.
- Leitura dos arquivos MAT.
- Validação de CSVs.
- Inferência em lote.
- Publicação consistente dos modelos.
- Verificação de sensores fora da faixa.
- Comportamento diante de falhas de leitura ou gravação.

Também foram realizadas verificações das dependências e da execução dos módulos a partir de diferentes diretórios.

---

# 18. Melhorias de Confiabilidade

Durante a auditoria do projeto, foram identificados e corrigidos problemas relacionados à execução e à manutenção do sistema.

## 18.1. Imports

Os imports foram organizados para permitir a execução correta dos módulos Python como parte do pacote `src`.

## 18.2. Consistência dos Alertas

A contagem de falhas utiliza diretamente a classificação `failure_predicted`, evitando divergências provocadas por arredondamentos da probabilidade.

## 18.3. Validação dos Sensores

Foram implementadas verificações para rejeitar entradas incompatíveis antes da execução dos modelos.

## 18.4. Persistência dos Modelos

A publicação dos dois modelos foi aprimorada para manter um conjunto consistente de artefatos.

## 18.5. Caminhos

Os caminhos dos arquivos foram organizados para reduzir a dependência do diretório atual de execução.

## 18.6. Dependências

O arquivo `requirements.txt` foi atualizado com as bibliotecas necessárias, incluindo `h5py`.

## 18.7. Desempenho

A inferência foi otimizada para processar amostras em lote, evitando recarregar os modelos a cada registro.

## 18.8. Treinamento

Foram adicionadas verificações para garantir que os dados e rótulos sejam compatíveis com os algoritmos utilizados.

## 18.9. CSV e MAT

A leitura dos arquivos inclui verificações de estrutura, tipos, dimensões e sequência temporal.

## 18.10. Testes

Foi criada e ampliada a suíte de testes automatizados para reduzir o risco de regressões.

---

# 19. Limitações do Projeto

Apesar dos resultados positivos no dataset de teste e do funcionamento da integração, algumas limitações precisam ser consideradas.

## 19.1. Natureza do Dataset

O AI4I 2020 é um dataset sintético de manutenção preditiva aplicado ao contexto de máquinas de usinagem.

Seus padrões não representam necessariamente o comportamento físico de qualquer motor elétrico.

## 19.2. Simulação Simplificada

O Simulink utiliza sinais simplificados de rotação, torque, temperatura e desgaste.

Não são modelados todos os fenômenos eletromecânicos e térmicos de um motor real.

## 19.3. Generalização

Os bons resultados do Random Forest foram obtidos no conjunto de teste do AI4I.

Eles não garantem desempenho equivalente em equipamentos industriais reais.

## 19.4. Falhas Não Detectadas

O aumento artificial de 15 K na temperatura do processo não gerou novos alertas do Random Forest ou Isolation Forest.

Isso demonstra que os modelos não reconhecem necessariamente toda perturbação de sensor como uma falha ou anomalia.

## 19.5. Dados Fora do Treinamento

O sistema apresenta avisos quando recebe valores fora das faixas observadas no treinamento.

Esses avisos indicam uma possível condição de extrapolação, mas não são diagnósticos físicos de falha.

## 19.6. Interpretação Temporal

Os modelos avaliam os atributos de cada amostra.

Eles não foram desenvolvidos especificamente para prever o instante futuro de ocorrência de uma falha nem para modelar a evolução temporal completa dos sensores.

Portanto, o projeto deve ser interpretado como uma aplicação de classificação de falhas e monitoramento de condições operacionais, com potencial aplicação em manutenção preditiva.

---

# 20. Conclusão

O **AutoPredict AI** demonstra a integração de técnicas de Machine Learning com dados de telemetria gerados por simulação.

O projeto reúne:

- Preparação e validação de dados.
- Engenharia de atributos.
- Treinamento supervisionado com Random Forest.
- Detecção não supervisionada de anomalias com Isolation Forest.
- Avaliação de desempenho.
- Persistência e reutilização dos modelos.
- Simulação simplificada de sinais operacionais no MATLAB/Simulink.
- Organização da telemetria utilizando CAN Bus.
- Integração com Python por meio de arquivos MAT.
- Inferência em lote.
- Verificação de valores fora das faixas de treinamento.
- Testes automatizados.

O Random Forest apresentou **ROC-AUC de 0,9703** no conjunto de teste do AI4I.

Na integração com o Simulink, o sistema processou corretamente os dois cenários, identificando condições associadas a falha durante os primeiros instantes da simulação e detectando valores de temperatura fora da faixa de treinamento após uma perturbação controlada.

Embora a perturbação térmica não tenha provocado novos alertas dos modelos, esse resultado permitiu observar limitações importantes de generalização e reforçou a necessidade de validação dos dados recebidos.

O projeto demonstra, assim, uma aplicação acadêmica de **scikit-learn para monitoramento de condições industriais**, integrando dados, modelos de inteligência artificial, simulação e validação de resultados.

---

**AutoPredict AI — Projeto acadêmico de Machine Learning e Monitoramento Industrial.**
