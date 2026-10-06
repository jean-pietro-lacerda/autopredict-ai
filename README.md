# AutoPredict AI — CAN Bus, Telemetria Industrial e Manutenção Preditiva

Projeto de integração **fim a fim** entre a simulação física de um centro de usinagem no **MATLAB/Simulink**, a comunicação de telemetria através de **CAN Bus (Controller Area Network)** e um pipeline de **Inteligência Artificial para manutenção preditiva em Python**.

O sistema simula diferentes condições de operação de uma máquina industrial, organiza suas variáveis em mensagens CAN e utiliza modelos de Machine Learning para **estimar o risco de falha e detectar anomalias**.

> **Simulink simula a máquina → CAN organiza a telemetria → Python processa os dados → IA avalia o risco → sistema identifica possíveis falhas.**

---

## Visão Geral

O **AutoPredict AI** foi desenvolvido como uma demonstração técnica de um pipeline de **monitoramento de condições (*Condition Monitoring*)** e **manutenção preditiva (*Predictive Maintenance*)**.

O projeto integra três componentes principais:

1. **MATLAB/Simulink** — simulação da dinâmica física de um centro de usinagem.
2. **CAN Bus** — empacotamento das variáveis da máquina em uma mensagem de telemetria.
3. **Python + Machine Learning** — processamento dos dados, treinamento dos modelos e avaliação do risco de falha.

O fluxo geral do sistema é:

```text
┌─────────────────────────┐
│     MATLAB / Simulink   │
│                         │
│   Simulação física da   │
│        máquina          │
└────────────┬────────────┘
             │
             │ CAN Bus
             ▼
┌─────────────────────────┐
│       Telemetria        │
│                         │
│ RPM / Torque /           │
│ Temperaturas / Tool Wear│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│          Python         │
│                         │
│ Processamento dos dados │
│ + Engenharia de Atributos│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│       Modelos de IA     │
│                         │
│ Random Forest           │
│ Isolation Forest        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Avaliação de Risco   │
│                         │
│ Probabilidade de falha  │
│ + Detecção de anomalias │
└─────────────────────────┘
```

---

# Arquitetura do Sistema

O projeto possui duas etapas principais:

### 1. Pré-Treinamento Offline

Os modelos de Machine Learning são treinados previamente utilizando o **AI4I 2020 Predictive Maintenance Dataset**.

```text
Dataset AI4I 2020
       │
       ▼
Limpeza e preparação
       │
       ▼
Engenharia de atributos
       │
       ▼
Treinamento
       │
       ├──► Random Forest
       │
       └──► Isolation Forest
       │
       ▼
Modelos .pkl
```

Os modelos treinados são armazenados em:

```text
models/
├── random_forest_model.pkl
└── isolation_forest_model.pkl
```

### 2. Inferência sobre a Telemetria Simulada

Após o treinamento, os modelos são utilizados para analisar a telemetria produzida pela simulação.

O Simulink gera as variáveis físicas, organiza esses dados em uma mensagem CAN e posteriormente a telemetria é exportada para arquivos `.mat`.

Quando o arquivo está no formato **MAT v7.3**, sua estrutura utiliza **HDF5**, permitindo que o Python faça sua leitura utilizando `h5py`.

```text
Simulink
   │
   ▼
Variáveis físicas
   │
   ▼
CAN Pack
   │
   ▼
CAN Message
   │
   ▼
Arquivo .mat
   │
   ▼
simulink_bridge.py
   │
   ▼
Modelos pré-treinados
   │
   ├──► Random Forest
   │
   └──► Isolation Forest
   │
   ▼
Risco de falha + Anomalias
```

> **Importante:** o arquivo `.mat` não é o protocolo CAN. O CAN é utilizado para estruturar a mensagem de telemetria, enquanto o `.mat` é utilizado para armazenar os dados exportados da simulação.

---

# Stack Tecnológica

| Camada               | Tecnologia                 | Descrição                                          |
| -------------------- | -------------------------- | -------------------------------------------------- |
| Simulação Física     | MATLAB / Simulink R2024b+  | Modelagem dos sinais físicos e dinâmica da máquina |
| Comunicação CAN      | Vehicle Network Toolbox    | Empacotamento e desempacotamento das mensagens CAN |
| Ambiente de Execução | WSL2 / Ubuntu 22.04 LTS    | Ambiente Python e desenvolvimento                  |
| Dataset              | AI4I 2020 / UCI Repository | Dataset utilizado para treinamento                 |
| Processamento MAT    | `h5py`                     | Leitura de arquivos `.mat` v7.3 baseados em HDF5   |
| Manipulação de Dados | Pandas / NumPy             | Estruturação e processamento dos dados             |
| Machine Learning     | Scikit-Learn               | Treinamento dos modelos de IA                      |
| Serialização         | Joblib                     | Salvamento e carregamento dos modelos              |
| Controle de Versão   | Git                        | Versionamento do projeto                           |

---

# Pipeline de Machine Learning

## Dataset — AI4I 2020

O treinamento utiliza o **AI4I 2020 Predictive Maintenance Dataset**, desenvolvido para representar problemas de manutenção preditiva em máquinas industriais.

O dataset possui **10.000 registros operacionais**, contendo variáveis relacionadas às condições de funcionamento da máquina e registros de falha.

Entre as principais variáveis utilizadas estão:

* `Rotational speed [rpm]`
* `Process temperature [K]`
* `Air temperature [K]`
* `Torque [Nm]`
* `Tool wear [min]`

---

## Pré-processamento

O módulo:

```text
src/data_loader.py
```

é responsável pelo carregamento e preparação inicial do dataset.

As principais etapas são:

* leitura do CSV;
* padronização dos nomes das colunas;
* remoção de identificadores irrelevantes;
* organização dos dados;
* geração do dataset processado.

Os identificadores removidos incluem:

```text
UDI
Product ID
```

O resultado é salvo em:

```text
data/processed/ai4i2020_processed.csv
```

---

# Engenharia de Atributos

O módulo:

```text
src/features.py
```

é responsável pela criação de atributos derivados das variáveis físicas.

## Potência Mecânica

A potência mecânica é calculada a partir do torque e da velocidade angular:

```text
P = τ × ω
```

onde:

* `P` = potência mecânica;
* `τ` = torque;
* `ω` = velocidade angular.

Como o dado original de rotação é fornecido em RPM, é necessário realizar a conversão para velocidade angular.

---

## Diferença de Temperatura

Também é calculada a diferença entre a temperatura do processo e a temperatura do ar:

```text
ΔT = Process Temperature - Air Temperature
```

Essa variável fornece uma indicação da diferença térmica entre o processo e o ambiente.

---

# Modelos de Inteligência Artificial

O sistema utiliza dois modelos com funções diferentes e complementares.

## Random Forest

O **Random Forest Classifier** é um modelo de aprendizado supervisionado utilizado para estimar a **probabilidade de falha** da máquina.

O modelo aprende relações entre as variáveis operacionais e os registros de falha presentes no dataset de treinamento.

O modelo treinado é salvo em:

```text
models/random_forest_model.pkl
```

---

## Isolation Forest

O **Isolation Forest** é utilizado para **detecção não supervisionada de anomalias**.

Diferentemente do Random Forest, seu objetivo não é diretamente classificar uma falha específica. Ele procura identificar observações que apresentam comportamento significativamente diferente do padrão considerado normal.

O modelo treinado é salvo em:

```text
models/isolation_forest_model.pkl
```

---

# Simulação Física no Simulink

A camada física do projeto é implementada no **MATLAB/Simulink**.

A simulação gera as principais variáveis de telemetria:

```text
RPM
Process Temperature
Torque
Air Temperature
Tool Wear
```

Essas variáveis são posteriormente encaminhadas para o bloco responsável pela montagem da mensagem CAN.

---

# Geração das Variáveis no Cenário Normal

O modelo Simulink utiliza sinais dinâmicos e valores de referência para representar o funcionamento da máquina.

## Process Temperature

A temperatura do processo possui um valor de referência de:

```text
295 K
```

Esse valor é fornecido por um bloco `Constant` e somado a uma componente dinâmica.

A dinâmica passa por:

```text
Gain = 0.015
```

seguido por um sistema de primeira ordem:

```text
        1
     --------
      10s + 1
```

Portanto, **295 K é o valor base**, e não necessariamente o valor final constante durante toda a simulação.

Conceitualmente:

```text
ProcessTemp = 295 K + componente dinâmica
```

---

## Torque

O torque também possui um valor base definido no Simulink:

```text
15 Nm
```

Esse valor é somado a uma componente dinâmica proveniente do sinal de entrada.

O modelo utiliza:

```text
Gain = 0.02
```

Portanto:

```text
Torque = 15 Nm + componente dinâmica
```

Isso permite que o torque varie durante a simulação.

---

## RPM

O RPM é gerado dinamicamente a partir do sinal de entrada através do ganho utilizado no modelo.

Dessa forma, o RPM representa uma variável operacional que pode variar durante a simulação, em vez de permanecer constante.

---

## Air Temperature

A temperatura do ar é definida diretamente através de um bloco `Constant`:

```text
AirTemp = 300 K
```

Portanto, no cenário mostrado no Simulink, a temperatura ambiente utilizada como referência é **300 K**.

---

## Tool Wear

O desgaste da ferramenta é representado por um sinal crescente.

Diferentemente de uma variável constante, o `ToolWear` aumenta ao longo da simulação, representando o **acúmulo progressivo do desgaste da ferramenta de corte**.

Conceitualmente:

```text
ToolWear
   │
   │        /
   │      /
   │    /
   │  /
   │/
   └──────────────────► Tempo
```

Assim, no cenário normal, o Tool Wear **não deve ser descrito simplesmente como `0 min` durante toda a simulação**.

---

# Mapeamento da Mensagem CAN

A telemetria é organizada em uma mensagem CAN padrão.

```text
Standard ID: 0x1A0
Decimal ID:  416
Payload:     8 bytes
```

No modelo Simulink, o bloco de empacotamento utiliza:

```text
Message: Engine data
Standard ID: 416
```

As variáveis são organizadas dentro dos 64 bits disponíveis no payload.

## Estrutura do Frame

| Sinal         | Bit Inicial | Tamanho | Tipo     | Intervalo Representável | Unidade |
| ------------- | ----------: | ------: | -------- | ----------------------: | ------- |
| `RPM`         |           0 | 16 bits | unsigned |              0 – 65.535 | RPM     |
| `ProcessTemp` |          16 | 12 bits | unsigned |               0 – 4.095 | K       |
| `Torque`      |          28 | 12 bits | unsigned |               0 – 4.095 | Nm      |
| `AirTemp`     |          40 | 12 bits | unsigned |               0 – 4.095 | K       |
| `ToolWear`    |          52 | 12 bits | unsigned |               0 – 4.095 | min     |

---

# Fluxo da Telemetria no Simulink

No cenário normal, o fluxo pode ser resumido como:

```text
                  Sinais da simulação
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
         RPM       ProcessTemp        Torque
                         │
          ┌──────────────┼──────────────┐
          │                             │
          ▼                             ▼
      AirTemp                        ToolWear
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  CAN Pack   │
                  │             │
                  │ ID: 416     │
                  │ 8 bytes     │
                  └──────┬──────┘
                         │
                         ▼
                    CAN Message
                         │
                         ▼
                 Exportação .mat
                         │
                         ▼
                telemetry_normal.mat
```

---

# Cenários de Validação

O sistema é validado utilizando dois cenários diferentes:

```text
telemetry_normal.mat
telemetry_fault.mat
```

O primeiro representa uma condição operacional normal, enquanto o segundo introduz condições deliberadamente severas para verificar a capacidade do sistema de identificar o aumento do risco.

---

# Cenário 1 — Operação Normal

Arquivo:

```text
telemetry_normal.mat
```

O cenário representa uma condição de operação normal da máquina.

### Principais características do modelo

```text
Process Temperature:
295 K como valor base
+ componente dinâmica

Air Temperature:
300 K

Torque:
15 Nm como valor base
+ componente dinâmica

RPM:
variável dinâmica

Tool Wear:
sinal crescente de desgaste
```

O objetivo desse cenário é representar uma máquina operando dentro das condições esperadas, permitindo verificar se o sistema mantém o nível de risco abaixo do limiar definido.

### Resultado

```text
Alertas detectados: 0 / 60
Probabilidade máxima de falha: 0,49
Limiar de risco: 0,50
```

O modelo não classificou os registros como condições de alto risco.

---

# Cenário 2 — Falha Induzida

Arquivo:

```text
telemetry_fault.mat
```

Neste cenário, são introduzidas condições severas de operação para testar a capacidade do pipeline de identificar uma situação de risco.

### Sobrecarga mecânica

```text
Torque: 85 – 132 Nm
```

O aumento significativo do torque representa uma condição de sobrecarga mecânica.

---

### Estresse térmico

```text
Process Temperature: 320 – 359 K
```

A temperatura elevada representa uma condição de estresse térmico muito superior à condição normal.

---

### Desgaste elevado

```text
Tool Wear: 215 – 315 min
```

O aumento do desgaste representa uma condição crítica de utilização da ferramenta.

---

### Resultado

```text
Alertas detectados: 60 / 60

Probabilidade média de falha: 0,754

Probabilidade máxima: 0,84
```

O modelo classificou todos os registros do cenário como condições de risco elevado.

---

# Estrutura do Repositório

```text
autopredict-ai/
│
├── README.md
├── main.py
├── requirements.txt
│
├── telemetry_normal.mat
├── telemetry_fault.mat
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       └── ai4i2020_processed.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── isolation_forest_model.pkl
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── features.py
    ├── train.py
    ├── predict.py
    └── simulink_bridge.py
```

### Principais arquivos

| Arquivo                      | Função                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                    | Orquestra o pipeline de treinamento                                          |
| `data_loader.py`             | Carrega e prepara o dataset                                                  |
| `features.py`                | Executa a engenharia de atributos                                            |
| `train.py`                   | Treina e salva os modelos                                                    |
| `predict.py`                 | Executa inferência e cálculo do risco                                        |
| `simulink_bridge.py`         | Faz a integração entre os dados exportados pelo Simulink e os modelos Python |
| `random_forest_model.pkl`    | Modelo supervisionado                                                        |
| `isolation_forest_model.pkl` | Modelo de detecção de anomalias                                              |

---

# Como Executar

## Pré-requisitos

* Python **3.10+**
* WSL2
* Ubuntu 22.04 LTS
* MATLAB/Simulink **R2024b+**
* Vehicle Network Toolbox

---

## 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/autopredict-ai.git
cd autopredict-ai
```

---

## 2. Criar o Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Instalar as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Executar o Treinamento

O `main.py` executa o pipeline de treinamento:

```text
Dataset bruto
     ↓
Pré-processamento
     ↓
Engenharia de atributos
     ↓
Treinamento
     ↓
Modelos .pkl
```

Execute:

```bash
python3 main.py
```

Os modelos serão gerados em:

```text
models/
├── random_forest_model.pkl
└── isolation_forest_model.pkl
```

---

## 5. Executar a Validação

Com os modelos treinados, execute:

```bash
python3 src/simulink_bridge.py
```

O script irá:

1. carregar os arquivos de telemetria;
2. interpretar os dados exportados pelo Simulink;
3. preparar as variáveis de entrada;
4. carregar os modelos previamente treinados;
5. executar a inferência;
6. calcular o risco de falha;
7. detectar possíveis anomalias;
8. apresentar os resultados dos cenários.

---

# Fluxo Completo

```text
                     TREINAMENTO OFFLINE
                             │
                             ▼
                  ┌─────────────────────┐
                  │   AI4I 2020 Dataset │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Pré-processamento  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Engenharia de       │
                  │ atributos           │
                  └──────────┬──────────┘
                             │
                             ▼
               ┌──────────────────────────┐
               │     Treinamento IA       │
               │                          │
               │ Random Forest            │
               │ Isolation Forest         │
               └────────────┬─────────────┘
                            │
                            ▼
                       Modelos .pkl
                            │
                            │
              ──────────────┼──────────────
                            │
                     SIMULAÇÃO
                            │
                            ▼
               ┌─────────────────────┐
               │   MATLAB/Simulink   │
               │                     │
               │ RPM                 │
               │ ProcessTemp         │
               │ Torque              │
               │ AirTemp             │
               │ ToolWear            │
               └──────────┬──────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │   CAN Pack   │
                  │   ID: 0x1A0  │
                  └──────┬───────┘
                         │
                         ▼
                   CAN Message
                         │
                         ▼
                    Arquivo .mat
                         │
                         ▼
                simulink_bridge.py
                         │
                         ▼
                ┌─────────────────┐
                │ Modelos de IA   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Risco de falha  │
                │ + Anomalias     │
                └─────────────────┘
```

---

# Objetivo do Projeto

O objetivo do **AutoPredict AI** é demonstrar como dados provenientes de uma máquina industrial podem ser integrados a um sistema de **monitoramento de condições e manutenção preditiva**.

A arquitetura conecta:

* **modelagem e simulação física**;
* **comunicação industrial via CAN Bus**;
* **telemetria de máquina**;
* **processamento de dados em Python**;
* **engenharia de atributos**;
* **Machine Learning supervisionado**;
* **detecção não supervisionada de anomalias**;
* **estimativa de risco de falha**.

O projeto demonstra, portanto, uma arquitetura simplificada em que a **camada física da máquina**, a **camada de comunicação** e a **camada de inteligência computacional** trabalham de forma integrada para identificar possíveis condições de falha antes que elas se tornem críticas.
