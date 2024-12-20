![MLens Logo](https://raw.githubusercontent.com/gscproseg/master/main/MLens.png)


## Acesse o MLens App

[![Acessar o MLens App](https://img.shields.io/badge/Acessar%20o%20MLens%20App-blue?style=for-the-badge&logo=appveyor)](https://mlensapp.streamlit.app/)

## Contextualização


A classe Myxozoa é composta por mais de 70 gêneros e cerca de 2.600 espécies, todas endoparasitas obrigatórios (Lom et al., 2013; Fiala, 2015). Muitos desses parasitas são responsáveis por causar doenças graves conhecidas como mixosporidioses (Jones, Bartholomew, Zhang, 2015). De acordo com Lom e Arthur (1989), variáveis morfológicas e morfométricas são essenciais para a definição dos mixosporos e seu agrupamento taxonômico.

Entre os myxozoários, os gêneros Henneguya (Carvalho et al., 2020) e Myxobolus desempenham papéis significativos como parasitas da ictiofauna (Zago et al., 2022). Esses gêneros, de natureza polifilética, têm ampla distribuição global e são os mais estudados, parasitando tanto peixes de água doce quanto marinhos, além de répteis e anfíbios (Lom & Dyková, 2006; Atkinson, 2011). A distribuição geográfica desses parasitas é influenciada por fatores ambientais como temperatura e qualidade da água, bem como pela disponibilidade de hospedeiros (Okamura, Gruhl, Bartholomew, 2015; Jerônimo et al., 2022; Lauringson et al., 2023).

Com o advento da Inteligência Artificial (IA), impulsionada por avanços contínuos no campo da ciência, tecnologias computacionais especializadas se tornaram cada vez mais presentes. Conforme Sichman et al. (2021), a IA tem desempenhado um papel fundamental na modernização da parasitologia, auxiliando na detecção de parasitas com uma precisão e eficiência antes inatingíveis. Tecnologias como as Convolutional Neural Networks (CNNs) têm sido aplicadas amplamente em visão computacional, destacando-se na classificação e detecção de objetos parasitários em imagens microscópicas, muitas vezes com intervenção mínima de especialistas humanos (Rahimzadeh & Attar, 2020; Yaacoub et al., 2020).

Neste trabalho, desenvolvemos o MLens, um WebAPP em versão beta projetado para realizar a detecção e classificação automatizada dos gêneros Myxobolus e Henneguya em imagens de microscopia de luz. Usando a rede neural YOLOv5, buscamos contribuir para uma maior eficiência e precisão no diagnóstico parasitológico.


| A Figura 1 representa nosso Pipeline |
![Figura 1](https://raw.githubusercontent.com/gscproseg/master/main/Figure1.png) |


A **Tabela 1**  resume os principais hiperparâmetros usados durante o treinamento dos modelos, conforme destacado por _Huang et al. (2018)_ e _Du (2018)_. Esses parâmetros foram essenciais para otimizar a detecção de myxozoários nos gêneros **Henneguya** e **Myxobolus**.

*Tabela 1 - Hiperparâmetros utilizados no treinamento*:

| Parâmetro              | Valor  |
|------------------------|--------|
| Tamanho da Imagem       | 640    |
| Épocas                  | 300    |
| Batch Size              | 16     |
| Taxa de Aprendizado     | 0.01   |
| Momentum                | 0.937  |
| Weight Decay            | 0.0005 |
| Otimizador              | SGD    |


A **Tabela 2** resume os resultados dos modelos YOLOv5n, YOLOv5s, YOLOv5m e YOLOv5l em termos de precisão (P), recall (R), e as métricas mAP@50 e mAP@50-95. Esses valores indicam a capacidade de cada modelo em detectar com eficácia os gêneros **Henneguya** e **Myxobolus**, conforme descrito nas seções anteriores. O desempenho superior do modelo YOLOv5l reflete-se em suas pontuações mais altas, especialmente em **mAP@50** e **mAP@50-95**, conforme esperado devido à sua maior complexidade.

*Tabela 2 - Resultados de validação dos modelos YOLOv5*:

| Modelo    | Imagens | Instâncias | P     | R     | mAP@50 | mAP@50-95 |
|-----------|---------|------------|-------|-------|--------|-----------|
| YOLOv5n   | 2000    | 12386      | 0.954 | 0.965 | 0.966  | 0.879     |
| YOLOv5s   | 2000    | 12386      | 0.972 | 0.958 | 0.976  | 0.860     |
| YOLOv5m   | 2000    | 12386      | 0.974 | 0.967 | 0.980  | 0.892     |
| YOLOv5l   | 2000    | 12386      | 0.994 | 0.970 | 0.993  | 0.905     |


A Tabela 2 resume os resultados dos modelos YOLOv5n, YOLOv5s, YOLOv5m e YOLOv5l em termos de precisão (P), recall (R), e as métricas mAP@50 e mAP@50-95. Esses valores indicam a capacidade de cada modelo em detectar com eficácia os gêneros Henneguya e Myxobolus, conforme descrito nas seções anteriores. O desempenho superior do modelo YOLOv5l reflete-se em suas pontuações mais altas, especialmente em mAP@50 e mAP@50-95, conforme esperado devido à sua maior complexidade.

Matrizes de Confusão
A matriz de confusão é uma tabela que apresenta a contagem desses valores, geralmente organizada da seguinte forma:

![Confusion_Matrix](https://raw.githubusercontent.com/gscproseg/master/main/Figure20.png)



Interpretação dos Valores na Matriz de Confusão
TP: Indica a eficácia do modelo em identificar corretamente as instâncias da classe positiva.
FP: Indica a quantidade de vezes que o modelo errou ao identificar uma instância como positiva.
FN: Indica a quantidade de instâncias positivas que o modelo não conseguiu identificar.
TN: Indica a eficácia do modelo em rejeitar corretamente as instâncias da classe negativa.


Resumo
A análise dos TP, FP, FN e TN fornece uma visão clara de como o modelo está se comportando em relação às classes que está tentando detectar. A matriz de confusão, por sua vez, oferece uma representação visual e numérica dessa análise, facilitando a compreensão da eficácia do modelo. Essa informação é crucial para ajustar e melhorar o desempenho do seu sistema de detecção de objetos.
