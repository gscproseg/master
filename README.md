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
| YOLOv5s   | 2000    | 12386      | 0.974 | 0.967 | 0.980  | 0.892     |
| YOLOv5m   | 2000    | 12386      | 0.972 | 0.958 | 0.976  | 0.860     |
| YOLOv5l   | 2000    | 12386      | 0.994 | 0.970 | 0.993  | 0.905     |
