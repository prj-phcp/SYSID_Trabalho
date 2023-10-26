# Trabalho de SYSTEM Identification 2022.2 (Predição de vazão usando Volve Dataset)

## Equipe

Felipe da Costa Pereira
Pedro Henrique Cardoso Paulo
Helon Ayala (Professor)

## Descrição

Este repositório é dedicado ao armazenamento dos arquivos usados para o trabalho de SYSID com o tema de identificação da vazão de líquido em um poço de petróleo usando o dataset de VOLVE, da Equinor.

## Estrutura dos arquivos

A estrutura dessa pasta é como se segue. É importante notar que os notebooks dos testes devem ser rodados após o de tratamento, e o de pós-processamento após todos os demais:

[data](./data): Pasta com os dados brutos usados no estudo.

[outputs](./outputs): Pasta com os arquivos de saída dos notebooks.

[presentations](./presentations): Pasta com a apresentação feita.

[Tratamento01.ipynb](./Tratamento01.ipynb): Notebook para processamento dos dados em Excel e geração do pickle de input.

[IntegratedSISO.EXPORT.W1.ipynb](./IntegratedSISO.EXPORT.W1.ipynb): Notebook para análise SISO do primeiro poço.

[IntegratedMISO.EXPORT.W1.ipynb](./IntegratedMISO.EXPORT.W1.ipynb): Notebook para análise MISO do primeiro poço.

[IntegratedSISO.EXPORT.W2.ipynb](./IntegratedSISO.EXPORT.W2.ipynb): Notebook para análise SISO do segundo poço.

[IntegratedMISO.EXPORT.W2.ipynb](./IntegratedMISO.EXPORT.W2.ipynb): Notebook para análise MISO do segundo poço.

[PosProcessamento.ipynb](./PosProcessamento.ipynb): Notebook para plot dos principais gráficos usados na apresentação.