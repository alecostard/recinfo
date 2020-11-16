# Implementação do trabalho final da disciplina Recuperação da Informação.

O relatório se encontra na raiz do projeto, com o nome `relatorio-alexandre.pdf`

Para diminuir o volume de dados trafegados, não foi incluído o dataset MovieLens 25M. Ele e os demais datasets relacionados ao MovieLens podem ser encontrados [na página oficial do projeto](https://grouplens.org/datasets/movielens/).

Antes de gerar os dados usados na análise, crie os diretórios `./predictions`, `./metrics/global`e `./metrics/user`. Eles são usados para armazenar as predições geradas e as medidas coletadas.

Para gerar as predições, inicie um console Python carregando o arquivo principal

    python -i trabalho.py

Então execute a função `run_experiments` passando como argumento quantas rodadas deseja executar. Essa função executará as rodadas em paralelo, usando todos os núcleos disponíveis.

    >>> run_experiments(12)

Para gerar as métricas

    >>> run_global_metrics()
    >>> run_user_metrics()

Essas duas funções também rodarão em paralelo. Finalmente, para gerar os gráficos usados

    >>> plot_global_metrics()
    >>> plot_user_metrics()