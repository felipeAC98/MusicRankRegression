# Music Rank Regression

Autor: Felipe Alves Cordeiro

Este repositório possui alguns scripts desenvolvidos com o objetivo geral analisar com o auxilio de técnicas de ML
e análise exploratória características que influenciam na popularidade geral das músicas
com intuito de beneficiar a produção de canções com mais chances de conseguirem maior
reconhecimento por parte do público.

## Resumo

A compreensão dos fatores que influenciam a popularidade musical são importantes
para toda a indústria da música. Este tipo de informação pode beneficiar diretamente
todos os envolvidos uma vez que, os conhecendo, as produções serão feitas de forma mais
direcionada e eficiente para atingir o público em geral. Técnicas de aprendizado de máquina tem se tornado cada vez mais comuns dentro do universo musical. Plataformas
de streaming de música como Spotify possuem como um de seus pilares a utilização de
técnicas de aprendizado de máquina para melhoria dos serviços para os usuários. Este
trabalho possui como objetivo construir um modelo de regressão para previsão da popularidade de uma música com base em características de áudio de alto nível e, a partir deste,
efetuar a análise dos fatores que mais influenciam em sua popularidade. Os algoritmos
de regressão utilizados para geração e aplicação do modelo foram: regressão linear, KNN,
árvore de decisão, Random Forest e XGBoost. As músicas selecionadas para comporem
o conjunto de dados foram as presentes no 4MuLA. A maior parte das características
destas músicas foram extraídas da plataforma de streaming Spotify, algumas outras foram
utilizadas diretamente do 4MuLA. O melhor desempenho foi obtido com a utilização do
Random Forest, que obteve um coeficiente de determinação ajustado de 0,65 e RMSE
de 12,3. Com estes valores é possível dizer que o modelo é capaz de prever de forma
satisfatória a popularidade de uma música e, partir da análise desse regressor, concluir
que dentre os fatores observados aquele que alcançou maior influência no sucesso de uma
faixa é o número de seguidores que o artista possui.

## Monografia

A monografia completa referente a este trabalho pode ser encontrada através deste link: https://drive.google.com/file/d/1RffvEdlt8fMRt_WU6B9Yj1vFFJ0a20MF/view?usp=sharing

## Como executar

Para executar os scripts python aqui presentes é necessário realizar a instalação do requirements.py, aconselha-se o uso de um virtual enviroment para isso.

### benchmark.py

 Este script tem como saída a acurácia do modelo, para executa-lo basta seguir o comando exemplo a seguir: python benchmark.py 
 Por padrão ele irá utilizar o algoritmo Random Forest e este pode demorar a depender da performance da máquina em questão, para objetivos de teste aconselha-se o uso do parâmetro --algorithm tree. Exemplo: python benchmark.py  --algorithm tree
 
#### Argumentos possíveis:

	--algorithm             #Algoritmo que sera utilizado para o teste, todos serao utilizados caso nao seja definido
	--dropDataTest          #Habilita ou nao o teste de drop dos dados do dataset para construcao doss modelos
	--set                   #Define se ira utilizar o conjunto de treino ou de testes para o teste do modelo
	--dropParams            #Define se ira utilizar somente os parametros principais definidos manualmente 
	--dropFollowers	        #Define se ira remover a caracteristica de total de seguidores, por padrao ela nao sera removida
	--dropArtPopularity	    #Define se ira remover a caracteristica popularidade do artista, por padrao ela SERA removida
