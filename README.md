\🛒 Preço Certo — Assistente Inteligente de Compras



Sistema híbrido de apoio à decisão que recomenda os estabelecimentos mais adequados para uma lista de compras com base no perfil logístico e financeiro do consumidor.



\🚀 Como Executar (Rápido — Windows)



Para testar a aplicação imediatamente sem precisar configurar comandos manualmente:



1\. Baixe ou clone este repositório.

2\. Dê um \*\*duplo clique\*\* no arquivo \*\*`Iniciar\_IA\_Base\_Preco\_Certo.bat`\*\*.

3\. O script instalará automaticamente as dependências necessárias e abrirá a interface web no seu navegador.



\*(Alternativa via terminal: `py -m streamlit run interface\_preco\_certo.py`)\*



\---



\🧠 O que a IA faz?



O \*\*Preço Certo\*\* combina Machine Learning clássico, regras de negócio e IA Generativa em quatro etapas:



1\. \*\*Cálculo Automático de Carga:\*\* Identifica o peso unitário de cada produto selecionado (por extração de texto e mediana histórica) e calcula o peso total do carrinho.

2\. \*\*Validação Logística e Regras de Segmento:\*\* Bloqueia cenários fisicamente inviáveis (ex.: excesso de peso para transporte a pé ou de moto, orçamento insuficiente) e filtra lojas compatíveis com os itens do carrinho.

3\. \*\*Classificação com Machine Learning (`Random Forest`):\*\* Calcula a probabilidade de adequação de cada estabelecimento cruzando distância, meio de transporte, orçamento, forma de pagamento, horário e clima atual (via \*\*Open-Meteo API\*\* e \*\*BrasilAPI\*\*).

4\. \*\*Explicação Inteligente (LLM + Fallback):\*\* Gera uma justificativa personalizada em linguagem natural (via API Groq / Llama 3.3, com \*fallback\* local automático caso nenhuma chave de API esteja configurada).



\---



\📁 Estrutura do Repositório



\* `Iniciar\_IA\_Base\_Preco\_Certo.bat` — Atalho automatizado para instalar dependências e abrir a interface.

\* `interface\_preco\_certo.py` — Aplicação web interativa desenvolvida em Streamlit.

\* `etapas\_ia\_preco\_certo.py` — Pipeline de limpeza de dados, análise exploratória e treinamento do modelo.

\* `modelo\_preco\_certo.pkl` — Modelo \*Random Forest\* treinado e pronto para inferência.

\* `dataset\_preco\_certo\_expandido\_sujo.csv` / `dataset\_preco\_certo\_limpo.csv` — Bases de dados bruta e tratada.

\* `metricas\_v1.json` / `graficos\_analise\_preco\_certo.png` — Métricas de avaliação (comparação com \*DummyClassifier\*) e visualizações exploratórias.

