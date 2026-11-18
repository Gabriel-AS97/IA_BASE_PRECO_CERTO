# 🛒 Preço Certo — Assistente Inteligente de Compras

Sistema híbrido de apoio à decisão que recomenda os estabelecimentos mais adequados para uma lista de compras com base no perfil logístico e financeiro do consumidor.

## 🚀 Como Executar (Rápido — Windows)

Para testar a aplicação imediatamente sem precisar configurar comandos manualmente:

1. Baixe ou clone este repositório.
2. Dê um **duplo clique** no arquivo `Iniciar_IA_Base_Preco_Certo.bat`.
3. O script instalará automaticamente as dependências necessárias e abrirá a interface web no seu navegador.

*(Alternativa via terminal: `py -m streamlit run interface_preco_certo.py`)*

---

## 🧠 O que a IA faz?

O **Preço Certo** combina Machine Learning clássico, regras de negócio e IA Generativa em quatro etapas:

1. **Cálculo Automático de Carga:** Identifica o peso unitário de cada produto selecionado (por extração de texto e mediana histórica) e calcula o peso total do carrinho.
2. **Validação Logística e Regras de Segmento:** Bloqueia cenários fisicamente inviáveis (ex.: excesso de peso para transporte a pé ou de moto, orçamento insuficiente) e filtra lojas compatíveis com os itens do carrinho.
3. **Classificação com Machine Learning (`Random Forest`):** Calcula a probabilidade de adequação de cada estabelecimento cruzando distância, meio de transporte, orçamento, forma de pagamento, horário e clima atual (via **Open-Meteo API** e **BrasilAPI**).
4. **Explicação Inteligente (LLM + Fallback):** Gera uma justificativa personalizada em linguagem natural (via API Groq / Llama 3.3, com *fallback* local automático caso nenhuma chave de API esteja configurada).

---

## 📁 Estrutura do Repositório

* `Iniciar_IA_Base_Preco_Certo.bat` — Atalho automatizado para instalar dependências e abrir a interface.
* `interface_preco_certo.py` — Aplicação web interativa desenvolvida em Streamlit.
* `etapas_ia_preco_certo.py` — Pipeline de limpeza de dados, análise exploratória e treinamento do modelo.
* `modelo_preco_certo.pkl` — Modelo *Random Forest* treinado e pronto para inferência.
* `dataset_preco_certo_expandido_sujo.csv` / `dataset_preco_certo_limpo.csv` — Bases de dados bruta e tratada.
* `metricas_v1.json` / `graficos_analise_preco_certo.png` — Métricas de avaliação (comparação com *DummyClassifier*) e visualizações exploratórias.
