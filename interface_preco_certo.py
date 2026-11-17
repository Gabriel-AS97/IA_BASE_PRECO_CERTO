import os
from google import genai
import streamlit as st
import pandas as pd
import joblib
import re
import urllib.request
import json

# Configuração da página
st.set_page_config(page_title="Preço Certo IA", page_icon="🛒", layout="centered")

st.title("🛒 Preço Certo — Assistente Inteligente de Compras")
st.markdown("Monte seu carrinho de compras e informe suas condições logísticas. Nossa IA calculará o peso da carga e indicará os melhores estabelecimentos compatíveis.")

# Carrega o modelo treinado e os dados limpos
@st.cache_resource
def carregar_recursos():
    modelo = joblib.load("modelo_preco_certo.pkl")
    df = pd.read_csv("dataset_preco_certo_limpo.csv")
    
    # Calcula o peso médio unitário real a partir do histórico limpo (fallback)
    df["peso_unitario"] = df["peso_total_estimado_kg"] / df["quantidade"]
    pesos_medios = df.groupby("produto")["peso_unitario"].median().to_dict()
    
    # Extrai o peso exato escrito no nome do produto (ex: "50kg", "90g", "500ml")
    for prod in df["produto"].unique():
        match_kg = re.search(r'(\d+(?:[.,]\d+)?)\s*kg', prod, re.IGNORECASE)
        match_g = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:g|ml)', prod, re.IGNORECASE)
        if match_kg:
            pesos_medios[prod] = float(match_kg.group(1).replace(',', '.'))
        elif match_g:
            pesos_medios[prod] = float(match_g.group(1).replace(',', '.')) / 1000.0
            
    return modelo, df, pesos_medios

try:
    modelo, df, pesos_medios = carregar_recursos()
except Exception:
    st.error("Erro ao carregar os arquivos. Execute primeiro o script 'etapas_ia_preco_certo_2.py'.")
    st.stop()
    
@st.cache_data(ttl=600)
def buscar_clima_atual():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=-22.92&longitude=-42.51&current_weather=true"
        req = urllib.request.urlopen(url, timeout=3)
        dados = json.loads(req.read().decode("utf-8"))
        codigo = dados["current_weather"]["weathercode"]
        temp = dados["current_weather"]["temperature"]
        
        # Traduz o código WMO da Open-Meteo para as categorias do modelo
        if codigo in [0, 1]:
            clima_str = "ensolarado"
        elif codigo in [2, 3, 45, 48]:
            clima_str = "nublado"
        elif codigo in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            clima_str = "chuva"
        elif codigo in [95, 96, 99]:
            clima_str = "tempestade"
        else:
            clima_str = "limpo"
            
        return clima_str, temp, True
    except Exception:
        return "limpo", None, False
    
@st.cache_data(ttl=3600)
def buscar_endereco_cep(cep):
    cep_limpo = re.sub(r'\D', '', cep)
    if len(cep_limpo) != 8:
        return None
    try:
        url = f"https://brasilapi.com.br/api/cep/v1/{cep_limpo}"
        req = urllib.request.urlopen(url, timeout=3)
        dados = json.loads(req.read().decode("utf-8"))
        bairro = dados.get("neighborhood", "")
        cidade = dados.get("city", "")
        uf = dados.get("state", "")
        return f"{bairro} - {cidade}/{uf}" if bairro else f"{cidade}/{uf}"
    except Exception:
        return None
    
def gerar_explicacao_llm(melhor_loja, resumo_itens, peso, veiculo, distancia, clima, orcamento, local_texto):
    texto_fallback = (
        f"**Por que esta escolha?** Para atender sua lista (**{resumo_itens}**) com carga total estimada em **{peso:.2f} kg**, "
        f"a IA avaliou seu perfil logístico ({veiculo}, {local_texto}até {distancia} km, clima *{clima}*) e aplicou filtros de disponibilidade de estoque, "
        f"identificando o **{melhor_loja}** como o estabelecimento mais adequado.\n\n"
        f"**Transparência e Limites:** Esta recomendação combina probabilidades calculadas pelo modelo *Random Forest* (treinado em base acadêmica) "
        f"com regras de restrição de segmento para evitar lojas incompatíveis com os produtos escolhidos.\n\n"
        f"**Próxima ação:** Confirme o estoque antes de se deslocar ou ajuste o meio de transporte caso o peso exceda a capacidade recomendada."
    )

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "COLE_SUA_CHAVE_gsk_AQUI":
        return texto_fallback, False

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }

        # 1. Busca dinamicamente um modelo de TEXTO/CHAT ativo na sua conta Groq
        req_models = urllib.request.Request("https://api.groq.com/openai/v1/models", headers=headers)
        with urllib.request.urlopen(req_models, timeout=5) as resp_m:
            lista_modelos = [m["id"] for m in json.loads(resp_m.read().decode("utf-8")).get("data", [])]
            # Ignora modelos de áudio (whisper), visão/guard ou sistemas internos
            modelos_chat = [m for m in lista_modelos if not any(x in m.lower() for x in ["whisper", "guard", "tts", "playai"])]
            modelo_escolhido = next((m for m in modelos_chat if "llama-3.3" in m or "llama-3.1" in m or "gemma" in m), modelos_chat[0])

        # 2. Gera a explicação com o modelo ativo encontrado
        url = "https://api.groq.com/openai/v1/chat/completions"
        prompt = (
            f"Você é o assistente do sistema 'Preço Certo'. Explique de forma clara e amigável (em português, exatamente 3 parágrafos curtos com negrito nos pontos principais) "
            f"por que a loja '{melhor_loja}' foi recomendada para: Lista: {resumo_itens} | Peso: {peso:.2f} kg | Transporte: {veiculo} (até {distancia} km) | Clima: {clima} | Orçamento: R$ {orcamento:.2f}. "
            f"Parágrafo 1: vantagem prática da loja para esse peso, clima e transporte. "
            f"Parágrafo 2: mencione que a escolha combinou Machine Learning (Random Forest) com filtros de segmento. "
            f"Parágrafo 3: escreva '**Próxima ação:**' seguido obrigatoriamente de uma recomendação prática de 1 frase completa para o comprador (ex: confirmar estoque ou horário antes de sair)."
        )
        payload = json.dumps({
            "model": modelo_escolhido,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            dados = json.loads(resp.read().decode("utf-8"))
            return dados["choices"][0]["message"]["content"], True
    
    except Exception as erro_groq:
        st.caption(f"Debug Groq: {erro_groq}")
        return texto_fallback, False

# --- REGRAS DE DOMÍNIO ---
def filtrar_lojas_compativeis(lista_produtos, todas_lojas):
    regras_segmento = {
        "cimento": ["Depósito Silva"],
        "tijolo": ["Depósito Silva"],
        "tinta": ["Depósito Silva", "Atacadão Central"],
        "ração": ["Petshop Animalia", "Supermercado Viva Bem", "Hipermercado Preço Bom"],
        "pet": ["Petshop Animalia"],
        "remédio": ["Farmácia Vida"],
        "dipirona": ["Farmácia Vida"],
        "sabonete": ["Farmácia Vida", "Supermercado Viva Bem", "Hipermercado Preço Bom", "Mercadinho da Esquina", "Mercadinho do Zé", "Empório Local"],
        "detergente": ["Supermercado Viva Bem", "Hipermercado Preço Bom", "Mercadinho da Esquina", "Mercadinho do Zé", "Empório Local", "Atacadão Central"],
        "banana": ["Varejão das Frutas", "Supermercado Viva Bem", "Hipermercado Preço Bom", "Mercadinho da Esquina", "Mercadinho do Zé"],
        "maçã": ["Varejão das Frutas", "Supermercado Viva Bem", "Hipermercado Preço Bom", "Mercadinho da Esquina", "Mercadinho do Zé"],
        "fruta": ["Varejão das Frutas", "Supermercado Viva Bem", "Hipermercado Preço Bom"],
        "mercearia": ["Supermercado Viva Bem", "Hipermercado Preço Bom", "Mercadinho da Esquina", "Mercadinho do Zé", "Empório Local", "Atacadão Central"],
    }
    
    lojas_validas = set(todas_lojas)
    
    for prod in lista_produtos:
        prod_lower = prod.lower()
        if any(p in prod_lower for p in ["arroz", "feijão", "feijao", "café", "cafe", "açúcar", "acucar", "óleo", "oleo", "macarrão", "macarrao", "leite"]):
            prod_lower = "mercearia"
        encontrou_regra = False
        for palavra_chave, permitidas in regras_segmento.items():
            if palavra_chave in prod_lower:
                lojas_validas = lojas_validas.intersection(set(permitidas))
                encontrou_regra = True
                break
        if not encontrou_regra:
            lojas_validas = lojas_validas - {"Petshop Animalia", "Depósito Silva", "Farmácia Vida"}
            
    if not lojas_validas:
        return ["Hipermercado Preço Bom", "Atacadão Central", "Supermercado Viva Bem"]
        
    return list(lojas_validas)

# --- INTERFACE: MONTE SEU CARRINHO ---
st.subheader("1. Monte sua Lista de Compras")
produtos_disponiveis = sorted(df["produto"].unique())
produtos_selecionados = st.multiselect(
    "Selecione um ou mais produtos:",
    options=produtos_disponiveis,
    default=[produtos_disponiveis[0]] if len(produtos_disponiveis) > 0 else None,
    placeholder="Escolha os itens da sua lista..."
)

carrinho = {}
peso_total_calculado = 0.0
qtd_total_itens = 0

if produtos_selecionados:
    st.caption("Defina a quantidade de cada item (o peso será calculado automaticamente pela IA):")
    cols_qtd = st.columns(min(len(produtos_selecionados), 3))
    for idx, prod in enumerate(produtos_selecionados):
        with cols_qtd[idx % 3]:
            qtd = st.number_input(f"Qtd: {prod}", min_value=1, max_value=50, value=2, key=f"qtd_{prod}")
            carrinho[prod] = qtd
            peso_unit = pesos_medios.get(prod, 1.0)
            peso_total_calculado += peso_unit * qtd
            qtd_total_itens += qtd

# --- INTERFACE: LOGÍSTICA E PAGAMENTO ---
with st.form("form_logistica"):
    st.subheader("2. Logística e Condições")
    col1, col2 = st.columns(2)
    with col1:
        veiculo = st.selectbox("Meio de transporte:", ["carro", "moto", "a_pe"], format_func=lambda x: {"carro": "🚗 Carro", "moto": "🏍️ Moto", "a_pe": "🚶 A pé"}[x])
        orcamento = st.number_input("Orçamento disponível (R$):", min_value=5.0, max_value=2000.0, value=150.0, step=10.0)
        pagamento = st.selectbox("Forma de pagamento:", sorted(df["forma_pagamento"].unique()))
        cep_usuario = st.text_input("CEP de partida (opcional - BrasilAPI):", value="28990-000", max_chars=9)
    with col2:
        distancia = st.number_input("Distância máxima aceitável (km):", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
        clima_api, temp_atual, api_ok = buscar_clima_atual()
        climas_validos = ["ensolarado", "limpo", "nublado", "chuva", "tempestade"]
        indice_padrao = climas_validos.index(clima_api) if clima_api in climas_validos else 1
        
        label_clima = f"Clima atual (Saquarema: {temp_atual}°C via Open-Meteo):" if api_ok else "Clima atual:"
        clima = st.selectbox(label_clima, climas_validos, index=indice_padrao)
        urgencia = st.selectbox("Nível de urgência:", sorted(df["urgencia"].unique()))
        horarios_validos = [h for h in sorted(df["horario_compra"].unique()) if h.lower() != "fechado"]
        horario = st.selectbox("Horário previsto:", horarios_validos)

    enviado = st.form_submit_button("🚀 Encontrar Melhor Estabelecimento", use_container_width=True)

# --- PROCESSAMENTO HÍBRIDO (ML + REGRAS DE NEGÓCIO + TRAVAS REALISTAS) ---
if enviado:
    endereco_formatado = buscar_endereco_cep(cep_usuario)
    local_texto = f"partindo de **{endereco_formatado}** " if endereco_formatado else ""
    
    if not produtos_selecionados:
        st.error("⚠️ Selecione pelo menos um produto no carrinho antes de analisar.")
        st.stop()

    # 1. Estimativa de custo mínimo realista por produto
    precos_base = {
        "cimento": 35.0, "tijolo": 2.5, "tinta": 85.0,
        "ração": 45.0, "remédio": 25.0, "dipirona": 12.0,
        "sabonete": 3.5, "detergente": 3.0, "banana": 6.0, "maçã": 8.0
    }
    custo_estimado = 0.0
    for prod, qtd in carrinho.items():
        preco_unit = 10.0
        for chave, valor in precos_base.items():
            if chave in prod.lower():
                preco_unit = valor
                break
        custo_estimado += preco_unit * qtd

    # 2. Validações de Bloqueio Crítico (Cenários Impossíveis)
    erros_bloqueio = []

    limites_peso = {
        "a_pe": (15.0, "A pé", "mude para 'Moto' ou 'Carro' ou reduza os itens"),
        "moto": (40.0, "Moto", "mude para 'Carro' ou reduza os itens"),
        "carro": (400.0, "Carro", "reduza a carga ou contrate frete especializado")
    }
    limite_max_peso, nome_veic, sug_peso = limites_peso[veiculo]
    if peso_total_calculado > limite_max_peso:
        erros_bloqueio.append(
            f"**Capacidade de Carga ({nome_veic}):** O carrinho pesa **{peso_total_calculado:.2f} kg** "
            f"(limite seguro: {limite_max_peso:.0f} kg). *Sugestão: {sug_peso}.*"
        )

    if veiculo == "a_pe":
        limite_dist_ape = 1.5 if any(c in clima.lower() for c in ["chuva", "tempestade"]) else 3.0
        if distancia > limite_dist_ape:
            erros_bloqueio.append(
                f"**Deslocamento a Pé Inviável:** Percorrer **{distancia:.1f} km** a pé (clima: *{clima}*) "
                f"ultrapassa o limite prático de **{limite_dist_ape:.1f} km**. *Sugestão: reduza a distância ou escolha Moto/Carro.*"
            )

    if orcamento < custo_estimado:
        erros_bloqueio.append(
            f"**Orçamento Insuficiente:** Sua lista tem custo mínimo estimado em **R$ {custo_estimado:.2f}**, "
            f"mas seu orçamento é de **R$ {orcamento:.2f}**. *Sugestão: aumente o orçamento ou remova itens.*"
        )

    if erros_bloqueio:
        mensagem_erro = "🚫 **Não foi possível recomendar uma loja devido a restrições práticas:**\n\n"
        for err in erros_bloqueio:
            mensagem_erro += f"- {err}\n"
        mensagem_erro += "\n👉 **Próxima ação:** Ajuste os parâmetros acima para tornar a compra viável e clique novamente."
        st.error(mensagem_erro)
        st.stop()

    # 3. Alertas Preventivos (Cenários Viáveis, mas com Risco Logístico/Operacional)
    avisos_risco = []
    if veiculo == "moto" and any(c in clima.lower() for c in ["chuva", "tempestade"]):
        avisos_risco.append(f"Deslocamento de **moto sob {clima}** exige atenção redobrada e embalagem impermeável.")
    if urgencia.lower() == "alta" and "rush" in horario.lower() and distancia > 10.0:
        avisos_risco.append(f"Urgência **alta** na **hora do rush** para **{distancia:.1f} km** pode sofrer atrasos severos de trânsito.")
    if veiculo == "moto" and distancia > 25.0:
        avisos_risco.append(f"Trajeto longo de moto (**{distancia:.1f} km**): avalie o custo-benefício do combustível.")

    if avisos_risco:
        texto_avisos = "⚠️ **Pontos de Atenção Logística:**\n" + "\n".join([f"- {a}" for a in avisos_risco])
        st.warning(texto_avisos)

    # 4. Execução da IA e Cruzamento com Segmento
    with st.spinner("Calculando peso da carga e cruzando probabilidades das lojas..."):
        produto_principal = max(carrinho, key=carrinho.get)
        
        entrada = pd.DataFrame([{
            "produto": produto_principal,
            "quantidade": qtd_total_itens,
            "veiculo": veiculo,
            "clima": clima,
            "urgencia": urgencia,
            "orcamento": orcamento,
            "forma_pagamento": pagamento,
            "horario_compra": horario,
            "distancia_loja_km": distancia,
            "peso_total_estimado_kg": round(peso_total_calculado, 2)
        }])
        
        classes_lojas = modelo.classes_
        probabilidades = modelo.predict_proba(entrada)[0]
        ranking_ia = dict(zip(classes_lojas, probabilidades))
        
        lojas_permitidas = filtrar_lojas_compativeis(produtos_selecionados, classes_lojas)
        ranking_filtrado = {loja: prob for loja, prob in ranking_ia.items() if loja in lojas_permitidas}
        
        lojas_ordenadas = sorted(ranking_filtrado.items(), key=lambda x: x[1], reverse=True)
        melhor_loja, score_vencedora = lojas_ordenadas[0]

    # --- EXIBIÇÃO RICA DOS RESULTADOS ---
    st.divider()       
    st.subheader("🎯 Resultado da Análise")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Itens no Carrinho", f"{qtd_total_itens} un.")
    m2.metric("Peso Calculado", f"{peso_total_calculado:.2f} kg")
    m3.metric("Orçamento", f"R$ {orcamento:.2f}")

    # Destaque Principal
    st.success(f"### 🏆 Melhor Opção: **{melhor_loja}**")
    
    # Quadro Azul Completo: Explicação detalhada (LLM ou Fallback)
    resumo_itens = ", ".join([f"{q}x {p}" for p, q in carrinho.items()])
    local_str = local_texto if "local_texto" in locals() else ""
    
    texto_quadro_azul, usou_llm = gerar_explicacao_llm(
        melhor_loja, resumo_itens, peso_total_calculado, veiculo, distancia, clima, orcamento, local_str
    )

    if usou_llm:
        texto_quadro_azul += "\n\n✨ *Explicação gerada dinamicamente via LLM (Llama 3.3 70B / Groq)*"

    st.info(texto_quadro_azul)

    # Ranking das Top 3 alternativas compatíveis
    st.markdown("#### 📊 Ranking de Compatibilidade (IA + Filtro de Segmento)")
    soma_probs = sum(p for _, p in lojas_ordenadas) or 1.0
    for pos, (loja, prob) in enumerate(lojas_ordenadas[:3], start=1):
        afinidade = (prob / soma_probs) * 100
        st.write(f"**{pos}º {loja}** — Afinidade relativa: `{afinidade:.1f}%`")
        st.progress(min(int(afinidade), 100))

    # Quadro Amarelo Final: Transparência Técnica, Métricas e Próxima Ação
    st.warning(
        "⚠️ **Transparência e Limites do Modelo:** Esta recomendação combina probabilidades calculadas pelo modelo *Random Forest* "
        "(acurácia de teste de ~10,3% na base acadêmica devido à dispersão dos dados sintéticos) com regras de restrição de segmento para evitar sugestões incompatíveis. "
        "A afinidade exibida representa a distribuição relativa entre as lojas válidas, não uma garantia de preço ou estoque em tempo real.\n\n"
        "👉 **Próxima ação:** Confirme a disponibilidade dos itens antes de se deslocar ou ajuste o meio de transporte acima caso o peso exceda a capacidade recomendada."
    )
    
    with st.expander("📊 Auditoria de Dados e Métricas do Modelo"):
        try:
            with open("metricas_v1.json", "r", encoding="utf-8") as f:
                metricas = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline (Dummy)", f"{metricas['baseline_dummy']['acuracia']*100:.2f}%")
            c2.metric("Random Forest (Acurácia)", f"{metricas['ia_random_forest']['acuracia']*100:.2f}%")
            c3.metric("Random Forest (F1-Macro)", f"{metricas['ia_random_forest']['f1_score_macro']*100:.2f}%")
        except Exception:
            st.caption("Execute o script de treino para carregar o metricas_v1.json.")
            
        try:
            st.image("graficos_analise_preco_certo.png", caption="Análise Exploratória gerada via Matplotlib e Seaborn")
        except Exception:
            pass