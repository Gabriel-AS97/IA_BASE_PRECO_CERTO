import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def executar_limpeza():
    print("A iniciar a limpeza e formatação dos dados...")
    
    # 1. Carrega o dataset original e remove duplicatas
    df = pd.read_csv("dataset_preco_certo_expandido_sujo.csv")
    total_inicial = len(df)
    
    df_limpo = df.drop_duplicates().copy()
    duplicatas_removidas = total_inicial - len(df_limpo)

    # 2. Filtros numéricos lógicos
    total_antes = len(df_limpo)
    df_limpo = df_limpo[(df_limpo['quantidade'] > 0) & (df_limpo['quantidade'] <= 100)]
    df_limpo = df_limpo[(df_limpo['orcamento'] > 0) & (df_limpo['orcamento'] <= 2000)]
    df_limpo = df_limpo[(df_limpo['distancia_loja_km'] >= 0) & (df_limpo['distancia_loja_km'] <= 50)]
    df_limpo = df_limpo[(df_limpo['peso_total_estimado_kg'] >= 0) & (df_limpo['peso_total_estimado_kg'] <= 500)]
    outliers_removidos = total_antes - len(df_limpo)

    # 3. Filtros categóricos (Produtos e Veículos válidos)
    total_antes = len(df_limpo)
    padrao_sujo = 'Mistério|Desconhecido|Item 99|N/A'
    df_limpo = df_limpo[~df_limpo['produto'].str.contains(padrao_sujo, case=False, na=False)]
    df_limpo = df_limpo[df_limpo['veiculo'].isin(['carro', 'a_pe', 'moto'])]
    prod_veic_invalidos = total_antes - len(df_limpo)
    
    # 4. Remoção de lojas fantasmas e pagamentos inviáveis
    total_antes = len(df_limpo)
    lojas_invalidas = ['Loja Inexistente', 'Mercado Fantasma', 'Nenhuma', 'Loja do Fim do Mundo']
    df_limpo = df_limpo[~df_limpo['loja_recomendada'].isin(lojas_invalidas)]
    
    pagamentos_invalidos = ['fiado', 'cheque', 'escambo', 'promissoria']
    df_limpo = df_limpo[~df_limpo['forma_pagamento'].isin(pagamentos_invalidos)]
    lojas_pagamentos_invalidos = total_antes - len(df_limpo)

    # 5. Tratamento rigoroso de Valores Nulos (NaN)
    total_antes = len(df_limpo)
    df_limpo = df_limpo.dropna(subset=['produto', 'loja_recomendada', 'veiculo', 'clima', 'urgencia', 'forma_pagamento'])
    moda_horario = df_limpo['horario_compra'].mode()[0]
    df_limpo['horario_compra'] = df_limpo['horario_compra'].fillna(moda_horario)
    nulos_removidos = total_antes - len(df_limpo)

    # 6. Salva a versão final sanitizada para alimentar o modelo de IA
    df_limpo.to_csv("dataset_preco_certo_limpo.csv", index=False)
    
    # --- RELATÓRIO VISUAL NO TERMINAL ---
    print("\n" + "="*45)
    print("      RELATÓRIO DE AUDITORIA DE DADOS")
    print("="*45)
    print(f"Total de linhas originais:       {total_inicial}")
    print(f"[-] Duplicatas removidas:        {duplicatas_removidas}")
    print(f"[-] Outliers numéricos:          {outliers_removidos}")
    print(f"[-] Produtos/Veículos inválidos: {prod_veic_invalidos}")
    print(f"[-] Lojas/Pagamentos inválidos:  {lojas_pagamentos_invalidos}")
    print(f"[-] Valores nulos essenciais:    {nulos_removidos}")
    print("-" * 45)
    print(f"TOTAL DE REGISTOS VÁLIDOS:       {len(df_limpo)}")
    print("="*45 + "\n")
    
    print("Exemplo do Dataset Sanitizado (Primeiras 3 linhas):")
    print(df_limpo.head(3).to_string(index=False) + "\n")
    
    print("Novo arquivo 'dataset_preco_certo_limpo.csv' gerado com sucesso.\n")
    
    return df_limpo

def gerar_analises(df_limpo):
    print("A gerar gráficos analíticos...")
    
    # Definição das entradas (X) e alvo (y) do projeto
    features = [
        "produto", "quantidade", "veiculo", "clima", 
        "urgencia", "orcamento", "forma_pagamento", 
        "horario_compra", "distancia_loja_km", "peso_total_estimado_kg"
    ]
    X = df_limpo[features]
    y = df_limpo["loja_recomendada"]

    # Configuração visual
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Frequência de Recomendações por Loja
    ordem_lojas = y.value_counts().index
    sns.countplot(y=y, order=ordem_lojas, palette='viridis', ax=axes[0], hue=y, legend=False)
    axes[0].set_title('1. Distribuição de Recomendações por Loja (Target y)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Quantidade de Indicações')
    axes[0].set_ylabel('Loja Recomendada')

    # Gráfico 2: Relação Logística (Veículo x Distância)
    sns.boxplot(data=X, x='veiculo', y='distancia_loja_km', palette='Set2', ax=axes[1], hue='veiculo', legend=False)
    axes[1].set_title('2. Distância das Lojas por Meio de Transporte', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Meio de Transporte')
    axes[1].set_ylabel('Distância (km)')

    plt.tight_layout()
    
    nome_imagem = 'graficos_analise_preco_certo.png'
    plt.savefig(nome_imagem, dpi=300)
    print(f"Análises finalizadas! Gráficos salvos como '{nome_imagem}'.")
    
def executar_treino_avaliacao():
    print("A iniciar o treino e avaliação do modelo...")
    
    # 1. Carrega os dados limpos gerados na etapa anterior
    df = pd.read_csv("dataset_preco_certo_limpo.csv")
    
    colunas_categoricas = ["produto", "veiculo", "clima", "urgencia", "forma_pagamento", "horario_compra"]
    colunas_numericas = ["quantidade", "orcamento", "distancia_loja_km", "peso_total_estimado_kg"]
    
    X = df[colunas_categoricas + colunas_numericas]
    y = df["loja_recomendada"]
    
    # 2. Divisão Treino/Teste (80% treino, 20% teste reservado sem vazamento)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Pré-processador (converte textos em colunas binárias e padroniza números)
    preprocessador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas),
            ('num', StandardScaler(), colunas_numericas)
        ]
    )
    
    # --- MODELO 1: BASELINE ---
    pipeline_baseline = Pipeline(steps=[
        ('prep', preprocessador),
        ('modelo', DummyClassifier(strategy="most_frequent"))
    ])
    pipeline_baseline.fit(X_treino, y_treino)
    y_pred_base = pipeline_baseline.predict(X_teste)
    
    # --- MODELO 2: IA PRINCIPAL ---
    pipeline_ia = Pipeline(steps=[
        ('prep', preprocessador),
        ('modelo', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline_ia.fit(X_treino, y_treino)
    y_pred_ia = pipeline_ia.predict(X_teste)
    
    # 4. Cálculo das Métricas
    resultados = {
        "baseline_dummy": {
            "acuracia": round(accuracy_score(y_teste, y_pred_base), 4),
            "f1_score_macro": round(f1_score(y_teste, y_pred_base, average='macro', zero_division=0), 4)
        },
        "ia_random_forest": {
            "acuracia": round(accuracy_score(y_teste, y_pred_ia), 4),
            "precisao_macro": round(precision_score(y_teste, y_pred_ia, average='macro', zero_division=0), 4),
            "recall_macro": round(recall_score(y_teste, y_pred_ia, average='macro', zero_division=0), 4),
            "f1_score_macro": round(f1_score(y_teste, y_pred_ia, average='macro', zero_division=0), 4)
        }
    }
    
    # 5. Salva as evidências em JSON e o modelo treinado
    with open("metricas_v1.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    joblib.dump(pipeline_ia, "modelo_preco_certo.pkl")
    
    # Exibe comparação no terminal
    print("\n" + "="*45)
    print("      COMPARAÇÃO: BASELINE VS IA PRINCIPAL")
    print("="*45)
    print(f"Baseline (Acurácia):       {resultados['baseline_dummy']['acuracia']*100:.2f}%")
    print(f"IA Random Forest (Acur.):  {resultados['ia_random_forest']['acuracia']*100:.2f}%")
    print(f"IA Random Forest (F1):     {resultados['ia_random_forest']['f1_score_macro']*100:.2f}%")
    print("="*45)
    print("Ficheiros 'metricas_v1.json' e 'modelo_preco_certo.pkl' salvos!\n")

def cli(etapa):
    if etapa == 'limpeza':
        print("=== A INICIAR ETAPA: LIMPEZA DE DADOS ===")
        df_tratado = executar_limpeza()
        gerar_analises(df_tratado)
        print("=== ETAPA LIMPEZA FINALIZADA COM SUCESSO ===")
    elif etapa == 'treinar':
        print("=== A INICIAR ETAPA: TREINO E AVALIAÇÃO ===")
        executar_treino_avaliacao()
        print("=== ETAPA TREINO FINALIZADA COM SUCESSO ===")
    elif etapa == 'tudo':
        df_tratado = executar_limpeza()
        gerar_analises(df_tratado)
        executar_treino_avaliacao()
    else:
        print(f"A etapa '{etapa}' ainda não foi configurada no sistema.")

if __name__ == '__main__':
    cli('tudo')