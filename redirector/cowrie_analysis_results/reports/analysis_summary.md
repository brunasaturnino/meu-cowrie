# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: 876
- **Comandos de teste**: 18
- **Iterações Random Walk**: 25
- **Fator de amortecimento (α)**: 0.6

## Expansão dos Conjuntos Semente
- **HonestyHumility_Positive**: 144 comandos
- **HonestyHumility_Negative**: 136 comandos
- **Emotionality_Positive**: 137 comandos
- **Emotionality_Negative**: 116 comandos
- **Extraversion_Positive**: 146 comandos
- **Extraversion_Negative**: 165 comandos
- **Agreeableness_Positive**: 130 comandos
- **Agreeableness_Negative**: 133 comandos
- **Conscientiousness_Positive**: 152 comandos
- **Conscientiousness_Negative**: 138 comandos
- **OpennessToExperience_Positive**: 161 comandos
- **OpennessToExperience_Negative**: 150 comandos

## Métricas de Classificação
- **Melhoria média nos escores**: 0.085
- **Classificações com melhoria positiva**: 61/108
- **Melhoria máxima**: 0.510

## Distribuição de Personalidade (Percentis)
- **HonestyHumility**: 3 comandos (16.7%)
- **Emotionality**: 2 comandos (11.1%)
- **Extraversion**: 3 comandos (16.7%)
- **Agreeableness**: 1 comandos (5.6%)
- **Conscientiousness**: 2 comandos (11.1%)
- **OpennessToExperience**: 7 comandos (38.9%)

## Perfil do Atacante (Baseado em Percentil da Média)

- **Personalidade dominante**: OpennessToExperience (POSITIVO)
- **Percentil da média**: 80.1%
- **Interpretação**: A média do atacante é mais opennesstoexperience que 80% dos comandos
- **Comandos analisados**: 18

### Scores Médios Brutos:
- **HonestyHumility**: 0.237
- **Emotionality**: 0.283
- **Extraversion**: 0.258
- **Agreeableness**: 0.234
- **Conscientiousness**: 0.258
- **OpennessToExperience**: 0.354

### Percentis da Média:
- **HonestyHumility**: 70.2% (POSITIVO)
- **Emotionality**: 72.4% (POSITIVO)
- **Extraversion**: 75.8% (POSITIVO)
- **Agreeableness**: 73.6% (POSITIVO)
- **Conscientiousness**: 71.7% (POSITIVO)
- **OpennessToExperience**: 80.1% (POSITIVO)

### Análise Detalhada:
→ Atacante criativo, emprega ferramentas inovadoras e exploratórias
→ A média dos seus comandos demonstra alta abertura a experiências

## Complexidade dos Comandos
- **Complexidade média**: 0.9 argumentos
- **Comandos simples (≤2 args)**: 18
- **Comandos complexos (>2 args)**: 0

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos (876 comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
- **Vectorizer**: TF-IDF global com gloss dos comandos

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*

- Δ médio (hold-out 20%): -0.018 (N=423)
