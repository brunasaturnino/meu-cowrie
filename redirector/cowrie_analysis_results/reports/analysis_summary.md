# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: 876
- **Comandos de teste**: 18
- **Iterações Random Walk**: 25
- **Fator de amortecimento (α)**: 0.6

## Expansão dos Conjuntos Semente
- **HonestyHumility_Positive**: 145 comandos
- **HonestyHumility_Negative**: 136 comandos
- **Emotionality_Positive**: 136 comandos
- **Emotionality_Negative**: 116 comandos
- **Extraversion_Positive**: 148 comandos
- **Extraversion_Negative**: 165 comandos
- **Agreeableness_Positive**: 129 comandos
- **Agreeableness_Negative**: 139 comandos
- **Conscientiousness_Positive**: 155 comandos
- **Conscientiousness_Negative**: 138 comandos
- **OpennessToExperience_Positive**: 167 comandos
- **OpennessToExperience_Negative**: 150 comandos

## Métricas de Classificação
- **Melhoria média nos escores**: 0.085
- **Classificações com melhoria positiva**: 61/108
- **Melhoria máxima**: 0.555

## Distribuição de Personalidade (Percentis)
- **HonestyHumility**: 3 comandos (16.7%)
- **Emotionality**: 3 comandos (16.7%)
- **Extraversion**: 1 comandos (5.6%)
- **Agreeableness**: 1 comandos (5.6%)
- **Conscientiousness**: 2 comandos (11.1%)
- **OpennessToExperience**: 8 comandos (44.4%)

## Perfil do Atacante (Baseado em Percentil da Média)

- **Personalidade dominante**: Agreeableness (POSITIVO)
- **Percentil da média**: 79.2%
- **Interpretação**: A média do atacante é mais agreeableness que 79% dos comandos
- **Comandos analisados**: 18

### Scores Médios Brutos:
- **HonestyHumility**: 0.249
- **Emotionality**: 0.331
- **Extraversion**: 0.264
- **Agreeableness**: 0.236
- **Conscientiousness**: 0.227
- **OpennessToExperience**: 0.359

### Percentis da Média:
- **HonestyHumility**: 71.7% (POSITIVO)
- **Emotionality**: 76.1% (POSITIVO)
- **Extraversion**: 77.6% (POSITIVO)
- **Agreeableness**: 79.2% (POSITIVO)
- **Conscientiousness**: 68.2% (POSITIVO)
- **OpennessToExperience**: 79.1% (POSITIVO)

### Análise Detalhada:
→ Atacante cooperativo, prefere comandos colaborativos e flexíveis
→ A média dos seus comandos demonstra alta cordialidade

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
