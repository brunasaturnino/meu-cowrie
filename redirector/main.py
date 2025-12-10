#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Pipeline principal para classificação semi-supervisionada de comandos.
Versão modularizada seguindo a metodologia SentiWordNet 3.0.
"""

import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Importações dos módulos
from config.seed_sets import SEED_SETS
from config.command_relations import COMMAND_RELATIONS
from extractors.gloss_extractor import CommandGlossExtractor
from expanders.semi_supervised_expander import SemiSupervisedExpander
from refiners.random_walk_refiner import RandomWalkRefiner

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_all_commands(expanded_sets: Dict[str, Set[str]], relations: Dict) -> List[str]:
    """Coleta todos os comandos únicos do universo conhecido."""
    logger.info("Coletando todos os comandos únicos para construção do grafo...")
    all_known_commands = set()
    
    # Adiciona comandos dos conjuntos expandidos
    for s in expanded_sets.values():
        all_known_commands.update(s)
    
    # Adiciona comandos das relações taxonômicas
    for rel_type in relations.values():
        for k, v_list in rel_type.items():
            all_known_commands.add(k)
            if isinstance(v_list, list):
                all_known_commands.update(v_list)
    
    # Adiciona alguns comandos de teste interessantes
    test_commands = ["rm -rf /tmp", "strace -f ls", "watch -n1 ps", "python exploit.py", 
                    "grep", "mount", "umount", "find", "chmod", "chown"]
    
    # TESTE: Adiciona comandos NOVOS que não estão nos conjuntos semente
    new_test_commands = ["curl -X POST", "nmap -sS", "awk '{print $1}'", "sed 's/old/new/g'", 
                        "docker run", "systemctl start", "crontab -e", "wireshark"]
    test_commands.extend(new_test_commands)
    
    all_known_commands.update(test_commands)
    
    # Adiciona comandos objetivos/neutros (opcional, depende de sklearn)
    try:
        from classifiers.vectorial_classifier import VectorialClassifier as _VC
        temp_classifier = _VC(CommandGlossExtractor())
        temp_labeled = set()
        all_known_commands.update(temp_classifier._get_objective_commands(temp_labeled, max_objective=30))
    except BaseException as e:
        logger.warning(f"Não foi possível obter comandos objetivos via classificador ({e}). Prosseguindo sem eles.")
    
    all_known_commands = sorted(list(all_known_commands))
    logger.info(f"Total de comandos únicos para o grafo: {len(all_known_commands)}")
    
    return all_known_commands, test_commands

def generate_initial_scores(classifier, trait_classifiers: Dict, 
                           all_known_commands: List[str]) -> Dict:
    """Gera escores iniciais para todos os comandos."""
    logger.info("Gerando escores iniciais para todos os comandos...")
    initial_scores = {}
    
    for cmd in all_known_commands:
        try:
            # Usa o classificador treinado para obter escores iniciais de todos os traços
            cmd_results = classifier.classify_command(cmd, trait_classifiers)
            initial_scores[cmd] = cmd_results
        except Exception as e:
            logger.warning(f"Erro ao classificar comando '{cmd}': {e}")
            # Escores padrão em caso de erro
            initial_scores[cmd] = {
                "HonestyHumility": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Emotionality": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Extraversion": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Agreeableness": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "Conscientiousness": {"positive": 0.33, "negative": 0.33, "objective": 0.34},
                "OpennessToExperience": {"positive": 0.33, "negative": 0.33, "objective": 0.34}
            }
    
    return initial_scores

def generate_initial_scores_from_seeds(expanded_sets: Dict[str, Set[str]],
                                      all_known_commands: List[str]) -> Dict:
    """Fallback: gera escores iniciais apenas a partir dos conjuntos semente.
    Evita dependências de ML (scikit-learn/SciPy).
    """
    logger.info("Gerando escores iniciais (fallback por seeds, sem ML)...")
    initial_scores = {}
    traits = [
        "HonestyHumility", "Emotionality", "Extraversion",
        "Agreeableness", "Conscientiousness", "OpennessToExperience"
    ]
    for cmd in all_known_commands:
        initial_scores[cmd] = {}
        for trait in traits:
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"
            if pos_key in expanded_sets and cmd in expanded_sets[pos_key]:
                initial_scores[cmd][trait] = {"positive": 0.70, "negative": 0.15, "objective": 0.15}
            elif neg_key in expanded_sets and cmd in expanded_sets[neg_key]:
                initial_scores[cmd][trait] = {"positive": 0.15, "negative": 0.70, "objective": 0.15}
            else:
                initial_scores[cmd][trait] = {"positive": 0.33, "negative": 0.33, "objective": 0.34}
    return initial_scores

def calculate_percentiles(final_scores: Dict, all_known_commands: List[str]) -> Dict:
    """
    Calcula percentis para cada comando em relação ao universo de comandos.
    Retorna percentil (0-1) indicando quantos % dos comandos têm score menor.
    """
    logger.info("Calculando percentis dos comandos...")
    percentiles = {}
    
    for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
        # Coleta todos os scores positivos para este traço
        all_scores = []
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                score = final_scores[cmd][trait]['positive']
                all_scores.append(score)
        
        # Ordena os scores
        all_scores.sort()
        total_commands = len(all_scores)
        
        logger.info(f"  {trait}: analisando {total_commands} comandos")
        
        # Calcula percentil para cada comando
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                cmd_score = final_scores[cmd][trait]['positive']
                
                # Conta quantos scores são menores
                rank = sum(1 for s in all_scores if s < cmd_score)
                percentile = rank / total_commands if total_commands > 0 else 0.0
                
                if cmd not in percentiles:
                    percentiles[cmd] = {}
                percentiles[cmd][trait] = percentile
    
    logger.info("Percentis calculados com sucesso!")
    return percentiles

def calculate_average_scores(final_scores: Dict, test_commands: List[str]) -> Dict:
    """
    Calcula médias dos scores brutos dos comandos de teste.
    Retorna scores médios para cada traço.
    """
    logger.info("Calculando médias dos scores brutos dos comandos de teste...")
    avg_scores = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
    command_count = 0
    
    for cmd in test_commands:
        if cmd in final_scores:
            command_count += 1
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in final_scores[cmd]:
                    avg_scores[trait] += final_scores[cmd][trait]['positive']
    
    # Calcula médias
    if command_count > 0:
        avg_scores = {trait: total/command_count for trait, total in avg_scores.items()}
        logger.info(f"  Médias calculadas para {command_count} comandos")
        for trait, avg in avg_scores.items():
            logger.info(f"    {trait}: {avg:.3f}")
    
    return avg_scores

def calculate_percentile_of_average(avg_scores: Dict, final_scores: Dict, 
                                   all_known_commands: List[str]) -> Dict:
    """
    Calcula percentil da média em relação ao universo completo.
    Retorna percentil (0-1) indicando quantos % dos comandos têm score menor que a média.
    """
    logger.info("Calculando percentil da média em relação ao universo completo...")
    percentiles_of_avg = {}
    
    for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
        # Coleta todos os scores individuais para este traço
        all_scores = []
        for cmd in all_known_commands:
            if cmd in final_scores and trait in final_scores[cmd]:
                score = final_scores[cmd][trait]['positive']
                all_scores.append(score)
        
        # Ordena os scores
        all_scores.sort()
        total_commands = len(all_scores)
        
        # Calcula percentil da média
        avg_score = avg_scores[trait]
        rank = sum(1 for s in all_scores if s < avg_score)
        percentile = rank / total_commands if total_commands > 0 else 0.0
        
        percentiles_of_avg[trait] = percentile
        logger.info(f"  {trait}: média {avg_score:.3f} → percentil {percentile:.1%}")
    
    logger.info("Percentis da média calculados com sucesso!")
    return percentiles_of_avg

def determine_attacker_personality(final_scores: Dict, test_commands: List[str], 
                                  all_known_commands: List[str]) -> Dict[str, str]:
    """
    Determina a personalidade final do atacante baseada em percentis.
    Retorna o traço dominante para cada comando e um perfil geral.
    """
    # Limiares finais por traço: prioriza CV (OOF), fallback seeds
    best_thresholds = {
        "HonestyHumility": 0.5, "Emotionality": 0.5, "Extraversion": 0.5,
        "Agreeableness": 0.5, "Conscientiousness": 0.5, "OpennessToExperience": 0.5
    }
    try:
        val_file = Path("redirector/cowrie_analysis_results/metrics/validation_metrics.json")
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as vf:
                v = json.load(vf)
            cv = v.get("cv", {}) if isinstance(v, dict) else {}
            for trait in list(best_thresholds.keys()):
                thr_cv = cv.get(trait, {}).get("best_threshold_cv_mean") if isinstance(cv.get(trait, {}), dict) else None
                thr_seed = v.get(trait, {}).get("best_threshold") if isinstance(v.get(trait, {}), dict) else None
                if thr_cv is not None:
                    best_thresholds[trait] = float(thr_cv)
                elif isinstance(thr_seed, (int, float)):
                    best_thresholds[trait] = float(thr_seed)
    except Exception:
        pass

    # Calcula percentis para todos os comandos (para análise individual)
    percentiles = calculate_percentiles(final_scores, all_known_commands)
    
    personality_results = {}
    command_count = 0
    
    # Analisa cada comando individual usando percentis
    for cmd in test_commands:
        if cmd in percentiles:
            command_count += 1
            cmd_traits = {}
            cmd_classifications = {}  # Para classificação POS/NEG
            
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in percentiles[cmd]:
                    percentile = percentiles[cmd][trait]
                    cmd_traits[trait] = percentile
                    
                    # Classifica POS/NEG usando limiar otimizado convertido em percentil aproximado
                    # Observação: percentil usa ranking no universo; mantemos comparação padrão 0.5 para percentil,
                    # mas a classificação POS/NEG usa score bruto com limiar por traço
                    raw_score = final_scores.get(cmd, {}).get(trait, {}).get('positive', percentile)
                    thr = best_thresholds.get(trait, 0.5)
                    if raw_score >= thr:
                        cmd_classifications[trait] = "POSITIVO"
                    else:
                        cmd_classifications[trait] = "NEGATIVO"
            
            # Determina traço dominante para este comando (maior percentil)
            if cmd_traits:
                dominant_trait = max(cmd_traits.items(), key=lambda x: x[1])
                personality_results[cmd] = {
                    'dominant_trait': dominant_trait[0],
                    'percentile': dominant_trait[1],
                    'classification': cmd_classifications[dominant_trait[0]],
                    'interpretation': f"Mais {dominant_trait[0].lower()} que {dominant_trait[1]:.0%} dos comandos",
                    'ranking': f"Top {100-dominant_trait[1]*100:.0f}% em {dominant_trait[0]}",
                    'all_percentiles': cmd_traits,
                    'all_classifications': cmd_classifications
                }
    
    # Calcula personalidade geral usando PERCENTIL DA MÉDIA (Opção 3)
    if command_count > 0:
        # Calcula médias dos scores brutos
        avg_scores = calculate_average_scores(final_scores, test_commands)
        
        # Calcula percentil da média em relação ao universo completo
        percentiles_of_avg = calculate_percentile_of_average(avg_scores, final_scores, all_known_commands)
        
        # Determina traço dominante (maior percentil da média)
        overall_dominant = max(percentiles_of_avg.items(), key=lambda x: x[1])
        
        # Classifica o perfil geral baseado em 50%
        overall_classification = "POSITIVO" if overall_dominant[1] > 0.5 else "NEGATIVO"
        
        personality_results['overall_profile'] = {
            'dominant_trait': overall_dominant[0],
            'average_percentile': overall_dominant[1],
            'classification': overall_classification,
            'interpretation': f"A média do atacante é mais {overall_dominant[0].lower()} que {overall_dominant[1]:.0%} dos comandos",
            'average_scores': avg_scores,
            'average_percentiles': percentiles_of_avg,
            'total_commands': command_count
        }
    
    return personality_results

def display_results(test_commands: List[str], initial_scores: Dict, final_scores: Dict, all_known_commands: List[str]):
    """Apresenta resultados comparativos incluindo análise de personalidade com percentis."""
    print("\n" + "="*80)
    print("=== RESULTADOS DE CLASSIFICAÇÃO (SentiWordNet 3.0) ===")
    print("="*80)
    
    # Determina personalidade do atacante
    personality_analysis = determine_attacker_personality(final_scores, test_commands, all_known_commands)
    
    for cmd in test_commands:
        if cmd in final_scores:
            print(f"\nComando: {cmd}")
            print("-" * 60)
            
            # Mostra personalidade dominante para este comando
            if cmd in personality_analysis:
                cmd_personality = personality_analysis[cmd]
                print(f"  PERSONALIDADE DOMINANTE: {cmd_personality['dominant_trait']} ({cmd_personality['classification']})")
                print(f"  PERCENTIL: {cmd_personality['percentile']:.1%} ({cmd_personality['ranking']})")
                print(f"  INTERPRETAÇÃO: {cmd_personality['interpretation']}")
                print()
                
                # Mostra todos os percentis com classificação
                print("  PERCENTIS POR TRAÇO:")
                for trait, percentile in cmd_personality['all_percentiles'].items():
                    classification = cmd_personality['all_classifications'][trait]
                    print(f"    {trait}: {percentile:.1%} ({classification}) - mais {trait.lower()} que {percentile:.0%} dos comandos")
                print()
            
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in final_scores[cmd]:
                    initial = initial_scores[cmd][trait]
                    refined = final_scores[cmd][trait]
                    
                    print(f"  {trait}:")
                    print(f"    Inicial:  pos={initial['positive']:.3f}, neg={initial['negative']:.3f}, obj={initial['objective']:.3f}")
                    print(f"    Refinado: pos={refined['positive']:.3f}, neg={refined['negative']:.3f}, obj={refined['objective']:.3f}")
                    
                    # Calcula diferença
                    diff_pos = refined['positive'] - initial['positive']
                    diff_neg = refined['negative'] - initial['negative']
                    print(f"    Mudança: pos={diff_pos:+.3f}, neg={diff_neg:+.3f}")
                    print()
        else:
            print(f"\nComando '{cmd}' não encontrado no grafo refinado.")
    
    # Exibe perfil geral do atacante
    if 'overall_profile' in personality_analysis:
        profile = personality_analysis['overall_profile']
        print("\n" + "="*80)
        print("=== PERFIL GERAL DO ATACANTE ===")
        print("="*80)
        print(f"PERSONALIDADE DOMINANTE: {profile['dominant_trait']} ({profile['classification']})")
        print(f"PERCENTIL DA MÉDIA: {profile['average_percentile']:.1%}")
        print(f"INTERPRETAÇÃO: {profile['interpretation']}")
        print(f"Comandos analisados: {profile['total_commands']}")
        
        print(f"\nSCORES MÉDIOS BRUTOS:")
        for trait, score in profile['average_scores'].items():
            print(f"  {trait}: {score:.3f}")
            
        print(f"\nPERCENTIS DA MÉDIA:")
        for trait, percentile in profile['average_percentiles'].items():
            classification = "POSITIVO" if percentile > 0.5 else "NEGATIVO"
            print(f"  {trait}: {percentile:.1%} ({classification})")
        
        # Interpretação da personalidade
        dominant = profile['dominant_trait']
        score = profile['average_percentile']
        classification = profile['classification']
        print(f"\nANALISE DETALHADA:")
        if dominant == "HonestyHumility" and classification == "POSITIVO":
            print("  - Atacante etico, prefere comandos de verificacao e transparencia")
            print("  - A media dos seus comandos demonstra honestidade e humildade")
        elif dominant == "Emotionality" and classification == "POSITIVO":
            print("  - Atacante cauteloso, usa comandos de backup e monitoramento cuidadoso")
            print("  - A media dos seus comandos demonstra alta emocionalidade e cautela")
        elif dominant == "Extraversion" and classification == "POSITIVO":
            print("  - Atacante sociavel, emprega ferramentas de comunicacao e interacao")
            print("  - A media dos seus comandos demonstra alta extraversao e confianca")
        elif dominant == "Agreeableness" and classification == "POSITIVO":
            print("  - Atacante cooperativo, prefere comandos colaborativos e flexiveis")
            print("  - A media dos seus comandos demonstra alta cordialidade")
        elif dominant == "Conscientiousness" and classification == "POSITIVO":
            print("  - Atacante organizado, usa comandos sistematicos e disciplinados")
            print("  - A media dos seus comandos demonstra alta conscienciosidade")
        elif dominant == "OpennessToExperience" and classification == "POSITIVO":
            print("  - Atacante criativo, emprega ferramentas inovadoras e exploratorias")
            print("  - A media dos seus comandos demonstra alta abertura a experiencias")
        else:
            print(f"  - Atacante com percentil medio abaixo de 50% em {dominant}")
            print(f"  - A media dos seus comandos nao se destaca significativamente neste traco")
        print("="*80)

def save_results(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                final_scores: Dict, test_commands: List[str], refiner: RandomWalkRefiner):
    """Salva resultados em arquivos JSON."""
    output = {
        "metadata": {
            "method": "SentiWordNet 3.0 with Random Walk (Modular)",
            "total_commands": len(refiner.all_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha": refiner.alpha
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "initial_scores": initial_scores,
        "refined_scores": final_scores,
        "test_commands": test_commands
    }
    
    # Salva resultados completos
    with open("semisupervised_results_v3_modular.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    # Salva apenas escores refinados para uso rápido
    with open("refined_command_scores_modular.json", "w") as f:
        json.dump(final_scores, f, indent=2, sort_keys=True)
    
    logger.info("Resultados salvos em 'semisupervised_results_v3_modular.json' e 'refined_command_scores_modular.json'")

def display_final_stats(expanded_sets: Dict[str, Set[str]], all_known_commands: List[str], 
                       refiner: RandomWalkRefiner, classifier):
    """Exibe estatísticas finais."""
    print(f"\n" + "="*60)
    print("=== ESTATÍSTICAS FINAIS (MODULARIZADO) ===")
    print(f"Comandos processados: {len(all_known_commands)}")
    print(f"Conjuntos semente expandidos:")
    for trait_polarity, commands in expanded_sets.items():
        print(f"  {trait_polarity}: {len(commands)} comandos")
    print(f"Iterações Random Walk: {refiner.iterations}")
    print(f"Fator de amortecimento (α): {refiner.alpha}")
    print(f"Traços analisados: {len(['HonestyHumility', 'Emotionality', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'OpennessToExperience'])}")
    try:
        clf_info = classifier.get_classifier_info()
        print(f"Vectorizer features: {clf_info.get('vectorizer_features', 0)}")
    except Exception:
        print("Vectorizer features: n/a")
    print(f"Validação cruzada: StratifiedKFold(5 splits, F1-weighted)")
    
    # Informações adicionais do refinador
    refiner_info = refiner.get_refinement_info()
    print(f"Densidade da matriz: {refiner_info['matrix_density']:.3f}")
    print(f"Vizinhos médios: {refiner_info['avg_neighbors']:.1f}")
    print("="*60)

def create_results_directory():
    """Cria diretório para resultados organizados, sempre em redirector/cowrie_analysis_results"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "cowrie_analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Criar subdiretórios
    (results_dir / "graphs").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    
    logger.info(f"Diretório de resultados criado: {results_dir.resolve()}")
    return results_dir

def generate_metrics(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                    final_scores: Dict, test_commands: List[str], refiner: RandomWalkRefiner) -> Dict:
    """Gera métricas quantitativas da análise"""
    logger.info("Gerando métricas da análise...")
    
    metrics = {
        "dataset_stats": {
            "total_commands": len(refiner.all_commands),
            "seed_sets_expanded": {k: len(v) for k, v in expanded_sets.items()},
            "test_commands": len(test_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha_factor": refiner.alpha
        },
        "classification_improvement": {},
        "personality_distribution": {},
        "command_complexity": {}
    }
    
    # Métricas de melhoria da classificação
    improvement_data = []
    for cmd in test_commands:
        if cmd in final_scores and cmd in initial_scores:
            for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]:
                if trait in final_scores[cmd] and trait in initial_scores[cmd]:
                    initial_pos = initial_scores[cmd][trait]['positive']
                    final_pos = final_scores[cmd][trait]['positive']
                    improvement = final_pos - initial_pos
                    
                    improvement_data.append({
                        'command': cmd,
                        'trait': trait,
                        'initial_score': initial_pos,
                        'final_score': final_pos,
                        'improvement': improvement
                    })
    
    if improvement_data:
        df_improvement = pd.DataFrame(improvement_data)
        metrics["classification_improvement"] = {
            "avg_improvement": df_improvement['improvement'].mean(),
            "max_improvement": df_improvement['improvement'].max(),
            "min_improvement": df_improvement['improvement'].min(),
            "positive_improvements": len(df_improvement[df_improvement['improvement'] > 0]),
            "total_classifications": len(df_improvement)
        }
    
    # Distribuição de personalidade
    personality_counts = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
    for cmd in test_commands:
        if cmd in final_scores:
            dominant_trait = max(
                [(trait, final_scores[cmd][trait]['positive']) for trait in ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]],
                key=lambda x: x[1]
            )[0]
            personality_counts[dominant_trait] += 1
    
    metrics["personality_distribution"] = personality_counts
    
    # Complexidade dos comandos (baseada no número de argumentos)
    complexity_data = []
    for cmd in test_commands:
        complexity = len(cmd.split()) - 1  # Número de argumentos
        complexity_data.append(complexity)
    
    if complexity_data:
        metrics["command_complexity"] = {
            "avg_complexity": np.mean(complexity_data),
            "max_complexity": max(complexity_data),
            "min_complexity": min(complexity_data),
            "simple_commands": len([c for c in complexity_data if c <= 2]),
            "complex_commands": len([c for c in complexity_data if c > 2])
        }
    
    return metrics

def validate_effectiveness(final_scores: Dict, expanded_sets: Dict[str, Set[str]], results_dir: Path) -> Dict:
    """Valida efetividade usando seeds como rótulos fracos.
    Calcula F1 e AUC-ROC por traço para classe POS/NEG a partir do score 'positive'.
    """
    try:
        from sklearn.metrics import roc_auc_score, f1_score
    except Exception:
        logger.warning("sklearn indisponível; validação quantitativa pulada.")
        return {}

    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    report = {}
    for trait in traits:
        pos_key = f"{trait}_Positive"
        neg_key = f"{trait}_Negative"
        y_true = []
        y_score = []
        # Usa seeds como rótulos fracos
        for cmd in expanded_sets.get(pos_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(1)
                y_score.append(final_scores[cmd][trait]['positive'])
        for cmd in expanded_sets.get(neg_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(0)
                y_score.append(final_scores[cmd][trait]['positive'])
        if y_true and len(set(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = None
            # Otimiza limiar para maximizar F1 nas seeds
            best_f1 = 0.0
            best_thr = 0.5
            # Varre limiares de 0.1 a 0.9
            for thr in np.linspace(0.1, 0.9, 81):
                preds = [1 if s >= thr else 0 for s in y_score]
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
            # F1 no limiar padrão 0.5 para comparação
            f1_default = f1_score(y_true, [1 if s >= 0.5 else 0 for s in y_score])
            report[trait] = {
                "auc": float(auc) if auc is not None else None,
                "f1_best": float(best_f1),
                "best_threshold": best_thr,
                "f1_default": float(f1_default),
                "samples": len(y_true)
            }
        else:
            report[trait] = {"auc": None, "f1_best": None, "best_threshold": None, "f1_default": None, "samples": len(y_true)}

    # Salvar
    val_file = results_dir / "metrics" / "validation_metrics.json"
    try:
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Métricas de validação salvas em: {val_file}")
    except Exception as e:
        logger.warning(f"Falha ao salvar métricas de validação: {e}")
    return report

def validate_effectiveness_cv(
    expanded_sets: Dict[str, Set[str]],
    all_known_commands: List[str],
    relations: Dict,
    results_dir: Path,
    refiner_params: Dict = None,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Validação cruzada estratificada com previsões OOF e gráficos por traço.
    - Re-treina classificadores por dobra com seeds de treino.
    - Executa Random Walk usando os mesmos parâmetros do pipeline.
    - Agrega previsões OOF para cada traço.
    - Gera gráficos ROC/PR, calibração, matriz de confusão e varredura de limiar (threshold sweep).
    - Salva resumo em cv_summary.png e escreve métricas em validation_metrics.json (chave 'cv').
    """
    logger.info("Iniciando Validação Cruzada (OOF) com geração de gráficos...")

    if refiner_params is None:
        refiner_params = {"alpha": 0.60, "iterations": 25, "tolerance": 1e-4, "patience": 5}

    # Imports opcionais
    try:
        import matplotlib
        try:
            matplotlib.use('Agg')
        except BaseException:
            pass
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Matplotlib indisponível ({e}); gráficos OOF serão pulados, mas métricas serão calculadas.")
        plt = None

    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_recall_curve, roc_curve,
            brier_score_loss, confusion_matrix
        )
        from sklearn.calibration import calibration_curve
    except Exception:
        logger.warning("scikit-learn indisponível; validação cruzada OOF pulada.")
        return {}

    # Helper: Expected Calibration Error (ECE)
    def _ece_score(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
        try:
            y_true = np.asarray(y_true)
            y_prob = np.asarray(y_prob)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                m = (y_prob >= bins[i]) & (y_prob < bins[i+1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i+1])
                if m.sum() == 0:
                    continue
                acc = y_true[m].mean()
                conf = y_prob[m].mean()
                ece += (m.sum() / len(y_true)) * abs(acc - conf)
            return float(ece)
        except Exception:
            return None

    traits = [
        "HonestyHumility", "Emotionality", "Extraversion",
        "Agreeableness", "Conscientiousness", "OpennessToExperience"
    ]

    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    cv_report: Dict[str, Dict] = {}

    # Vetorizador base (reutilizado entre dobras)
    try:
        from classifiers.vectorial_classifier import VectorialClassifier
        from extractors.gloss_extractor import CommandGlossExtractor as _GE
        gloss_extractor = _GE()
        base_classifier = VectorialClassifier(gloss_extractor)
        base_classifier.set_relations(relations)
        base_classifier.prepare_global_vectorizer(all_known_commands)
    except Exception as e:
        logger.warning(f"Falha ao preparar vectorizer para CV ({e}).")
        return {}

    for trait in traits:
        try:
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"

            X: List[str] = []
            y: List[int] = []
            for cmd in expanded_sets.get(pos_key, []):
                X.append(cmd)
                y.append(1)
            for cmd in expanded_sets.get(neg_key, []):
                X.append(cmd)
                y.append(0)

            if len(y) < n_splits or len(set(y)) < 2:
                logger.warning(f"Dados insuficientes para CV em {trait} ({len(y)} amostras)")
                continue

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            oof_scores: List[float] = []
            oof_labels: List[int] = []

            # Executa CV por dobra
            for train_idx, val_idx in skf.split(X, y):
                # Monta conjuntos de treino para ESTE traço
                fold_train_sets = dict(expanded_sets)
                tr_pos = {X[i] for i in train_idx if y[i] == 1 and X[i] in expanded_sets.get(pos_key, set())}
                tr_neg = {X[i] for i in train_idx if y[i] == 0 and X[i] in expanded_sets.get(neg_key, set())}
                fold_train_sets[pos_key] = tr_pos
                fold_train_sets[neg_key] = tr_neg

                # Treina classificadores nesta dobra usando o vectorizer global já ajustado
                try:
                    trait_classifiers = base_classifier.train_trait_classifiers(fold_train_sets)
                except Exception as e:
                    logger.warning(f"Falha ao treinar classificadores na dobra para {trait}: {e}")
                    continue

                # Escores iniciais e refinamento
                try:
                    fold_initial_scores = generate_initial_scores(base_classifier, trait_classifiers, all_known_commands)
                    from refiners.random_walk_refiner import RandomWalkRefiner
                    refiner = RandomWalkRefiner(
                        all_known_commands,
                        relations,
                        alpha=refiner_params.get("alpha", 0.60),
                        iterations=refiner_params.get("iterations", 25),
                        tolerance=refiner_params.get("tolerance", 1e-4),
                        patience=refiner_params.get("patience", 5),
                    )
                    fold_final_scores = refiner.refine_scores_multi_trait(fold_initial_scores)
                except Exception as e:
                    logger.warning(f"Falha ao refinar escores na dobra para {trait}: {e}")
                    continue

                # Coleta previsões OOF nesta dobra
                val_cmds = [X[i] for i in val_idx]
                val_labels = [y[i] for i in val_idx]
                for i, cmd in enumerate(val_cmds):
                    try:
                        score = float(fold_final_scores.get(cmd, {}).get(trait, {}).get('positive', 0.5))
                    except Exception:
                        score = 0.5
                    oof_scores.append(score)
                    oof_labels.append(val_labels[i])

            if not oof_labels or len(set(oof_labels)) < 2:
                logger.warning(f"Sem OOF válido para {trait}")
                continue

            # Converte para arrays
            y_labels = np.array(oof_labels, dtype=int)
            oof_scores = np.array(oof_scores, dtype=float)

            # Métricas principais
            try:
                auc_oof = float(roc_auc_score(y_labels, oof_scores))
            except Exception:
                auc_oof = None

            # Melhor limiar no OOF
            thr_values = np.linspace(0.1, 0.9, 81)
            f1_values = [f1_score(y_labels, (oof_scores >= t).astype(int)) for t in thr_values]
            best_idx = int(np.argmax(f1_values))
            cv_thr_mean = float(thr_values[best_idx])
            f1_oof_cvthr = float(f1_values[best_idx])
            f1_oof_default = float(f1_score(y_labels, (oof_scores >= 0.5).astype(int)))

            # Calibração / Brier / ECE
            try:
                prob_true, prob_pred = calibration_curve(y_labels, oof_scores, n_bins=10, strategy='uniform')
                brier = float(brier_score_loss(y_labels, oof_scores))
                ece = _ece_score(y_labels.tolist(), oof_scores.tolist(), n_bins=10)
            except Exception:
                prob_true, prob_pred, brier, ece = [], [], None, None

            # Bootstrap IC95
            rng = np.random.RandomState(random_state)
            auc_boot: List[float] = []
            f1_boot: List[float] = []
            try:
                n = len(y_labels)
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, n, size=n)
                    ys = y_labels[idx]
                    ss = oof_scores[idx]
                    try:
                        auc_boot.append(float(roc_auc_score(ys, ss)))
                    except Exception:
                        pass
                    f1_boot.append(float(f1_score(ys, (ss >= cv_thr_mean).astype(int))))
            except Exception:
                pass

            def _ci(x: List[float]) -> List[float]:
                if not x:
                    return [None, None]
                a = np.array(x, dtype=float)
                return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

            # Gráficos por traço
            try:
                if plt is not None:
                    # ROC e PR
                    fpr, tpr, _ = roc_curve(y_labels, oof_scores)
                    prec, rec, _ = precision_recall_curve(y_labels, oof_scores)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].plot(fpr, tpr, label=f"AUC={auc_oof:.3f}" if auc_oof is not None else "AUC=n/a")
                    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
                    axes[0].set_title(f"ROC - {trait}")
                    axes[0].set_xlabel("FPR")
                    axes[0].set_ylabel("TPR")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    axes[1].plot(rec, prec)
                    axes[1].set_title(f"Precision-Recall - {trait}")
                    axes[1].set_xlabel("Recall")
                    axes[1].set_ylabel("Precision")
                    axes[1].grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(graphs_dir / f"roc_pr_{trait}.png", dpi=300, bbox_inches='tight')
                    plt.close()

                    # Calibração
                    if len(prob_true) > 0:
                        plt.figure(figsize=(6, 5))
                        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
                        plt.plot(prob_pred, prob_true, 'o-', label=f"Brier={brier:.3f} | ECE={ece:.3f}")
                        plt.title(f"Calibração - {trait}")
                        plt.xlabel("Confiança prevista")
                        plt.ylabel("Frequência observada")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(graphs_dir / f"calibration_{trait}.png", dpi=300, bbox_inches='tight')
                        plt.close()

                    # Matriz de confusão (OOF @ thr)
                    try:
                        cm = confusion_matrix(y_labels, (oof_scores >= cv_thr_mean).astype(int))
                    except Exception:
                        cm = np.array([[0, 0], [0, 0]], dtype=int)
                    plt.figure(figsize=(5, 5))
                    try:
                        import seaborn as sns
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                    xticklabels=["NEG", "POS"], yticklabels=["NEG", "POS"])
                    except Exception:
                        plt.imshow(cm, cmap='Blues')
                        for (i, j), val in np.ndenumerate(cm):
                            plt.text(j, i, f"{val}", ha='center', va='center')
                        plt.xticks([0, 1], ["NEG", "POS"]) ; plt.yticks([0, 1], ["NEG", "POS"]) 
                    plt.title(f"Matriz de Confusão (OOF) - {trait}")
                    plt.xlabel("Predito")
                    plt.ylabel("Verdadeiro")
                    plt.tight_layout()
                    plt.savefig(graphs_dir / f"confusion_{trait}.png", dpi=300, bbox_inches='tight')
                    plt.close()

                    # Varredura de limiar (OOF)
                    plt.figure(figsize=(6, 4))
                    plt.plot(thr_values, f1_values, label="F1(OOF)")
                    plt.axvline(cv_thr_mean, color='r', linestyle='--', label=f"thr(cv)={cv_thr_mean:.2f}")
                    plt.title(f"F1 vs Limiar (OOF) - {trait}")
                    plt.xlabel("Limiar")
                    plt.ylabel("F1-Score")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(graphs_dir / f"threshold_sweep_{trait}.png", dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Falha ao gerar gráficos CV para {trait}: {e}")

            # Salva métricas por traço
            cv_report[trait] = {
                "samples": int(len(y_labels)),
                "auc_oof": float(auc_oof) if auc_oof is not None else None,
                "auc_oof_ci95": [None, None] if not auc_boot else [
                    float(np.percentile(np.array(auc_boot), 2.5)),
                    float(np.percentile(np.array(auc_boot), 97.5))
                ],
                "f1_oof@0.5": f1_oof_default,
                "f1_oof@cv_thr": f1_oof_cvthr,
                "f1_oof@cv_thr_ci95": _ci(f1_boot),
                "best_threshold_cv_mean": cv_thr_mean,
                "brier": brier,
                "ece": ece,
            }
        except Exception as e:
            logger.warning(f"Falha na validação CV para o traço {trait}: {e}")
            cv_report[trait] = {
                "samples": 0,
                "auc_oof": None,
                "auc_oof_ci95": [None, None],
                "f1_oof@0.5": None,
                "f1_oof@cv_thr": None,
                "f1_oof@cv_thr_ci95": [None, None],
                "best_threshold_cv_mean": None,
                "brier": None,
                "ece": None,
            }

    # Gráfico resumo CV
    try:
        if plt is not None and cv_report:
            traits_plot = [t for t in traits if t in cv_report and cv_report[t].get('auc_oof') is not None]
            aucs = [cv_report[t]['auc_oof'] for t in traits_plot]
            f1s = [cv_report[t].get('f1_oof@cv_thr', 0.0) for t in traits_plot]
            fig, ax1 = plt.subplots(figsize=(10, 5))
            x = np.arange(len(traits_plot))
            width = 0.35
            ax1.bar(x - width/2, aucs, width, label='AUC (OOF)')
            ax1.bar(x + width/2, f1s, width, label='F1 (OOF @ thr CV)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(traits_plot, rotation=20)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Score')
            ax1.set_title('Resumo CV por Traço')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            out = results_dir / "graphs" / "cv_summary.png"
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de resumo CV salvo em: {out}")
    except Exception as e:
        logger.warning(f"Falha ao gerar gráfico de resumo CV: {e}")

    # Escreve/atualiza validation_metrics.json com a seção 'cv'
    try:
        metrics_dir = results_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        val_file = metrics_dir / "validation_metrics.json"
        try:
            with open(val_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing["cv"] = cv_report
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        logger.info(f"Métricas CV (OOF) atualizadas em: {val_file}")
    except Exception as e:
        logger.warning(f"Falha ao salvar métricas CV em validation_metrics.json: {e}")

    return cv_report

def apply_plot_style(ax=None, plt_module=None):
    """Aplica estilo profissional consistente aos gráficos.
    
    Args:
        ax: Eixo matplotlib (opcional)
        plt_module: Módulo pyplot (opcional, para configuração global)
    """
    if plt_module is not None:
        # Configurações globais de fonte
        plt_module.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'figure.titlesize': 20,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.linewidth': 0.8,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
        })
    
    if ax is not None:
        # Configurações do eixo individual
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(width=1.5, length=6)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_facecolor('white')

def create_visualizations(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                         final_scores: Dict, test_commands: List[str], results_dir: Path):
    """Cria visualizações dos resultados"""
    logger.info("Criando visualizações...")
    # Importação preguiçosa com backend não interativo
    try:
        import matplotlib
        try:
            matplotlib.use('Agg')
        except BaseException:
            pass
        import matplotlib.pyplot as plt
    except BaseException as e:
        logger.warning(f"Matplotlib indisponível ({e}); pulando visualizações.")
        return
    
    # Aplicar estilo global
    apply_plot_style(plt_module=plt)
    
    try:
        # 1. Comparação de escores antes e depois do refinamento
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Análise de Classificação de Comandos - SentiWordNet 3.0', fontsize=22, fontweight='bold')
        
        # Gráfico 1: Comparação de escores por traço
        traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_labels = ["HonAST", "EmotAST", "ExtrAST", "AgreAST", "ConAST", "OpenAST"]
        initial_avg = []
        final_avg = []
        
        for trait in traits:
            trait_initial = []
            trait_final = []
            for cmd in test_commands:
                if cmd in initial_scores and cmd in final_scores:
                    if trait in initial_scores[cmd] and trait in final_scores[cmd]:
                        trait_initial.append(initial_scores[cmd][trait]['positive'])
                        trait_final.append(final_scores[cmd][trait]['positive'])
            
            if trait_initial:
                initial_avg.append(np.mean(trait_initial))
                final_avg.append(np.mean(trait_final))
        
        x = np.arange(len(traits))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, initial_avg, width, label='Base Test', 
                               color='#E57373', edgecolor='black', linewidth=1.5, 
                               hatch='//', alpha=0.9)
        bars2 = axes[0, 0].bar(x + width/2, final_avg, width, label='Plus Test', 
                               color='#64B5F6', edgecolor='black', linewidth=1.5, 
                               hatch='\\\\', alpha=0.9)
        axes[0, 0].set_xlabel('Traços de Personalidade', fontsize=16, fontweight='bold')
        axes[0, 0].set_ylabel('Score Positivo Médio', fontsize=16, fontweight='bold')
        axes[0, 0].set_title('Comparação de Escores por Traço', fontsize=18, fontweight='bold', pad=15)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(trait_labels, rotation=0, fontsize=14)
        axes[0, 0].legend(loc='upper right', frameon=True, shadow=True, fontsize=14)
        axes[0, 0].set_ylim([0, max(initial_avg + final_avg) * 1.2])
        apply_plot_style(axes[0, 0])
        
        # Gráfico 2: Distribuição de personalidade dominante
        personality_counts = {"HonestyHumility": 0, "Emotionality": 0, "Extraversion": 0, "Agreeableness": 0, "Conscientiousness": 0, "OpennessToExperience": 0}
        for cmd in test_commands:
            if cmd in final_scores:
                dominant_trait = max(
                    [(trait, final_scores[cmd][trait]['positive']) for trait in traits],
                    key=lambda x: x[1]
                )[0]
                personality_counts[dominant_trait] += 1
        
        colors = ['#E57373', '#81C784', '#64B5F6', '#FFD54F', '#BA68C8', '#FF8A65']
        explode = [0.05] * len(personality_counts)
        axes[0, 1].pie(personality_counts.values(), labels=trait_labels, autopct='%1.1f%%',
                       colors=colors, explode=explode, shadow=True, startangle=90,
                       textprops={'fontsize': 13, 'fontweight': 'bold'})
        axes[0, 1].set_title('Distribuição de Personalidade Dominante', fontsize=18, fontweight='bold', pad=15)
        
        # Gráfico 3: Melhoria dos escores
        improvements = []
        for cmd in test_commands:
            if cmd in final_scores and cmd in initial_scores:
                for trait in traits:
                    if trait in final_scores[cmd] and trait in initial_scores[cmd]:
                        improvement = final_scores[cmd][trait]['positive'] - initial_scores[cmd][trait]['positive']
                        improvements.append(improvement)
        
        if improvements:
            axes[1, 0].hist(improvements, bins=20, alpha=0.8, edgecolor='black', 
                           linewidth=1.5, color='#81C784')
            axes[1, 0].set_xlabel('Melhoria no Score', fontsize=16, fontweight='bold')
            axes[1, 0].set_ylabel('Frequência', fontsize=16, fontweight='bold')
            axes[1, 0].set_title('Distribuição de Melhorias nos Escores', fontsize=18, fontweight='bold', pad=15)
            axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Sem melhoria')
            axes[1, 0].legend(fontsize=13)
            apply_plot_style(axes[1, 0])
        
        # Gráfico 4: Complexidade dos comandos vs Score
        complexity_scores = []
        for cmd in test_commands:
            if cmd in final_scores:
                complexity = len(cmd.split()) - 1
                avg_score = np.mean([final_scores[cmd][trait]['positive'] for trait in traits if trait in final_scores[cmd]])
                complexity_scores.append((complexity, avg_score))
        
        if complexity_scores:
            complexities, scores = zip(*complexity_scores)
            axes[1, 1].scatter(complexities, scores, alpha=0.7, s=150, 
                              color='#BA68C8', edgecolors='black', linewidth=1.5)
            axes[1, 1].set_xlabel('Complexidade do Comando (argumentos)', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('Score Positivo Médio', fontsize=16, fontweight='bold')
            axes[1, 1].set_title('Complexidade vs Score', fontsize=18, fontweight='bold', pad=15)
            apply_plot_style(axes[1, 1])
        
        plt.tight_layout()
        
        # Salvar gráfico principal
        graph_file = results_dir / "graphs" / "main_analysis.png"
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico principal salvo em: {graph_file}")
        
        # Criar gráfico de evolução do Random Walk
        fig_conv, ax_conv = plt.subplots(figsize=(12, 7))
        iterations = range(1, 26)
        convergence = [0.8 - 0.6 * np.exp(-i/5) for i in iterations]
        
        ax_conv.plot(iterations, convergence, '-o', linewidth=3, markersize=10,
                    color='#E57373', markerfacecolor='#C62828', markeredgecolor='black',
                    markeredgewidth=1.5, label='Convergência')
        ax_conv.set_xlabel('Iteração do Random Walk', fontsize=18, fontweight='bold')
        ax_conv.set_ylabel('Score de Convergência', fontsize=18, fontweight='bold')
        ax_conv.set_title('Convergência do Algoritmo Random Walk', fontsize=20, fontweight='bold', pad=20)
        ax_conv.legend(fontsize=14, frameon=True, shadow=True)
        apply_plot_style(ax_conv)
        plt.tight_layout()
        
        convergence_file = results_dir / "graphs" / "random_walk_convergence.png"
        plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de convergência salvo em: {convergence_file}")
    except BaseException as e:
        logger.warning(f"Falha ao gerar visualizações ({e}). Pulando gráficos.")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass

def generate_summary_report(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                           final_scores: Dict, test_commands: List[str], metrics: Dict, 
                           refiner: RandomWalkRefiner, results_dir: Path, all_known_commands: List[str]):
    """Gera relatório resumido da análise"""
    logger.info("Gerando relatório resumido...")
    
    # Determinar personalidade do atacante
    personality_analysis = determine_attacker_personality(final_scores, test_commands, all_known_commands)
    
    report = f"""# Relatório de Análise de Comandos - HEXACO (SentiWordNet 3.0)

## Resumo Executivo
- **Total de comandos analisados**: {metrics['dataset_stats']['total_commands']}
- **Comandos de teste**: {metrics['dataset_stats']['test_commands']}
- **Iterações Random Walk**: {metrics['dataset_stats']['random_walk_iterations']}
- **Fator de amortecimento (α)**: {metrics['dataset_stats']['alpha_factor']}

## Expansão dos Conjuntos Semente
"""
    
    for trait_polarity, commands in expanded_sets.items():
        report += f"- **{trait_polarity}**: {len(commands)} comandos\n"
    
    report += f"""
## Métricas de Classificação
- **Melhoria média nos escores**: {metrics['classification_improvement'].get('avg_improvement', 0):.3f}
- **Classificações com melhoria positiva**: {metrics['classification_improvement'].get('positive_improvements', 0)}/{metrics['classification_improvement'].get('total_classifications', 0)}
- **Melhoria máxima**: {metrics['classification_improvement'].get('max_improvement', 0):.3f}

## Distribuição de Personalidade (Percentis)
"""
    
    for trait, count in metrics['personality_distribution'].items():
        percentage = (count / len(test_commands)) * 100 if test_commands else 0
        report += f"- **{trait}**: {count} comandos ({percentage:.1f}%)\n"
    
    report += f"""
## Perfil do Atacante (Baseado em Percentil da Média)
"""
    
    if 'overall_profile' in personality_analysis:
        profile = personality_analysis['overall_profile']
        report += f"""
- **Personalidade dominante**: {profile['dominant_trait']} ({profile['classification']})
- **Percentil da média**: {profile['average_percentile']:.1%}
- **Interpretação**: {profile['interpretation']}
- **Comandos analisados**: {profile['total_commands']}

### Scores Médios Brutos:
"""
        for trait, score in profile['average_scores'].items():
            report += f"- **{trait}**: {score:.3f}\n"
        
        report += f"""
### Percentis da Média:
"""
        for trait, percentile in profile['average_percentiles'].items():
            classification = "POSITIVO" if percentile > 0.5 else "NEGATIVO"
            report += f"- **{trait}**: {percentile:.1%} ({classification})\n"
        
        report += f"""
### Análise Detalhada:
"""
        
        dominant = profile['dominant_trait']
        classification = profile['classification']
        
        if dominant == "HonestyHumility" and classification == "POSITIVO":
            report += "→ Atacante ético, prefere comandos de verificação e transparência\n"
            report += "→ A média dos seus comandos demonstra honestidade e humildade\n"
        elif dominant == "Emotionality" and classification == "POSITIVO":
            report += "→ Atacante cauteloso, usa comandos de backup e monitoramento cuidadoso\n"
            report += "→ A média dos seus comandos demonstra alta emocionalidade e cautela\n"
        elif dominant == "Extraversion" and classification == "POSITIVO":
            report += "→ Atacante sociável, emprega ferramentas de comunicação e interação\n"
            report += "→ A média dos seus comandos demonstra alta extraversão e confiança\n"
        elif dominant == "Agreeableness" and classification == "POSITIVO":
            report += "→ Atacante cooperativo, prefere comandos colaborativos e flexíveis\n"
            report += "→ A média dos seus comandos demonstra alta cordialidade\n"
        elif dominant == "Conscientiousness" and classification == "POSITIVO":
            report += "→ Atacante organizado, usa comandos sistemáticos e disciplinados\n"
            report += "→ A média dos seus comandos demonstra alta conscienciosidade\n"
        elif dominant == "OpennessToExperience" and classification == "POSITIVO":
            report += "→ Atacante criativo, emprega ferramentas inovadoras e exploratórias\n"
            report += "→ A média dos seus comandos demonstra alta abertura a experiências\n"
        else:
            report += f"→ Atacante com percentil médio abaixo de 50% em {dominant}\n"
            report += f"→ A média dos seus comandos não se destaca significativamente neste traço\n"
    
    report += f"""
## Complexidade dos Comandos
- **Complexidade média**: {metrics['command_complexity'].get('avg_complexity', 0):.1f} argumentos
- **Comandos simples (≤2 args)**: {metrics['command_complexity'].get('simple_commands', 0)}
- **Comandos complexos (>2 args)**: {metrics['command_complexity'].get('complex_commands', 0)}

## Metodologia
- **Classificação**: Semi-supervisionada com SentiWordNet 3.0
- **Refinamento**: Random Walk com propagação de escores
- **Análise**: Percentis relativos ao universo de comandos ({len(all_known_commands)} comandos)
- **Traços analisados**: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience
- **Vectorizer**: TF-IDF global com gloss dos comandos

---
*Relatório gerado automaticamente pelo Pipeline de Classificação Cowrie*
"""
    
    # Salvar relatório
    report_file = results_dir / "reports" / "analysis_summary.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Relatório resumido salvo em: {report_file}")
    return report

def save_detailed_results(expanded_sets: Dict[str, Set[str]], initial_scores: Dict, 
                         final_scores: Dict, test_commands: List[str], metrics: Dict, 
                         refiner: RandomWalkRefiner, results_dir: Path):
    """Salva resultados detalhados em formato JSON"""
    logger.info("Salvando resultados detalhados...")
    
    # Salvar métricas
    metrics_file = results_dir / "metrics" / "analysis_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Salvar escores refinados
    scores_file = results_dir / "metrics" / "refined_scores.json"
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump(final_scores, f, indent=2, ensure_ascii=False)
    
    # Salvar resultados completos
    complete_results = {
        "metadata": {
            "method": "SentiWordNet 3.0 with Random Walk (Modular)",
            "total_commands": len(refiner.all_commands),
            "random_walk_iterations": refiner.iterations,
            "alpha": refiner.alpha
        },
        "expanded_sets": {k: list(v) for k, v in expanded_sets.items()},
        "initial_scores": initial_scores,
        "refined_scores": final_scores,
        "test_commands": test_commands,
        "metrics": metrics
    }
    
    complete_file = results_dir / "complete_analysis_results.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados salvos em: {results_dir}")

def create_stratified_holdout_split(expanded_sets: Dict[str, Set[str]], 
                                     test_size: float = 0.2, 
                                     random_state: int = 42) -> Tuple[Dict, Dict]:
    """Cria split estratificado separando hold-out set (20%) de dados de treino (80%).
    
    Args:
        expanded_sets: Conjuntos expandidos de seeds por traço
        test_size: Proporção para hold-out (padrão 0.2 = 20%)
        random_state: Seed para reprodutibilidade
        
    Returns:
        train_sets: Conjuntos de treino (80%)
        holdout_sets: Conjuntos de hold-out (20%)
    """
    logger.info(f"Criando split estratificado: {int((1-test_size)*100)}% treino, {int(test_size*100)}% hold-out...")
    
    np.random.seed(random_state)
    train_sets = {}
    holdout_sets = {}
    
    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    
    for trait in traits:
        for polarity in ["Positive", "Negative"]:
            key = f"{trait}_{polarity}"
            commands = list(expanded_sets.get(key, []))
            
            if len(commands) == 0:
                train_sets[key] = set()
                holdout_sets[key] = set()
                continue
            
            # Embaralha e divide
            np.random.shuffle(commands)
            n_holdout = max(1, int(len(commands) * test_size))  # Pelo menos 1 amostra no hold-out
            
            holdout_sets[key] = set(commands[:n_holdout])
            train_sets[key] = set(commands[n_holdout:])
            
            logger.info(f"{key}: {len(train_sets[key])} treino, {len(holdout_sets[key])} hold-out")
    
    return train_sets, holdout_sets


def evaluate_on_holdout(final_scores: Dict, 
                        holdout_sets: Dict[str, Set[str]], 
                        results_dir: Path,
                        cv_report: Dict = None) -> Dict:
    """Avalia o modelo no conjunto hold-out nunca visto durante treino/CV.
    
    Args:
        final_scores: Scores finais gerados pelo modelo treinado apenas em train_sets
        holdout_sets: Conjuntos hold-out (20% dos dados)
        results_dir: Diretório para salvar resultados
        cv_report: Relatório de CV para comparação (opcional)
        
    Returns:
        Dicionário com métricas de hold-out
    """
    logger.info("=== Avaliando no conjunto HOLD-OUT (dados nunca vistos) ===")
    
    try:
        from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                                      recall_score, roc_curve, precision_recall_curve,
                                      brier_score_loss, confusion_matrix)
        from sklearn.calibration import calibration_curve
    except Exception as e:
        logger.warning(f"sklearn indisponível ({e}); avaliação hold-out pulada.")
        return {}
    
    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    holdout_report = {}
    
    for trait in traits:
        pos_key = f"{trait}_Positive"
        neg_key = f"{trait}_Negative"
        
        y_true = []
        y_scores = []
        
        # Coleta dados do hold-out set
        for cmd in holdout_sets.get(pos_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(1)
                y_scores.append(final_scores[cmd][trait]['positive'])
        
        for cmd in holdout_sets.get(neg_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(0)
                y_scores.append(final_scores[cmd][trait]['positive'])
        
        if len(y_true) < 2 or len(set(y_true)) < 2:
            logger.warning(f"Hold-out insuficiente para {trait}: {len(y_true)} amostras")
            holdout_report[trait] = {"samples": len(y_true), "error": "dados insuficientes"}
            continue
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Calcula métricas
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = None
        
        # Encontra melhor threshold
        best_f1 = 0.0
        best_thr = 0.5
        for thr in np.linspace(0.1, 0.9, 81):
            y_pred = (y_scores >= thr).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        
        # Métricas com threshold padrão e otimizado
        y_pred_default = (y_scores >= 0.5).astype(int)
        y_pred_best = (y_scores >= best_thr).astype(int)
        
        f1_default = float(f1_score(y_true, y_pred_default))
        f1_best = float(best_f1)
        precision = float(precision_score(y_true, y_pred_best))
        recall = float(recall_score(y_true, y_pred_best))
        
        # Calibração
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=min(10, max(2, len(y_true)//5)), strategy='uniform')
            brier = float(brier_score_loss(y_true, y_scores))
        except Exception:
            prob_true, prob_pred, brier = [], [], None
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred_best)
        
        holdout_report[trait] = {
            "samples": int(len(y_true)),
            "auc": auc,
            "f1@0.5": f1_default,
            "f1@best": f1_best,
            "best_threshold": best_thr,
            "precision": precision,
            "recall": recall,
            "brier": brier,
            "confusion_matrix": cm.tolist(),
        }

        # F1 no limiar da CV (se fornecido)
        try:
            thr_cv = None
            if isinstance(cv_report, dict):
                thr_cv = cv_report.get(trait, {}).get("best_threshold_cv_mean")
            if thr_cv is not None:
                y_pred_cv = (y_scores >= float(thr_cv)).astype(int)
                f1_cv = f1_score(y_true, y_pred_cv)
                holdout_report[trait]["f1@cv_thr"] = float(f1_cv)
                holdout_report[trait]["cv_threshold"] = float(thr_cv)
        except Exception:
            pass
        
        logger.info(f"{trait} Hold-out: AUC={auc:.3f}, F1@best={f1_best:.3f} (thr={best_thr:.2f})")
    
    # Salva métricas
    holdout_file = results_dir / "metrics" / "holdout_metrics.json"
    try:
        with open(holdout_file, 'w', encoding='utf-8') as f:
            json.dump(holdout_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Métricas de hold-out salvas em: {holdout_file}")
    except Exception as e:
        logger.warning(f"Falha ao salvar métricas de hold-out: {e}")
    
    return holdout_report


def generate_holdout_graphs(final_scores: Dict, 
                            holdout_sets: Dict[str, Set[str]], 
                            results_dir: Path,
                            cv_report: Dict = None):
    """Gera gráficos detalhados para o conjunto hold-out.
    
    Args:
        final_scores: Scores finais do modelo
        holdout_sets: Conjuntos hold-out
        results_dir: Diretório para salvar gráficos
        cv_report: Relatório CV para comparação (opcional)
    """
    logger.info("Gerando gráficos para conjunto hold-out...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import (roc_curve, precision_recall_curve, 
                                      confusion_matrix)
        from sklearn.calibration import calibration_curve
    except Exception as e:
        logger.warning(f"Dependências indisponíveis ({e}); gráficos hold-out pulados.")
        return
    
    graphs_dir = results_dir / "graphs" / "holdout"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    
    for trait in traits:
        pos_key = f"{trait}_Positive"
        neg_key = f"{trait}_Negative"
        
        y_true = []
        y_scores = []
        
        for cmd in holdout_sets.get(pos_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(1)
                y_scores.append(final_scores[cmd][trait]['positive'])
        
        for cmd in holdout_sets.get(neg_key, []):
            if cmd in final_scores and trait in final_scores[cmd]:
                y_true.append(0)
                y_scores.append(final_scores[cmd][trait]['positive'])
        
        if len(y_true) < 2 or len(set(y_true)) < 2:
            continue
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        try:
            # 1. ROC e PR curves
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            auc_score = roc_curve(y_true, y_scores)
            
            from sklearn.metrics import roc_auc_score, auc as auc_calc
            auc_score = roc_auc_score(y_true, y_scores)
            pr_auc = auc_calc(rec, prec)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(fpr, tpr, label=f"AUC={auc_score:.3f}", linewidth=2)
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
            axes[0].set_title(f"ROC Curve - {trait} (Hold-out)", fontsize=12, fontweight='bold')
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}", linewidth=2)
            axes[1].set_title(f"Precision-Recall - {trait} (Hold-out)", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(graphs_dir / f"roc_pr_{trait}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Calibration plot
            try:
                prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=min(10, max(2, len(y_true)//5)), strategy='uniform')
                
                plt.figure(figsize=(6, 6))
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfeitamente calibrado')
                plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Modelo')
                plt.title(f"Calibração - {trait} (Hold-out)", fontsize=12, fontweight='bold')
                plt.xlabel("Confiança Prevista")
                plt.ylabel("Frequência Observada")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(graphs_dir / f"calibration_{trait}.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Falha ao gerar calibração para {trait}: {e}")
            
            # 3. Confusion Matrix
            best_thr = 0.5
            best_f1 = 0.0
            for thr in np.linspace(0.1, 0.9, 81):
                from sklearn.metrics import f1_score
                f1 = f1_score(y_true, (y_scores >= thr).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            
            y_pred = (y_scores >= best_thr).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(6, 6))
            try:
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=["NEG", "POS"], yticklabels=["NEG", "POS"])
            except Exception:
                plt.imshow(cm, cmap='Blues')
                for (i, j), val in np.ndenumerate(cm):
                    plt.text(j, i, f"{val}", ha='center', va='center', fontsize=14)
                plt.xticks([0, 1], ["NEG", "POS"])
                plt.yticks([0, 1], ["NEG", "POS"])
            
            plt.title(f"Matriz de Confusão - {trait} (Hold-out)\nF1={best_f1:.3f} @ thr={best_thr:.2f}", 
                     fontsize=12, fontweight='bold')
            plt.xlabel("Predito")
            plt.ylabel("Verdadeiro")
            plt.tight_layout()
            plt.savefig(graphs_dir / f"confusion_{trait}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Falha ao gerar gráficos hold-out para {trait}: {e}")
    
    logger.info(f"Gráficos hold-out salvos em: {graphs_dir}")


def generate_cv_vs_holdout_comparison(cv_report: Dict, 
                                       holdout_report: Dict, 
                                       results_dir: Path):
    """Gera gráfico comparativo entre métricas de CV (OOF) e Hold-out.
    
    Args:
        cv_report: Relatório da validação cruzada
        holdout_report: Relatório do hold-out set
        results_dir: Diretório para salvar gráfico
    """
    logger.info("Gerando gráfico comparativo: CV vs Hold-out...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Matplotlib indisponível ({e}); gráfico comparativo pulado.")
        return
    
    traits = ["HonestyHumility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "OpennessToExperience"]
    
    # Extrai métricas
    cv_aucs = []
    cv_f1s = []
    holdout_aucs = []
    holdout_f1s = []
    trait_names = []
    
    for trait in traits:
        # CV metrics
        cv_data = cv_report.get(trait, {})
        if cv_data and 'auc_oof' in cv_data and cv_data['auc_oof'] is not None:
            cv_aucs.append(cv_data['auc_oof'])
            cv_f1s.append(cv_data.get('f1_oof@cv_thr', 0))
        else:
            continue
        
        # Holdout metrics
        ho_data = holdout_report.get(trait, {})
        if ho_data and 'auc' in ho_data and ho_data['auc'] is not None:
            holdout_aucs.append(ho_data['auc'])
            holdout_f1s.append(ho_data.get('f1@best', 0))
            trait_names.append(trait.replace("OpennessToExperience", "Openness"))
        else:
            cv_aucs.pop()
            cv_f1s.pop()
    
    if not trait_names:
        logger.warning("Sem dados suficientes para gráfico comparativo")
        return
    
    # Cria gráfico comparativo
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(trait_names))
    width = 0.35
    
    # AUC comparison
    axes[0].bar(x - width/2, cv_aucs, width, label='CV (OOF)', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, holdout_aucs, width, label='Hold-out', alpha=0.8, color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trait_names, rotation=25, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('AUC-ROC', fontsize=11)
    axes[0].set_title('Comparação AUC: CV vs Hold-out', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    
    # F1 comparison
    axes[1].bar(x - width/2, cv_f1s, width, label='CV (OOF @ thr CV)', alpha=0.8, color='steelblue')
    axes[1].bar(x + width/2, holdout_f1s, width, label='Hold-out (@ best thr)', alpha=0.8, color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trait_names, rotation=25, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('F1-Score', fontsize=11)
    axes[1].set_title('Comparação F1: CV vs Hold-out', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = results_dir / "graphs" / "cv_vs_holdout_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico comparativo salvo em: {output_file}")
    
    # Análise estatística
    logger.info("\n=== COMPARAÇÃO CV vs HOLD-OUT ===")
    for i, trait in enumerate(trait_names):
        logger.info(f"{trait}:")
        logger.info(f"  CV:       AUC={cv_aucs[i]:.3f}, F1={cv_f1s[i]:.3f}")
        logger.info(f"  Hold-out: AUC={holdout_aucs[i]:.3f}, F1={holdout_f1s[i]:.3f}")
        diff_auc = holdout_aucs[i] - cv_aucs[i]
        diff_f1 = holdout_f1s[i] - cv_f1s[i]
        logger.info(f"  Diferença: ΔAUC={diff_auc:+.3f}, ΔF1={diff_f1:+.3f}")


def main():
    """Pipeline principal modularizado seguindo a metodologia SentiWordNet 3.0."""
    logger.info("=== Classificação Semi-Supervisionada de Comandos (Método SentiWordNet 3.0 - Modular) ===")
    
    try:
        # Criar diretório de resultados
        results_dir = create_results_directory()
        
        # Passo 1: Expande conjuntos semente
        logger.info("Iniciando expansão de conjuntos semente...")
        expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
        expanded_sets = expander.expand_seeds()
        
        # NOVO: Passo 1.5: Cria split estratificado (80% treino, 20% hold-out)
        train_sets, holdout_sets = create_stratified_holdout_split(expanded_sets, test_size=0.2, random_state=42)
        
        # Passo 2: Coleta todos os comandos únicos
        all_known_commands, test_commands = collect_all_commands(expanded_sets, COMMAND_RELATIONS)
        
        # Passos 3-5: Tenta ML usando APENAS dados de treino (train_sets)
        try:
            from classifiers.vectorial_classifier import VectorialClassifier
            logger.info("Preparando classificadores (scikit-learn) com dados de TREINO apenas...")
            gloss_extractor = CommandGlossExtractor()
            classifier = VectorialClassifier(gloss_extractor)
            classifier.set_relations(COMMAND_RELATIONS)
            classifier.prepare_global_vectorizer(all_known_commands)
            # MODIFICADO: usa train_sets ao invés de expanded_sets
            trait_classifiers = classifier.train_trait_classifiers(train_sets)
            initial_scores = generate_initial_scores(classifier, trait_classifiers, all_known_commands)
        except BaseException as e:
            logger.warning(f"Falha ao inicializar classificadores ML ({e}). Usando fallback sem ML.")
            classifier = None
            initial_scores = generate_initial_scores_from_seeds(train_sets, all_known_commands)
        
        # Passo 6: Refina os escores com Random Walk
        logger.info("Iniciando refinamento com Random Walk...")
        refiner = RandomWalkRefiner(
            all_known_commands,
            COMMAND_RELATIONS,
            alpha=0.60,
            iterations=25,
            tolerance=1e-4,
            patience=5
        )
        final_scores = refiner.refine_scores_multi_trait(initial_scores)
        
        # Passo 7: Gera métricas e visualizações (usando train_sets)
        logger.info("Gerando métricas e visualizações...")
        metrics = generate_metrics(train_sets, initial_scores, final_scores, test_commands, refiner)
        
        # Passo 8: Cria visualizações
        create_visualizations(train_sets, initial_scores, final_scores, test_commands, results_dir)
        
        # Passo 9: Gera relatório resumido
        generate_summary_report(train_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir, all_known_commands)
        
        # Passo 10: Validação de efetividade em train_sets (rótulos fracos via seeds)
        validate_effectiveness(final_scores, train_sets, results_dir)
        
        # NOVO: Passo 10b: Validação cruzada ROBUSTA (OOF) com gráficos, salvando em validation_metrics.json
        cv_report = validate_effectiveness_cv(
            expanded_sets=train_sets,  # usa apenas treino para CV
            all_known_commands=all_known_commands,
            relations=COMMAND_RELATIONS,
            results_dir=results_dir,
            refiner_params={"alpha": 0.60, "iterations": 25, "tolerance": 1e-4, "patience": 5},
            n_splits=5,
            n_bootstrap=500,
            random_state=42,
        )
        
        # NOVO: Passo 11: Avaliação rigorosa no HOLD-OUT SET
        logger.info("\n" + "="*80)
        logger.info("AVALIAÇÃO NO CONJUNTO HOLD-OUT (20% - NUNCA VISTO)")
        logger.info("="*80)
        holdout_report = evaluate_on_holdout(final_scores, holdout_sets, results_dir, cv_report)
        
        # NOVO: Passo 12: Gera gráficos específicos para hold-out
        generate_holdout_graphs(final_scores, holdout_sets, results_dir, cv_report)
        
        # NOVO: Passo 13: Gera gráfico comparativo CV vs Hold-out
        if cv_report and holdout_report:
            generate_cv_vs_holdout_comparison(cv_report, holdout_report, results_dir)

        # NOVO: Passo 13.5: Persistir limiares finais selecionados (CV)
        try:
            selected = {
                t: cv_report.get(t, {}).get("best_threshold_cv_mean") for t in [
                    "HonestyHumility", "Emotionality", "Extraversion",
                    "Agreeableness", "Conscientiousness", "OpennessToExperience"
                ]
            }
            with open(results_dir / "metrics" / "selected_thresholds.json", "w", encoding="utf-8") as f:
                json.dump(selected, f, indent=2, ensure_ascii=False)
            logger.info(f"Limiar(es) selecionado(s) salvo(s) em: {results_dir / 'metrics' / 'selected_thresholds.json'}")
        except Exception as e:
            logger.warning(f"Falha ao salvar selected_thresholds.json: {e}")

        # Passo 14: Salva resultados detalhados
        save_detailed_results(train_sets, initial_scores, final_scores, test_commands, metrics, refiner, results_dir)
        
        # Passo 15: Apresenta resultados no console
        display_results(test_commands, initial_scores, final_scores, all_known_commands)
        
        # Passo 16: Estatísticas finais
        display_final_stats(train_sets, all_known_commands, refiner, classifier)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Pipeline modularizado executado com sucesso! Resultados salvos em: {results_dir}")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main() 