#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_graphs.py
Gera gráficos e visualizações independentes a partir dos resultados do pipeline.
Lê os arquivos JSON gerados pelo main.py e cria visualizações personalizadas.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_plot_style(ax=None, plt_module=None):
    """Aplica estilo profissional consistente aos gráficos."""
    if plt_module is not None:
        plt_module.rcParams.update({
            'font.size': 18,
            'axes.titlesize': 24,
            'axes.labelsize': 22,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'figure.titlesize': 26,
            'axes.grid': True,
            'grid.alpha': 0.5,
            'grid.linestyle': '-',
            'grid.linewidth': 1.2,
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(width=1.5, length=6)
        ax.set_axisbelow(True)  # Grade fica atrás dos elementos
        ax.grid(True, alpha=0.5, linestyle='-', linewidth=1.2)
        ax.set_facecolor('white')


def load_data(results_dir: Path) -> Dict:
    """Carrega todos os dados necessários dos arquivos JSON."""
    logger.info("Carregando dados dos arquivos JSON...")
    
    data = {}
    
    # Carrega resultados completos
    complete_file = results_dir / "complete_analysis_results.json"
    if complete_file.exists():
        with open(complete_file, 'r', encoding='utf-8') as f:
            data['complete'] = json.load(f)
    else:
        logger.warning(f"Arquivo não encontrado: {complete_file}")
        data['complete'] = {}
    
    # Carrega métricas
    metrics_file = results_dir / "metrics" / "analysis_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data['metrics'] = json.load(f)
    else:
        logger.warning(f"Arquivo não encontrado: {metrics_file}")
        data['metrics'] = {}
    
    # Carrega métricas de validação
    validation_file = results_dir / "metrics" / "validation_metrics.json"
    if validation_file.exists():
        with open(validation_file, 'r', encoding='utf-8') as f:
            data['validation'] = json.load(f)
    else:
        logger.warning(f"Arquivo não encontrado: {validation_file}")
        data['validation'] = {}
    
    # Carrega métricas de hold-out
    holdout_file = results_dir / "metrics" / "holdout_metrics.json"
    if holdout_file.exists():
        with open(holdout_file, 'r', encoding='utf-8') as f:
            data['holdout'] = json.load(f)
    else:
        logger.warning(f"Arquivo não encontrado: {holdout_file}")
        data['holdout'] = {}
    
    # Carrega scores refinados
    scores_file = results_dir / "metrics" / "refined_scores.json"
    if scores_file.exists():
        with open(scores_file, 'r', encoding='utf-8') as f:
            data['scores'] = json.load(f)
    else:
        logger.warning(f"Arquivo não encontrado: {scores_file}")
        data['scores'] = {}
    
    logger.info("Dados carregados com sucesso!")
    return data


def create_personality_radar_chart(data: Dict, output_dir: Path):
    """Cria gráfico radar mostrando o perfil de personalidade."""
    logger.info("Gerando gráfico radar de personalidade...")
    
    try:
        scores = data.get('scores', {})
        if not scores:
            logger.warning("Sem dados de scores para gráfico radar")
            return
        
        # Calcula médias por traço
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_labels = ["Honestidade\ne Humildade", "Emocionalidade", 
                       "Extraversão", "Cordialidade", 
                       "Conscienciosidade", "Abertura à\nExperiência"]
        
        avg_scores = []
        for trait in traits:
            trait_scores = []
            for cmd, cmd_data in scores.items():
                if trait in cmd_data:
                    trait_scores.append(cmd_data[trait]['positive'])
            avg_scores.append(np.mean(trait_scores) if trait_scores else 0.33)
        
        # Cria gráfico radar
        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
        avg_scores_plot = avg_scores + [avg_scores[0]]  # Fecha o polígono
        angles += angles[:1]  # Fecha o polígono
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Plota dados
        ax.plot(angles, avg_scores_plot, 'o-', linewidth=3, color='#E57373', 
                markersize=12, markerfacecolor='#C62828', markeredgecolor='white',
                markeredgewidth=2, label='Perfil do Atacante')
        ax.fill(angles, avg_scores_plot, alpha=0.25, color='#E57373')
        
        # Linha de referência neutra
        neutral = [0.5] * (len(traits) + 1)
        ax.plot(angles, neutral, '--', linewidth=2, color='gray', alpha=0.5, label='Neutro (0.5)')
        
        # Configurações
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(trait_labels, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
        ax.set_title('Perfil de Personalidade HEXACO do Atacante', 
                    fontsize=22, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14, 
                 frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        plt.tight_layout()
        output_file = output_dir / "personality_radar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico radar salvo em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico radar: {e}")


def create_trait_distribution_heatmap(data: Dict, output_dir: Path):
    """Cria heatmap mostrando distribuição de scores por traço."""
    logger.info("Gerando heatmap de distribuição de traços...")
    
    try:
        scores = data.get('scores', {})
        test_commands = data.get('complete', {}).get('test_commands', [])
        
        if not scores or not test_commands:
            logger.warning("Sem dados para heatmap")
            return
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_short = ["Hon-Hum", "Emoc", "Extr", "Cord", "Consc", "Abert"]
        
        # Matriz de scores
        matrix = []
        valid_commands = []
        
        for cmd in test_commands[:20]:  # Limita a 20 comandos para visualização
            if cmd in scores:
                row = []
                for trait in traits:
                    if trait in scores[cmd]:
                        row.append(scores[cmd][trait]['positive'])
                    else:
                        row.append(0.33)
                if row:
                    matrix.append(row)
                    # Abrevia comando para visualização
                    short_cmd = cmd[:30] + "..." if len(cmd) > 30 else cmd
                    valid_commands.append(short_cmd)
        
        if not matrix:
            logger.warning("Sem dados válidos para matriz")
            return
        
        matrix = np.array(matrix)
        
        # Cria heatmap
        fig, ax = plt.subplots(figsize=(14, max(8, len(valid_commands) * 0.4)))
        
        try:
            import seaborn as sns
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                       xticklabels=trait_short, yticklabels=valid_commands,
                       cbar_kws={'label': 'Score Positivo'}, ax=ax,
                       linewidths=0.5, linecolor='gray', vmin=0, vmax=1,
                       annot_kws={"size": 10, "weight": "bold"})
        except ImportError:
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(trait_short)))
            ax.set_yticks(np.arange(len(valid_commands)))
            ax.set_xticklabels(trait_short)
            ax.set_yticklabels(valid_commands)
            plt.colorbar(im, ax=ax, label='Score Positivo')
        
        ax.set_title('Distribuição de Scores HEXACO por Comando', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Traços HEXACO', fontsize=16, fontweight='bold')
        ax.set_ylabel('Comandos', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "trait_distribution_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap salvo em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar heatmap: {e}")


def create_validation_comparison(data: Dict, output_dir: Path):
    """Cria gráfico comparando métricas de validação."""
    logger.info("Gerando gráfico de comparação de validação...")
    
    try:
        validation = data.get('validation', {})
        holdout = data.get('holdout', {})
        
        if not validation and not holdout:
            logger.warning("Sem dados de validação")
            return
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_short = ["Hon-Hum", "Emoc", "Extr", "Cord", "Consc", "Abert"]
        
        # Extrai métricas
        cv_data = validation.get('cv', {}) if isinstance(validation, dict) else {}
        
        cv_aucs = []
        cv_f1s = []
        ho_aucs = []
        ho_f1s = []
        valid_traits = []
        
        for i, trait in enumerate(traits):
            cv_trait = cv_data.get(trait, {})
            ho_trait = holdout.get(trait, {})
            
            if cv_trait.get('auc_oof') is not None or ho_trait.get('auc') is not None:
                cv_aucs.append(cv_trait.get('auc_oof', 0))
                cv_f1s.append(cv_trait.get('f1_oof@cv_thr', 0))
                ho_aucs.append(ho_trait.get('auc', 0))
                ho_f1s.append(ho_trait.get('f1@best', 0))
                valid_traits.append(trait_short[i])
        
        if not valid_traits:
            logger.warning("Sem métricas válidas para comparação")
            return
        
        # Cria gráfico
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        apply_plot_style(plt_module=plt)
        
        x = np.arange(len(valid_traits))
        width = 0.35
        
        # AUC
        bars1 = axes[0].bar(x - width/2, cv_aucs, width, label='Validação Cruzada', 
                           color='#7E57C2', edgecolor='black', linewidth=1.5, 
                           alpha=0.9)
        bars2 = axes[0].bar(x + width/2, ho_aucs, width, label='Hold-out', 
                           color='#FFEB3B', edgecolor='black', linewidth=1.5, 
                           alpha=0.9)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(valid_traits, rotation=0, fontweight='bold', fontsize=18)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('AUC-ROC', fontsize=26, fontweight='bold')
        axes[0].set_xlabel('Traços HEXACO', fontsize=26, fontweight='bold')
        axes[0].set_title('Comparação de AUC', fontsize=30, fontweight='bold', pad=20)
        
        # Configura legendas em negrito
        legend0 = axes[0].legend(fontsize=16, frameon=True, shadow=True, 
                                prop={'weight': 'bold', 'size': 16})
        for text in legend0.get_texts():
            text.set_fontweight('bold')
        
        # Deixa os ticks do eixo Y em negrito e aumenta o tamanho
        for label in axes[0].get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(22)
        
        apply_plot_style(axes[0])
        
        # F1
        bars3 = axes[1].bar(x - width/2, cv_f1s, width, label='Validação Cruzada', 
                           color='#7E57C2', edgecolor='black', linewidth=1.5, 
                           alpha=0.9)
        bars4 = axes[1].bar(x + width/2, ho_f1s, width, label='Hold-out', 
                           color='#FFEB3B', edgecolor='black', linewidth=1.5, 
                           alpha=0.9)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(valid_traits, rotation=0, fontweight='bold', fontsize=18)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel('F1-Score', fontsize=26, fontweight='bold')
        axes[1].set_xlabel('Traços HEXACO', fontsize=26, fontweight='bold')
        axes[1].set_title('Comparação de F1-Score', fontsize=30, fontweight='bold', pad=20)
        
        # Configura legendas em negrito
        legend1 = axes[1].legend(fontsize=16, frameon=True, shadow=True,
                                prop={'weight': 'bold', 'size': 16})
        for text in legend1.get_texts():
            text.set_fontweight('bold')
        
        # Deixa os ticks do eixo Y em negrito e aumenta o tamanho
        for label in axes[1].get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(22)
        
        apply_plot_style(axes[1])
        
        plt.tight_layout()
        output_file = output_dir / "validation_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de comparação salvo em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de comparação: {e}")


def create_score_evolution(data: Dict, output_dir: Path):
    """Cria gráfico mostrando evolução dos scores (inicial vs refinado)."""
    logger.info("Gerando gráfico de evolução de scores...")
    
    try:
        complete = data.get('complete', {})
        initial_scores = complete.get('initial_scores', {})
        refined_scores = complete.get('refined_scores', {})
        test_commands = complete.get('test_commands', [])
        
        if not initial_scores or not refined_scores:
            logger.warning("Sem dados de scores para evolução")
            return
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_short = ["Hon-Hum", "Emoc", "Extr", "Cord", "Consc", "Abert"]
        
        # Calcula médias
        initial_avg = []
        refined_avg = []
        improvements = []
        
        for trait in traits:
            initial_vals = []
            refined_vals = []
            
            for cmd in test_commands:
                if cmd in initial_scores and cmd in refined_scores:
                    if trait in initial_scores[cmd] and trait in refined_scores[cmd]:
                        initial_vals.append(initial_scores[cmd][trait]['positive'])
                        refined_vals.append(refined_scores[cmd][trait]['positive'])
            
            if initial_vals and refined_vals:
                init_avg = np.mean(initial_vals)
                ref_avg = np.mean(refined_vals)
                initial_avg.append(init_avg)
                refined_avg.append(ref_avg)
                improvements.append(ref_avg - init_avg)
        
        # Cria gráfico
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        apply_plot_style(plt_module=plt)
        
        x = np.arange(len(traits))
        width = 0.35
        
        # Comparação Inicial vs Refinado
        bars1 = axes[0].bar(x - width/2, initial_avg, width, label='Scores Iniciais', 
                           color='#E57373', edgecolor='black', linewidth=1.5, 
                           hatch='//', alpha=0.9)
        bars2 = axes[0].bar(x + width/2, refined_avg, width, label='Scores Refinados', 
                           color='#64B5F6', edgecolor='black', linewidth=1.5, 
                           hatch='\\\\', alpha=0.9)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(trait_short, rotation=0)
        axes[0].set_ylabel('Score Médio', fontsize=18, fontweight='bold')
        axes[0].set_xlabel('Traços HEXACO', fontsize=18, fontweight='bold')
        axes[0].set_title('Evolução dos Scores: Inicial vs Refinado', 
                         fontsize=20, fontweight='bold', pad=20)
        axes[0].legend(fontsize=14, frameon=True, shadow=True)
        axes[0].set_ylim(0, max(initial_avg + refined_avg) * 1.15)
        apply_plot_style(axes[0])
        
        # Melhorias
        colors = ['#81C784' if imp > 0 else '#E57373' for imp in improvements]
        bars3 = axes[1].bar(trait_short, improvements, color=colors, 
                           edgecolor='black', linewidth=1.5, alpha=0.9)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=2)
        axes[1].set_ylabel('Melhoria no Score', fontsize=18, fontweight='bold')
        axes[1].set_xlabel('Traços HEXACO', fontsize=18, fontweight='bold')
        axes[1].set_title('Melhoria após Refinamento (Random Walk)', 
                         fontsize=20, fontweight='bold', pad=20)
        apply_plot_style(axes[1])
        
        # Adiciona valores nas barras
        for i, (bar, val) in enumerate(zip(bars3, improvements)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:+.3f}', ha='center', va='bottom' if val > 0 else 'top',
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "score_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de evolução salvo em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de evolução: {e}")


def create_top_commands_analysis(data: Dict, output_dir: Path):
    """Cria gráfico mostrando os comandos com scores mais altos/baixos."""
    logger.info("Gerando análise de top comandos...")
    
    try:
        scores = data.get('scores', {})
        
        if not scores:
            logger.warning("Sem dados de scores")
            return
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        
        # Calcula score médio por comando
        cmd_avg_scores = {}
        for cmd, cmd_data in scores.items():
            trait_scores = []
            for trait in traits:
                if trait in cmd_data:
                    trait_scores.append(cmd_data[trait]['positive'])
            if trait_scores:
                cmd_avg_scores[cmd] = np.mean(trait_scores)
        
        if not cmd_avg_scores:
            logger.warning("Sem scores válidos para análise")
            return
        
        # Top 10 e Bottom 10
        sorted_commands = sorted(cmd_avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_commands[:10]
        bottom_10 = sorted_commands[-10:]
        
        # Cria gráfico
        fig, axes = plt.subplots(2, 1, figsize=(14, 14))
        apply_plot_style(plt_module=plt)
        
        # Top 10
        top_cmds = [cmd[:40] + "..." if len(cmd) > 40 else cmd for cmd, _ in top_10]
        top_scores = [score for _, score in top_10]
        
        bars1 = axes[0].barh(top_cmds, top_scores, color='#81C784', 
                            edgecolor='black', linewidth=1.5, alpha=0.9)
        axes[0].set_xlabel('Score Médio HEXACO', fontsize=16, fontweight='bold')
        axes[0].set_title('Top 10 Comandos com Maiores Scores', 
                         fontsize=20, fontweight='bold', pad=20)
        axes[0].set_xlim(0, 1)
        axes[0].invert_yaxis()
        apply_plot_style(axes[0])
        
        # Adiciona valores
        for i, (bar, val) in enumerate(zip(bars1, top_scores)):
            axes[0].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
        
        # Bottom 10
        bottom_cmds = [cmd[:40] + "..." if len(cmd) > 40 else cmd for cmd, _ in bottom_10]
        bottom_scores = [score for _, score in bottom_10]
        
        bars2 = axes[1].barh(bottom_cmds, bottom_scores, color='#E57373', 
                            edgecolor='black', linewidth=1.5, alpha=0.9)
        axes[1].set_xlabel('Score Médio HEXACO', fontsize=16, fontweight='bold')
        axes[1].set_title('Top 10 Comandos com Menores Scores', 
                         fontsize=20, fontweight='bold', pad=20)
        axes[1].set_xlim(0, 1)
        axes[1].invert_yaxis()
        apply_plot_style(axes[1])
        
        # Adiciona valores
        for i, (bar, val) in enumerate(zip(bars2, bottom_scores)):
            axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "top_commands_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Análise de top comandos salva em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar análise de top comandos: {e}")


def create_metrics_summary_table(data: Dict, output_dir: Path):
    """Cria tabela visual resumindo todas as métricas."""
    logger.info("Gerando tabela resumo de métricas...")
    
    try:
        validation = data.get('validation', {})
        holdout = data.get('holdout', {})
        metrics = data.get('metrics', {})
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_short = ["Hon-Hum", "Emoc", "Extr", "Cord", "Consc", "Abert"]
        
        # Prepara dados da tabela
        cv_data = validation.get('cv', {}) if isinstance(validation, dict) else {}
        
        table_data = []
        for i, trait in enumerate(traits):
            cv_trait = cv_data.get(trait, {})
            ho_trait = holdout.get(trait, {})
            
            row = [
                trait_short[i],
                f"{cv_trait.get('auc_oof', 0):.3f}" if cv_trait.get('auc_oof') else "N/A",
                f"{cv_trait.get('f1_oof@cv_thr', 0):.3f}" if cv_trait.get('f1_oof@cv_thr') else "N/A",
                f"{ho_trait.get('auc', 0):.3f}" if ho_trait.get('auc') else "N/A",
                f"{ho_trait.get('f1@best', 0):.3f}" if ho_trait.get('f1@best') else "N/A",
            ]
            table_data.append(row)
        
        # Cria figura para tabela
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        columns = ['Traço', 'AUC (CV)', 'F1 (CV)', 'AUC (Hold-out)', 'F1 (Hold-out)']
        
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15, 0.15, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Estiliza cabeçalho
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4A90E2')
            cell.set_text_props(weight='bold', color='white', fontsize=14)
        
        # Estiliza células
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
        
        plt.title('Resumo de Métricas de Validação por Traço HEXACO', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = output_dir / "metrics_summary_table.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Tabela resumo salva em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar tabela resumo: {e}")


def create_roc_pr_curves_custom(data: Dict, output_dir: Path):
    """Cria curvas ROC e Precision-Recall customizadas para cada traço.
    
    Gera gráficos similares aos do main.py mas com estilo personalizado,
    usando dados de hold-out ou expanded sets.
    """
    logger.info("Gerando curvas ROC e Precision-Recall customizadas...")
    
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
    except ImportError:
        logger.warning("scikit-learn não disponível, pulando curvas ROC/PR")
        return
    
    try:
        scores = data.get('scores', {})
        complete = data.get('complete', {})
        expanded_sets = complete.get('expanded_sets', {})
        
        if not scores or not expanded_sets:
            logger.warning("Sem dados suficientes para curvas ROC/PR")
            return
        
        traits = ["HonestyHumility", "Emotionality", "Extraversion", 
                  "Agreeableness", "Conscientiousness", "OpennessToExperience"]
        trait_full_names = {
            "HonestyHumility": "Honestidade-Humildade",
            "Emotionality": "Emocionalidade",
            "Extraversion": "Extraversão",
            "Agreeableness": "Cordialidade",
            "Conscientiousness": "Conscienciosidade",
            "OpennessToExperience": "Abertura à Experiência"
        }
        
        for trait in traits:
            pos_key = f"{trait}_Positive"
            neg_key = f"{trait}_Negative"
            
            y_true = []
            y_scores = []
            
            # Coleta dados dos expanded sets
            for cmd in expanded_sets.get(pos_key, []):
                if cmd in scores and trait in scores[cmd]:
                    y_true.append(1)
                    y_scores.append(scores[cmd][trait]['positive'])
            
            for cmd in expanded_sets.get(neg_key, []):
                if cmd in scores and trait in scores[cmd]:
                    y_true.append(0)
                    y_scores.append(scores[cmd][trait]['positive'])
            
            if len(y_true) < 2 or len(set(y_true)) < 2:
                logger.warning(f"Dados insuficientes para {trait}")
                continue
            
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            
            # Calcula curvas
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            
            auc_roc = roc_auc_score(y_true, y_scores)
            auc_pr = auc(rec, prec)
            
            # Cria gráfico
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # ROC Curve
            axes[0].plot(fpr, tpr, linewidth=3.5, color='#7E57C2', 
                        label=f"AUC={auc_roc:.3f}")
            axes[0].fill_between(fpr, tpr, alpha=0.2, color='#7E57C2')
            axes[0].set_title("Curva ROC", fontsize=30, fontweight='bold', pad=20)
            axes[0].set_xlabel("TFP", fontsize=26, fontweight='bold')
            axes[0].set_ylabel("TVP", fontsize=26, fontweight='bold')
            
            # Configura legendas em negrito
            legend0 = axes[0].legend(fontsize=16, frameon=True, shadow=True, loc='lower right', 
                                    prop={'weight': 'bold', 'size': 16})
            for text in legend0.get_texts():
                text.set_fontweight('bold')
            
            axes[0].set_xlim([-0.02, 1.02])
            axes[0].set_ylim([-0.02, 1.02])
            
            # Deixa os ticks em negrito e aumenta tamanho
            for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(22)
            
            apply_plot_style(axes[0])
            
            # PR Curve
            axes[1].plot(rec, prec, linewidth=3.5, color='#FFEB3B', 
                        label=f"AUC-PR={auc_pr:.3f}")
            axes[1].fill_between(rec, prec, alpha=0.2, color='#FFEB3B')
            axes[1].set_title("Curva PR", fontsize=30, fontweight='bold', pad=20)
            axes[1].set_xlabel("Revocação", fontsize=26, fontweight='bold')
            axes[1].set_ylabel("Precisão", fontsize=26, fontweight='bold')
            
            # Configura legendas em negrito
            legend1 = axes[1].legend(fontsize=16, frameon=True, shadow=True, loc='best',
                                    prop={'weight': 'bold', 'size': 16})
            for text in legend1.get_texts():
                text.set_fontweight('bold')
            
            axes[1].set_xlim([-0.02, 1.02])
            axes[1].set_ylim([-0.02, 1.02])
            
            # Deixa os ticks em negrito e aumenta tamanho
            for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(22)
            
            apply_plot_style(axes[1])
            
            plt.tight_layout()
            output_file = output_dir / f"roc_pr_{trait}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ROC/PR para {trait}: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")
        
        logger.info(f"Curvas ROC/PR salvas em: {output_dir}")
    except Exception as e:
        logger.error(f"Erro ao gerar curvas ROC/PR: {e}")


def generate_all_graphs(results_dir: Path = None):
    """Função principal que gera todos os gráficos."""
    logger.info("="*80)
    logger.info("Iniciando geração de gráficos personalizados")
    logger.info("="*80)
    
    # Define diretório de resultados
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent / "cowrie_analysis_results"
    else:
        results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Diretório de resultados não encontrado: {results_dir}")
        return
    
    # Cria diretório para gráficos personalizados
    custom_graphs_dir = results_dir / "graphs" / "custom"
    custom_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega dados
    data = load_data(results_dir)
    
    # Gera cada tipo de gráfico
    logger.info("\nGerando gráficos personalizados...")
    
    # Gráficos principais selecionados
    create_validation_comparison(data, custom_graphs_dir)
    create_roc_pr_curves_custom(data, custom_graphs_dir)
    
    # Outros gráficos desabilitados (descomente para gerar)
    # create_personality_radar_chart(data, custom_graphs_dir)
    # create_trait_distribution_heatmap(data, custom_graphs_dir)
    # create_score_evolution(data, custom_graphs_dir)
    # create_top_commands_analysis(data, custom_graphs_dir)
    # create_metrics_summary_table(data, custom_graphs_dir)
    
    logger.info("\n" + "="*80)
    logger.info(f"Todos os gráficos foram gerados em: {custom_graphs_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    # Executa geração de gráficos
    generate_all_graphs()

