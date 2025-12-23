"""
Module de visualisation pour la maintenance prédictive.
Génère des graphiques professionnels avec Matplotlib et Seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

# Configuration du style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass  # Utiliser le style par défaut

import config
from demo_mock import StrategyResult


class MaintenanceVisualizer:
    """
    Classe pour générer des visualisations professionnelles
    des résultats de maintenance prédictive.
    """
    
    def __init__(
        self,
        save_dir: str = config.PLOTS_SAVE_DIR,
        figsize: tuple = config.FIGURE_SIZE,
        dpi: int = config.FIGURE_DPI
    ):
        """
        Initialise le visualiseur.
        
        Args:
            save_dir: Répertoire de sauvegarde des figures
            figsize: Taille des figures
            dpi: Résolution
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        
        # Créer le répertoire si nécessaire
        os.makedirs(save_dir, exist_ok=True)
        
        # Palette de couleurs professionnelle
        self.colors = {
            'periodic': '#E74C3C',    # Rouge vif
            'threshold': '#F39C12',   # Orange
            'rl': '#27AE60',          # Vert
            'failure': '#8E44AD',     # Violet
            'neutral': '#3498DB',     # Bleu
            'dark': '#2C3E50',        # Gris foncé
            'light': '#ECF0F1'        # Gris clair
        }
    
    def plot_cost_comparison(
        self,
        strategies: Dict[str, StrategyResult],
        title: str = "Comparaison des Coûts de Maintenance",
        save_name: Optional[str] = "cost_comparison.png",
        show: bool = True
    ) -> plt.Figure:
        """
        Génère un graphique en barres comparant les coûts des stratégies.
        
        Fig 1: Comparaison des coûts (Bar Chart)
        
        Args:
            strategies: Dictionnaire des résultats par stratégie
            title: Titre du graphique
            save_name: Nom du fichier de sauvegarde
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extraire les données
        names = []
        costs = []
        errors = []
        colors = []
        
        order = ['periodic', 'threshold', 'rl']
        for key in order:
            if key in strategies:
                result = strategies[key]
                names.append(result.name)
                costs.append(result.mean_cost)
                errors.append(result.std_cost)
                colors.append(self.colors[key])
        
        # Créer le graphique
        x = np.arange(len(names))
        bars = ax.bar(x, costs, yerr=errors, capsize=8, color=colors,
                      edgecolor='white', linewidth=2, alpha=0.9)
        
        # Ajouter les valeurs sur les barres
        for bar, cost, error in zip(bars, costs, errors):
            height = bar.get_height()
            ax.annotate(f'{cost:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=14, fontweight='bold',
                       color=self.colors['dark'])
        
        # Personnalisation
        ax.set_ylabel('Coût Moyen par Cycle de Vie (€)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Stratégie de Maintenance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        
        # Ligne de référence
        min_cost = min(costs)
        ax.axhline(y=min_cost, color=self.colors['rl'], linestyle='--', 
                   alpha=0.5, label=f'Coût optimal: {min_cost:.1f}€')
        ax.legend(loc='upper right', fontsize=10)
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(costs) * 1.3)
        
        # Grille légère
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"📊 Figure sauvegardée: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_engine_lifecycle(
        self,
        lifecycle_data: Dict[str, Any],
        title: str = "Cycle de Vie du Moteur",
        save_name: Optional[str] = "engine_lifecycle.png",
        show: bool = True
    ) -> plt.Figure:
        """
        Génère une courbe de vie moteur avec annotations des interventions.
        
        Fig 2: Courbe de vie moteur (RUL vs Temps)
        
        Args:
            lifecycle_data: Données de simulation du cycle de vie
            title: Titre du graphique
            save_name: Nom du fichier
            show: Afficher
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        strategy = lifecycle_data['strategy']
        cycles = lifecycle_data['cycles']
        rul_curve = lifecycle_data['rul_curve']
        interventions = lifecycle_data['interventions']
        
        # Courbe de dégradation
        color = self.colors.get(strategy, self.colors['neutral'])
        ax.plot(cycles, rul_curve, color=color, linewidth=2.5, 
                label=f'RUL ({strategy.upper()})', alpha=0.9)
        
        # Zone de danger (RUL < 10)
        ax.axhspan(0, 10, color=self.colors['failure'], alpha=0.15,
                   label='Zone critique (RUL < 10)')
        
        # Zone de seuil
        ax.axhline(y=config.THRESHOLD_RUL, color=self.colors['threshold'],
                   linestyle='--', alpha=0.6, label=f'Seuil ({config.THRESHOLD_RUL})')
        
        # Annotations des interventions
        for i, intervention in enumerate(interventions):
            cycle = intervention['cycle']
            rul = intervention['rul_at_intervention']
            is_optimal = intervention.get('optimal', False)
            
            # Marqueur de maintenance
            marker_color = self.colors['rl'] if is_optimal else self.colors['periodic']
            ax.scatter([cycle], [rul], s=200, c=marker_color, marker='v',
                      zorder=5, edgecolors='white', linewidth=2)
            
            # Annotation
            label = "Optimal" if is_optimal else "Maintenance"
            ax.annotate(f'{label}\nRUL={rul:.0f}',
                       xy=(cycle, rul),
                       xytext=(10, 30),
                       textcoords='offset points',
                       fontsize=9,
                       ha='left',
                       arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=marker_color, alpha=0.9))
        
        # Personnalisation
        ax.set_xlabel('Cycle d\'opération', fontsize=12, fontweight='bold')
        ax.set_ylabel('Durée de Vie Restante (RUL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, len(cycles))
        ax.set_ylim(0, config.MAX_RUL + 10)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"📊 Figure sauvegardée: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multi_strategy_lifecycle(
        self,
        lifecycle_simulations: Dict[str, Dict[str, Any]],
        title: str = "Comparaison des Stratégies - Cycle de Vie",
        save_name: Optional[str] = "multi_lifecycle.png",
        show: bool = True
    ) -> plt.Figure:
        """
        Compare les 3 stratégies sur un même graphique.
        
        Args:
            lifecycle_simulations: Dictionnaire des simulations
            title: Titre
            save_name: Fichier de sauvegarde
            show: Afficher
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        strategies = ['periodic', 'threshold', 'rl']
        titles = [
            'Maintenance Périodique (Conservatrice)',
            'Maintenance sur Seuil (Réactive)',
            'Agent RL (Optimale)'
        ]
        
        for ax, strategy, strat_title in zip(axes, strategies, titles):
            if strategy not in lifecycle_simulations:
                continue
            
            data = lifecycle_simulations[strategy]
            cycles = data['cycles']
            rul_curve = data['rul_curve']
            interventions = data['interventions']
            
            color = self.colors[strategy]
            
            # Courbe RUL
            ax.fill_between(cycles, 0, rul_curve, alpha=0.3, color=color)
            ax.plot(cycles, rul_curve, color=color, linewidth=2, label='RUL')
            
            # Zone critique
            ax.axhspan(0, 10, color=self.colors['failure'], alpha=0.1)
            ax.axhline(y=10, color=self.colors['failure'], linestyle=':', alpha=0.5)
            
            # Interventions - différencier maintenances et pannes
            for intervention in interventions:
                cycle = intervention['cycle']
                rul = intervention.get('rul_at_intervention', 0)
                intervention_type = intervention.get('type', 'maintenance')
                
                if intervention_type == 'failure':
                    # Panne: X rouge
                    ax.axvline(x=cycle, color=self.colors['failure'], linestyle='-', alpha=0.5, linewidth=2)
                    ax.scatter([cycle], [rul], s=200, c=self.colors['failure'], marker='X',
                              zorder=5, edgecolors='white', linewidth=2, label='Panne' if cycle == interventions[0]['cycle'] else '')
                else:
                    # Maintenance préventive: triangle vert
                    ax.axvline(x=cycle, color=color, linestyle='--', alpha=0.7)
                    ax.scatter([cycle], [rul], s=150, c=color, marker='v',
                              zorder=5, edgecolors='white', linewidth=2)
            
            # Titre et labels
            ax.set_title(strat_title, fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel('RUL', fontsize=10)
            ax.set_ylim(0, config.MAX_RUL + 5)
            
            # Coûts dans le coin avec explication
            total_cost = data['total_cost']
            num_cycles = len(cycles)
            cost_per_cycle = total_cost / num_cycles if num_cycles > 0 else 0
            
            cost_text = (
                f'Coût total: {total_cost:.0f}€\n'
                f'Coût/cycle: {cost_per_cycle:.2f}€'
            )
            ax.text(0.98, 0.95, cost_text,
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        axes[-1].set_xlabel('Cycle d\'opération', fontsize=12, fontweight='bold')
        
        # Ajouter une note explicative détaillée en bas
        fig.text(0.5, 0.02, 
                '📌 IMPORTANT: Ces graphiques montrent UNE simulation de démonstration continue de 2000 cycles.\n'
                'Les coûts affichés ici sont cumulatifs sur toute la durée (2000 cycles), contrairement au "Résumé Exécutif"\n'
                'qui présente le coût MOYEN par épisode (calculé sur 50-100 épisodes jusqu\'à panne). '
                'C\'est pourquoi les valeurs ne correspondent pas directement.',
                ha='center', fontsize=8.5, style='italic', color='#333',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF9E6', alpha=0.9, edgecolor='#F39C12', linewidth=2))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"📊 Figure sauvegardée: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_rul_utilization(
        self,
        strategies: Dict[str, StrategyResult],
        title: str = "Utilisation de la Durée de Vie (RUL)",
        save_name: Optional[str] = "rul_utilization.png",
        show: bool = True
    ) -> plt.Figure:
        """
        Graphique montrant l'efficacité d'utilisation du RUL.
        
        Args:
            strategies: Résultats des stratégies
            title: Titre
            save_name: Fichier
            show: Afficher
            
        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        order = ['periodic', 'threshold', 'rl']
        names = []
        utilizations = []
        colors = []
        
        for key in order:
            if key in strategies:
                result = strategies[key]
                names.append(result.name)
                utilizations.append(result.mean_rul_utilization * 100)
                colors.append(self.colors[key])
        
        # Barres horizontales
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, utilizations, color=colors, height=0.6,
                       edgecolor='white', linewidth=2)
        
        # Valeurs sur les barres
        for bar, util in zip(bars, utilizations):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{util:.1f}%', ha='left', va='center',
                   fontsize=13, fontweight='bold')
        
        # Ligne objectif 100%
        ax.axvline(x=100, color=self.colors['dark'], linestyle='--',
                   alpha=0.5, label='Utilisation maximale (100%)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Utilisation du RUL (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 110)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"📊 Figure sauvegardée: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_full_report(
        self,
        demo_data: Dict[str, Any],
        show: bool = True
    ) -> List[plt.Figure]:
        """
        Génère le rapport complet de visualisation.
        
        Args:
            demo_data: Données de démonstration
            show: Afficher les figures
            
        Returns:
            Liste des figures générées
        """
        figures = []
        
        print("\n" + "=" * 50)
        print("🎨 GÉNÉRATION DU RAPPORT VISUEL")
        print("=" * 50 + "\n")
        
        strategies = demo_data['strategies']
        
        # Figure 1: Comparaison des coûts
        print("📊 Génération: Comparaison des coûts...")
        fig1 = self.plot_cost_comparison(strategies, show=show)
        figures.append(fig1)
        
        # Figure 2: Utilisation RUL
        print("📊 Génération: Utilisation du RUL...")
        fig2 = self.plot_rul_utilization(strategies, show=show)
        figures.append(fig2)
        
        # Figure 3: Comparaison multi-stratégies
        if 'lifecycle_simulations' in demo_data:
            print("📊 Génération: Comparaison des cycles de vie...")
            fig3 = self.plot_multi_strategy_lifecycle(
                demo_data['lifecycle_simulations'],
                show=show
            )
            figures.append(fig3)
            
            # Figure 4: Cycle de vie RL détaillé
            print("📊 Génération: Cycle de vie RL optimal...")
            fig4 = self.plot_engine_lifecycle(
                demo_data['lifecycle_simulations']['rl'],
                title="Cycle de Vie - Agent RL Optimal",
                save_name="rl_lifecycle_detail.png",
                show=show
            )
            figures.append(fig4)
        
        print("\n✅ Rapport visuel généré avec succès!")
        print(f"   Figures sauvegardées dans: {self.save_dir}/")
        
        return figures


def generate_demo_visualizations(show: bool = True):
    """
    Fonction utilitaire pour générer toutes les visualisations de démo.
    """
    from demo_mock import get_demo_data
    
    demo_data = get_demo_data(force_mock=True)
    visualizer = MaintenanceVisualizer()
    
    return visualizer.create_full_report(demo_data, show=show)


if __name__ == "__main__":
    print("=== Test du Module de Visualisation ===\n")
    
    # Générer les visualisations de démo
    figures = generate_demo_visualizations(show=True)
    
    print(f"\n{len(figures)} figures générées.")
