"""
Génération des graphiques de comparaison pour le README.
Compare les performances de PPO, DQN, Seuil et Périodique.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Créer le dossier pour les images
os.makedirs("plots/comparison", exist_ok=True)

# Style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Couleurs pour chaque stratégie
COLORS = {
    'PPO': '#00D4AA',
    'DQN': '#667EEA',
    'Seuil': '#FFA726',
    'Périodique': '#EF5350'
}

# Données mockup (cohérentes avec streamlit_app.py)
np.random.seed(42)

strategies = ['PPO', 'DQN', 'Seuil', 'Périodique']

# Métriques moyennes
mean_costs = [45.2, 54.8, 84.6, 109.3]
std_costs = [8.1, 9.2, 10.5, 14.8]
failure_rates = [2.0, 4.0, 8.0, 15.0]  # En pourcentage
rul_utilization = [92, 88, 75, 55]  # En pourcentage
maintenance_counts = [3.8, 4.0, 4.2, 5.5]

# =============================================================================
# 1. Graphique des coûts moyens
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(strategies, mean_costs, 
              color=[COLORS[s] for s in strategies],
              edgecolor='white', linewidth=2)

# Ajouter les barres d'erreur
ax.errorbar(strategies, mean_costs, yerr=std_costs, 
            fmt='none', color='#333', capsize=5, capthick=2)

# Annotations
for bar, cost, std in zip(bars, mean_costs, std_costs):
    ax.annotate(f'{cost:.1f}€\n±{std:.1f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Coût Moyen (€)')
ax.set_title('Comparaison des Coûts de Maintenance par Stratégie', fontweight='bold', pad=20)
ax.set_ylim(0, max(mean_costs) * 1.3)

plt.tight_layout()
plt.savefig('plots/comparison/cout_moyen.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ cout_moyen.png généré")

# =============================================================================
# 2. Graphique du taux de panne
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(strategies, failure_rates,
              color=[COLORS[s] for s in strategies],
              edgecolor='white', linewidth=2)

for bar, rate in zip(bars, failure_rates):
    ax.annotate(f'{rate}%',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Taux de Panne (%)')
ax.set_title('Taux de Panne par Stratégie de Maintenance', fontweight='bold', pad=20)
ax.set_ylim(0, max(failure_rates) * 1.3)

# Ligne de référence
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Objectif < 5%')
ax.legend()

plt.tight_layout()
plt.savefig('plots/comparison/taux_panne.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ taux_panne.png généré")

# =============================================================================
# 3. Graphique d'utilisation RUL
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(strategies, rul_utilization,
              color=[COLORS[s] for s in strategies],
              edgecolor='white', linewidth=2)

for bar, util in zip(bars, rul_utilization):
    ax.annotate(f'{util}%',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Utilisation RUL (%)')
ax.set_title('Efficacité d\'Utilisation de la Durée de Vie Utile', fontweight='bold', pad=20)
ax.set_ylim(0, 110)

# Zone optimale
ax.axhspan(85, 100, alpha=0.2, color='green', label='Zone Optimale (>85%)')
ax.legend()

plt.tight_layout()
plt.savefig('plots/comparison/utilisation_rul.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ utilisation_rul.png généré")

# =============================================================================
# 4. Graphique radar de performance globale
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Catégories (inversées pour certaines métriques où moins = mieux)
categories = ['Coût\n(inversé)', 'Fiabilité\n(1-panne)', 'Utilisation\nRUL', 'Efficacité\nMaintenance']
N = len(categories)

# Normaliser les données (0-100, où 100 = meilleur)
# Coût: inversé et normalisé
max_cost = max(mean_costs)
cost_scores = [(max_cost - c) / max_cost * 100 for c in mean_costs]

# Fiabilité: 100 - taux de panne
reliability_scores = [100 - f for f in failure_rates]

# RUL utilization: déjà en %
rul_scores = rul_utilization

# Efficacité maintenance: moins de maintenances = mieux (inversé)
max_maint = max(maintenance_counts)
maint_scores = [(max_maint - m) / max_maint * 100 + 50 for m in maintenance_counts]

# Angles
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, strategy in enumerate(strategies):
    values = [cost_scores[i], reliability_scores[i], rul_scores[i], maint_scores[i]]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=COLORS[strategy])
    ax.fill(angles, values, alpha=0.25, color=COLORS[strategy])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 100)
ax.set_title('Performance Globale des Stratégies', fontweight='bold', pad=20, size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('plots/comparison/radar_performance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ radar_performance.png généré")

# =============================================================================
# 5. Évolution RUL simulée
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

def simulate_rul(strategy, n_steps=300):
    """Simule l'évolution du RUL pour une stratégie."""
    np.random.seed({'PPO': 100, 'DQN': 200, 'Seuil': 300, 'Périodique': 400}[strategy])
    
    rul_history = []
    maintenance_events = []
    current_rul = 125.0
    
    if strategy == 'PPO':
        threshold = 22
    elif strategy == 'DQN':
        threshold = 28
    elif strategy == 'Seuil':
        threshold = 25
    else:
        threshold = None
        interval = 80
    
    cycles = 0
    for step in range(n_steps):
        rul_history.append(current_rul)
        
        do_maintenance = False
        if strategy == 'Périodique':
            cycles += 1
            if cycles >= interval:
                do_maintenance = True
                cycles = 0
        else:
            if current_rul <= threshold:
                do_maintenance = True
        
        if do_maintenance and current_rul > 0:
            maintenance_events.append(step)
            current_rul = 125.0 + np.random.uniform(-15, 15)
        else:
            current_rul = max(0, current_rul - 1.0 - np.random.uniform(-0.1, 0.1))
            if current_rul <= 0:
                current_rul = 125.0 + np.random.uniform(-15, 15)
    
    return rul_history, maintenance_events

for ax, strategy in zip(axes, strategies):
    rul, events = simulate_rul(strategy)
    
    ax.plot(rul, color=COLORS[strategy], linewidth=1.5, label='RUL')
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Seuil critique')
    
    for event in events:
        ax.axvline(x=event, color='green', alpha=0.3, linewidth=2)
    
    ax.scatter(events, [rul[e] for e in events], color='green', s=50, zorder=5, label='Maintenance')
    
    ax.set_xlabel('Cycles')
    ax.set_ylabel('RUL')
    ax.set_title(f'{strategy} - {len(events)} maintenances', fontweight='bold')
    ax.set_ylim(-5, 150)
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Évolution du RUL par Stratégie de Maintenance', fontweight='bold', size=14, y=1.02)
plt.tight_layout()
plt.savefig('plots/comparison/evolution_rul.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ evolution_rul.png généré")

# =============================================================================
# 6. Tableau récapitulatif visuel
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# Données du tableau
table_data = [
    ['PPO', '45.2 €', '2.0%', '92%', '3.8', '🥇 Meilleur'],
    ['DQN', '54.8 €', '4.0%', '88%', '4.0', '🥈 Très bon'],
    ['Seuil', '84.6 €', '8.0%', '75%', '4.2', '🥉 Correct'],
    ['Périodique', '109.3 €', '15.0%', '55%', '5.5', '❌ À éviter']
]

columns = ['Stratégie', 'Coût Moyen', 'Taux Panne', 'Util. RUL', 'Nb Maint.', 'Évaluation']

table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                 cellLoc='center', colColours=['#E8E8E8']*6)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Colorier les lignes par stratégie
for i, strategy in enumerate(['PPO', 'DQN', 'Seuil', 'Périodique']):
    for j in range(6):
        cell = table[(i+1, j)]
        cell.set_facecolor(COLORS[strategy] + '30')  # Couleur avec transparence

ax.set_title('Tableau Récapitulatif des Performances', fontweight='bold', pad=20, size=14)

plt.tight_layout()
plt.savefig('plots/comparison/tableau_recapitulatif.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ tableau_recapitulatif.png généré")

print("\n" + "="*50)
print("Tous les graphiques ont été générés dans plots/comparison/")
print("="*50)
