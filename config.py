"""
Configuration globale pour le système de maintenance prédictive.
Contient tous les hyperparamètres, coûts et paramètres d'environnement.
"""

# =============================================================================
# PARAMÈTRES DE L'ENVIRONNEMENT
# =============================================================================

# Fenêtre glissante pour l'observation (nombre de cycles historiques)
WINDOW_SIZE = 30

# Nombre maximum de cycles avant panne (RUL max)
MAX_RUL = 125

# Nombre de capteurs simulés
NUM_SENSORS = 14

# =============================================================================
# CONFIGURATION DATASET NASA C-MAPSS
# =============================================================================

# Sous-ensemble à utiliser: "FD001", "FD002", "FD003", "FD004"
CMAPSS_SUBSET = "FD001"

# Répertoire des données
CMAPSS_DATA_DIR = "data/cmapss"

# Utiliser des données réelles (True) ou synthétiques (False)
USE_REAL_DATA = True


# =============================================================================
# STRUCTURE DES COÛTS
# =============================================================================

# Coût fixe de maintenance préventive
COST_MAINTENANCE_FIXED = 50.0

# Coefficient alpha pour le coût variable (pénalité maintenance précoce)
# Plus le RUL est élevé lors de la maintenance, plus on "gaspille" de vie utile
COST_ALPHA = 0.5

# Coût de panne catastrophique (doit être >> coût maintenance max)
# Règle: COST_FAILURE > 1.5 * max(COST_MAINTENANCE)
# Max maintenance cost ~ COST_MAINTENANCE_FIXED + COST_ALPHA * MAX_RUL
COST_FAILURE = 200.0

# Récompense pour fonctionnement normal (petit bonus par cycle)
REWARD_RUNNING = 1.0

# =============================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT RL
# =============================================================================

# Algorithme à utiliser: "PPO" ou "DQN"
RL_ALGORITHM = "PPO"

# Nombre total de timesteps d'entraînement
TOTAL_TIMESTEPS = 50000

# Taille du buffer pour DQN
BUFFER_SIZE = 10000

# Taille du batch pour l'entraînement
BATCH_SIZE = 64

# Taux d'apprentissage
LEARNING_RATE = 3e-4

# Facteur de discount (gamma)
GAMMA = 0.99

# Nombre d'environnements parallèles pour PPO
N_ENVS = 4

# =============================================================================
# PARAMÈTRES DES BASELINES (STRATÉGIES CLASSIQUES)
# =============================================================================

# Maintenance périodique: intervalle en cycles
PERIODIC_INTERVAL = 80

# Maintenance basée sur seuil: seuil RUL pour déclencher maintenance
THRESHOLD_RUL = 25

# =============================================================================
# PARAMÈTRES DE SIMULATION
# =============================================================================

# Nombre de moteurs pour la simulation comparative
NUM_ENGINES_SIMULATION = 50

# Seed pour la reproductibilité
RANDOM_SEED = 42

# =============================================================================
# PARAMÈTRES DE VISUALISATION
# =============================================================================

# Style des graphiques
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# Taille des figures (largeur, hauteur) en pouces
FIGURE_SIZE = (12, 6)

# DPI pour les exports
FIGURE_DPI = 150

# Couleurs pour les stratégies
COLORS = {
    "periodic": "#E74C3C",      # Rouge
    "threshold": "#F39C12",     # Orange
    "rl": "#27AE60",            # Vert
    "failure": "#8E44AD",       # Violet
}

# =============================================================================
# PARAMÈTRES DE DÉMO
# =============================================================================

# Activer le mode démo garantie (données synthétiques si entraînement incomplet)
DEMO_MODE = True

# Seuil de performance pour considérer l'entraînement comme "réussi"
# Si le coût moyen RL > ce seuil * coût périodique, utiliser les données mock
DEMO_PERFORMANCE_THRESHOLD = 0.8

# =============================================================================
# CHEMINS DE FICHIERS
# =============================================================================

# Répertoire pour sauvegarder les modèles entraînés
MODEL_SAVE_DIR = "models"

# Répertoire pour les graphiques générés
PLOTS_SAVE_DIR = "plots"

# Nom du fichier modèle
MODEL_FILENAME = "ppo_maintenance_agent"
