# Planificateur de Maintenance Prédictive par RL 🚀

Ce projet implémente un système de maintenance prédictive intelligent utilisant l'apprentissage par renforcement profond (Deep Reinforcement Learning). Il compare les algorithmes **PPO** et **DQN** face aux stratégies classiques (Maintenance Périodique et à Seuils) sur un problème inspiré du dataset NASA C-MAPSS.

## 📂 Structure du Projet

- `agent.py`: Logique des agents RL (PPO, DQN) et baselines.
- `maintenance_env.py`: Environnement Gym personnalisé simulant la dégradation.
- `compare_rl_algorithms.py`: Script principal pour entraîner et comparer les modèles.
- `config.py`: Configuration centralisée (hyperparamètres, coûts, paramètres physiques).
- `visualization.py`: Utilitaires pour générer les graphiques de résultats.
- `rapport_academique_moderne.tex`: **Rapport final complet** (Code source LaTeX).

## 🚀 Utilisation Rapide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Lancer la Comparaison Complète

Pour entraîner les modèles et générer les rapports :

```bash
python compare_rl_algorithms.py
```

Cela va :
1. Entraîner PPO et DQN.
2. Évaluer leurs performances.
3. Les comparer aux baselines (Périodique, Seuils).
4. Générer des graphiques dans `rl_comparison/` et un rapport HTML.

### 3. Compiler le Rapport

Pour obtenir le document PDF final :

```bash
pdflatex rapport_academique_moderne.tex
pdflatex rapport_academique_moderne.tex
```

## 📊 Résultats Clés

| Stratégie | Coût Moyen | Taux de Panne | RUL Utilisé |
| :--- | :---: | :---: | :---: |
| Périodique | 140€ | 5% | 45% |
| Seuils | 100€ | 2% | 70% |
| **PPO (IA)** | **65€** | **2%** | **92%** |

---
**Auteur :** Amine AMLLAL  
**Encadrant :** M. Tawfik MASROUR
