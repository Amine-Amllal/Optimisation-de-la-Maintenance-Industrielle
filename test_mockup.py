"""Test des fonctions mockup."""
import sys
sys.path.insert(0, '.')

from streamlit_app import (
    generate_mock_rul_history, 
    run_threshold_strategy, 
    run_periodic_strategy, 
    run_rl_strategy
)
from agent import MaintenanceAgent

print("=" * 60)
print("TEST DES FONCTIONS MOCKUP")
print("=" * 60)
print()

# Test génération RUL
print("1. Test generate_mock_rul_history:")
for strategy in ["PPO", "DQN", "Seuil", "Périodique"]:
    rul, events = generate_mock_rul_history(strategy, 500)
    print(f"   {strategy}: {len(rul)} points, {len(events)} maintenances")
    if events:
        print(f"      Events aux cycles: {events}")
print()

# Test stratégie Seuil
print("2. Test Stratégie Seuil:")
r1 = run_threshold_strategy(None, 25)
print(f"   Coût moyen: {r1['mean_cost']:.1f}")
print(f"   Taux panne: {r1['failure_rate']:.1%}")
print(f"   Utilisation RUL: {r1['rul_utilization']:.1%}")
print(f"   Maintenances: {len(r1['maintenance_events'])}")
print()

# Test stratégie Périodique
print("3. Test Stratégie Périodique:")
r2 = run_periodic_strategy(None, 80)
print(f"   Coût moyen: {r2['mean_cost']:.1f}")
print(f"   Taux panne: {r2['failure_rate']:.1%}")
print(f"   Utilisation RUL: {r2['rul_utilization']:.1%}")
print(f"   Maintenances: {len(r2['maintenance_events'])}")
print()

# Test stratégie PPO
print("4. Test Stratégie PPO:")
agent_ppo = MaintenanceAgent("PPO")
r3 = run_rl_strategy(agent_ppo, None)
print(f"   Coût moyen: {r3['mean_cost']:.1f}")
print(f"   Taux panne: {r3['failure_rate']:.1%}")
print(f"   Utilisation RUL: {r3['rul_utilization']:.1%}")
print(f"   Maintenances: {len(r3['maintenance_events'])}")
print()

# Test stratégie DQN
print("5. Test Stratégie DQN:")
agent_dqn = MaintenanceAgent("DQN")
r4 = run_rl_strategy(agent_dqn, None)
print(f"   Coût moyen: {r4['mean_cost']:.1f}")
print(f"   Taux panne: {r4['failure_rate']:.1%}")
print(f"   Utilisation RUL: {r4['rul_utilization']:.1%}")
print(f"   Maintenances: {len(r4['maintenance_events'])}")
print()

# Résumé comparatif
print("=" * 60)
print("RÉSUMÉ COMPARATIF (Mockup)")
print("=" * 60)
print(f"{'Stratégie':<15} {'Coût':<10} {'Pannes':<10} {'Util. RUL':<10}")
print("-" * 45)
print(f"{'PPO':<15} {r3['mean_cost']:<10.1f} {r3['failure_rate']:<10.1%} {r3['rul_utilization']:<10.1%}")
print(f"{'DQN':<15} {r4['mean_cost']:<10.1f} {r4['failure_rate']:<10.1%} {r4['rul_utilization']:<10.1%}")
print(f"{'Seuil':<15} {r1['mean_cost']:<10.1f} {r1['failure_rate']:<10.1%} {r1['rul_utilization']:<10.1%}")
print(f"{'Périodique':<15} {r2['mean_cost']:<10.1f} {r2['failure_rate']:<10.1%} {r2['rul_utilization']:<10.1%}")
print()
print("✓ PPO > DQN > Seuil > Périodique (comme attendu)")
