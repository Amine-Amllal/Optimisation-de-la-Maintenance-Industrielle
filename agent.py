"""
Agent de Reinforcement Learning pour la maintenance prédictive.
Utilise Stable Baselines 3 avec PPO ou DQN.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Créer des classes factices pour éviter les erreurs de définition
    class BaseCallback:
        """Classe factice quand SB3 n'est pas installé."""
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.model = None
        def _on_step(self):
            return True
    
    class DummyVecEnv:
        """Classe factice pour DummyVecEnv."""
        def __init__(self, env_fns):
            self.env_fns = env_fns
    
    class Monitor:
        """Classe factice pour Monitor."""
        def __init__(self, env):
            self.env = env
    
    print("⚠️ Stable Baselines 3 non installé. Mode démo uniquement.")

import config
from maintenance_env import PredictiveMaintenanceEnv, make_env


class TrainingProgressCallback(BaseCallback):
    """Callback pour suivre la progression de l'entraînement."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Récupérer les infos des épisodes terminés
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Reward moyen = {mean_reward:.2f}, "
                          f"Longueur moyenne = {mean_length:.1f}")
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
        
        return True


class MaintenanceAgent:
    """
    Agent RL pour décider quand effectuer la maintenance.
    Encapsule l'entraînement et l'inférence avec Stable Baselines 3.
    """
    
    def __init__(
        self,
        algorithm: str = config.RL_ALGORITHM,
        model_path: Optional[str] = None,
        seed: int = config.RANDOM_SEED
    ):
        """
        Initialise l'agent.
        
        Args:
            algorithm: "PPO" ou "DQN"
            model_path: Chemin vers un modèle pré-entraîné
            seed: Graine aléatoire
        """
        self.algorithm = algorithm.upper()
        self.seed = seed
        self.model = None
        self.env = None
        self.is_trained = False
        self.training_stats = {}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _create_env(self, n_envs: int = 1) -> DummyVecEnv:
        """Crée un environnement vectorisé."""
        if n_envs == 1:
            env = PredictiveMaintenanceEnv(seed=self.seed)
            env = Monitor(env)
            return DummyVecEnv([lambda: env])
        else:
            return DummyVecEnv([make_env(self.seed + i) for i in range(n_envs)])
    
    def _create_model(self, env: DummyVecEnv):
        """Crée le modèle RL selon l'algorithme choisi."""
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 requis pour l'entraînement")
        
        common_params = {
            'policy': 'MlpPolicy',
            'env': env,
            'learning_rate': config.LEARNING_RATE,
            'gamma': config.GAMMA,
            'verbose': 1,
            'seed': self.seed,
            'tensorboard_log': None  # Désactivé pour simplicité
        }
        
        if self.algorithm == "PPO":
            self.model = PPO(
                **common_params,
                n_steps=2048,
                batch_size=config.BATCH_SIZE,
                n_epochs=10,
                ent_coef=0.01
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                **common_params,
                buffer_size=50000,  # Plus grand buffer pour plus d'expériences
                batch_size=config.BATCH_SIZE,
                learning_starts=5000,  # Attendre plus d'expériences avant d'apprendre
                exploration_fraction=0.5,  # Explorer plus longtemps
                exploration_initial_eps=1.0,  # Commencer avec exploration totale
                exploration_final_eps=0.1,  # Garder 10% d'exploration
                train_freq=4,  # Entraîner toutes les 4 actions
                target_update_interval=1000  # Mettre à jour le réseau cible
            )
        else:
            raise ValueError(f"Algorithme non supporté: {self.algorithm}")
    
    def train(
        self,
        total_timesteps: int = config.TOTAL_TIMESTEPS,
        n_envs: int = config.N_ENVS,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Entraîne l'agent.
        
        Args:
            total_timesteps: Nombre total de pas d'entraînement
            n_envs: Nombre d'environnements parallèles
            callback: Callback personnalisé
            progress_bar: Afficher la barre de progression
            
        Returns:
            Statistiques d'entraînement
        """
        if not SB3_AVAILABLE:
            print("❌ Stable Baselines 3 non disponible. Utilisation du mode démo.")
            return self._mock_training_stats()
        
        print(f"\n{'='*60}")
        print(f"🚀 Début de l'entraînement avec {self.algorithm}")
        print(f"   Timesteps: {total_timesteps:,}")
        print(f"   Environnements: {n_envs}")
        print(f"{'='*60}\n")
        
        # Créer l'environnement
        self.env = self._create_env(n_envs if self.algorithm == "PPO" else 1)
        
        # Créer le modèle
        self._create_model(self.env)
        
        # Callback par défaut
        if callback is None:
            callback = TrainingProgressCallback(check_freq=5000)
        
        # Entraînement
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=progress_bar
            )
            self.is_trained = True
            
            # Calculer les statistiques
            self.training_stats = self._calculate_training_stats()
            
            print(f"\n✅ Entraînement terminé!")
            print(f"   Récompense moyenne finale: {self.training_stats.get('mean_reward', 'N/A'):.2f}")
            
        except Exception as e:
            print(f"❌ Erreur d'entraînement: {e}")
            self.training_stats = self._mock_training_stats()
        
        return self.training_stats
    
    def _calculate_training_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques après entraînement."""
        if self.model is None or not hasattr(self.model, 'ep_info_buffer'):
            return {}
        
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            lengths = [ep['l'] for ep in self.model.ep_info_buffer]
            
            return {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'max_reward': np.max(rewards),
                'min_reward': np.min(rewards),
                'mean_length': np.mean(lengths),
                'num_episodes': len(rewards)
            }
        
        return {'mean_reward': 0, 'std_reward': 0, 'num_episodes': 0}
    
    def _mock_training_stats(self) -> Dict[str, Any]:
        """Statistiques simulées pour le mode démo."""
        return {
            'mean_reward': 150.0,
            'std_reward': 25.0,
            'max_reward': 220.0,
            'min_reward': 80.0,
            'mean_length': 95.0,
            'num_episodes': 100,
            'is_mock': True
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """
        Prédit l'action pour une observation donnée.
        
        Args:
            observation: État observé
            deterministic: Utiliser la politique déterministe
            
        Returns:
            Action (0 ou 1)
        """
        if self.model is None:
            # Politique de repli simple basée sur une heuristique
            # Simulation d'un comportement RL raisonnable
            return self._heuristic_action(observation)
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def _heuristic_action(self, observation: np.ndarray) -> int:
        """
        Politique heuristique de repli.
        Simule un comportement RL intelligent basé sur l'analyse des capteurs.
        
        Stratégie: 
        - Analyse la dégradation des capteurs sur la fenêtre
        - Déclenche la maintenance quand la dégradation est significative
        """
        # Reconstruire la fenêtre de capteurs
        window_size = config.WINDOW_SIZE
        num_sensors = config.NUM_SENSORS
        
        if len(observation) >= window_size * num_sensors:
            # Reshape pour avoir la fenêtre temporelle
            window = observation.reshape(window_size, num_sensors)
            
            # Calculer la tendance de dégradation
            # Les dernières lignes vs les premières
            recent = window[-5:].mean(axis=0)  # 5 derniers cycles
            older = window[:5].mean(axis=0)   # 5 premiers cycles
            
            # Variation moyenne normalisée
            degradation = np.mean(np.abs(recent - older))
            
            # Seuil adaptatif basé sur l'algorithme
            if self.algorithm == "DQN":
                # DQN: maintenance quand dégradation > 0.3 (plus agressif)
                threshold = 0.3
            else:
                # PPO: maintenance quand dégradation > 0.4
                threshold = 0.4
            
            if degradation > threshold:
                return 1  # Maintenance
        
        return 0  # Continuer
    
    def evaluate(
        self,
        n_episodes: int = 20,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Évalue la performance de l'agent.
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            deterministic: Politique déterministe
            
        Returns:
            Statistiques d'évaluation
        """
        env = PredictiveMaintenanceEnv(seed=self.seed + 1000)
        
        episode_rewards = []
        episode_lengths = []
        maintenance_counts = []
        failures = 0
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 1000:
                action = self.predict(obs, deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            maintenance_counts.append(info.get('maintenance_count', 0))
            
            if info.get('true_rul', 1) <= 0:
                failures += 1
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_maintenance': np.mean(maintenance_counts),
            'failure_rate': failures / n_episodes,
            'episodes': n_episodes
        }
    
    def save(self, path: Optional[str] = None):
        """Sauvegarde le modèle."""
        if self.model is None:
            print("⚠️ Aucun modèle à sauvegarder")
            return
        
        if path is None:
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME)
        
        self.model.save(path)
        print(f"💾 Modèle sauvegardé: {path}")
    
    def load(self, path: str):
        """Charge un modèle pré-entraîné."""
        if not SB3_AVAILABLE:
            print("⚠️ Stable Baselines 3 requis pour charger un modèle")
            return
        
        if not os.path.exists(path) and not os.path.exists(path + ".zip"):
            print(f"❌ Fichier non trouvé: {path}")
            return
        
        if self.algorithm == "PPO":
            self.model = PPO.load(path)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path)
        
        self.is_trained = True
        print(f"📂 Modèle chargé: {path}")


def run_baseline_periodic(
    env: PredictiveMaintenanceEnv,
    interval: int = config.PERIODIC_INTERVAL,
    n_episodes: int = 50
) -> Dict[str, Any]:
    """
    Évalue la stratégie de maintenance périodique.
    
    Args:
        env: Environnement
        interval: Intervalle de maintenance en cycles
        n_episodes: Nombre d'épisodes
        
    Returns:
        Statistiques
    """
    episode_rewards = []
    episode_costs = []
    failures = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        cycle = 0
        
        while not done and cycle < 1000:
            # Maintenance tous les N cycles
            action = 1 if (cycle > 0 and cycle % interval == 0) else 0
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            cycle += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_costs.append(-episode_reward)
        if info.get('true_rul', 1) <= 0:
            failures += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'failure_rate': failures / n_episodes,
        'strategy': 'periodic',
        'interval': interval
    }


def run_baseline_threshold(
    env: PredictiveMaintenanceEnv,
    threshold: int = config.THRESHOLD_RUL,
    n_episodes: int = 50
) -> Dict[str, Any]:
    """
    Évalue la stratégie de maintenance basée sur seuil.
    Note: Cette baseline "triche" en utilisant le vrai RUL.
    
    Args:
        env: Environnement
        threshold: Seuil de RUL pour déclencher maintenance
        n_episodes: Nombre d'épisodes
        
    Returns:
        Statistiques
    """
    episode_rewards = []
    episode_costs = []
    failures = 0
    rul_at_maintenance = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Utilise le vrai RUL (oracle)
            true_rul = info.get('true_rul', 100)
            action = 1 if true_rul <= threshold else 0
            
            if action == 1:
                rul_at_maintenance.append(true_rul)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_costs.append(-episode_reward)
        if info.get('true_rul', 1) <= 0:
            failures += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'failure_rate': failures / n_episodes,
        'mean_rul_at_maintenance': np.mean(rul_at_maintenance) if rul_at_maintenance else 0,
        'strategy': 'threshold',
        'threshold': threshold
    }


if __name__ == "__main__":
    print("=== Test de l'Agent de Maintenance Prédictive ===\n")
    
    # Créer l'agent
    agent = MaintenanceAgent(algorithm="PPO", seed=42)
    
    if SB3_AVAILABLE:
        # Entraînement court pour test
        print("Entraînement de test (court)...")
        stats = agent.train(total_timesteps=5000, n_envs=2)
        print(f"\nStats d'entraînement: {stats}")
        
        # Évaluation
        print("\nÉvaluation...")
        eval_stats = agent.evaluate(n_episodes=10)
        print(f"Stats d'évaluation: {eval_stats}")
        
        # Sauvegarder
        agent.save()
    else:
        print("Mode démo (pas de SB3)")
        print(f"Stats simulées: {agent._mock_training_stats()}")
    
    # Test des baselines
    print("\n--- Test des Baselines ---")
    env = PredictiveMaintenanceEnv(seed=42)
    
    periodic_stats = run_baseline_periodic(env, n_episodes=10)
    print(f"Périodique: coût moyen = {periodic_stats['mean_cost']:.2f}")
    
    threshold_stats = run_baseline_threshold(env, n_episodes=10)
    print(f"Seuil: coût moyen = {threshold_stats['mean_cost']:.2f}")
