"""
Environnement Gymnasium pour la maintenance prédictive.
Simule un moteur avec dégradation et permet à un agent RL de décider
quand effectuer la maintenance.

Supporte deux modes de données:
- Synthétique: données générées algorithmiquement
- C-MAPSS: données réelles NASA (si disponibles)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

import config
from data_generator import SyntheticEngineDataGenerator

# Import conditionnel du loader C-MAPSS
try:
    from cmapss_loader import CMAPSSDataLoader
    CMAPSS_AVAILABLE = True
except ImportError:
    CMAPSS_AVAILABLE = False


class PredictiveMaintenanceEnv(gym.Env):
    """
    Environnement de maintenance prédictive pour Reinforcement Learning.
    
    L'agent observe une fenêtre glissante de lectures de capteurs et doit
    décider quand effectuer une maintenance préventive pour maximiser
    l'utilisation du moteur tout en évitant les pannes.
    
    Actions:
        0: Continuer (ne rien faire)
        1: Effectuer maintenance
        
    Observations:
        Fenêtre glissante des dernières lectures de capteurs (window_size x num_sensors)
        aplatie en vecteur 1D
        
    Récompenses:
        - Fonctionnement normal: +REWARD_RUNNING par cycle
        - Maintenance: -(COST_MAINTENANCE_FIXED + COST_ALPHA * RUL_actuel)
        - Panne: -COST_FAILURE (pénalité massive)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        window_size: int = config.WINDOW_SIZE,
        max_rul: int = config.MAX_RUL,
        num_sensors: int = config.NUM_SENSORS,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        use_real_data: bool = None
    ):
        """
        Initialise l'environnement.
        
        Args:
            window_size: Taille de la fenêtre d'observation
            max_rul: RUL maximum
            num_sensors: Nombre de capteurs
            render_mode: Mode de rendu
            seed: Graine aléatoire
            use_real_data: Utiliser les données C-MAPSS (défaut: config.USE_REAL_DATA)
        """
        super().__init__()
        
        self.window_size = window_size
        self.max_rul = max_rul
        self.num_sensors = num_sensors
        self.render_mode = render_mode
        self.seed_value = seed
        
        # Déterminer la source de données
        if use_real_data is None:
            use_real_data = getattr(config, 'USE_REAL_DATA', False)
        
        self.use_real_data = use_real_data and CMAPSS_AVAILABLE
        
        # Espace d'actions: 0 = Continuer, 1 = Maintenance
        self.action_space = spaces.Discrete(2)
        
        # Initialiser la source de données
        if self.use_real_data:
            self.cmapss_loader = CMAPSSDataLoader(
                data_dir=config.CMAPSS_DATA_DIR,
                subset=config.CMAPSS_SUBSET,
                max_rul=max_rul,
                window_size=window_size
            )
            self.cmapss_loader.load_data()
            self.cmapss_loader.add_rul_to_train()
            
            # Utiliser le nombre de capteurs du dataset
            self.num_sensors = len(self.cmapss_loader.get_sensor_columns())
            self.data_generator = None
            
            print(f"🔢 Environnement C-MAPSS initialisé ({self.num_sensors} capteurs)")
        else:
            self.cmapss_loader = None
            self.data_generator = SyntheticEngineDataGenerator(
                max_rul=max_rul,
                num_sensors=num_sensors,
                seed=seed
            )
        
        # Espace d'observations: fenêtre aplatie
        obs_dim = window_size * self.num_sensors
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # État interne
        self.current_cycle = 0
        self.total_life = 0
        self.sensor_history = None
        self.true_rul = 0
        self.done = False
        self.episode_reward = 0
        self.maintenance_count = 0
        
        # Données C-MAPSS du moteur actuel
        self.current_engine_sensors = None
        self.current_engine_rul = None
        self.current_engine_id = None
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Historique pour le debugging
        self.history = {
            'cycles': [],
            'rul': [],
            'actions': [],
            'rewards': []
        }

    
    def _get_observation(self) -> np.ndarray:
        """
        Construit l'observation à partir de l'historique des capteurs.
        
        Returns:
            Vecteur d'observation aplati
        """
        # Remplir avec des zéros si pas assez d'historique
        if len(self.sensor_history) < self.window_size:
            padding = np.zeros((self.window_size - len(self.sensor_history), self.num_sensors))
            window = np.vstack([padding, np.array(self.sensor_history)])
        else:
            window = np.array(self.sensor_history[-self.window_size:])
        
        # Normalisation simple (z-score sur la fenêtre)
        mean = window.mean()
        std = window.std() + 1e-8
        normalized = (window - mean) / std
        
        return normalized.flatten().astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calcule la récompense pour une action donnée.
        
        Formule de récompense:
        - Continuer: +REWARD_RUNNING (petit bonus)
        - Maintenance: -(COST_FIXED + alpha * RUL_actuel)
        - Panne (RUL=0 sans maintenance): -COST_FAILURE
        
        Args:
            action: 0 (continuer) ou 1 (maintenance)
            
        Returns:
            Récompense
        """
        if action == 1:  # Maintenance
            # Coût = fixe + pénalité pour maintenance précoce
            cost = config.COST_MAINTENANCE_FIXED + config.COST_ALPHA * self.true_rul
            return -cost
        
        else:  # Continuer
            if self.true_rul <= 0:  # Panne!
                return -config.COST_FAILURE
            else:
                return config.REWARD_RUNNING
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Réinitialise l'environnement pour un nouvel épisode.
        
        Args:
            seed: Graine aléatoire
            options: Options supplémentaires
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Réinitialisation de l'état commun
        self.current_cycle = 0
        self.sensor_history = []
        self.done = False
        self.episode_reward = 0
        self.maintenance_count = 0
        
        if self.use_real_data and self.cmapss_loader is not None:
            # Charger un moteur aléatoire depuis C-MAPSS
            unit_id, sensors, rul = self.cmapss_loader.get_random_engine(self.rng)
            
            self.current_engine_id = unit_id
            self.current_engine_sensors = sensors
            self.current_engine_rul = rul
            self.total_life = len(rul)
            self.true_rul = rul[0] if len(rul) > 0 else self.max_rul
            
            # Première lecture depuis les données réelles
            if len(sensors) > 0:
                self.sensor_history.append(sensors[0])
        else:
            # Mode synthétique
            if self.data_generator is not None:
                if seed is not None:
                    self.data_generator.rng = np.random.default_rng(seed)
                
                variation = self.data_generator.rng.integers(-20, 21)
                self.total_life = max(50, self.max_rul + variation)
                self.true_rul = min(self.total_life, self.max_rul)
                
                # Première lecture synthétique
                initial_reading = self.data_generator.get_sensor_reading(
                    self.current_cycle, self.total_life
                )
                self.sensor_history.append(initial_reading)
        
        # Réinitialiser l'historique
        self.history = {
            'cycles': [],
            'rul': [],
            'actions': [],
            'rewards': []
        }
        
        observation = self._get_observation()
        info = {
            'true_rul': self.true_rul,
            'cycle': self.current_cycle,
            'total_life': self.total_life,
            'data_source': 'cmapss' if self.use_real_data else 'synthetic',
            'engine_id': self.current_engine_id if self.use_real_data else None
        }
        
        return observation, info

    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: 0 (continuer) ou 1 (maintenance)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Épisode terminé. Appelez reset().")
        
        # Enregistrer l'état avant action
        self.history['cycles'].append(self.current_cycle)
        self.history['rul'].append(self.true_rul)
        self.history['actions'].append(action)
        
        # Calculer la récompense
        reward = self._calculate_reward(action)
        self.history['rewards'].append(reward)
        self.episode_reward += reward
        
        terminated = False
        truncated = False
        
        if action == 1:  # Maintenance effectuée
            self.maintenance_count += 1
            
            if self.use_real_data and self.cmapss_loader is not None:
                # Charger un nouveau moteur depuis C-MAPSS
                unit_id, sensors, rul = self.cmapss_loader.get_random_engine(self.rng)
                self.current_engine_id = unit_id
                self.current_engine_sensors = sensors
                self.current_engine_rul = rul
                self.total_life = len(rul)
                self.current_cycle = 0
                self.true_rul = rul[0] if len(rul) > 0 else self.max_rul
                self.sensor_history = []
                if len(sensors) > 0:
                    self.sensor_history.append(sensors[0])
            else:
                # Mode synthétique
                variation = self.rng.integers(-20, 21)
                self.total_life = max(50, self.max_rul + variation)
                self.current_cycle = 0
                self.true_rul = min(self.total_life, self.max_rul)
                self.sensor_history = []
                
                if self.data_generator is not None:
                    initial_reading = self.data_generator.get_sensor_reading(0, self.total_life)
                    self.sensor_history.append(initial_reading)
            
        else:  # Continuer
            if self.true_rul <= 0:  # Panne!
                terminated = True
                self.done = True
            else:
                # Avancer d'un cycle
                self.current_cycle += 1
                
                if self.use_real_data and self.current_engine_sensors is not None:
                    # Utiliser les données C-MAPSS
                    if self.current_cycle < len(self.current_engine_sensors):
                        self.sensor_history.append(self.current_engine_sensors[self.current_cycle])
                        self.true_rul = self.current_engine_rul[self.current_cycle]
                    else:
                        # Fin des données du moteur
                        self.true_rul = 0
                        terminated = True
                        self.done = True
                else:
                    # Mode synthétique
                    self.true_rul = max(0, min(self.total_life - self.current_cycle, self.max_rul))
                    
                    if self.data_generator is not None:
                        reading = self.data_generator.get_sensor_reading(
                            self.current_cycle, self.total_life
                        )
                        self.sensor_history.append(reading)
        
        observation = self._get_observation()
        
        info = {
            'true_rul': self.true_rul,
            'cycle': self.current_cycle,
            'total_life': self.total_life,
            'maintenance_count': self.maintenance_count,
            'episode_reward': self.episode_reward,
            'data_source': 'cmapss' if self.use_real_data else 'synthetic'
        }
        
        return observation, reward, terminated, truncated, info

    
    def render(self) -> Optional[str]:
        """
        Affiche l'état actuel de l'environnement.
        """
        if self.render_mode == "human" or self.render_mode == "ansi":
            status = "🔧" if self.maintenance_count > 0 else "⚙️"
            rul_bar = "█" * min(20, int(20 * self.true_rul / self.max_rul))
            rul_bar += "░" * (20 - len(rul_bar))
            
            output = (
                f"\n{status} Cycle: {self.current_cycle:4d} | "
                f"RUL: [{rul_bar}] {self.true_rul:3d} | "
                f"Maintenances: {self.maintenance_count}"
            )
            
            if self.render_mode == "human":
                print(output)
            return output
        
        return None
    
    def close(self):
        """Nettoie les ressources."""
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de l'épisode.
        
        Returns:
            Dictionnaire de statistiques
        """
        return {
            'total_cycles': self.current_cycle,
            'maintenance_count': self.maintenance_count,
            'episode_reward': self.episode_reward,
            'final_rul': self.true_rul,
            'failed': self.true_rul <= 0 and not any(
                a == 1 for a in self.history['actions'][-1:]
            )
        }


def make_env(seed: Optional[int] = None) -> PredictiveMaintenanceEnv:
    """
    Factory function pour créer l'environnement.
    Compatible avec les VecEnv de Stable Baselines 3.
    """
    def _init():
        env = PredictiveMaintenanceEnv(seed=seed)
        return env
    return _init


if __name__ == "__main__":
    # Test de l'environnement
    print("=== Test de l'Environnement de Maintenance Prédictive ===\n")
    
    env = PredictiveMaintenanceEnv(render_mode="human", seed=42)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Info initiale: {info}\n")
    
    # Simulation avec politique aléatoire
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            env.render()
        
        if terminated or truncated:
            print(f"\n🛑 Épisode terminé après {step+1} steps")
            print(f"Stats: {env.get_episode_stats()}")
            break
    
    print(f"\nRécompense totale: {total_reward:.2f}")
    env.close()
