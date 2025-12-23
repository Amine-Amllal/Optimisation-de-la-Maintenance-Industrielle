"""
Générateur de données synthétiques type NASA C-MAPSS simplifié.
Génère des données de dégradation de moteurs avec capteurs et RUL.
"""

import numpy as np
from typing import Tuple, Optional
import config


class SyntheticEngineDataGenerator:
    """
    Générateur de données synthétiques simulant la dégradation d'un moteur.
    Inspiré du dataset NASA C-MAPSS (Turbofan Engine Degradation Simulation).
    """
    
    def __init__(
        self,
        max_rul: int = config.MAX_RUL,
        num_sensors: int = config.NUM_SENSORS,
        seed: Optional[int] = None
    ):
        """
        Initialise le générateur.
        
        Args:
            max_rul: Durée de vie maximale en cycles
            num_sensors: Nombre de capteurs simulés
            seed: Graine pour la reproductibilité
        """
        self.max_rul = max_rul
        self.num_sensors = num_sensors
        self.rng = np.random.default_rng(seed)
        
        # Paramètres de dégradation pour chaque capteur
        # Chaque capteur a: valeur_initiale, taux_dégradation, bruit_std
        self._init_sensor_params()
    
    def _init_sensor_params(self):
        """Initialise les paramètres de simulation des capteurs."""
        self.sensor_params = []
        
        # Capteurs de température (3)
        for i in range(3):
            self.sensor_params.append({
                'name': f'temp_{i+1}',
                'initial': 300 + self.rng.uniform(-10, 10),
                'degradation_rate': 0.1 + self.rng.uniform(0, 0.05),
                'noise_std': 2.0,
                'type': 'increasing'
            })
        
        # Capteurs de pression (3)
        for i in range(3):
            self.sensor_params.append({
                'name': f'pressure_{i+1}',
                'initial': 100 + self.rng.uniform(-5, 5),
                'degradation_rate': -0.05 + self.rng.uniform(-0.02, 0.02),
                'noise_std': 1.0,
                'type': 'decreasing'
            })
        
        # Capteurs de vibration (3)
        for i in range(3):
            self.sensor_params.append({
                'name': f'vibration_{i+1}',
                'initial': 0.5 + self.rng.uniform(-0.1, 0.1),
                'degradation_rate': 0.02 + self.rng.uniform(0, 0.01),
                'noise_std': 0.05,
                'type': 'increasing'
            })
        
        # Capteurs de vitesse/rotation (3)
        for i in range(3):
            self.sensor_params.append({
                'name': f'speed_{i+1}',
                'initial': 9000 + self.rng.uniform(-100, 100),
                'degradation_rate': -2.0 + self.rng.uniform(-0.5, 0.5),
                'noise_std': 20.0,
                'type': 'decreasing'
            })
        
        # Capteurs d'efficacité (2)
        for i in range(2):
            self.sensor_params.append({
                'name': f'efficiency_{i+1}',
                'initial': 0.95 + self.rng.uniform(-0.02, 0.02),
                'degradation_rate': -0.002 + self.rng.uniform(-0.0005, 0),
                'noise_std': 0.01,
                'type': 'decreasing'
            })
        
        # Ajuster au nombre exact de capteurs demandé
        self.sensor_params = self.sensor_params[:self.num_sensors]
    
    def generate_engine_data(
        self,
        total_cycles: Optional[int] = None,
        include_failure: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère les données complètes d'un moteur jusqu'à la panne.
        
        Args:
            total_cycles: Nombre total de cycles (si None, utilise max_rul + variation)
            include_failure: Si True, le moteur tombe en panne à la fin
            
        Returns:
            sensor_data: Array (cycles, num_sensors) des lectures capteurs
            rul_data: Array (cycles,) du RUL restant à chaque cycle
        """
        if total_cycles is None:
            # Variation naturelle de la durée de vie
            variation = self.rng.integers(-20, 21)
            total_cycles = max(50, self.max_rul + variation)
        
        sensor_data = np.zeros((total_cycles, self.num_sensors))
        
        for cycle in range(total_cycles):
            # Facteur de dégradation non-linéaire (accélère vers la fin)
            progress = cycle / total_cycles
            degradation_factor = progress ** 1.5  # Dégradation exponentielle
            
            for i, params in enumerate(self.sensor_params):
                # Valeur de base avec dégradation
                base_value = params['initial']
                degradation = params['degradation_rate'] * cycle * (1 + degradation_factor)
                
                # Ajout de bruit
                noise = self.rng.normal(0, params['noise_std'])
                
                # Valeur finale
                sensor_data[cycle, i] = base_value + degradation + noise
        
        # Calcul du RUL (Remaining Useful Life)
        rul_data = np.arange(total_cycles - 1, -1, -1)
        
        # Clip RUL au maximum configuré (comme dans C-MAPSS)
        rul_data = np.minimum(rul_data, self.max_rul)
        
        return sensor_data, rul_data
    
    def generate_dataset(
        self,
        num_engines: int,
        seed: Optional[int] = None
    ) -> list:
        """
        Génère un dataset complet avec plusieurs moteurs.
        
        Args:
            num_engines: Nombre de moteurs à simuler
            seed: Graine pour la reproductibilité
            
        Returns:
            Liste de tuples (sensor_data, rul_data) pour chaque moteur
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        dataset = []
        for _ in range(num_engines):
            sensor_data, rul_data = self.generate_engine_data()
            dataset.append((sensor_data, rul_data))
        
        return dataset
    
    def get_sensor_reading(
        self,
        cycle: int,
        total_life: int
    ) -> np.ndarray:
        """
        Génère une lecture de capteur unique pour un cycle donné.
        Utile pour la simulation en temps réel dans l'environnement Gym.
        
        Args:
            cycle: Cycle actuel
            total_life: Durée de vie totale prévue du moteur
            
        Returns:
            Array des valeurs des capteurs
        """
        readings = np.zeros(self.num_sensors)
        progress = cycle / total_life
        degradation_factor = progress ** 1.5
        
        for i, params in enumerate(self.sensor_params):
            base_value = params['initial']
            degradation = params['degradation_rate'] * cycle * (1 + degradation_factor)
            noise = self.rng.normal(0, params['noise_std'])
            readings[i] = base_value + degradation + noise
        
        return readings
    
    def normalize_data(
        self,
        data: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normalise les données des capteurs.
        
        Args:
            data: Données à normaliser (cycles, sensors)
            method: 'minmax' ou 'zscore'
            
        Returns:
            Données normalisées
        """
        if method == 'minmax':
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Éviter division par zéro
            return (data - min_vals) / range_vals
        
        elif method == 'zscore':
            mean_vals = data.mean(axis=0)
            std_vals = data.std(axis=0)
            std_vals[std_vals == 0] = 1
            return (data - mean_vals) / std_vals
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")


def create_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée un jeu de données d'exemple pour les tests.
    
    Returns:
        sensor_data, rul_data pour un moteur
    """
    generator = SyntheticEngineDataGenerator(seed=config.RANDOM_SEED)
    return generator.generate_engine_data()


if __name__ == "__main__":
    # Test du générateur
    print("=== Test du Générateur de Données Synthétiques ===\n")
    
    generator = SyntheticEngineDataGenerator(seed=42)
    
    # Générer données pour un moteur
    sensor_data, rul_data = generator.generate_engine_data()
    
    print(f"Forme des données capteurs: {sensor_data.shape}")
    print(f"Forme des données RUL: {rul_data.shape}")
    print(f"\nPremières valeurs capteurs:\n{sensor_data[:5, :3]}")
    print(f"\nPremières valeurs RUL: {rul_data[:10]}")
    print(f"Dernières valeurs RUL: {rul_data[-10:]}")
    
    # Générer un dataset
    dataset = generator.generate_dataset(num_engines=5, seed=42)
    print(f"\nDataset généré: {len(dataset)} moteurs")
    for i, (s, r) in enumerate(dataset):
        print(f"  Moteur {i+1}: {s.shape[0]} cycles, RUL initial={r[0]}")
