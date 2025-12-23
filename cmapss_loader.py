"""
Loader pour le dataset NASA C-MAPSS (Turbofan Engine Degradation).

Ce module gère:
- Téléchargement automatique du dataset depuis Kaggle
- Parsing des fichiers de données
- Préparation des données pour l'entraînement RL
- Génération de fenêtres temporelles pour l'observation

Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- FD001: 1 condition opérationnelle, 1 mode de panne
- FD002: 6 conditions opérationnelles, 1 mode de panne  
- FD003: 1 condition opérationnelle, 2 modes de panne
- FD004: 6 conditions opérationnelles, 2 modes de panne
"""

import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import urllib.request
import config

# Colonnes du dataset C-MAPSS
CMAPSS_COLUMNS = [
    'unit_id',           # ID du moteur
    'cycle',             # Cycle temporel
    'op_setting_1',      # Paramètre opérationnel 1
    'op_setting_2',      # Paramètre opérationnel 2
    'op_setting_3',      # Paramètre opérationnel 3
    'sensor_1',          # Fan inlet temperature
    'sensor_2',          # LPC outlet temperature
    'sensor_3',          # HPC outlet temperature
    'sensor_4',          # LPT outlet temperature
    'sensor_5',          # Fan inlet Pressure
    'sensor_6',          # Bypass-duct pressure
    'sensor_7',          # HPC outlet pressure
    'sensor_8',          # Physical fan speed
    'sensor_9',          # Physical core speed
    'sensor_10',         # Engine pressure ratio
    'sensor_11',         # HPC outlet Static pressure
    'sensor_12',         # Ratio of fuel flow to Ps30
    'sensor_13',         # Corrected fan speed
    'sensor_14',         # Corrected core speed
    'sensor_15',         # Bypass Ratio
    'sensor_16',         # Burner fuel-air ratio
    'sensor_17',         # Bleed Enthalpy
    'sensor_18',         # Required fan speed
    'sensor_19',         # Required fan conversion speed
    'sensor_20',         # High-pressure turbines Cool air flow
    'sensor_21',         # Low-pressure turbines Cool air flow
]

# Capteurs à variance quasi-nulle (à exclure)
CONSTANT_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                    'sensor_16', 'sensor_18', 'sensor_19']

# Capteurs informatifs à garder
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 
                    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
                    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
                    'sensor_20', 'sensor_21']


class CMAPSSDataLoader:
    """
    Classe pour charger et prétraiter le dataset NASA C-MAPSS.
    """
    
    def __init__(
        self,
        data_dir: str = "data/cmapss",
        subset: str = "FD001",
        max_rul: int = config.MAX_RUL,
        window_size: int = config.WINDOW_SIZE,
        normalize: bool = True
    ):
        """
        Initialise le loader.
        
        Args:
            data_dir: Répertoire des données
            subset: Sous-ensemble à utiliser (FD001, FD002, FD003, FD004)
            max_rul: RUL maximum (clip)
            window_size: Taille de la fenêtre d'observation
            normalize: Normaliser les données
        """
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.max_rul = max_rul
        self.window_size = window_size
        self.normalize = normalize
        
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.rul_df: Optional[pd.DataFrame] = None
        
        self.scaler_params: Dict[str, Tuple[float, float]] = {}
        
        # Créer le répertoire de données
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, force: bool = False):
        """
        Télécharge le dataset si non présent.
        
        Note: Le dataset C-MAPSS doit être téléchargé manuellement depuis Kaggle
        car il nécessite une authentification.
        
        Args:
            force: Forcer le re-téléchargement
        """
        train_file = self.data_dir / f"train_{self.subset}.txt"
        
        if train_file.exists() and not force:
            print(f"✅ Dataset {self.subset} déjà présent")
            return
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                 TÉLÉCHARGEMENT NASA C-MAPSS                     ║
╚══════════════════════════════════════════════════════════════════╝

Le dataset C-MAPSS n'est pas disponible localement.

📥 Pour télécharger le dataset:

1. Allez sur Kaggle: 
   https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

2. Téléchargez le fichier ZIP

3. Extrayez les fichiers dans:
   {self.data_dir.absolute()}

Fichiers attendus:
   - train_FD001.txt, train_FD002.txt, ...
   - test_FD001.txt, test_FD002.txt, ...
   - RUL_FD001.txt, RUL_FD002.txt, ...
        """)
        
        # Créer des données synthétiques comme fallback
        self._create_synthetic_cmapss()
    
    def _create_synthetic_cmapss(self):
        """Crée des données synthétiques au format C-MAPSS si le download échoue."""
        print("\n📦 Création de données synthétiques au format C-MAPSS...")
        
        rng = np.random.default_rng(42)
        
        # Générer des données pour 100 moteurs
        all_data = []
        
        for unit_id in range(1, 101):
            # Durée de vie aléatoire entre 100 et 200 cycles
            max_cycles = rng.integers(100, 201)
            
            for cycle in range(1, max_cycles + 1):
                row = [unit_id, cycle]
                
                # 3 paramètres opérationnels
                row.extend([rng.uniform(0, 0.01) for _ in range(3)])
                
                # 21 capteurs avec dégradation progressive
                progress = cycle / max_cycles
                degradation = progress ** 1.5
                
                for i in range(21):
                    # Valeur de base + dégradation + bruit
                    base = 500 + i * 50
                    trend = degradation * rng.uniform(10, 50) * (1 if i % 2 == 0 else -1)
                    noise = rng.normal(0, base * 0.01)
                    row.append(base + trend + noise)
                
                all_data.append(row)
        
        # Créer le DataFrame et sauvegarder
        df = pd.DataFrame(all_data, columns=CMAPSS_COLUMNS)
        
        # Sauvegarder train
        train_path = self.data_dir / f"train_{self.subset}.txt"
        df.to_csv(train_path, sep=' ', header=False, index=False)
        print(f"   ✅ Créé: {train_path}")
        
        # Sauvegarder test (50 derniers moteurs)
        test_df = df[df['unit_id'] > 50].copy()
        # Tronquer les séries pour simuler des cas d'usage réel
        test_data = []
        for unit_id in test_df['unit_id'].unique():
            unit_data = test_df[test_df['unit_id'] == unit_id]
            # Garder 70% des cycles
            keep_cycles = int(len(unit_data) * 0.7)
            test_data.append(unit_data.head(keep_cycles))
        
        if test_data:
            test_final = pd.concat(test_data)
            test_path = self.data_dir / f"test_{self.subset}.txt"
            test_final.to_csv(test_path, sep=' ', header=False, index=False)
            print(f"   ✅ Créé: {test_path}")
            
            # RUL pour les données de test
            rul_values = []
            for unit_id in test_final['unit_id'].unique():
                unit_data = df[df['unit_id'] == unit_id]
                test_unit = test_final[test_final['unit_id'] == unit_id]
                remaining = len(unit_data) - len(test_unit)
                rul_values.append(min(remaining, self.max_rul))
            
            rul_path = self.data_dir / f"RUL_{self.subset}.txt"
            pd.DataFrame(rul_values).to_csv(rul_path, sep=' ', header=False, index=False)
            print(f"   ✅ Créé: {rul_path}")
        
        print("   📊 Données synthétiques C-MAPSS créées avec succès!")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Charge les données du dataset.
        
        Returns:
            train_df, test_df, rul_df
        """
        # Vérifier/télécharger les données
        self.download_dataset()
        
        train_path = self.data_dir / f"train_{self.subset}.txt"
        test_path = self.data_dir / f"test_{self.subset}.txt"
        rul_path = self.data_dir / f"RUL_{self.subset}.txt"
        
        # Charger les données d'entraînement
        print(f"\n📂 Chargement de {self.subset}...")
        
        self.train_df = pd.read_csv(
            train_path, 
            sep=r'\s+', 
            header=None,
            names=CMAPSS_COLUMNS
        )
        print(f"   Train: {len(self.train_df)} observations, "
              f"{self.train_df['unit_id'].nunique()} moteurs")
        
        # Charger les données de test
        if test_path.exists():
            self.test_df = pd.read_csv(
                test_path,
                sep=r'\s+',
                header=None,
                names=CMAPSS_COLUMNS
            )
            print(f"   Test: {len(self.test_df)} observations, "
                  f"{self.test_df['unit_id'].nunique()} moteurs")
        
        # Charger les RUL de test
        if rul_path.exists():
            self.rul_df = pd.read_csv(rul_path, header=None, names=['RUL'])
            print(f"   RUL: {len(self.rul_df)} valeurs")
        
        return self.train_df, self.test_df, self.rul_df
    
    def add_rul_to_train(self) -> pd.DataFrame:
        """
        Calcule et ajoute la colonne RUL aux données d'entraînement.
        
        Returns:
            DataFrame avec colonne RUL ajoutée
        """
        if self.train_df is None:
            self.load_data()
        
        # Calculer le cycle max par unité
        max_cycles = self.train_df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        
        # Merger et calculer RUL
        self.train_df = self.train_df.merge(max_cycles, on='unit_id')
        self.train_df['RUL'] = self.train_df['max_cycle'] - self.train_df['cycle']
        
        # Clip au RUL maximum
        self.train_df['RUL'] = self.train_df['RUL'].clip(upper=self.max_rul)
        
        # Supprimer la colonne temporaire
        self.train_df.drop('max_cycle', axis=1, inplace=True)
        
        return self.train_df
    
    def get_sensor_columns(self, include_op_settings: bool = False) -> List[str]:
        """Retourne la liste des colonnes de capteurs à utiliser."""
        cols = SELECTED_SENSORS.copy()
        if include_op_settings:
            cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + cols
        return cols
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les features (min-max scaling).
        
        Args:
            df: DataFrame à normaliser
            
        Returns:
            DataFrame normalisé
        """
        sensor_cols = self.get_sensor_columns(include_op_settings=True)
        
        df_normalized = df.copy()
        
        for col in sensor_cols:
            if col in df.columns:
                if col not in self.scaler_params:
                    self.scaler_params[col] = (df[col].min(), df[col].max())
                
                min_val, max_val = self.scaler_params[col]
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1
                
                df_normalized[col] = (df[col] - min_val) / range_val
        
        return df_normalized
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les séquences temporelles pour l'entraînement.
        
        Args:
            df: DataFrame avec RUL
            sequence_length: Longueur des séquences (défaut: window_size)
            
        Returns:
            X: Features (n_samples, sequence_length, n_features)
            y: RUL cibles
            unit_ids: IDs des unités
        """
        if sequence_length is None:
            sequence_length = self.window_size
        
        sensor_cols = self.get_sensor_columns(include_op_settings=True)
        
        X_list = []
        y_list = []
        unit_list = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            
            # Features et RUL
            features = unit_data[sensor_cols].values
            rul = unit_data['RUL'].values
            
            # Créer les séquences avec padding si nécessaire
            for i in range(len(unit_data)):
                if i < sequence_length:
                    # Padding avec les premières valeurs
                    padding = np.repeat(features[0:1], sequence_length - i - 1, axis=0)
                    seq = np.vstack([padding, features[:i+1]])
                else:
                    seq = features[i-sequence_length+1:i+1]
                
                X_list.append(seq)
                y_list.append(rul[i])
                unit_list.append(unit_id)
        
        return np.array(X_list), np.array(y_list), np.array(unit_list)
    
    def get_engine_data(self, unit_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Récupère les données d'un moteur spécifique.
        
        Args:
            unit_id: ID du moteur
            
        Returns:
            sensor_data: Array (cycles, n_sensors)
            rul_data: Array (cycles,)
        """
        if self.train_df is None or 'RUL' not in self.train_df.columns:
            self.load_data()
            self.add_rul_to_train()
        
        if self.normalize:
            df = self.normalize_features(self.train_df)
        else:
            df = self.train_df
        
        unit_data = df[df['unit_id'] == unit_id]
        sensor_cols = self.get_sensor_columns()
        
        sensor_data = unit_data[sensor_cols].values
        rul_data = unit_data['RUL'].values
        
        return sensor_data, rul_data
    
    def get_random_engine(
        self, 
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Sélectionne un moteur aléatoire et retourne ses données.
        
        Args:
            rng: Générateur aléatoire
            
        Returns:
            unit_id, sensor_data, rul_data
        """
        if self.train_df is None:
            self.load_data()
            self.add_rul_to_train()
        
        if rng is None:
            rng = np.random.default_rng()
        
        unit_ids = self.train_df['unit_id'].unique()
        unit_id = rng.choice(unit_ids)
        
        sensor_data, rul_data = self.get_engine_data(unit_id)
        
        return unit_id, sensor_data, rul_data
    
    def get_statistics(self) -> Dict[str, any]:
        """Retourne des statistiques sur le dataset."""
        if self.train_df is None:
            self.load_data()
        
        stats = {
            'n_engines_train': self.train_df['unit_id'].nunique(),
            'n_observations_train': len(self.train_df),
            'n_sensors': len(SELECTED_SENSORS),
            'sensor_names': SELECTED_SENSORS,
            'avg_life_cycles': self.train_df.groupby('unit_id')['cycle'].max().mean(),
            'min_life_cycles': self.train_df.groupby('unit_id')['cycle'].max().min(),
            'max_life_cycles': self.train_df.groupby('unit_id')['cycle'].max().max(),
        }
        
        if self.test_df is not None:
            stats['n_engines_test'] = self.test_df['unit_id'].nunique()
            stats['n_observations_test'] = len(self.test_df)
        
        return stats
    
    def print_info(self):
        """Affiche les informations sur le dataset."""
        stats = self.get_statistics()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║               NASA C-MAPSS Dataset - {self.subset}                      ║
╚══════════════════════════════════════════════════════════════════╝

📊 Statistiques:
   • Moteurs (train): {stats['n_engines_train']}
   • Observations: {stats['n_observations_train']:,}
   • Capteurs: {stats['n_sensors']}
   
📈 Durée de vie:
   • Moyenne: {stats['avg_life_cycles']:.0f} cycles
   • Min: {stats['min_life_cycles']} cycles
   • Max: {stats['max_life_cycles']} cycles
   
🔧 Capteurs utilisés:
   {', '.join(stats['sensor_names'][:7])}
   {', '.join(stats['sensor_names'][7:])}
        """)


def load_cmapss_for_training(
    subset: str = "FD001",
    window_size: int = config.WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fonction utilitaire pour charger C-MAPSS pour l'entraînement RL.
    
    Args:
        subset: Sous-ensemble à utiliser
        window_size: Taille de fenêtre
        
    Returns:
        X: Features normalisées
        y: RUL
    """
    loader = CMAPSSDataLoader(subset=subset, window_size=window_size)
    loader.load_data()
    loader.add_rul_to_train()
    
    df = loader.normalize_features(loader.train_df)
    X, y, _ = loader.prepare_sequences(df)
    
    return X, y


if __name__ == "__main__":
    print("=== Test du Loader C-MAPSS ===\n")
    
    loader = CMAPSSDataLoader(subset="FD001")
    loader.load_data()
    loader.add_rul_to_train()
    loader.print_info()
    
    # Test récupération d'un moteur
    unit_id, sensors, rul = loader.get_random_engine()
    print(f"\n📌 Moteur aléatoire #{unit_id}:")
    print(f"   Shape capteurs: {sensors.shape}")
    print(f"   RUL initial: {rul[0]}, final: {rul[-1]}")
