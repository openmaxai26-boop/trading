"""
UTILITAIRE DE JOURNALISATION
=============================
Ce module configure un système de logs professionnel.

POURQUOI LES LOGS ?
- Tracer chaque action du système en temps réel
- Diagnostiquer les erreurs facilement
- Garder un historique des décisions de trading
- Obligatoire dans un système de production

EXEMPLE DE LOG :
2024-01-15 10:30:45 | INFO     | DataCollector | AAPL : 252 jours téléchargés ✓
2024-01-15 10:30:46 | WARNING  | RiskManager   | Drawdown -12% → réduction des positions
2024-01-15 10:30:47 | ERROR    | Prediction    | Données manquantes pour BTC-USD
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    """
    Crée un logger configuré pour un module donné.

    PARAMÈTRES :
    - name : Nom du module (ex: "DataCollector", "RiskManager")
    - log_dir : Dossier où stocker les fichiers de logs

    RETOURNE :
    - Un objet Logger prêt à l'emploi

    UTILISATION :
        logger = get_logger("MonModule")
        logger.info("Tout va bien")
        logger.warning("Attention !")
        logger.error("Erreur !")
    """
    # Créer le dossier de logs s'il n'existe pas
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Créer le logger avec le nom donné
    logger = logging.getLogger(name)

    # Éviter les doublons si le logger existe déjà
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format des messages : date | niveau | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1 : Affichage dans le terminal (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2 : Écriture dans un fichier (une ligne par jour)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(log_dir) / f"trading_{today}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
