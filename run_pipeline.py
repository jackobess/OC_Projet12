"""run_pipeline.py : preprocess -> train -> evaluate, en une commande. Code retour 1 dès qu'une étape échoue.

Usage : python src/data_preprocessing/run_pipeline.py [config.yaml] [--steps preprocess train evaluate]
"""
import argparse
import logging
import sys
import time

import evaluate
import preprocess
import train
from common import setup_logging

log = logging.getLogger("run_pipeline")
STEPS = {"preprocess": preprocess.run, "train": train.run, "evaluate": evaluate.run}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    a = ap.parse_args()
    setup_logging()
    for step in a.steps:
        t0 = time.time()
        log.info("=== étape : %s ===", step)
        try:
            STEPS[step](a.config)
        except Exception:                      # on logge la trace complète puis on stoppe : pas d'étape suivante sur un état douteux
            log.exception("ÉCHEC de l'étape '%s'", step)
            sys.exit(1)
        log.info("étape '%s' terminée en %.0f s", step, time.time() - t0)
    log.info("pipeline terminé")


if __name__ == "__main__":
    main()
