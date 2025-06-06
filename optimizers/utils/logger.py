"""
utils/logger.py

Logging utility for evolutionary optimization processes.

This module provides the OptimizerLogger class, which enables detailed tracking and recording of optimization runs across various algorithms (GA, ES, DE, etc.).
The logger can output both:
    - Per-generation statistical summaries (e.g., best, worst, average fitness)
    - Full population snapshots with individual phenotypes and fitnesses (for the purpose of visualization of 1D/2D problem)

Logs can be saved in CSV and JSON formats, suitable for further analysis or visualization.

Dependencies:
    - numpy
    - pandas
    - DEAP
    - pathlib
    - json

Classes:
    OptimizerLogger: Main logger for tracking evolutionary optimizer progress.
"""

import numpy as np
import json
from pathlib import Path
from deap import tools
import pandas as pd
from .formatting import make_json_serializable

class OptimizerLogger:
    """
    A logging utility for recording optimization progress.

    Supports logging both generation summaries and full population data.

    Logs:
        - Generation statistics (best/worst/avg/std fitness) as CSV
        - Population snapshots (phenotypes + fitnesses) as JSON

    Attributes:
        enable_summary (bool): Enable logging generation statistics.
        enable_population (bool): Enable logging detailed population data.
        log_dir (Path): Directory where logs are saved.
        run_id (str): Identifier for current run, used in filenames.
        stats (deap.tools.Statistics): DEAP statistics object for tracking fitness metrics.
        logbook (deap.tools.Logbook): DEAP logbook for storing summary records.
        population_history (list): Stored list of population data per generation.
    """

    def __init__(self, log_dir="logs", run_id=None, enable_summary=True, enable_population=True):
        """
        Initializes the OptimizerLogger.

        Args:
            log_dir (str): Path to the directory where logs will be saved.
            run_id (str, optional): Unique ID for the run (used as filename prefix).
            enable_summary (bool): If True, logs generation-level fitness stats.
            enable_population (bool): If True, logs full population data each generation.
        """
        self.enable_summary = enable_summary
        self.enable_population = enable_population
        self.log_dir = Path(log_dir)
        self.run_id = run_id or "run"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup statistics tracker
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # Setup logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.stats.fields

        # Reserve empty list for storing the data for plotting
        self.population_history = []

    def log_population(self, generation, individuals, fitnesses):
        """
        Logs population snapshot for the current generation.

        Args:
            generation (int): Generation number.
            individuals (List[List[float]]): Phenotypic representation of individuals.
            fitnesses (List[float]): Corresponding fitness values.
        """
        if not self.enable_population:
            return
        self.population_history.append({
            "generation": generation,
            "individuals": [list(map(float, ind)) for ind in individuals],
            "fitnesses": list(map(float, fitnesses))
        })

    def save(self):
        """
        Saves logs to disk.

        Outputs:
            - summary CSV file: <log_dir>/<run_id>_summary.csv
            - population JSON file: <log_dir>/<run_id>_population.json
        """
        if self.enable_summary:
            # Convert logbook to a DataFrame
            df = pd.DataFrame(self.logbook)
            # Write to CSV
            df.to_csv(self.log_dir / f"{self.run_id}_summary.csv", index=False)

        if self.enable_population and self.population_history:
            pop_file = self.log_dir / f"{self.run_id}_population.json"
            with open(pop_file, "w") as f:
                json.dump(make_json_serializable(self.population_history), f, indent=2)