import csv
import numpy as np
import json
from pathlib import Path
from deap import tools
import pandas as pd
from .formatting import make_json_serializable

class OptimizerLogger:
    """
    A logging utility for recording optimization progress.

    Logs:
    - Generation summary (best/worst/avg fitness) to CSV
    - Population snapshot (phenotype + fitness) to JSON

    Supports multiple optimizers (GA, ES, DE, etc.).

    Attributes:
        enable_summary (bool): Enable logging generation summaries.
        enable_population (bool): Enable logging full population data.
    """

    def __init__(self, log_dir="logs", run_id=None, enable_summary=True, enable_population=True):
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

    def log_generation_summary(self, generation, best, worst, avg):
        """
        Logs generation-level summary if enabled.

        Args:
            generation (int): Generation index.
            best (float): Best fitness.
            worst (float): Worst fitness.
            avg (float): Average fitness.
        """
        if not self.enable_summary:
            return
        self.summary.append({
            "generation": generation,
            "best_fitness": best,
            "worst_fitness": worst,
            "avg_fitness": avg,
        })

    def log_population(self, generation, individuals, fitnesses):
        """
        Logs population phenotypes and fitnesses if enabled.

        Args:
            generation (int): Generation index.
            individuals (List[List[float]]): Phenotype representation.
            fitnesses (List[float]): Fitness values.
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
        Saves summary (CSV) and population history (JSON) if available.
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