import csv
import json
from pathlib import Path
from .formatting import make_json_serializable

class OptimizerLogger:
    """
    A logging utility class for recording optimization progress across generations.
    
    Supports both:
    - CSV output for lightweight, generation-level summary (best/worst/avg fitness).
    - JSON output for full population history (phenotype and fitness only).

    Designed to be independent of encoding mechanisms (e.g., genotype/phenotype),
    making it suitable for GA, ES, DE, PSO, etc.

    Attributes:
        log_dir (Path): Directory where log files will be saved.
        run_id (str): Identifier for the current optimization run (used as filename prefix).
        summary (List[Dict]): List of generation-level fitness summary records.
        population_history (List[Dict]): List of full population snapshots per generation.
    """
    def __init__(self, log_dir="logs", run_id=None):
        """
        Initializes the logger with a given directory and optional run ID.

        Args:
            log_dir (str or Path): Directory path to save logs.
            run_id (str): Optional identifier to differentiate multiple runs.
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id or "run"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary = []
        self.population_history = []

    def log_generation_summary(self, generation, best, worst, avg):
        """
        Logs a summary of the generation-level fitness metrics.

        Args:
            generation (int): Generation index.
            best (float): Best fitness value in current generation.
            worst (float): Worst fitness value in current generation.
            avg (float): Average fitness value in current generation.
        """
        self.summary.append({
            "generation": generation,
            "best_fitness": best,
            "worst_fitness": worst,
            "avg_fitness": avg,
        })

    def log_population(self, generation, individuals, fitnesses):
        """
        Logs the full population state.

        Args:
            generation (int): Generation index.
            individuals (List[List[float]] or np.ndarray): Phenotypes of individuals.
            fitnesses (List[float] or np.ndarray): Corresponding fitness values.
        """
        self.population_history.append({
            "generation": generation,
            "individuals": [list(map(float, ind)) for ind in individuals],
            "fitnesses": list(map(float, fitnesses))
        })

    def save(self):
        """
        Saves all logged data to disk:
        - summary as a CSV file
        - population history as a JSON file
        """
        with open(self.log_dir / f"{self.run_id}_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary[0].keys())
            writer.writeheader()
            writer.writerows(self.summary)

        # Save population
        with open(self.log_dir / f"{self.run_id}_population.json", "w") as f:
            json.dump(make_json_serializable(self.population_history), f, indent=2)
