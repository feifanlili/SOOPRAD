# SOOPRAD
OpenRadioss + SciPy/DEAP for Structural Optimization

# Optimizer Module

A modular Python framework for evolutionary and heuristic optimization algorithms, including Genetic Algorithms (GA), Evolution Strategies (ES), and SciPy-based optimizers. The module supports easy customization, logging, and visualization of optimization processes.

---

## Features

- Implementations of various optimizers:
  - Genetic Algorithm (GA) with DEAP
  - Evolution Strategy (ES) with DEAP
  - SciPy optimizers
- Suite of benchmark objective functions (sphere, ackley, levy, etc.)
- Built-in logging system to track population and generation statistics
- Visualization tools for 1D and 2D optimization runs, including animated replays
- Configurable parameters with sensible defaults for rapid experimentation

---

## Installation

This module depends on several external Python packages. To install all dependencies, run:

```bash
pip install -r requirements.txt