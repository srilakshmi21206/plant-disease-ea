# genetic_algorithm.py
import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── GA SETTINGS ────────────────────────────────────────────
N_FEATURES = 112        # total features from feature_extractor
POP_SIZE = 30           # population size
N_GENERATIONS = 20      # number of generations
CXPB = 0.7             # crossover probability
MUTPB = 0.2            # mutation probability

def evaluate(individual, X, y):
    """
    Fitness function:
    - Select features where gene = 1
    - Train a quick classifier
    - Return cross-val accuracy
    """
    selected = [i for i, bit in enumerate(individual) if bit == 1]

    # If no features selected, return worst fitness
    if len(selected) == 0:
        return (0.0,)

    X_selected = X[:, selected]

    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
    return (scores.mean(),)


def run_genetic_algorithm(X, y):
    """
    Run GA to find the best subset of features.
    Returns: list of selected feature indices
    """

    # ── Setup DEAP ──────────────────────────────────────────
    # Avoid re-registering if already exists
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Each gene is 0 or 1 (feature selected or not)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, N_FEATURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ── Run GA ──────────────────────────────────────────────
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)  # keep best individual

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("\n🧬 Starting Genetic Algorithm...")
    print(f"   Population: {POP_SIZE} | Generations: {N_GENERATIONS}")
    print(f"   Total features: {N_FEATURES}\n")

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=CXPB, mutpb=MUTPB,
        ngen=N_GENERATIONS,
        stats=stats, halloffame=hof,
        verbose=True
    )

    # ── Extract best features ────────────────────────────────
    best_individual = hof[0]
    selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]

    print(f"\n✅ GA Complete!")
    print(f"   Selected {len(selected_features)} out of {N_FEATURES} features")
    print(f"   Best fitness (accuracy): {best_individual.fitness.values[0]:.4f}")

    return selected_features, log