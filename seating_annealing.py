"""
seating_annealing.py
====================
Simulated annealing optimizer for seating arrangements.

Problem
-------
Given a rectangular grid of seats (some of which may be blocked), place a
group of people so that everyone is as far apart from their nearest neighbour
as possible.  The quality of an arrangement is measured by the *sum* of each
person's minimum Manhattan distance to any other person – the larger this
value, the more "spread-out" (and therefore preferable) the arrangement is.

Algorithm overview
------------------
1. Build the grid and remove blocked seats.
2. Seed an initial random arrangement: assign each person a distinct available
   seat chosen at random.
3. Run simulated annealing:
   a. Generate a *neighbour* by swapping the seats of two randomly chosen
      people.
   b. Accept the neighbour unconditionally if it improves the score.
   c. Accept it with probability exp(Δ / T) when it does not improve the
      score (Metropolis criterion) – this lets the algorithm escape local
      optima.
   d. Cool the temperature T by a constant factor each iteration.
4. Return (and display) the best arrangement found.

State representation
--------------------
An arrangement is a plain Python dict  { person_name: (row, col), ... }.
Example: {"Alice": (0, 0), "Bob": (2, 3)}

Each seat coordinate appears at most once across all values (hard constraint
maintained by the swap move).

Distance metric
---------------
Manhattan distance between two grid positions (r1, c1) and (r2, c2):
    d = |r1 - r2| + |c1 - c2|

Scoring
-------
score(arrangement) = Σ  min_{j ≠ i}  manhattan_distance(pos_i, pos_j)
                     i

A higher score means people are more spread apart from their nearest
neighbour, which is the desired outcome.
"""

import math
import random


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Return the Manhattan distance between two (row, col) coordinates."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_score(arrangement: dict) -> int:
    """
    Score an arrangement by summing each person's distance to their nearest
    neighbour.  With only one person the score is 0 (no neighbours to compare).

    A higher score indicates a more spread-out arrangement.
    """
    positions = list(arrangement.values())
    n = len(positions)

    if n <= 1:
        return 0

    total = 0
    for i in range(n):
        # Find the closest other person
        min_dist = min(
            manhattan_distance(positions[i], positions[j])
            for j in range(n)
            if j != i
        )
        total += min_dist

    return total


# ---------------------------------------------------------------------------
# Neighbour generation  (swap-based)
# ---------------------------------------------------------------------------

def generate_neighbour(arrangement: dict) -> dict:
    """
    Create a neighbouring arrangement by swapping the seats of two randomly
    chosen people.  The swap preserves the one-person-per-seat constraint.

    Returns a new dict; the original is not modified.
    """
    people = list(arrangement.keys())

    if len(people) < 2:
        # Nothing to swap – return a copy unchanged
        return dict(arrangement)

    # Pick two distinct people at random
    person_a, person_b = random.sample(people, 2)

    # Build the new arrangement with the seats swapped
    neighbour = dict(arrangement)
    neighbour[person_a], neighbour[person_b] = (
        neighbour[person_b],
        neighbour[person_a],
    )
    return neighbour


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    initial_arrangement: dict,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
    max_iterations: int = 10_000,
) -> tuple:
    """
    Optimise a seating arrangement using simulated annealing.

    Parameters
    ----------
    initial_arrangement : dict
        Starting state  { person: (row, col), ... }.
    initial_temp : float
        Starting temperature – controls how eagerly bad moves are accepted
        at the beginning of the search.
    cooling_rate : float
        Multiplicative factor applied to the temperature after each iteration
        (should be < 1, e.g. 0.995).
    min_temp : float
        Stop early when the temperature falls below this threshold.
    max_iterations : int
        Hard upper limit on the number of iterations.

    Returns
    -------
    best_arrangement : dict
        The arrangement with the highest score encountered.
    best_score : int
        The score of that arrangement.
    """
    current = dict(initial_arrangement)
    current_score = calculate_score(current)

    # Track the globally best solution seen so far
    best = dict(current)
    best_score = current_score

    temperature = initial_temp

    for iteration in range(max_iterations):
        # Stop when the system is effectively frozen
        if temperature < min_temp:
            break

        # --- Generate a candidate neighbour via a seat swap ---
        neighbour = generate_neighbour(current)
        neighbour_score = calculate_score(neighbour)

        # --- Accept / reject (Metropolis criterion) ---
        delta = neighbour_score - current_score

        if delta > 0:
            # Always accept improvements
            current = neighbour
            current_score = neighbour_score
        else:
            # Accept a worse solution with a temperature-dependent probability
            # so the search can escape local optima
            acceptance_probability = math.exp(delta / temperature)
            if random.random() < acceptance_probability:
                current = neighbour
                current_score = neighbour_score

        # --- Update the global best ---
        if current_score > best_score:
            best = dict(current)
            best_score = current_score

        # --- Cool down ---
        temperature *= cooling_rate

    return best, best_score


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def build_available_seats(rows: int, cols: int, blocked: set) -> list:
    """
    Return a list of all (row, col) positions in a rows × cols grid
    that are not in *blocked*.
    """
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked
    ]


def display_arrangement(
    arrangement: dict,
    rows: int,
    cols: int,
    blocked: set,
) -> None:
    """
    Print the seating grid to stdout.

    Legend
    ------
    X   – blocked seat
    P   – first character of a person's name
    ·   – empty available seat
    """
    # Reverse map: coordinate → person name
    seat_to_person = {pos: name for name, pos in arrangement.items()}

    # Top border
    separator = "+" + "---+" * cols
    print(separator)

    for r in range(rows):
        row_str = "|"
        for c in range(cols):
            pos = (r, c)
            if pos in blocked:
                row_str += " X |"
            elif pos in seat_to_person:
                # Show up to one character so that the cell width stays fixed
                label = seat_to_person[pos][0]
                row_str += f" {label} |"
            else:
                row_str += " · |"
        print(row_str)
        print(separator)


# ---------------------------------------------------------------------------
# Interactive entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 50)
    print("  Seating Arrangement Optimizer")
    print("  (Simulated Annealing)")
    print("=" * 50)
    print()

    # --- Grid dimensions ---
    rows = int(input("Number of rows    : "))
    cols = int(input("Number of columns : "))

    total_seats = rows * cols
    print(f"\nGrid has {total_seats} seat(s) total.")

    # --- Blocked seats ---
    blocked: set = set()
    print(
        "\nEnter seats to BLOCK (row and column are 0-indexed)."
        "\nPress Enter with no input when done."
    )
    while True:
        raw = input("  Block seat (row,col) or Enter to finish: ").strip()
        if not raw:
            break
        try:
            r_str, c_str = raw.split(",")
            r, c = int(r_str.strip()), int(c_str.strip())
            if 0 <= r < rows and 0 <= c < cols:
                blocked.add((r, c))
                print(f"  → Seat ({r},{c}) blocked.")
            else:
                print(f"  ✗ ({r},{c}) is outside the grid. Try again.")
        except ValueError:
            print("  ✗ Invalid format – use row,col  e.g. 1,2")

    available = build_available_seats(rows, cols, blocked)
    print(f"\n{len(available)} seat(s) available after blocking.")

    if not available:
        print("No available seats – exiting.")
        return

    # --- Number of people ---
    max_people = len(available)
    num_people = int(
        input(f"Number of people to seat (1 – {max_people}): ")
    )
    if not 1 <= num_people <= max_people:
        print(f"Must be between 1 and {max_people}. Exiting.")
        return

    # --- Collect names ---
    names: list = []
    print("\nEnter a name for each person (press Enter to use a default).")
    for i in range(num_people):
        name = input(f"  Person {i + 1} name: ").strip()
        if not name:
            name = f"P{i + 1}"
        names.append(name)

    # --- Initial random arrangement ---
    random.seed()  # Initialize RNG from system entropy for non-deterministic runs
    chosen_seats = random.sample(available, num_people)
    initial_arrangement = dict(zip(names, chosen_seats))

    print("\n--- Initial arrangement (random) ---")
    display_arrangement(initial_arrangement, rows, cols, blocked)
    print(f"Score : {calculate_score(initial_arrangement)}")

    # --- Simulated annealing ---
    print("\nRunning simulated annealing …")
    best_arrangement, best_score = simulated_annealing(initial_arrangement)

    print("\n--- Optimised arrangement ---")
    display_arrangement(best_arrangement, rows, cols, blocked)
    print(f"Score : {best_score}")

    # --- Show per-person details ---
    print("\nFinal seat assignments:")
    seat_to_person = {pos: name for name, pos in best_arrangement.items()}
    positions = list(best_arrangement.values())
    for name, pos in sorted(best_arrangement.items()):
        others = [p for p in positions if p != pos]
        if others:
            nearest_dist = min(manhattan_distance(pos, p) for p in others)
        else:
            nearest_dist = 0
        print(
            f"  {name:20s}  seat ({pos[0]},{pos[1]})  "
            f"nearest-neighbour distance = {nearest_dist}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
