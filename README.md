# SilentSeetSeak
Simple program for fun that finds best seat(s) in a library.

## Where is the code?

The main program is **[`seating_annealing.py`](seating_annealing.py)**.

It implements a **simulated annealing** optimizer that finds the seating
arrangement in which everyone is as far apart from their nearest neighbour
as possible.

## How to run

```bash
python seating_annealing.py
```

The program will interactively prompt you for:

1. **Grid size** – number of rows and columns.
2. **Blocked seats** – any seats that are unavailable (0-indexed `row,col`).
3. **Number of people** to seat.
4. **Names** for each person.

It then prints the initial random layout, runs the optimizer, and shows the
optimized seating arrangement together with each person's nearest-neighbour
distance.

## Requirements

Python 3 standard library only (`math`, `random`) — no extra packages needed.
