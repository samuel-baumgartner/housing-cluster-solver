# Housing Cluster Solver (Python prototype)

Python reference implementation of the **bottom-up cluster housing packer** from
`my-colony-sim` (`housing_cluster_solver.gd`). Use this to experiment, visualize,
and later translate back to GDScript.

## Setup

```bash
cd ~/projects/housing-cluster-solver
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step-by-step animation

```bash
python animate.py                  # deep 28×16 zone (default)
python animate.py --zone-size 20x12   # custom width×height
python animate.py --zone-size 32x20 --zone-min 6,10
python animate.py --zone-max 40,28    # explicit bounds
python animate.py --zone pocket    # interior pocket scenario
python animate.py --zone shallow   # shallow frontage strip
python animate.py --save-gif out.gif
```

**Controls:** `←/→` or `Space` step forward · `Home`/`End` jump · **Prev/Next/Auto** buttons

### What each step shows

| Step kind | Meaning |
|-----------|---------|
| `init` | Green zone painted |
| `landscape` | Large natural circles carved from the zone (before any paths or homes) |
| `evaluate` | Frontier (white dots), top scored candidates, highlighted best footprint |
| `place` | Home committed + door access path (numbered orange tiles) |
| `reject` | Candidate failed strand/path check |
| `street` | Interior street row or boundary spine opened |
| `done` | No more valid placements |

## Headless run

```bash
python run_demo.py
```

## Algorithm (mirrors GDScript)

1. **Seed** homes adjacent to TH path network
2. **Frontier** — BFS from cluster edge; full zone when fill is still sparse
3. **Score** each candidate: `4×touch + door_proximity − 3×exposed_green` + line mix
4. **Strand reject** — block placements that disconnect a ≥16-cell green pocket from paths
5. **Commit** — reserve footprint + shortest green path to network (ties hug buildings)
6. **Interior streets** — open row north of built frontage, or boundary spine, when stuck
7. Repeat until saturated

## Project layout

```
housing/
  types.py       # House, Candidate, SolveStep
  footprints.py  # Small / Big / L cell offsets + rotations
  grid.py        # Zone, paths, routing BFS
  scoring.py     # Frontier, score, strand check
  solver.py      # Main solve loop + step recorder
animate.py       # Matplotlib step player
run_demo.py      # CLI summary
```

## Mapping to Godot

| Python | GDScript |
|--------|----------|
| `housing/solver.py` → `solve()` | `housing_cluster_solver.gd` |
| `housing/scoring.py` | scoring + frontier helpers in cluster solver |
| `housing/grid.py` | `housing_district_placer.gd` path helpers |
| `housing/footprints.py` | `building_dimension_log.gd` + `housing_footprint_rules.gd` |
| `SolveStep` events | hook into `plan_zone` for in-game debug overlay |
