r"""Tutorial 05 -- Transactional State Update and Rollback.

Demonstrates
------------
RELIC's state update is transactional at two levels:
  - one apply_updates batch is validated and committed together
  - the simulator wraps all group calls of a step in one savepoint and
    restores the captured internal state if any part fails
This script demonstrates the batch level with a manual snapshot restore.
Declared bounds are enforced by clipping (with a recorded clip loss); the
Scenario B rollback rule below is a script-level rule, not a RELIC rule.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Cumulative update with bounds projection:

    x^{n+1} = P_{[L,U]}(x^n + Sum_i Delta x_i)         ... (1)

where P_{[L,U]} is the componentwise clamping projection.

Clip-loss diagnostic:
    Delta x_clip = (x^n + Sum_i Delta x_i) - x^{n+1}   ... (2)

Transactional semantics (Theorem 3.6.6, coertheory.md):
  - If all constraints pass: commit new state
  - If any constraint fails: restore pre-step snapshot

References
----------
1. coertheory.md, Theorems 3.6.3 and 3.6.6
2. spatiotemporal_scale_topology.md, Section 7.3 (Proposition 7.1)

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.state.manager.StateManager (checkpoint / restore / apply_updates)
- relic.core.state.types.StateVariableSpec (schema definition)
- relic.core.state.types.UpdateItem / UpdateMode (increment payloads)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    add_legend_outside,
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)

# -----------------------------------------------------------------------
# 0. Import real RELIC state management components
# -----------------------------------------------------------------------
from relic.core.state.manager import StateManager
from relic.core.state.types import (
    StateVariableSpec,
    UpdateItem,
    UpdateMode,
)


# -----------------------------------------------------------------------
# 1. Helper: build a StateManager with carbon pool schema
# -----------------------------------------------------------------------
def _create_carbon_state_manager(
    init_vals: dict,
    bounds: dict,
) -> StateManager:
    """Create a StateManager with carbon pool variables.

    Reference: coertheory.md Theorem 3.6.6
    Math: U_{dt} either commits all increments or rolls back entirely.

    Parameters
    ----------
    init_vals : dict
        Mapping variable_name -> initial value (float).
    bounds : dict
        Mapping variable_name -> (lower, upper) bounds tuple.
    """
    schema = {}
    for name, val in init_vals.items():
        lo, hi = bounds[name]
        schema[name] = StateVariableSpec(
            name=name,
            units="g C m-2",
            bounds=(lo, hi),
            conserved=True,
            description=f"Carbon pool: {name}",
        )

    # Disable contract autoload -- tutorial only needs explicit schema
    sm = StateManager(schema=schema, autoload_contract_registry=False)
    # Initialize state variables using torch tensors
    state_dict = {k: torch.tensor(float(v)) for k, v in init_vals.items()}
    sm.initialize_state(state_dict)
    return sm


def _get_state_array(sm: StateManager, names: list) -> np.ndarray:
    """Extract state variables as a numpy array in specified order."""
    state = sm.get_state(clone=True)
    return np.array([float(state[n]) for n in names])


# -----------------------------------------------------------------------
# 2. Scenario A: Successful commit using real StateManager
# -----------------------------------------------------------------------
names = ["Leaf_C", "Litter_C", "SOM_C"]
init_vals_a = {"Leaf_C": 200.0, "Litter_C": 150.0, "SOM_C": 300.0}
bounds_map = {"Leaf_C": (0.0, 1000.0), "Litter_C": (0.0, 1000.0), "SOM_C": (0.0, 1000.0)}

sm = _create_carbon_state_manager(init_vals_a, bounds_map)

n_steps = 50
history_pass = np.zeros((n_steps + 1, 3))
history_pass[0] = _get_state_array(sm, names)

for step in range(n_steps):
    # Reference: coertheory.md Theorem 3.6.6
    # Math: checkpoint() saves x^n; apply_updates applies Sum_i Delta x_i
    # then commit with bounds projection P_{[L,U]}

    # Create checkpoint (snapshot) before the step
    snapshot = sm.checkpoint()

    # Build increments: Litterfall (Leaf -> Litter) + Decomposition (Litter -> SOM)
    # Reference: coertheory.md Theorem 3.6.3
    # Math: x_proposed = x^n + Sum_i Delta x_i
    updates = {
        "Leaf_C": UpdateItem(
            value=torch.tensor(-4.0),
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Litterfall",
        ),
        "Litter_C": UpdateItem(
            value=torch.tensor(4.0 - 3.0),  # +4 from litterfall, -3 to decomp
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Litterfall+Decomposition",
        ),
        "SOM_C": UpdateItem(
            value=torch.tensor(3.0),
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Decomposition",
        ),
    }

    # Transactional apply: will clamp to bounds and check conservation
    sm.apply_updates(updates, dt=1.0, check_conservation=True, rollback_on_failure=False)
    history_pass[step + 1] = _get_state_array(sm, names)

total_c_pass = history_pass.sum(axis=1)
print(f"[OK] Scenario A (valid): total C drift = {np.abs(total_c_pass - total_c_pass[0]).max():.2e}")

# -----------------------------------------------------------------------
# 3. Scenario B: Rollback on constraint failure
# -----------------------------------------------------------------------
init_vals_b = {"Leaf_C": 50.0, "Litter_C": 20.0, "SOM_C": 300.0}
sm2 = _create_carbon_state_manager(init_vals_b, bounds_map)

n_steps_b = 30
history_fail = np.zeros((n_steps_b + 1, 3))
history_fail[0] = _get_state_array(sm2, names)
rollback_steps = []

for step in range(n_steps_b):
    # Save snapshot for potential rollback
    # Reference: coertheory.md Theorem 3.6.6 (transactional semantics)
    snapshot = sm2.checkpoint()

    # Aggressive decomposition that depletes Litter pool
    updates = {
        "Leaf_C": UpdateItem(
            value=torch.tensor(-3.0),
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Litterfall",
        ),
        "Litter_C": UpdateItem(
            value=torch.tensor(3.0 - 8.0),  # +3 litterfall, -8 decomp (drains fast)
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Litterfall+Decomposition",
        ),
        "SOM_C": UpdateItem(
            value=torch.tensor(8.0),
            mode=UpdateMode.ACCUMULATE,
            element="carbon",
            source_functor="Decomposition",
        ),
    }

    # Apply the updates
    sm2.apply_updates(updates, dt=1.0, check_conservation=True, rollback_on_failure=False)
    current = _get_state_array(sm2, names)

    # Script-level rule (not a RELIC runtime rule): if Litter_C reaches zero,
    # restore the pre-step snapshot manually. RELIC's apply_updates clips to
    # the declared bounds and records the clip loss; it rejects a batch only
    # for errors such as a conservation violation configured to raise.
    proposed_litter = current[1]
    if proposed_litter < 0.0 + 1e-10:
        # Constraint failure -> rollback to pre-step snapshot
        sm2.restore(snapshot)
        rollback_steps.append(step)
        current = _get_state_array(sm2, names)

    history_fail[step + 1] = current

print(f"[OK] Scenario B (rollback): {len(rollback_steps)} steps rolled back")
print(f"  State after rollback = {_get_state_array(sm2, names)}"
      f"  (unchanged from pre-failure snapshot)")

# -----------------------------------------------------------------------
# 4. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.32))
time_a = np.arange(n_steps + 1)
time_b = np.arange(n_steps_b + 1)

# --- Panel (a): Successful commit ---
ax = axes[0]
for i in range(3):
    ax.plot(time_a, history_pass[:, i], label=names[i], linewidth=1.2)
ax.plot(time_a, total_c_pass, "k--", linewidth=1.8, label="Total C")
ax.set_xlabel("Step")
ax.set_ylabel("Carbon [g C m$^{-2}$]")
ax.set_title("Successful Commits", fontsize=9)
ax.legend(fontsize=5, ncol=1, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "a")

# --- Panel (b): Rollback scenario ---
ax = axes[1]
for i in range(3):
    ax.plot(time_b, history_fail[:, i], label=names[i], linewidth=1.2)

# Mark rollback steps
for rs in rollback_steps:
    ax.axvline(rs, color="red", alpha=0.3, linewidth=0.8, linestyle="--")

if rollback_steps:
    ax.axvline(rollback_steps[0], color="red", alpha=0.3, linewidth=0.8,
               linestyle="--", label="Rollback")
ax.set_xlabel("Step")
ax.set_ylabel("Carbon [g C m$^{-2}$]")
ax.set_title("Script-Level Rollback Rule", fontsize=9)
ax.legend(fontsize=5, ncol=1, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "b")

# --- Panel (c): Comparison: proposed vs actual after rollback ---
ax = axes[2]

# Show what would happen without rollback vs with rollback
# at step where first rollback occurred
if rollback_steps:
    step = rollback_steps[0]
    pre = history_fail[step]
    post = history_fail[step + 1]
    proposed = pre + np.array([-3.0, 3.0 - 8.0, 8.0])

    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, pre, width, color=CB_PALETTE[5], label="Pre-step")
    ax.bar(x, proposed, width, color=CB_PALETTE[1], label="Proposed (rejected)")
    ax.bar(x + width, post, width, color=CB_PALETTE[2], label="Actual (rolled back)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=5, ncol=1, loc="upper right", frameon=False,
              prop={"weight": "bold"}, borderaxespad=0.3)

ax.set_ylabel("Carbon [g C m$^{-2}$]")
ax.set_title("Proposed vs Actual State", fontsize=9)
add_panel_label(ax, "c")

fig.suptitle("Transactional State Update: All-or-Nothing Commit with Rollback",
             fontsize=10, y=1.03)
plt.tight_layout()
savefig(fig, "05_transactional_state_update")
plt.close(fig)

print("\n[OK] Tutorial 05 completed successfully.")
print("  Demonstrated: RELIC StateManager checkpoint/restore/apply_updates")
print("  Used: relic.core.state.manager.StateManager (real transactional API)")
