import pandas as pd

def generate_spread_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """
    Generate spread position from z-scores.

    Position convention:
     1 = long spread
    -1 = short spread
     0 = flat
    """

    positions = []
    current_position = 0

    for z in zscore:
        if current_position == 0:
            if z < -entry_threshold:
                current_position = 1
            elif z > entry_threshold:
                current_position = -1

        else:
            if abs(z) < exit_threshold:
                current_position = 0

        positions.append(current_position)

    positions = pd.Series(
        positions,
        index=zscore.index,
        name=f"{zscore.name}_position",
    )

    return positions