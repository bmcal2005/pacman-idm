"""
Action labels for Atari-HEAD Ms. Pac-Man.

Atari-HEAD stores actions as FULL ALE ids (see data/raw/action_enums.txt):
0 NOOP, 1 FIRE, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN, 6-9 diagonals, 10-17 FIRE combos.

Every mapping below is derived from action NAMES, never hard-coded numbers.
That is what makes the off-by-one bug in the original repo impossible here.
"""

ALE_ACTIONS = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
    'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
    'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE',
]

H_NAMES = ['LEFT', 'NONE', 'RIGHT']   # horizontal axis, index 0/1/2
V_NAMES = ['UP', 'NONE', 'DOWN']      # vertical axis, index 0/1/2


def to_hv(raw):
    """Raw ALE id -> (h, v). FIRE is ignored: it does nothing in Ms. Pac-Man."""
    name = ALE_ACTIONS[raw].replace('FIRE', '')
    h = 0 if 'LEFT' in name else 2 if 'RIGHT' in name else 1
    v = 0 if 'UP' in name else 2 if 'DOWN' in name else 1
    return h, v


def to_9(raw):
    """Raw ALE id -> one of 9 classes, laid out as a 3x3 grid: index = 3*v + h."""
    h, v = to_hv(raw)
    return 3 * v + h


def _class_name(h, v):
    name = (V_NAMES[v] if v != 1 else '') + (H_NAMES[h] if h != 1 else '')
    return name or 'NOOP'


# index -> name: UPLEFT, UP, UPRIGHT, LEFT, NOOP, RIGHT, DOWNLEFT, DOWN, DOWNRIGHT
CLASS_NAMES = [_class_name(h, v) for v in range(3) for h in range(3)]


if __name__ == '__main__':
    for raw, name in enumerate(ALE_ACTIONS):
        h, v = to_hv(raw)
        print(f'{raw:2d} {name:14s} -> h={H_NAMES[h]:5s} v={V_NAMES[v]:4s} class={CLASS_NAMES[to_9(raw)]}')
