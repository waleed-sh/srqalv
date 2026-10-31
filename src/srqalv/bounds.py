import math


def reduced_sigma(signs: dict[tuple[int, int, int], int]) -> int:
    return (
        int(signs[(0, 1, 2)])
        - int(signs[(0, 1, 3)])
        + int(signs[(0, 2, 3)])
        - int(signs[(1, 2, 3)])
    )


def analytic_lambda_for_sector(two_js: tuple[int, ...], sigma: int) -> float:
    if len(two_js) != 4:
        raise ValueError("This experiment is restricted to 4-valent vertices.")
    if sigma == 0:
        return 0.0
    js = tuple(two_j / 2.0 for two_j in two_js[:3])
    return 6.0 * abs(float(sigma)) * math.prod(math.sqrt(j * (j + 1.0)) for j in js)

