def assert_fairness(value: float, threshold: float):
    assert value <= threshold, f"Fairness threshold exceeded: {value} > {threshold}"