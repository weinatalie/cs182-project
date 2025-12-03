import csv
import random

def generate_linear_regression_icl_csv(
    path="linreg_icl_100mb.csv",
    n_tasks=100000,
    min_context=5,
    max_context=8,
    slope_range=(-5.0, 5.0),
    intercept_range=(-5.0, 5.0),
    noise_std_range=(0.1, 1.0),
    x_range=(-10.0, 10.0),
    seed=0,
):
    """
    Generate a CSV for ICL-style linear regression tasks.

    Each row is one task:
      - Sample (m, b, noise_std) for the task
      - Sample L in [min_context, max_context]
      - Sample L context pairs (x_i, y_i)
      - Sample a query x_query, y_query

    Columns:
      task_id, slope, intercept, noise_std, L,
      x1..x_max_context, y1..y_max_context, x_query, y_query
    """
    assert 1 <= min_context <= max_context, "min_context must be <= max_context"
    random.seed(seed)

    # Build header
    header = [
        "task_id",
        "slope",
        "intercept",
        "noise_std",
        "L",
    ]

    # x1..x_max_context, y1..y_max_context
    header += [f"x{i+1}" for i in range(max_context)]
    header += [f"y{i+1}" for i in range(max_context)]

    # query
    header += ["x_query", "y_query"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for task_id in range(n_tasks):
            # Sample task parameters
            m = random.uniform(*slope_range)
            b = random.uniform(*intercept_range)
            noise_std = random.uniform(*noise_std_range)

            # Context length for this task, in [min_context, max_context]
            L = random.randint(min_context, max_context)

            # Sample context x's
            xs = [random.uniform(*x_range) for _ in range(L)]
            # Corresponding y's with Gaussian noise
            ys = [
                m * x + b + random.gauss(0.0, noise_std)
                for x in xs
            ]

            # Pad context to max_context with empty strings
            xs_padded = xs + [""] * (max_context - L)
            ys_padded = ys + [""] * (max_context - L)

            # Query point
            xq = random.uniform(*x_range)
            yq = m * xq + b + random.gauss(0.0, noise_std)

            row = [task_id, m, b, noise_std, L] \
                  + xs_padded + ys_padded \
                  + [xq, yq]

            writer.writerow(row)

    print(f"Wrote {n_tasks} tasks to {path}")

if __name__ == "__main__":
    generate_linear_regression_icl_csv()
