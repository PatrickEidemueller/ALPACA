from matplotlib import pyplot as plt
from dataclasses import dataclass


@dataclass
class PerformanceHistory:
    performances: list[float]
    label: str


def plot_RMSE_over_iterations(
    performance_histories: list[PerformanceHistory], filename: str
) -> None:
    # plot RMSE over n_queries
    scale = min(1.0, len(performance_histories) / 2)
    fig, ax = plt.subplots(figsize=(8.5 * scale, 6 * scale), dpi=300)
    for hist in performance_histories:
        ax.plot(hist.performances, label=hist.label)
        ax.scatter(range(len(hist.performances)), hist.performances)
    ax.grid(True)
    plt.legend(loc="upper left", frameon=True)
    ax.set_title("Incremental RMSE")
    ax.set_xlabel("Query iteration")
    ax.set_ylabel("RMSE")
    #    plt.show()
    plt.savefig(filename)
