import matplotlib.pyplot as plt

WIN_RATES = [0.700000, 0.850000, 0.950000, 0.850000, 1.000000]

if __name__ == "__main__":
    steps = list(range(1, len(WIN_RATES) + 1))
    plt.plot(steps, WIN_RATES, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Win rate vs baseline")
    plt.title("ResNet vs baseline win rate")
    plt.grid(True, alpha=0.3)
    plt.savefig("resnet_baseline_winrate.png")
    print("saved resnet_baseline_winrate.png")