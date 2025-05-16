import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Optional in notebooks, can omit if causing issues


def plot(scores, mean_scores):
    plt.clf()  # Clear current figure
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()

    display.clear_output(wait=True)
    plt.pause(0.001)  # Tiny pause to allow UI to update
