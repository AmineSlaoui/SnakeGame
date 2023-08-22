import matplotlib.pyplot as plt
from IPython import display

plt.ion() # Interactive mode is on

def plot(score, mean_score):
    # print(score)
    # print(mean_score)
    # print("Plotting")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf() # Prevents overlapping of values
    plt.ylabel("Score")
    plt.xlabel("Number of games")
    plt.plot(score)
    plt.plot(mean_score)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.show(block=False)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean_score)-1, mean_score[-1], str(mean_score[-1]))
    plt.pause(0.1)
