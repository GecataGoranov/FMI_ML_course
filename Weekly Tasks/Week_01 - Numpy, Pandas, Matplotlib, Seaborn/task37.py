import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(123)
    all_walks = []

    made_more_than_60 = 0
    made_less_than_60 = 0

    for _ in range(500):
        walk = do_walk()

        if walk[-1] <= 60:
            made_less_than_60 += 1
        
        else:
            made_more_than_60 += 1

        all_walks.append(walk)


    all_walks_np = np.array(all_walks)

    plt.hist(all_walks_np[:, -1])

    plt.title("Random walks", fontsize = 16)
    plt.xlabel("End step")

    plt.show()

    # print("Odds:", made_more_than_60, "/", made_less_than_60)
    # Using this line, I have calculated, that the odds of making it to more then 60 steps are 293 / 207


def do_walk():
    steps = [0]
    step = 0

    for _ in range(100):
        slip = np.random.rand()

        if slip <= 0.005:
            step = 0
            steps.append(step)
            continue

        diceroll = np.random.randint(1, 7)

        if diceroll == 6:
            new_diceroll = np.random.randint(1, 7)
            step += new_diceroll
        
        elif diceroll <= 2:
            step -= 1

        else:
            step += 1
        
        steps.append(step)
    
    return steps


if __name__ == "__main__":
    main()