import numpy as np


def main():
    np.random.seed(123)

    steps = [0]
    step = 0

    for _ in range(100):
        diceroll = np.random.randint(1, 7)

        if diceroll == 6:
            new_diceroll = np.random.randint(1, 7)
            step += new_diceroll
        
        elif diceroll <= 2:
            if step > 0:
                step -= 1

        else:
            step += 1
        
        steps.append(step)

    print(steps)


if __name__ == "__main__":
    main()