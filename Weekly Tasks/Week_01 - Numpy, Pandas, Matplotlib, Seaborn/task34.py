import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(123)

    all_walks = []

    for _ in range(5):
        all_walks.append(do_walk())

    print(all_walks)
    

def do_walk():
    steps = [0]
    step = 0

    for _ in range(100):
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