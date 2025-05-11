import numpy as np


def main():
    np.random.seed(123)

    print("Random float:", np.random.rand())
    print("Ranom integer 1:", np.random.randint(1, 7))
    print("Ranom integer 2:", np.random.randint(1, 7))

    step = 50
    print("Before throw step =", step)

    diceroll = np.random.randint(1, 7)
    print("After throw dice =", diceroll)
    
    if diceroll == 6:
        new_diceroll = np.random.randint(1, 7)
        step += new_diceroll

    elif diceroll <= 2:
        step -= 1

    else:
        step += 1

    print("After throw step =", step)
    


if __name__ == "__main__":
    main()