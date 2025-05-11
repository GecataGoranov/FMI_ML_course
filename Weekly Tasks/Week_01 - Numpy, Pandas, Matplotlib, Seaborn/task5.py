import numpy as np


def main():
    baseball = [[180, 78.4],
                [215, 102.7],
                [210, 98.5],
                [188, 75.2]]
    
    np_baseball = np.array(baseball)

    print("Type: ", type(np_baseball))

    print("Shape: ", np_baseball.shape)

if __name__ == "__main__":
    main()