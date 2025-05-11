import numpy as np


def main():
    arr = np.array([True, 1, 2]) + np.array([3, 4, False])
    answers_dict = {"A": np.array([True, 1, 2, 3, 4, False]),
                    "B": np.array([4, 3, 0]) + np.array([0, 2, 2]),
                    "C": np.array([1, 1, 2]) + np.array([3, 4, -1]),
                    "D": np.array([0, 1, 2, 3, 4, 5])}\
                    
    for answer, expression in answers_dict.items():

        if expression.shape != arr.shape:
            continue
        
        if (expression == arr).all():
            print("Answer: ", answer)


if __name__ == "__main__":
    main()