from HeartLabTest.src.main import create_datasets, split_dataset, preprocess
import os
data_dir = "chest_xray"

def test_split_dataset(dataset):
    X, y = split_dataset(dataset)
    test = False
    # test shuffle
    for i in range(300):
        if y[i] != 0:
            test = True
    assert test, "dataset not shuffled"

    # test X, y contains all examples
    assert X.shape[0] == 990, "dataset missing examples"
    assert y.shape[0] == 990, "dataset missing examples"

    # check image size is 256 x 256
    assert X.shape[1:] == (256,256), "images are {} not 256x256".format(X.shape[1:])

def test_preprocess(dataset):
    X, y = split_dataset(dataset)
    X, y = preprocess(X, y)

    # test images are reshaped correctly
    assert X.shape == (990, 256, 256, 1)
    # test X normalized between 0 - 1
    for sample in X:
        for value in sample:
            assert value[0] <= 1


def main():
    dataset = create_datasets(data_dir)
    test_split_dataset(dataset)
    test_preprocess(dataset)


if __name__ == '__main__':
    main()
