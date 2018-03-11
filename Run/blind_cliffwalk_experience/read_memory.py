import numpy as np

if __name__ == '__main__':
    m = np.load("./memory_25.npy")

    print(m)
    mm = m[:,:]
    print(mm)
    print(mm.shape[0])

    batchIndex = np.random.choice(mm.shape[0], size=10)
    batchSample = mm[batchIndex, :]

    for i in batchSample:
        a = i

    print(a)
    print(a[2])