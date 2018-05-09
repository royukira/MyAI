import numpy as np

a = np.zeros((10,6))
a[1,4:6] = [2,3]

b = a[1,4]
print(b)

check = np.array([2,3])


for i in range(a.shape[0]):
    t = int(a[i,4])
    idx = int(a[i,5])
    if t == 2 and idx == 4:
        a = np.delete(a,i,0)
        break
    else:
        continue

print(a.shape)
