
import pandas as pd
import numpy as np

a = ("up", "down", "left", "right")
test = pd.Series(
                    [1]*4,
                    index=a,
                    name=str([1,2,3,4]))
print(test)

t = pd.DataFrame(columns=["up","down","left","right"],dtype=np.int64)
t2 = pd.DataFrame([[1,2,3,4],[0,0,0,0]],columns=["up","down","left","right"],dtype=np.int64)
t = t.append(t2)
print(t)
t = t.append(test)
print(t)

if "state" in t.index:
    print("yes")

ll = t.loc[1,:]
rl = ll.reindex(np.random.permutation(t.index))  # Conform DataFrame to new index with optional filling logic,
                                                  # placing NA/NaN in locations having no value in the previous index.

print(pd.Series(ll,dtype=np.int64).idxmax())
ddd = pd.Series(ll,dtype=np.int64).all()
if ddd == 0:
    print("yes")

#print(ll.idxmax())