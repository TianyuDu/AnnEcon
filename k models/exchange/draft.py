o = object_remover(d)

[np.diff(d, i, axis=0) == o._difference(d, order=i) for i in range(100)]

o._difference(d, order=100) == np.diff(d, 100, axis=0)
