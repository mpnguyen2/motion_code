def func(a, x):
    return (x**2)

def eval(func, y):
    return func(y)

print(eval(func(a=2), 2))