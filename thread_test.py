import threading

n = 0

def add():
    global n
    for _ in range(1000000):
        n += 1

def sub():
    global n
    for _ in range(1000000):
        n -= 1

th1 = threading.Thread(target=add)
th2 = threading.Thread(target=sub)

th1.start()
th2.start()

th1.join()
th2.join()
print(n)
