import numpy as np


def gradient_desent(X, Y):
    m_cur = b_cur = 0
    iterations = 1000
    n = len(X)
    learning_rate = 0.08

    for i in range(iterations):
        y_predict = m_cur * X + b_cur
        cost = (1 / n) * sum([val**2 for val in (Y - y_predict)])

        md = -(2 / n) * sum(X * (Y - y_predict))
        bd = -(2 / n) * sum(Y - y_predict)
        m_cur = m_cur - learning_rate * md
        b_cur = b_cur - learning_rate * bd
        print("m {}, b {},cost {}, iteration {}".format(m_cur, b_cur, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_desent(x, y)
