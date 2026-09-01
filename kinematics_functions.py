import numpy as np
import matplotlib.pyplot as plt
import time

def sinX(alpha):
    if (abs(np.sin(alpha)) < 1e-14) and (abs(alpha) < np.pi):
        y = 1
    else:
        y = np.sin(alpha) / alpha

    return y


def A_btSpring(la, lb, lc, w):
    rotz = lambda t: np.array([[np.cos(t), -np.sin(t), 0],
                               [np.sin(t), np.cos(t), 0],
                               [0, 0, 1]])
    roty = lambda t: np.array([[np.cos(t), 0, np.sin(t)],
                               [0, 1, 0],
                               [-np.sin(t), 0, np.cos(t)]])

    l_avg = (la + lb + lc) / 3
    th = 2 * np.sqrt(la**2 + lb**2 + lc**2 - (la*lb + la*lc + lb*lc)) / (3 * w)
    # r = l_avg/th
    phi = np.arctan2(np.sqrt(3)*(lc-lb), lb+lc-2*la)

    x =  l_avg * sinX(th/2) * np.cos((np.pi-th)/2) * np.sin(phi - np.pi/2)
    y = -l_avg * sinX(th/2) * np.cos((np.pi-th)/2) * np.cos(phi - np.pi/2)
    z =  l_avg * sinX(th/2) * np.sin((np.pi-th)/2)

    o_bt = np.array([x, y, z])
    R_bt = rotz(phi) @ roty(-th) @ rotz(-phi)

    return np.row_stack((np.column_stack((R_bt, o_bt)), [0, 0, 0, 1]))


def invKspace_cyl(r, z, phi, theta_flag=False):
    h1 = 4.2
    h2 = 4.2
    # h3 = 7.1 + 7.6
    h3 = 7.1 + 2.9 + 26

    R = 18.75

    rad = (h1**2 - 2*h1*z + h2*h1 - h3**2 - h2*h3 + r**2 + z**2 - h2*z) / (2 * r) - \
        (h2 * (h1 - h3 - z + (h1**2 - 2*h1*h3 - 2*h1*z + h3**2 + 2*h3*z + r**2 + z**2)**0.5)) / (2*r)

    th = 2 * np.arctan((h1 - h3 - z + (h1**2 - 2*h1*h3 - 2*h1*z + h3**2 + 2*h3*z + r**2 + z**2)**.5) / r)

    per = lambda ph: h1+h2+h3 + 2 * (rad + R * np.cos(ph)) * th
    # len2jtheta = lambda p: (-2.468*p + 316.1)
    len2jtheta = lambda p: (-26.03780*p + 4488.62157)

    p1 = per(np.pi-phi)
    p2 = per(np.pi/3-phi)
    p3 = per(-np.pi/3-phi)

    if theta_flag: return np.array([int(len2jtheta(p1)), int(len2jtheta(p2)), int(len2jtheta(p3))])
    else: return np.array([p1, p2, p3])
    # # j1 = len2jtheta(p1)
    # # j2 = len2jtheta(p2)
    # # j3 = len2jtheta(p3)

    # return np.column_stack([j1, j2, j3])

def invKspace_car(x, y, z, theta_flag=True):
    r = np.sqrt(x**2 + y**2)
    # phi = np.arctan2(y, x) + np.pi
    phi = np.arctan2(y, -x)

    if r < 1e-6:
        if theta_flag: return np.array([int(len2jtheta(z)), int(len2jtheta(z)), int(len2jtheta(z))])
        else: return np.array([z, z, z])
    else:
        return invKspace_cyl(r, z, phi, theta_flag=theta_flag)


def T_beModule(p, base, fig, ax):

    h1 = 4.2
    h2 = 4.2
    h3 = 7.1 + 2.9 + 26
    # h3 = 7.1 + 7.6
    R = 18.75

    res = 10

    if len(base) == 0:
        base = np.eye(4)

    la = (p[0] - (h1+h2+h3)) / 2
    lb = (p[1] - (h1+h2+h3)) / 2
    lc = (p[2] - (h1+h2+h3)) / 2

    def T_mat(val):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, val],
            [0, 0, 0, 1]
        ])

    T_b0 = base @ T_mat(h1)
    A_01 = A_btSpring(la, lb, lc, R)
    T_12 = T_mat(h2)
    A_23 = A_01
    T_3e = T_mat(h3)

    T_be = T_b0 @ A_01 @ T_12 @ A_23 @ T_3e

    if fig > 0:
        s = np.column_stack((base @ np.array([0, 0, 0, 1]),
                         T_b0 @ np.array([0, 0, 0, 1])))

        for i in range(1, res + 1):
            s = np.column_stack((s, T_b0 @ (A_btSpring(i * la / res, i * lb / res, i * lc / res, R) @ np.array([0, 0, 0, 1]))))

        s = np.column_stack((s, T_b0 @ (A_01 @ (T_12 @ np.array([0, 0, 0, 1])))))

        for i in range(1, res + 1):
            s = np.column_stack((s, T_b0 @ A_01 @ T_12 @ A_btSpring(i * la / res, i * lb / res, i * lc / res, R) @ np.array([0, 0, 0, 1])))

        s = np.column_stack((s, T_b0 @ A_01 @ T_12 @ A_23 @ np.array([0, 0, 0, 1]), T_be @ np.array([0, 0, 0, 1])))

        ax.plot(s[0, 0:2], s[1, 0:2], s[2, 0:2], 'r', linewidth=2)
        ax.plot(s[0, 2:2 + res], s[1, 2:2 + res], s[2, 2:2 + res], 'b', linewidth=2)
        ax.plot(s[0, 2 + res:3 + res], s[1, 2 + res:3 + res], s[2, 2 + res:3 + res], 'r', linewidth=2)
        ax.plot(s[0, 3 + res:3 + 2 * res], s[1, 3 + res:3 + 2 * res], s[2, 3 + res:3 + 2 * res], 'b', linewidth=2)
        ax.plot(s[0, 3 + 2 * res:], s[1, 3 + 2 * res:], s[2, 3 + 2 * res:], 'r', linewidth=2)
        ax.plot(s[0, -1], s[1, -1], s[2, -1], 'ko', markersize=5)

        plt.pause(0.05)

    return T_be

def jtheta2len(p):
    return (p - 4488.62157) / -26.03760

def len2jtheta(p):
    return (-26.03780*p + 4488.62157)

if __name__ == "__main__":
    pt = np.array([800, 1220, 1220])
    # pt = np.array([65, 65, 65])
    jtheta2len = lambda p: (p - 4488.62157) / -26.03760 # writeMicroseconds, polaris readings
    # jtheta2len = lambda p: (p - 316.1) / -2.468

    # print(jtheta2len(pt))
    print(T_beModule(jtheta2len(pt), [], 0, 0)[:3,3])

    pt = np.array([125.5, 125.5, 125.5])
    # len2jtheta = lambda p: (-2.468*p + 316.1)
    len2jtheta = lambda p: (-26.03780*p + 4488.62157)
    # print(len2jtheta(pt))
    # print(pt)

    print(invKspace_cyl(30, 126, np.pi))
    print(T_beModule(invKspace_cyl(30, 126, np.pi), [], 0, 0)[:3,3])
    print(invKspace_car(30, 0, 126))
    print(T_beModule(invKspace_car(30, 0, 126), [], 0, 0)[:3,3])
