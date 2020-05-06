import numpy as np
from tabulate import tabulate
from scipy.stats import f, t, ttest_ind, norm
from _pydecimal import Decimal, ROUND_UP, ROUND_FLOOR
from pyDOE2 import ccdesign
from sklearn.linear_model import LinearRegression


class full_factor_experiment:
    def __init__(self, N, m, q, y_only_int=True):
        self.N = N
        self.m = m
        self.q = q
        self.x = np.array([[-10, 50], [-20, 40], [-20, -15]])
        x_cp_min, x_cp_max = self.x.T[0].mean(), self.x.T[1].mean()
        self.y_max = 200 + x_cp_max
        self.y_min = 200 + x_cp_min
        self.d = 0
        self.l = np.sqrt(m)
        self.d_a = 0
        self.y_only_int = y_only_int


    def y_func(self, x1, x2, x3):
        return 4.5 + 4.2 * x1 + 7.5 * x2 + 5.6 * x3 + 2.6 * x1 * x1 + 0.1 * x2 * x2 + 7.6 * x3 * x3 + 9.9 * x1 * x2 + 0.3 * x1 * x3 +3.1 * x2 * x3 + 8.7 * x1 * x2 * x3

    def get_cohren_value(self):
        size_of_selections = self.N + 1
        qty_of_selections = self.m - 1
        significance = self.q
        partResult1 = significance / (size_of_selections - 1)
        params = [partResult1, qty_of_selections, (size_of_selections - 1 - 1) * qty_of_selections]
        fisher = f.isf(*params)
        result = fisher / (fisher + (size_of_selections - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    def get_student_value(self):
        f3 = (self.m - 1) * self.N
        significance = self.q
        return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

    def get_fisher_value(self):
        f3 = (self.m - 1) * self.N
        f4 = self.N - self.d
        significance = self.q
        return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()

    def cohren_crit(self):
        return (np.max(self.y_std) / np.sum(self.y_std)) < self.get_cohren_value()

    def student_crit(self):
        y_std_mean = np.mean(self.y_std)
        self.S_2b = y_std_mean
        b_test = [np.sum(self.y_mean * self.extended.T[i]) / self.N for i in range(self.extended.shape[1])]
        t = np.array([np.abs(b) / np.sqrt(y_std_mean / (self.N * self.m)) for b in b_test])
        ret = np.where(t > self.get_student_value())
        for i in range(self.b.shape[0]):
            if i not in ret[0]:
                print(f"Коефіцієнт {self.b[i]} - незначимий")
                self.b[i] = 0
        self.d_a += self.d
        self.d = len(ret[0])

        self.test = []
        for i in range(len(self.y_mean)):
            self.test.append((self.b[0] + np.sum(self.b[1:] * self.extended_real[i]) - self.y_mean[i]) ** 2)

    def fisher_crit(self):
        if self.d == self.N:
            self.N += 1
        f4 = self.N - self.d
        S_2_ad = self.m / (self.N - self.d) * np.sum(self.test)
        F_p = S_2_ad / self.S_2b
        return F_p < self.get_fisher_value()

    def gen_matrix(self):
        self.seq = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3], [1, 1], [2, 2], [3, 3]]

        delta_x = [np.abs(m[1] - m[0]) / 2 for m in self.x]
        x_i_0 = [(m[1] + m[0]) / 2 for m in self.x]

        self.extended = ccdesign(3, center=(0, 0))
        # self.extended = np.delete(self.extended, -1, 0)

        self.extended_real = np.zeros((14, 3))
        for i in range(self.extended.shape[0]):
            for j in range(self.extended.shape[1]):
                if self.extended[i][j] == 1:
                    self.extended_real[i][j] = self.x[j][1]
                elif self.extended[i][j] == -1:
                    self.extended_real[i][j] = self.x[j][0]
                elif self.extended[i][j] == 0:
                    self.extended_real[i][j] = x_i_0[j]
                elif self.extended[i][j] > 0:
                    self.extended[i][j] = self.l
                    self.extended_real[i][j] = self.l * delta_x[j] + x_i_0[j]
                else:
                    self.extended_real[i][j] = -self.l * delta_x[j] + x_i_0[j]
                    self.extended[i][j] = -self.l

        self.y = np.array(
            [[self.y_func(*self.extended_real[i][:3]) + np.random.rand() * 10 + 5 for j in range(self.m)] for i in
             range(self.N)])
        self.y_mean = [i.mean() for i in self.y]
        self.y_std = [np.std(i) for i in self.y]

        num = self.extended.shape[1]
        seq = [[1, 2], [1, 3], [2, 3], [1, 2, 3], [1, 1], [2, 2], [3, 3]]
        for i in seq:
            app = np.array([1] * self.extended.shape[0])
            app_real = np.array([1] * self.extended.shape[0])
            for j in i:
                app = app * self.extended.T[j - 1]
                app_real = app_real * self.extended_real.T[j - 1]
            self.extended = np.insert(self.extended, num, app.T, axis=1)
            self.extended_real = np.insert(self.extended_real, num, app_real.T, axis=1)
            num += 1
        # print(tabulate(self.extended_real))

    def find_b(self):
        self.regression = LinearRegression()
        self.regression.fit(self.extended_real, self.y_mean)
        self.b = [self.regression.intercept_]
        self.b.extend(self.regression.coef_)
        self.true_b = np.copy(self.b)
        self.b = np.round(self.b, decimals=3)
        # print(self.regression.intercept_ + np.sum(self.regression.coef_ * np.array([[-5, -10, -7, 50, 35, 70, -350, 25, 100, 49]])) / self.y_mean[0])
        self.regression_norm = LinearRegression()
        self.regression_norm.fit(self.extended, self.y_mean)
        self.b_norm = [self.regression_norm.intercept_]
        self.b_norm.extend(self.regression_norm.coef_)
        self.b_norm = np.round(self.b_norm, decimals=3)
        # print(np.mean(regression.predict(np.array([[-5, -10, -7, 50, 35, 70, -350, 25, 100, 49]]))))

    def check(self):
        for i in range(len(self.y_mean)):
            predicted = np.round(self.b[0] + np.sum(self.b[1:] * self.extended_real[i]), decimals=3)
            print(f"y{i + 1} = ", predicted, " ~ ", np.round(self.y_mean[i], decimals=3))
            del predicted

    def model(self):
        self.gen_matrix()
        while True:
            if not self.cohren_crit():
                print("Дисперсія неоднорідна за критерієм Кохрена")
                self.gen_matrix()
            else:
                col = []
                for i in range(len(self.y_mean)):
                    col.append([self.y_mean[i]])
                to_print = np.hstack((self.extended_real, self.y, col))
                headers = ['x1', 'x2', 'x3', 'x1x2', 'x1x3', 'x2x3', 'x1x2x3', "x1^2", "x2^2", "x3^2"]
                for i in range(self.m):
                    headers.append(f'y{i + 1}')
                headers.append('y')
                headers.append('S(y)')
                print("Матриця планування експерименту")
                print(tabulate(list(to_print), headers=headers, tablefmt="fancy_grid"))
                to_print = np.hstack((self.extended, self.y, col))
                print("Матриця планування експерименту з нормованими значеннями")
                print(tabulate(list(to_print), headers=headers, tablefmt="fancy_grid"))
                print("Дисперсія однорідна за критерієм Кохрена")

                self.find_b()

                print("Рівняння регресії",
                      f"y = {self.b[0]} + {self.b[1]}*x1 + {self.b[2]}*x2 + {self.b[3]}*x3 + {self.b[4]}*x1x2 + {self.b[5]}*x1x3 + {self.b[6]}*x2x3 + {self.b[7]}*x1x2x3 + {self.b[8]}*x1^2 + {self.b[9]}*x2^2 + {self.b[10]}*x3^2")
                print("Рівняння регресії для кодованих значень",
                      f"y = {self.b_norm[0]} + {self.b_norm[1]}*x1 + {self.b_norm[2]}*x2 + {self.b_norm[3]}*x3 + {self.b_norm[4]}*x1x2 + {self.b_norm[5]}*x1x3 + {self.b_norm[6]}*x2x3 + {self.b_norm[7]}*x1x2x3 + {self.b_norm[8]}*x1^2 + {self.b_norm[9]}*x2^2 + {self.b_norm[10]}*x3^2")

                self.check()

                self.student_crit()
                if self.fisher_crit():
                    print("Рівняння регресії адекватно оригіналу")
                    break
                else:
                    print("Рівняння регресії неадекватно оригіналу")
                    break



m = full_factor_experiment(14, 3, 0.05)
for i in range(100):
	m.model()
print("середнє значення усіх значимих коефіцієнтів=",m.d_a / 100)
