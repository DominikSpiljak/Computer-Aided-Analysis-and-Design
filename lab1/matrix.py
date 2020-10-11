from copy import deepcopy
from typing import Optional


class Matrix:

    matrix: list

    def __init__(self, f=None):
        """Constructor for Matrix class

        Args:
            f (string, optional): Path to file containing matrix. Defaults to None.
        """
        if f:
            self.matrix = self.read_from_file(f)

    def read_from_file(self, f) -> list:
        """Parses file and returns a matrix

        Args:
            f (string): Path to file containing matrix

        Returns:
            list: Parsed matrix
        """
        with open(f, 'r') as inp:
            return [list(map(lambda x: float(x), line.replace('\n', '').replace('\t', ' ').split())) for line in inp.readlines()]

    def get_column(self, index):
        column = Matrix()
        column.matrix = [[self[i][index]] for i in range(len(self))]
        return column

    @staticmethod
    def zeros(shape: [tuple, list, set]):
        """Creates a matrix full of zeros with given shape

        Args:
            shape (iterable): shape of matrix

        Returns:
            Matrix: matrix full of zeros with given shape
        """
        matrix = Matrix()
        matrix.matrix = [[0 for _ in range(shape[1])] for _ in range(shape[0])]
        return matrix

    @staticmethod
    def eye(n: int):
        """Creates an matrix of indexes with ones on diagonal

        Args:
            n(int): dimension of matrix

        Returns:
            Matrix: matrix of indexes with ones on diagonal
        """
        matrix = Matrix()
        matrix.matrix = [
            [1 if i == j else 0 for j in range(n)] for i in range(n)]
        return matrix

    @staticmethod
    def merge_columns(columns):
        rows = [[] for _ in columns]
        for i, column in enumerate(columns):
            for j, element in enumerate(column):
                rows[j].append(element[0])
        mat = Matrix()
        mat.matrix = rows
        return mat

    def swap_rows(self, row_index1, row_index2):
        self[row_index1], self[row_index2] = self[row_index2], self[row_index1]

    def shape(self) -> tuple:
        """Returns dimensions of matrix

        Returns:
            tuple: Matrix dimensions
        """
        return len(self.matrix), len(self.matrix[0])

    def reshape(self, shape: [tuple, list, set]) -> None:
        """Reshapes matrix into another shape

        Args:
            shape (iterable): Iterable with 2 elements representing new matrix shape

        Raises:
            ValueError: In case shape doesn't match number of elements in matrix,
                         ValueError is raised
        """
        dims = self.shape()
        elements = dims[0] * dims[1]
        if elements != shape[0] * shape[1]:
            raise ValueError(
                'Cannot reshape {} elements into shape {}'.format(elements, shape))
        iterator = iter(self.flatten())
        self.matrix = [[next(iterator) for column in range(shape[1])]
                       for line in range(shape[0])]

    def flatten(self) -> list:
        """Flattens matrix into a 1-d list

        Returns:
            list: Flattened matrix
        """
        flat = []
        for line in self.matrix:
            flat.extend(line)
        return flat

    def __getitem__(self, indices: [tuple, list, set]):
        """Overloads and handles getter indexing

        Args:
            indices (tuple, list, set): Indices of element inside matrix

        Returns:
            Object at given indices
        """
        return self.matrix[indices]

    def __setitem__(self, indices: [tuple, list, set], item):
        """Overloads and handles setter indexing

        Args:
            indices (tuple, list, set): Indices of element inside matrix
        """
        self.matrix[indices] = item

    def __add__(self, other: Optional['Matrix']) -> Optional['Matrix']:
        """Overloads and handles addition

        Args:
            other (Matrix or a number): Other member of addition

        Raises:
            ValueError: In case of adding 2 Matrices,
             if their shapes don't match, raise ValueError

        Returns:
            Matrix: result of addition
        """
        selfshape = self.shape()
        othershape = other.shape()
        if selfshape != othershape:
            raise ValueError("Shapes mismatch: {} != {}".format(
                selfshape, othershape))

        mat_copy = Matrix()
        mat_copy.__dict__ = deepcopy(self.__dict__)

        for i in range(selfshape[0]):
            for j in range(selfshape[1]):
                mat_copy[i][j] += other[i][j]

        return mat_copy

    def __iter__(self):
        """Returns iterator
        """
        return iter(self.matrix)

    def __sub__(self, other: Optional['Matrix']) -> Optional['Matrix']:
        """Overloads and handles substraction

        Args:
            other (Matrix or a number): Other member of substraction

        Raises:
            ValueError: In case of substracting 2 Matrices,
             if their shapes don't match, raise ValueError

        Returns:
            Matrix: result of substraction
        """
        selfshape = self.shape()
        othershape = other.shape()
        if selfshape != othershape:
            raise ValueError("Shapes mismatch: {} != {}".format(
                selfshape, othershape))

        mat_copy = Matrix()
        mat_copy.__dict__ = deepcopy(self.__dict__)

        for i in range(selfshape[0]):
            for j in range(selfshape[1]):
                mat_copy[i][j] -= other[i][j]

        return mat_copy

    def __mul__(self, other: Optional['Matrix']) -> Optional['Matrix']:
        """Overloads and handles multiplication

        Args:
            other (Matrix or a number): Other member of multiplication

        Raises:
            ValueError: In case of multiplying 2 Matrices,
             if their shapes don't match, raise ValueError

        Returns:
            Matrix: result of multiplication
        """
        selfshape = self.shape()

        if isinstance(other, Matrix):
            othershape = other.shape()
            if selfshape[1] != othershape[0]:
                raise ValueError("Shapes mismatch: {} and {}".format(
                    selfshape, othershape))
            mat = Matrix.zeros((selfshape[0], othershape[1]))
            for i in range(selfshape[0]):
                for j in range(othershape[1]):
                    mat[i][j] = sum([self[i][k] * other[k][j]
                                     for k in range(selfshape[1])])

            return mat

        else:
            mat = Matrix()
            mat.__dict__ = deepcopy(self.__dict__)
            for i in range(selfshape[0]):
                for j in range(selfshape[1]):
                    mat[i][j] *= other
            return mat

    __rmul__ = __mul__  # Assign multiplication function to right multiplication overloading

    def __str__(self) -> str:
        """Returns string representation of matrix

        Returns:
            str: String representation of matrix
        """
        return '\n'.join([' '.join(list(map(lambda x: str(x), line)))
                          for line in self.matrix])

    def __len__(self) -> int:
        return len(self.matrix)

    def LU_decomposition(self) -> None:
        """Perform inline LU decomposition

        Raises:
            ZeroDivisionError: In case pivot element is zero, ZeroDivisionError is raised
        """
        n = self.shape()[0]
        for i in range(n - 1):
            if self[i][i] == 0:
                raise ZeroDivisionError(
                    'Found pivot with value 0, try with LUP_decomposition')
            for j in range(i + 1, n):
                self[j][i] /= self[i][i]
                for k in range(i + 1, n):
                    self[j][k] -= self[j][i] * self[i][k]

    def LUP_decomposition(self, return_swap_count=False) -> Optional['Matrix']:
        def find_pivot(matrix, index):
            pivot = None
            for j in range(index, selfshape[1]):
                if pivot is None:
                    max_element = abs(self[j][index])
                    pivot = j

                elif abs(self[j][index]) > max_element:
                    max_element = abs(self[j][index])
                    pivot = j
            return pivot

        selfshape = self.shape()
        indexes = Matrix.eye(selfshape[0])
        swaps = 0
        for i in range(selfshape[0]):
            pivot = find_pivot(self, i)
            if self[i][pivot] == 0:
                raise ZeroDivisionError(
                    'Found pivot with value 0, can\'t solve for singular matrix')
            if pivot != i:
                self.swap_rows(i, pivot)
                indexes.swap_rows(i, pivot)
                swaps += 1
            for j in range(i + 1, selfshape[0]):
                self[j][i] /= self[i][i]
                for k in range(i + 1, selfshape[0]):
                    self[j][k] -= self[j][i] * self[i][k]

        if return_swap_count:
            return indexes, swaps
        else:
            return indexes

    def forward_supstitution(self, b: Optional['Matrix'], indexes=None) -> Optional['Matrix']:
        """Performs forward supstitution L * y = b

        Raises:
            ValueError: In case height of b doesn't match height of L, raise ValueError

        Returns:
            Matrix: y calculated from forward supstitution
        """

        n = self.shape()[0]
        if b.shape() != (n, 1):
            raise ValueError(
                'Shape of b doesn\'t match required shape {}'.format((n, 1)))

        y = Matrix()
        y.__dict__ = deepcopy(b.__dict__)

        if indexes is not None:
            y = indexes * y

        for i in range(len(y)):
            for j in range(i):
                y[i][0] -= self[i][j] * y[j][0]

        return y

    def backward_supstitution(self, y: Optional['Matrix']) -> Optional['Matrix']:
        """Performs backward supstitution U * x = y

        Raises:
            ValueError: In case height of y doesn't match height of U, raise ValueError

        Returns:
            Matrix: x calculated from backward supstitution
        """
        n = self.shape()[0]
        if y.shape() != (n, 1):
            raise ValueError(
                'Shape of y doesn\'t match required shape {}'.format((n, 1)))

        x = Matrix()
        x.__dict__ = deepcopy(y.__dict__)

        for i in reversed(range(len(x))):
            for j in range(i, len(x) - 1):
                x[i][0] -= self[i][j + 1] * x[j + 1][0]
            try:
                x[i][0] /= self[i][i]
            except ZeroDivisionError:
                raise ZeroDivisionError(
                    'Found element on diagonal with value 0 during backward supstitution')

        return x

    def LUP_invert(self):
        indexes = self.LUP_decomposition()
        columns = []
        for i in range(self.shape()[1]):
            b = indexes.get_column(i)
            y = self.forward_supstitution(b)
            columns.append(self.backward_supstitution(y))
        return Matrix.merge_columns(columns)

    @staticmethod
    def get_LU_matrix_determinant(matrix):
        det = 1
        for i in range(len(matrix)):
            det *= matrix[i][i]
        return det

    def LUP_determinant(self):
        A = Matrix()
        A.__dict__ = deepcopy(self.__dict__)
        indexes, swap_count = A.LUP_decomposition(return_swap_count=True)
        return pow(-1, swap_count) * Matrix.get_LU_matrix_determinant(A)


def print_task_num(num):
    print()
    print('=' * 20, 'zadatak ', num, '=' * 20)
    print()


if __name__ == "__main__":
    print_task_num(1)
    print('{} == {} ( = {}) => {}'.format(
        7.666, "7.666/3.51 * 3.51", 7.666/3.51 * 3.51, 7.666 == 7.666/3.51 * 3.51))
    print('round({}, 6) == round({}, 6) => {}'.format(
        7.666, "7.666/3.51 * 3.51", round(7.666, 6) == round(7.666/3.51 * 3.51, 6)))

    print_task_num(2)
    A = Matrix()
    A.matrix = [[3, 9, 6], [4, 12, 12], [1, -1, 1]]
    b = Matrix()
    b.matrix = [[12], [12], [1]]
    print('LU dekompozicija:')
    print()
    try:
        A.LU_decomposition()
        y = A.forward_supstitution(b)
        x = A.backward_supstitution(y)
        print('Riješeno!\n', x)
    except ZeroDivisionError as e:
        print(e)
        print()
        A = Matrix()
        A.matrix = [[3, 9, 6], [4, 12, 12], [1, -1, 1]]
        b = Matrix()
        b.matrix = [[12], [12], [1]]
        print('LUP dekompozicija:')
        print()
        try:
            indexes = A.LUP_decomposition()
            y = A.forward_supstitution(b, indexes)
            x = A.backward_supstitution(y)
            print('Riješeno!\n', x)
        except ZeroDivisionError as e:
            print(e)

    print_task_num(3)
    A = Matrix()
    A.matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print('LU dekompozicija:')
    print()
    try:
        A.LU_decomposition()
        print('LU dekompozicija uspješna\n', A)
        b = Matrix()
        b.matrix = [[6], [15], [24]]
        print('b = \n', b)
        y = A.forward_supstitution(b)
        x = A.backward_supstitution(y)
        print('Riješeno za x = \n', x)
    except ZeroDivisionError as e:
        print(e)
        print()
        A = Matrix()
        A.matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        print('LUP dekompozicija:')
        print()
        try:
            indexes = A.LUP_decomposition()
            print('LUP dekompozicija uspješna\n', A)
            b = Matrix()
            b.matrix = [[6], [15], [24]]
            print('b = \n', b)
            y = A.forward_supstitution(b, indexes)
            x = A.backward_supstitution(y)
            print('Riješeno za x = \n', x)
        except ZeroDivisionError as e:
            print(e)

    print_task_num(4)
    A = Matrix()
    A.matrix = [[0.000001, 3000000, 2000000], [
        1000000, 2000000, 3000000], [2000000, 1000000, 2000000]]
    b = Matrix()
    b.matrix = [[12000000.000001], [14000000], [10000000]]
    print('Riješavanje LU dekompozicijom')
    A.LU_decomposition()
    y = A.forward_supstitution(b)
    x = A.backward_supstitution(y)
    print('x = \n', x)

    A = Matrix()
    A.matrix = [[0.000001, 3000000, 2000000], [
        1000000, 2000000, 3000000], [2000000, 1000000, 2000000]]
    b = Matrix()
    b.matrix = [[12000000.000001], [14000000], [10000000]]
    print('Riješavanje LUP dekompozicijom')
    indexes = A.LUP_decomposition()
    y = A.forward_supstitution(b, indexes)
    x = A.backward_supstitution(y)
    print('x = \n', x)

    print()
    print('Razlika je u tome što tijekom LUP dekompozicije ko pivot uzimamo element s najvećom apsolutnom vrijednosti te se smanjuje mogućnost greške zbog premalih brojeva')

    print_task_num(5)
    A = Matrix()
    A.matrix = [[0, 1, 2], [2, 0, 3], [3, 5, 1]]
    b = Matrix()
    b.matrix = [[6], [9], [3]]
    print('Riješavanje LUP dekompozicijom')
    indexes = A.LUP_decomposition()
    y = A.forward_supstitution(b, indexes)
    x = A.backward_supstitution(y)
    print('x = \n', x)
    print()
    print('Došlo je do razlike u rezultatu zbog takozvanog "šuma" zbog zaokruživanja brojeva te x1 i x2 nisu točno 0')

    print_task_num(6)
    A = Matrix()
    A.matrix = [[4000000000, 1000000000, 3000000000], [
        4, 2, 7], [.0000000003, .0000000005, .0000000002]]
    div = Matrix.zeros((3, 3))
    div[0][0] = 10**-6
    div[1][1] = 10**6
    div[2][2] = 10**-6
    print('Pomnozimo jednadzbe sa:\n{}'.format(div))
    b = Matrix()
    b.matrix = [[9000000000], [15], [.0000000015]]
    A = div * A
    b = div * b
    print('Riješavanje LUP dekompozicijom')
    indexes = A.LUP_decomposition()
    y = A.forward_supstitution(b, indexes)
    x = A.backward_supstitution(y)
    print('x = \n', x)

    print_task_num(7)
    A = Matrix()
    A.matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(A.LUP_invert())

    print_task_num(8)
    A = Matrix()
    A.matrix = [[4, -5, -2], [5, -6, -2], [-8, 9, 3]]
    print(A.LUP_invert())

    print_task_num(9)
    A = Matrix()
    A.matrix = [[4, -5, -2], [5, -6, -2], [-8, 9, 3]]
    print(A.LUP_determinant())

    print_task_num(10)
    A = Matrix()
    A.matrix = [[3, 9, 6], [4, 12, 12], [1, -1, 1]]
    print(A.LUP_determinant())
