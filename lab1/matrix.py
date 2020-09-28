from copy import deepcopy


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

    def shape(self) -> tuple:
        """Returns dimensions of matrix

        Returns:
            tuple: Matrix dimensions
        """
        return len(self.matrix), len(self.matrix[0])

    def reshape(self, shape) -> None:
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

    def __getitem__(self, indices):
        """Overloads and handles indexing

        Args:
            indices (int): Indices of element inside matrix

        Returns:
            Object at given indices
        """
        return self.matrix[indices]

    def __add__(self, other: Matrix) -> Matrix:
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

    def __sub__(self, other: Matrix) -> Matrix:
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

    def __mul__(self, other: Matrix) -> Matrix:
        """Overloads and handles multiplication

        Args:
            other (Matrix or a number): Other member of multiplication

        Raises:
            ValueError: In case of multiplying 2 Matrices,
             if their shapes don't match, raise ValueError

        Returns:
            Matrix: result of multiplication
        """
        mat_copy = Matrix()
        mat_copy.__dict__ = deepcopy(self.__dict__)
        selfshape = self.shape()

        if isinstance(other, Matrix):
            othershape = other.shape()
            if selfshape != othershape:
                raise ValueError("Shapes mismatch: {} != {}".format(
                    selfshape, othershape))

            for i in range(selfshape[0]):
                for j in range(selfshape[1]):
                    mat_copy[i][j] *= other[i][j]

            return mat_copy

        else:
            for i in range(selfshape[0]):
                for j in range(selfshape[1]):
                    mat_copy[i][j] *= other
            return mat_copy

    __rmul__ = __mul__

    def __str__(self) -> str:
        """Returns string representation of matrix

        Returns:
            str: String representation of matrix
        """
        return '\n'.join([' '.join(list(map(lambda x: str(x), line)))
                          for line in self.matrix])

    def LU_decomposition(self) -> None:
        """Perform inline LU decomposition
        """
        n = self.shape()[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                self[j][i] /= self[i][i]
                for k in range(i + 1, n):
                    self[j][k] -= self[j][i] * self[i][k]

    def forward_supstitution(self, b: Matrix) -> Matrix:
        n = self.shape()[0]
        if b.shape() != (n, 1):
            raise ValueError(
                'Shape of b doesn\'t match required shape {}'.format((n, 1)))

        y = Matrix()
        y.__dict__ = deepcopy(b.__dict__)

        # TODO: Finish forward supsitiution

# TODO: LUR decomposition, backward supstitution, LUP inverse and LUP determinant
