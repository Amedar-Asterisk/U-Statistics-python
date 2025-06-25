from dataclasses import dataclass
from .U2V import get_all_partitions, partition_weight
from typing import List, Set
import numpy as np

from .._utils import get_backend


def get_sign_backend():
    return get_backend().sign


def get_einsum_backend():
    return get_backend().einsum


def get_prod_backend():
    return get_backend().prod


def get_zeros_backend():
    return get_backend().zeros


@dataclass
class _Expression:

    input_expression: List[str]
    output_expression: str
    contracted_indices: List[str]

    @property
    def formula(self) -> str:
        """
        Returns Einsum formula for the expression.
        The formula is constructed in the form of "input_expression->output_expression".
        """
        return f"{','.join(self.input_expression)}->{self.output_expression}"

    def subexpression(self, partition: List[Set[str]]) -> "_Expression":
        """
        Create a subexpression by partitioning the input expression.
        """
        mapping = {}
        new_contracted_indices = []
        for s in partition:
            representative = min(s)
            new_contracted_indices.append(representative)
            for element in s:
                if element != representative:
                    mapping[element] = representative

        new_input = []
        for pair in self.input_expression:
            new_pair = [mapping.get(index, index) for index in pair]
            new_pair = "".join(new_pair)
            new_input.append(new_pair)
        return _Expression(
            input_expression=new_input,
            output_expression=self.output_expression,
            contracted_indices=new_contracted_indices,
        )

    @staticmethod
    def get_contracted_indices(inputs: List[str], output: str) -> Set[str]:
        """
        Extracts contracted indices from the input and output expressions.
        """
        contracted_indices = set()
        for pair in inputs:
            contracted_indices.update(pair)
        for index in output:
            contracted_indices.discard(index)
        return contracted_indices


# -----------Spearman's rho statistics----------------
class _SpearmanExpressions:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.S = _Expression(
                input_expression=["ijp", "ikq"],
                output_expression="pq",
                contracted_indices=["i", "j", "k"],
            )
            self.K = _Expression(
                input_expression=["ijp", "ijq"],
                output_expression="pq",
                contracted_indices=["i", "j"],
            )
            self._initialized = True


def _get_spearman_expressions():
    return _SpearmanExpressions()


def _spearman_kernel(X: np.ndarray) -> np.ndarray:
    """
    Computes the Spearman's kernel for the given data.
    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.
    Returns
    -------
    np.ndarray
        A n x n x p tensor representing the Spearman's kernel.
        T[i, j, p] = sign(X[i, p] - X[j, p]) for all i, j in {0, ..., n-1}
    and p in {0, ..., p-1}.
    """
    Xi = X[:, np.newaxis, :]
    Xj = X[np.newaxis, :, :]
    T = np.sign(Xi - Xj)
    return T


def _spearman_hatrho(T: np.ndarray) -> np.ndarray:
    """
    Computes the first part of the Spearman's rho, which is
    a 3-th order U-statistic.

    Parameters
    ----------
    T : np.ndarray
        A n x n x p tensor of spearman's kernel, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    np.ndarray
        A p x p matrix representing the first part of the Spearman's rho.
    """
    n, _, p = T.shape
    expr = _get_spearman_expressions()
    partitions = get_all_partitions(expr.S.contracted_indices)
    result = get_zeros_backend()((p, p))
    for partition in partitions:
        subexpr = expr.S.subexpression(partition)
        weight = partition_weight(partition)
        result += weight * get_einsum_backend()(subexpr.formula, T, T)
    return result


def _spearman_tau(T: np.ndarray):
    """
    Computes the second part of the Spearman's rho, which is
    a 2-th order U-statistic.

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    np.ndarray
        A p x p matrix representing the second part of the Spearman's rho.
    """
    n, _, p = T.shape
    expr = _get_spearman_expressions()
    partitions = get_all_partitions(expr.K.contracted_indices)
    result = get_zeros_backend()((p, p))
    for partition in partitions:
        subexpr = expr.K.subexpression(partition)
        weight = partition_weight(partition)
        result += weight * get_einsum_backend()(subexpr.formula, T, T)
    result = result / get_prod_backend()(range(n, n - 2, -1))
    return result


def spearman_rho(X: np.ndarray) -> np.ndarray:
    """
    Computes the Spearman's rho for the given data

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    np.ndarray
        A p x p matrix representing the Spearman's rho.
    """

    n, p = X.shape
    T = _spearman_kernel(X)
    hatrho = _spearman_hatrho(T)
    tau = _spearman_tau(T)
    return 3 * (hatrho + tau) / get_prod_backend()(range(n, n - 3, -1))


# -----------Bergsma-Dassios t* statistics----------------
class _BergsmaDassiostExpressions:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.expressions: List[_Expression] = []
            left_input_expressions_list = []
            left_input_expressions_list.append(["abp", "adp", "cbp", "cdp"])
            left_input_expressions_list.append(["bap", "dap", "bcp", "dcp"])
            left_input_expressions_list.append(["acp", "adp", "bcp", "bdp"])
            left_input_expressions_list.append(["cap", "dap", "cbp", "dbp"])

            right_input_expressions_list = []
            right_input_expressions_list.append(["abq", "adq", "cbq", "cdq"])
            right_input_expressions_list.append(["baq", "daq", "bcq", "dcq"])
            right_input_expressions_list.append(["acq", "adq", "bcq", "bdq"])
            right_input_expressions_list.append(["caq", "daq", "cbq", "dbq"])
            output_expression = "pq"
            contracted_indices = ["a", "b", "c", "d"]

            for left_input in left_input_expressions_list:
                for right_input in right_input_expressions_list:
                    self.expressions.append(
                        _Expression(
                            input_expression=left_input + right_input,
                            output_expression=output_expression,
                            contracted_indices=contracted_indices,
                        )
                    )

            signs = [1, 1, -1, -1]
            self.signs = []
            for left_sign in signs:
                for right_sign in signs:
                    self.signs.append(left_sign * right_sign)
            self._initialized = True


def _get_bergsma_dassios_expressions():
    return _BergsmaDassiostExpressions()


def _bergsma_dassios_kernel(X: np.ndarray) -> np.ndarray:
    """Computes the lessness Bergsma-Dassios kernel for the given data.

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    T: np.ndarray
        A n x n x p tensor representing the Bergsma-Dassios kernel.
        T[i, j, p] = 1(X[i, p] < X[j, p]) for all i, j in {0, ..., n-1}
    and p in {0, ..., p-1}.
    """

    Xi = X[:, np.newaxis, :]
    Xj = X[np.newaxis, :, :]
    T = (Xi < Xj).astype(int)
    return T


def bergsma_dassios_t(X: np.ndarray) -> np.ndarray:
    """
    Computes the Bergsma-Dassios t* for the given data.

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    np.ndarray
        A p x p matrix representing the Bergsma-Dassios tau.
    """
    exprs = _get_bergsma_dassios_expressions().expressions
    signs = _get_bergsma_dassios_expressions().signs
    contracted_indices = exprs[0].contracted_indices
    T = _bergsma_dassios_kernel(X)
    tensors = [T] * 8
    n, _, p = T.shape
    result = get_zeros_backend()((p, p))
    partitions = get_all_partitions(contracted_indices)

    for expr, sign in zip(exprs, signs):
        t_result = get_zeros_backend()((p, p))
        for partition in partitions:
            subexpr = expr.subexpression(partition)
            weight = partition_weight(partition)
            t_result += weight * get_einsum_backend()(subexpr.formula, *tensors)
        result = sign * t_result

    result = result / get_prod_backend()(range(n, n - 4, -1))
    return result


# ---------------Hoeffding's D statistics----------------
class _HoeffdingDExpressions:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.expressions: List[_Expression] = []
            intputsbase = []
            intputsbase.append(["ab", "ad"])
            intputsbase.append(["ab", "ae"])
            intputsbase.append(["ac", "ad"])
            intputsbase.append(["ac", "ae"])
            output_expression = "pq"

            left_input_expressions_list = []
            right_input_expressions_list = []

            for inputs in intputsbase:
                left_input_expressions_list.append([pair + ["p"] for pair in inputs])
                right_input_expressions_list.append([pair + ["q"] for pair in inputs])

            for left_input in left_input_expressions_list:
                for right_input in right_input_expressions_list:
                    input_expression = left_input + right_input
                    self.expressions.append(
                        _Expression(
                            input_expression=input_expression,
                            output_expression=output_expression,
                            contracted_indices=_Expression.get_contracted_indices(
                                input_expression, output_expression
                            ),
                        )
                    )

            signs = [1, -1, -1, 1]
            self.signs = []
            for left_sign in signs:
                for right_sign in signs:
                    self.signs.append(left_sign * right_sign)

            self._initialized = True


def _hoeffding_d_kernel(X: np.ndarray) -> np.ndarray:
    """Computes the lessness Hoeffding D kernel for the given data.

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    T: np.ndarray
        A n x n x p tensor representing the Hoeffding D kernel.
        T[i, j, p] = 1(X[i, p] > X[j, p]) for all i, j in {0, ..., n-1}
    and p in {0, ..., p-1}.
    """

    Xi = X[:, np.newaxis, :]
    Xj = X[np.newaxis, :, :]
    T = (Xi > Xj).astype(int)
    return T


def hoeffding_d(X: np.ndarray) -> np.ndarray:
    """
    Computes the Hoeffding's D for the given data.

    Parameters
    ----------
    X : np.ndarray
        A n x p matrix of data points, where n is the number of samples
        and p is the number of features.

    Returns
    -------
    np.ndarray
        A p x p matrix representing the Hoeffding's D.
    """
    exprs = _HoeffdingDExpressions().expressions
    signs = _HoeffdingDExpressions().signs
    T = _hoeffding_d_kernel(X)
    tensors = [T] * 8
    n, _, p = T.shape
    result = get_zeros_backend()((p, p))

    for expr, sign in zip(exprs, signs):
        t_result = get_zeros_backend()((p, p))
        partitions = get_all_partitions(expr.contracted_indices)
        for partition in partitions:
            subexpr = expr.subexpression(partition)
            weight = partition_weight(partition)
            t_result += weight * get_einsum_backend()(subexpr.formula, *tensors)
        result = sign * t_result

    result = result / get_prod_backend()(range(n, n - 5, -1))
    return result
