import numpy as np
from computable_sequence import ComputableSequence


#####
def unique_index_indices(self):
    def unique_element_indices(lst):
        element_count = {}
        unique_indices = []

        for index, element in enumerate(lst):
            if element in element_count:
                element_count[element][0] += 1
            else:
                element_count[element] = [1, index]

        for element, (count, index) in element_count.items():
            if count == 1:
                unique_indices.append(index)

        return unique_indices

    return unique_element_indices(self._index_sequence)


def repeated_pair_indices(self):  # TODO
    def repeated_element_indices(lst):
        element_count = {}
        repeated_indices = []

        for index, element in enumerate(lst):
            if element in element_count:
                element_count[element][0] += 1
                element_count[element].append(index)
            else:
                element_count[element] = [1, index]

        for element, (count, *indices) in element_count.items():
            if count > 1:
                repeated_indices.extend(indices)

        return repeated_indices

    return repeated_element_indices(self._pair_sequence)


#####
def V_statisitics(matrix_list: list, computing_sequence: ComputableSequence):
    if len(matrix_list) != len(computing_sequence.pair_sequence):
        raise ValueError(
            "The length of the matrix list must be the same as the pair sequence"
        )
    if len(matrix_list) == 1:
        if (
            computing_sequence.pair_sequence[0][0]
            == computing_sequence.pair_sequence[0][1]
        ):
            return np.sum(np.diag(matrix_list[0]))
        else:
            return np.sum(matrix_list[0])
    else:
        ### check if there are unique indices
        if unique_indices := computing_sequence.unique_index_indices():
            for indice in unique_indices:
                if (
                    neighbour_indice := computing_sequence.neighbour_indice(indice)
                ) in unique_indices:
                    return np.sum(
                        np.diag(matrix_list[computing_sequence.pair_indice(indice)[0]])
                    ) * V_statisitics(
                        matrix_list.pop(computing_sequence.pair_indice(indice)[0]),
                        computing_sequence.pop(
                            computing_sequence.pair_indice(indice)[0]
                        ),
                    )
                else:
                    sum_vector = np.sum(
                        matrix_list[computing_sequence.pair_indice(indice)[0]],
                        axis=computing_sequence.pair_indice(indice)[1],
                    )
                    matrix_list.pop(computing_sequence.pair_indice(indice)[0])
                    for k in range(0, len(computing_sequence.pair_sequence)):
                        if computing_sequence.index((k, 0)) == computing_sequence.index(
                            neighbour_indice
                        ):
                            matrix_list[k] = sum_vector * matrix_list[k]
                            return V_statisitics(
                                matrix_list,
                                computing_sequence.pop(k),
                            )
                        elif computing_sequence.index(
                            (k, 1)
                        ) == computing_sequence.index(neighbour_indice):
                            matrix_list[k] = matrix_list[k] * sum_vector
                            return V_statisitics(
                                matrix_list,
                                computing_sequence.pop(k),
                            )
        ### check if there are repeated pairs
        for pair_indice in range(0, len(computing_sequence.pair_sequence)):
            for compared_indice in range(
                pair_indice + 1, len(computing_sequence.pair_sequence)
            ):
                if set(now_pair := computing_sequence.pair(pair_indice)) == set(
                    compared_pair := computing_sequence.pair(compared_indice)
                ):
                    if now_pair == compared_pair:
                        matrix_list[pair_indice] = (
                            matrix_list[pair_indice] * matrix_list[compared_indice]
                        )
                        return V_statisitics(
                            matrix_list.pop(compared_indice),
                            computing_sequence.pop(compared_indice),
                        )
                    else:
                        matrix_list[pair_indice] = matrix_list[
                            pair_indice
                        ] * np.transpose(matrix_list[compared_indice])
                        return V_statisitics(
                            matrix_list.pop(compared_indice),
                            computing_sequence.pop(compared_indice),
                        )

        ### check if there are pairs with single same index
        # such as (1, 2) and (1, 3) or (1, 2) and (3, 1)
