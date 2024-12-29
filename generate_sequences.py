from itertools import combinations


def generate_sequences(l, k):
    def helper(sequence, k_list):
        if not k_list:
            return [sequence]

        k1 = k_list[0]
        remaining_k = k_list[1:]
        results = []

        for indices in combinations(range(len(sequence)), k1):
            new_sequence = sequence.copy()
            for index in indices[1:]:
                new_sequence[index] = new_sequence[indices[0]]
            remaining_elements = [
                new_sequence[i] for i in range(len(new_sequence)) if i not in indices
            ]
            for result in helper(remaining_elements, remaining_k):
                for i, index in enumerate(indices):
                    result.insert(index, new_sequence[indices[0]])
                results.append(result)

        return results

    return helper(l, k)


l = [1, 2, 3, 4, 5, 6]
k = [2, 2, 2]
# k = [3, 3]

sequences = generate_sequences(l, k)
for seq in sequences:
    print(seq)
