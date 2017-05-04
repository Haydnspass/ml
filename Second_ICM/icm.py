import numpy
import copy

def iterated_conditional_modes(unaries, beta, labels=None):
    def phiP(v1, v2, beta = beta):
        if v1 == v2:
            return 0
        else:
            return beta

    shape = unaries.shape[0:2]
    n_labels = unaries.shape[2]

    if labels is None:
        labels = numpy.argmin(unaries, axis=2)
        originalLabel = copy.deepcopy(labels)
        ax1.imshow(labels, cmap = 'gray')

    continue_search = True
    loopIndex = 0

    while (continue_search):
        loopIndex += 1
        continue_search = False

        for x0 in range(1, shape[0] - 1):
            for x1 in range(1, shape[1] - 1):

                current_label = labels[x0, x1]
                min_energy = float('inf')
                best_label = None

                for l in range(n_labels):

                    # evaluate cost
                    energy = 0.0

                    # unary terms
                    energy += unaries[x0, x1, l]

                    # pairwise terms
                    energy += phiP(l, labels[x0 - 1, x1]) + \
                              phiP(l, labels[x0 + 1, x1]) + \
                              phiP(l, labels[x0, x1 - 1]) + \
                              phiP(l, labels[x0, x1 + 1])

                    if energy < min_energy:
                        min_energy = energy
                        best_label = l

                if best_label != current_label:
                    labels[x0, x1] = best_label
                    continue_search = True

    return labels, originalLabel, loopIndex


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))

    shape = [100, 100]
    n_labels = 2
    # unaries
    unaries = numpy.random.rand(shape[0], shape[1], n_labels)
    # regularizer strength
    beta = 1
    labels, originalLabel, totalLoops = iterated_conditional_modes(unaries, beta=beta)
    print(totalLoops)

    ax2.imshow(labels, cmap = 'gray')

    ax3.imshow(abs(labels - originalLabel), cmap = 'gray')
    ax1.set_title('data')
    ax2.set_title('mrf')
    ax3.set_title('difference')
    plt.show()
