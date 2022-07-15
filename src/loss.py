from torch import tensor, lgamma, randn, inner, dot, prod, log, rand


def norm_constant(output):

    # Implementation according to : http://www2.stat.duke.edu/~st118/sta941/dirichlet.pdf

    """
    ouput: torch tensor of shpe b, 1. Ouptut of the final layer from the model
    """
    demoninator = lgamma(sum(output))
    nominator = prod(lgamma(output), 0)
    print(f"demoninator of the norm constant is {demoninator}")
    print(f"nominator of the norm constant is {nominator}")
    result = nominator / demoninator
    print(f"the normalisation constant is {result}")
    return result


def dirichlet_ll(output, target, epsilon=0.01):

    """
    ouput : torch tensor of shape b (batch_size), 1
    target: torch tensor of shape b (batch_size), 1
    """
    print(norm_constant(output))
    normalisation_constant = log(norm_constant(output))

    target = log((target + epsilon))

    print((output - 1).squeeze())
    print((target).squeeze())
    first_part = dot((output - 1).squeeze(), (target).squeeze())
    print(first_part)
    print(normalisation_constant)

    result = first_part - log(normalisation_constant)

    return result


if __name__ == "__main__":

    target = tensor([[0.25], [0.2], [0.4], []])
    print(target)
    output = rand(4, 1)
    print(output)

    print(
        dirichlet_ll(
            output,
            target,
        )
    )
