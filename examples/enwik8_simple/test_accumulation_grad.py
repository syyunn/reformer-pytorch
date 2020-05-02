import torch


def test_backward_accumulates_gradients(inp, f1, f2, f1_mult, f2_mult):
    f1.backward()
    assert inp.grad.numpy()[0] == f1_mult

    f2.backward()
    assert inp.grad.numpy()[0] == f1_mult + f2_mult

    print("accumulation verified!")

if __name__ == "__main__":
    x = torch.ones(1, requires_grad=True)

    f_multiplier = 2
    g_multiplier = 3

    f = x ** f_multiplier
    g = x ** g_multiplier

    # f = x.detach() ** f_multiplier
    # g = x.detach() ** g_multiplier

    test_backward_accumulates_gradients(x, f, g, f_multiplier, g_multiplier)
    pass
