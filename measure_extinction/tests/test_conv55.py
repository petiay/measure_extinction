from measure_extinction.extdata import conv55toAv, conv55toRv


def test_conv55toAv():

    A55 = [4.84, 0.02]
    E4455 = [1.60, 0.02]
    Av = conv55toAv(A55, E4455)

    assert Av[0] == 4.7616
    assert Av[1] == 0.020023995605273192


def test_conv55toRv():

    R55 = [3.03, 0.04]
    Rv = conv55toRv(R55)

    assert Rv[0] == 3.10979
    assert Rv[1] == 0.0404
