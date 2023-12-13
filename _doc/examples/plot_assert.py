"""
Time to check a condition
=========================

Case 0: string length and print
+++++++++++++++++++++++++++++++
"""
import pandas
from onnx_extended.ext_test_case import measure_time

data = []
for i in range(1000, 10001, 1000):
    info = list([a + 0.5 for a in range(i)])
    obs = measure_time(lambda f=info: f"{f}", number=10)
    obs["i"] = i
    data.append(obs)

df = pandas.DataFrame(data)
print(df)
ax = (
    df[["i", "average"]]
    .set_index("i")
    .plot(title="Time to format a list based on its length")
)
ax.get_figure().savefig("plot_assert_print.png")


#################################
# Case 1: if then
# +++++++++++++++


info = list([a + 0.5 for a in range(10000000)])
data = []


def case1(a, b):
    if a == b:
        raise RuntimeError(f"{a} != {b}, {info}")
    return a - b


def call1(n, a, b):
    for i in range(n):
        case1(a, b)


data.append(measure_time(lambda: call1(1000, 4, 5), repeat=100, number=200))
data[-1]["case"] = "if-then"
print(data[-1])

#################################
# Case 2: assert
# ++++++++++++++


def case2(a, b):
    assert a != b, f"{a} != {b}, {info}"
    return a - b


def call2(n, a, b):
    for i in range(n):
        case2(a, b)


data.append(measure_time(lambda: call2(1000, 4, 5), repeat=100, number=200))
data[-1]["case"] = "assert"
print(data[-1])


#################################
# Case 3: enforce
# +++++++++++++++


def enforce(cond, msg, exc):
    if not cond:
        raise exc(msg())


def case3(a, b):
    enforce(a != b, lambda: f"{a} != {b}, {info}", RuntimeError)
    return a - b


def call3(n, a, b):
    for i in range(n):
        case3(a, b)


data.append(measure_time(lambda: call3(1000, 4, 5), repeat=100, number=200))
data[-1]["case"] = "enforce"
print(data[-1])

#################################
# Case 4: no check
# ++++++++++++++++


def case4(a, b):
    return a - b


def call4(n, a, b):
    for i in range(n):
        case4(a, b)


data.append(measure_time(lambda: call4(1000, 4, 5), repeat=100, number=200))
data[-1]["case"] = "none"
print(data[-1])

###############################
# Conclusion
# ++++++++++
#
# The first case (if-then) is fast but leaves a line not covered by unit tests
# if the exception is not checked. The second case is as fast as the first one.
# The message seems to be evaluated only the condition is wrong and the code coverage
# looks goot too. The last (enforce) is much slowed. So, if an exception is
# not covered by a unit test, `assert` should be used.

df = pandas.DataFrame(data)
print(df)
ax = (
    df[["case", "average"]]
    .set_index("case")
    .plot.barh(title="Comparison of assert time")
)
ax.get_figure().savefig("plot_assert.png")
