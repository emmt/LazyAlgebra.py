# Notes about the implementation of `lazyalgebra` in Python

## Simplification rules

### First stage rules

A few simple simplifications are automatically done when processing simple
expressions involving an operator `A`:

* Unary plus: `+A -> A` for all operators (unless `__pos__` is overwritten by a
  sub-class).

* Unary minus: `-A -> (-1)*A`, this is achieved by calling `A.scaled_by(-1)` so
  only this method needs to be overwritten if needed.

* Call: `A(x) -> A.apply(x)` by providing the `__call_` method.

* Addition: `A + B -> add(A, B)`, with `B` another operator, yields an operator
  implementing the effects of the sum of operators `A + B`. There may be any
  number of terms in a sum. The private `add` function (described below) is
  responsible for further simplifications (e.g. `A + 0*B -> A` and `A + B + A
  -> 2*A + B`). Addition of an operator and a non-operator is considered as an
  error.

* Subtraction: `A - B -> add(A, -B)`.

* Multiplication, there are 3 possible cases (dealt with by `__mul__` and
  `__rmul__`):

  * `alpha * A` (or `A * alpha`) with a scalar `alpha` yields a *scaled*
    operator if `alpha != 1` and yields `A` otherwise. Multiplying a *scaled*
    operator by another scalars is optimized.

  * `A * x -> A.apply(x)` with `x` a (multi-)dimensional array yields the
    result of applying the operator `A` to the *vector* `x`.

  * `A * B -> compose(A, B)`, with `B` another operator, yields an operator
     implementing the effects of the composition of operators `A ∘ B`. There
     may be any number of terms in a composition. The private `compose`
     function (described below) is responsible for further simplifications
     (`A*Id -> A`, `Id/A > inv(A)`, etc).

* Division, there are 2 possible cases (dealt with by `__truediv__` and
  `__rtruediv__`):

  * `A / alpha -> (1/alpha) * A` with `alpha` a scalar multiplier. Note that
    `alpha / A` is considered as an error, write `alpha * Id / A` or `alpha *
    inv(A)` if that correspond to your intention.

  * `A / B -> A * inv(B) -> compose(A, inv(B))` with is `B` another operator.

* Adjoint by `adj(A)` of `A.adj()` and inverse by `inv(A)` or `A.inv()` may
  perform some simplification (like `adj(adj(A)) -> A` and `inv(inv(A)) -> A`)
  and permutations (i.e. `adj(inv(A)) -> inv(adj(A))`) so that `Adjoint` and
  `Inverse` *decorators* can only appear in the inverse-adjoint order.

### Simplification of sums

Private function `lazyalgebra.operators.add(A, B)` returns the operator that
implements an equivalent, possibly simplified, of the addition `A + B`. This
involves the following stages:

1. All terms of the sum are collected (`A` and `B` may also be sums
   themselves) as `SumTerm` instances. This yields a list `S` of terms.
   `SumTerm` objects are equivalent to an operator but in a form that
   is more suitable for simplifying a sum of operators.

2. The terms of the sum are sorted in a specific order so that
   operators that may be identical (as checked by the `==` operator
   based on the `__eq__` method) are grouped together.

3. Identical terms are reduced (by adding their multipliers) and null operators
   (those with a zero multiplier) are deleted (except that the first null
   operator is preserved if there are no non-null operators).
   Depending on the number of remaining terms, a single (possibly null) operator
   or a `Sum` instance is returned.


### Simplification of products

Private function `lazyalgebra.operators.compose(A, B)` returns the operator
that implements an equivalent, possibly simplified, of the composition `A ∘ B`.
This involves the following stages:

1. All terms of the composition are collected (`A` and `B` may also be
   compositions themselves) in order, factorizing the multipliers of *scaled*
   operators appearing in this collection and suppressing all identities. This
   yields a list `C` of collected terms.

2. For each operator `C[i]` of the collected list, the method
   `C[i].try_simplify_prod(C, i)` is called to attempt to simplify the
   composition around the `i`-th term in `C`. This method shall return `None`
   if no simplification can be done. Otherwise, if a simplification is
   possible, `try_simplify_prod` shall return the result of `simplify_prod(C,
   j, n, ...)` with `j` and `n` the index of the first term and the number of
   terms to replace in `C` by the result of the simplification represented by
   subsequent arguments `...` which may be an optional scalar multiplier
   `alpha`, and operator(s). The global multiplier is multiplied by `alpha` if
   specified, and the terms `C[j:j+n]` are replaced by the returned operator(s)
   (there may be none). Note that `j` may be different from `i`, For example `j
   = i - 1`, if the preceding term simplifies with the `i`-th one. Each time a
   simplification is successful, the process is restarted at the term
   immediately preceding the first modified one. (Because it is assumed that
   simplifications concerning a term involve at most one of its predecessor.)

3. Depending on the number of collected terms, return the identity (no terms
   collected), the first term (a single term collected), or an instance of
   `Product` (two or more terms collected) possibly scaled by the multiplier
   factorized during the collection.

Some operator classes implement trivial simplification via their `__mul__` and
`__rmul__` method. To alleviate the work of the `compose` function. This is for
example the case of the multiplication of an operator by the identity `Id`. Or
the multiplication of an operator by its inverse. However such simplifications
rules shall also be implemented for `try_simplify_prod` because, e.g., it may
happen that one operator and its inverse become direct neighbors after other
simplifications.

## Benchmark

```python
import timeit
n=1000000;timeit.timeit("lazyalgebra.utils.adapt_multiplier(np.float64, 1)", number=n,setup="import numpy as np; import lazyalgebra")/n
```

yields 1.5 μs

```python
n=10000;timeit.timeit("np.multiply(a, 1)", number=n, setup="import numpy as np; a = np.ones((1000,1000),np.float32)")/n
   ```

takes 300 μs
