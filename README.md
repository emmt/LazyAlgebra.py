# Lazy linear operators for Python

`lazyalgebra` is a Python package designed for dealing with linear algebra and
linear operators. With this package, you can:

* Write readable code involving linear operators with a syntax similar to
  linear algebra equations. For example (and using an arrow `->` to denote the
  result of an operation):

  * `A*x` and `A(x)` yield the result of applying the operator `A` to the array `x`.

  * `B = alpha*A` and `B = A*alpha`, with `alpha` a scalar and `A` an operator,
     yield a *scaled* operator `B` such that `B*x -> alpha*(A*x)`.

  * `S = A + B + … + D` (for operators `A`, `B`, …, and `D`) yields an operator
    `S` whose behavior is the same as the *sum* `A + B + … + D`, that is to say
    `S*x -> A⋅x + B⋅x + … + D⋅x`.

  * `C = A * B * … * D` (for operators `A`, `B`, …, and `D`) yields an operator
    `C` whose behavior is the same as the *composition* `A ∘ B ∘ … ∘ D`, that
    is to say `C*x -> A⋅(B⋅…⋅(D*x))`.

  * `A / B` return an operator equivalent to `A*inv(B)`.

  * `A.adj()` of `adj(A)` yields the adjoint of `A`.

  * `A.inv()` of `inv(A)` yields the inverse of `A`.

* Build linear operators of arbitrary complexity by combining other operators.
  Operators can be scaled, added, subtracted, composed, or divided to build an
  other operators. During this construction, some simplifications are
  automatically performed. For example, with operator `A`, `A + A` yields
  `2*A`, `A*inv(A)` and `inv(A)*A` both yield the identity `Id`, `adj(adj(A))`
  and `inv(inv(A)` both yield `A`, `A*Id` and `Id*A` yield `A`, `Id/A` yields
  `inv(A)`, etc.

* Manipulate operators in a lazy way, i.e. without actually computing their
  coefficients (except diagonal operators). For example, `B = inv(A)` does not
  compute the inverse of `A` but yields a thin object `B` which compute `B*x`by
  calling the method `A.apply_inverse(x)`. In comparison, the result of `A*x`
  is computed by `A.apply(x)`. A similar trick is done for `adj(A)`, the
  adjoint of `A` and for sums and compositions of operators which are simplify
  stored into lightweight objects that memorize the structure and know how to
  apply the resulting operator to an input argument. These objects can be used
  themselves as the building-blocks of other constructions.


## Operators

Existing operators are:

* `Id` or `Identity()` is the universal identity. `Id*x -> x` whatever the shape
  of `x`.

* `A = Diag(w)` creates a diagonal operator with coefficients `w`. For any
  array `x` of suitable shape, `A*x -> w*x` the element-wise multiplication of
  `x` by `w`. Attribute `A.coefs` yields the coefficients (that is `w` here) of
  the operator.

* `D = Diff()` is the universal finite difference operator. `D*x -> y` such
   that `y[...,d]` is the finite differences of `x` along its `d`-th dimension.
   Hence, `y` has an additional trailing dimension compared to `x` and the
   leading dimensions of `y` are the same as those of of `x`.

* The construction `adj(D)*D` with `D = Diff()` is automatically simplified
  into a specially optimized Gram operator.

* `H = Conv(h)` creates a circulant convolution operator whose effect is to
  convole its argument by the *point spread function* (PSF) `h`. The PSF must
  have the same shape as the arguments of the operator `H` and must have its
  origin at index `0` as expected by the FFT.

* `S = Rank1(u)` and `R = Rank1(u, v)` yield a symmetric rank-1 and a rank-1
  operators such that `S*x -> (u'⋅x)⋅u` while `R*x -> (v'⋅x)⋅u` where `v'⋅x`
  denotes the scalar product of `v` by `x`.
