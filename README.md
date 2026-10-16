# Lazy linear operators for Python

`lazyalgebra` is a Python package designed for dealing with linear algebra and
linear operators. Provided features are:

- Write readable code involving linear operators with a syntax similar to
  linear algebra equations.

- Build linear operators of arbitrary complexity by combining other operators.

- Manipulate operators in a lazy way, i.e. without computing their coefficients.

- Provide methods to apply the operators, their adjoint, and, if this makes
  sense, their inverse or their inverse-adjoint.

Operator objects implement the following behavior:

* `alpha*A` or `A*alpha` yields a *scaled* operator (`alpha` is a scalar and `A` is an operator).
* `A*x` and `A(x)` yields `A` applied to `x` (`A` is an operator and `x` is an array).
* `A*B*...` yields a *composition* operator (`A`, `B`, etc. are operators).
* `A / B` return an operator equivalent to `A*inv(B)`.
* `A + B + ...` return an equivalent operator.
* `A.adj()` of `adj(A)` take the adjoint of `A`.
* `A.inv()` of `inv(A)` yields the inverse of `A`.

Operators can be scaled, added, subtracted, composed, or divided to build an equivalent
operator. During this construction, some simplifications are automatically performed (e.g.
`A + A` yields `2*A`, `A*Id` yields `A` etc.).
