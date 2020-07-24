 min J(x, x') <=> y_i = sum(beta_j*x_ij) for i = 0...n and j = 0...p with x_i0 = 1.
beta

Switch to a space "of redescription" (equivalent to initial space):
1. replace x by phi(x)
2. J(x, x') is now J(phi(x), phi(x'))
   => J(x, x') was a function of <x, x'>, so, J(phi(x), phi(x')) is now a function of <phi(x), phi(x')>
                                                                                      \_______________/
                                                                                        kernel(x, x')

linear classification: the kernel is a bilinear function (= linear combination of linear terms).
nonlinear classification: the kernel is not a bilinear function (= combination of nonlinear terms - quadratic, ...).
