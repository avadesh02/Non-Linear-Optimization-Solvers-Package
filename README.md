This repository is a non linear optimization solver package that contains an implementation of various constrained and unconstrained optimization solvers that are described in Nocedal's Numerical Optimization Books. This repository was developed as a part of the final project for the course - Convex and Non Smooth Optimization offered at Courant Institute of Mathematical Sciences by Prof. Overton. 

The repository contains the following optimization solvers. 
### Unconstrained Solvers :
1. Back tracking Line search with steepest descent
2. Line search that satisfies wolfe conditions with steepest descent
3. Conjugate gradient methods (FR, PR)
4. Quasi Newton Method (DFP, BFGS)

### Constrained Solvers :
1. Projected conjugate gradient method
2. Projected gradient method for convex problems
3. Non linear projected gradient method (trust region method)
4. Augmented Lagrangian Method (Bound Constrained Formulation)

This non linear solvers package uses pytorch to compute derivatives and uses Quasi Newton methods for hessian approximations. The theory for each solver can be obtained in Nocedal's Numerical Optimization Book. 

Inside the demo folder there exist a simple example that shows how each solver can be used along with results obtained from the solver when minimizing a simple objective function. 

Please cite this package if used for any research purposes. 

**Author**: Avadesh Meduri 
