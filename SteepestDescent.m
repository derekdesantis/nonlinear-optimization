function [x, iter] = SteepestDescent (A,b,x0 , maxIter , tol )
%Given a SPD matrix A, use Steepest Descent algorithm to solve Ax = b.
%A description of steepest descent as follows:
%We seek to minimize f(x) = .5 x' * A * x  - x' b
%A is SPD so the problem has a unique solution, call it x
%Given a guess x_i, the residual is r_i = b - A*x_i
%The residual is negative the gradient of f at x_i, and thus the direction to walk
%The residual r_i is orthogonal to r_j for i != j
%alpha_i is the distance we walk in the direction of r_i
%alpha_i can be computed to be dot(r_i,r_i)/dot(A r_i, r_i)
%It follows r_{i+1} = r_i - alpha_i *A*r_i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize variables
iter = 0;
x=x0;
r = b - A*x;


for i=1:maxIter
    if dot(r,r)<tol %Check to see if residual is under tolerance
        break
    else
	iter = iter+1; %Increase iteration count
    Ar = A*r; %Compute the one matrix\vector multiplication
    alpha = dot(r,r)/dot(Ar,r);
    x = x + alpha*r; %Find new direction and compute next iteration
    r = r - alpha* Ar; %Compute the next residual
    end
end