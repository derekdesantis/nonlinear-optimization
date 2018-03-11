function [x, iter] = conjgrad(A,b,x0 , maxIter , tol )
%Given a SPD matrix A, use Conjugate Gradient  algorithm to solve Ax = b.
%A description of conjugate gradient is as follows:

%Two vectors u,v are conjugate with respect to A if u'*(A*v) = 0
%Since A is SPD, the above defines a new inner product
%If p_1, ... p_n are mutually conjugate, then they form an orthogonal basis for R^n with respect to the A-inner product
%Solving Ax = b with respect to this inner product is easy: x = sum_{i=1}^n dot(p_i,b)/dot(A*p_i, p_i) p_i
%x can be computed iteratively, as in file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Set variables.  For the first iteration, we are using the residual as the
%initial search direction. 
iter = 0;
x = x0;
r =  b-A*x;
p = r;


for i=1:maxIter
    if dot(r,r)<tol %Check to see if residual is under tol
        iter = i;
        break
    else
	iter = iter+1;
        Ap = A*p;
        alpha = dot(r,r)/dot(Ap,p);  %Distance to move
        x = x+alpha*p; %Move x
        rnew = r - alpha*Ap; %Define a new residual to be used to compute a new p.  We will need a distance to walk, beta
        beta = dot(rnew,rnew)/dot(r,r);
        r = rnew; %Overwrite old r with new r
        p = r + beta*p;
    end
end
