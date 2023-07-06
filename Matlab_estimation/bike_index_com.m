% functions
% % % % % % % % 
% m*A + 6*s + 2*l + 0*p = 5
% m*A + 6*s + 1*l + 0*p = 4
% m*A + 5*s + 2*l + 0*p = 4
% m*A + 5*s + 1*l + 0*p = 4
% m*A + 5*s + 0*l + 0*p = 3
% m*A + 3*s + 2*l + 1*p = 4
% m*A + 3*s + 2*l + 0*p = 3
%  m*A + 3*s + 2*l + 0*p = 3
% m*A + 3*s + 0*l + 0*p = 3   // evry wher which has 30 speed
% % % % % % % % % % % 

% m*B + 5*s + 2*l + 0*p = 4
% m*B + 3*s + 2*l + 0*p = 4
% m*B + 3*s + 1*l + 0*p = 3

% % % % % % % % % % 
% m*c + 3*s + 1*l + 0*p = 3
% m*c + 3*s + 2*l + 0*p = 4
% m*c + 3*s + 0*l + 0*p = 3

clear all; 
close all; 
clc;
%set up Unkown parameter 
syms K A B C s l p

% Observation vector
 L = [3;5;4;4;4;3;4;3;3;4;4;3;3;4;3];
 len_L=length(L);

% F_1 = m*A + 6*s + 2*l + 0*p 
% F_2 = m*A + 6*s + 1*l + 0*p 
% F_3 = m*A + 5*s + 2*l + 0*p 
% F_4 = m*A + 5*s + 1*l + 0*p 
% F_5 = m*A + 5*s + 0*l + 0*p 
% F_6 = m*A + 3*s + 2*l + 1*p 
% F_7 = m*A + 3*s + 0*l + 0*p 
% F_9 = m*B + 5*s + 2*l + 0*p
% F_10 = m*B + 3*s + 2*l + 0*p 
% F_11 = m*B + 3*s + 1*l + 0*p 
% F_12 = m*C + 3*s + 1*l + 0*p 
% F_13 = m*C + 3*s + 2*l + 0*p
% F_14 = m*C + 3*s + 0*l + 0*p
X = [K; A; B; C; s; l; p];
F_0 = K;
F_1 = A + 6*s + 2*l + 0*p ;
F_2 = A + 6*s + 1*l + 0*p ;
F_3 = A + 5*s + 2*l + 0*p ;
F_4 = A + 5*s + 1*l + 1*p ;
F_5 = A + 5*s + 1*l + 0*p ;
F_6 = A + 3*s + 2*l + 1*p ;
F_7 = A + 3*s + 2*l + 0*p ;
F_8 = A + 3*s + 1*l + 0*p ;
F_9 = B + 5*s + 2*l + 0*p;
F_10 = B + 3*s + 2*l + 0*p;
F_11 = B + 3*s + 1*l + 0*p ;
F_12 = C + 3*s + 1*l + 0*p ;
F_13 = C + 3*s + 2*l + 0*p;
F_14 = C + 3*s + 0*l + 0*p;

% 
J=jacobian([F_0;F_1;F_2;F_3;F_4;F_5;F_6;F_7;F_8;F_9;F_10; F_11; F_12; F_13; F_14],[K A B C s l p]);
A=eval(J);
% 
 Z=eye(14,7);
 N=A'*A;
 n=A'*L;
 X_est=inv(N)*n;
writematrix(X_est,'X_est.txt')

