clc
clear

% Symbolic parameters
syms mu_h b_SH T_MH beta b_IH b_AH T_HM chi
syms omega_h alpha d_h gamma_I gamma_A mu_m omega_m
syms Lambda_H NH0 Lambda_M NM0
syms SH_DFE VH_DFE NH_DFE SM_DFE
syms a b c d e
% F matrix
F = [
    0, 0, 0, 0, b_SH*T_MH*(SH_DFE + beta*VH_DFE)/NH_DFE;
    0, 0, 0, 0, 0;
    0, 0, 0, 0, 0;
    0, b_IH*T_HM*SM_DFE/NH_DFE, ...
       b_AH*T_HM*chi*SM_DFE/NH_DFE, 0, 0;
    0, 0, 0, 0, 0
];
latex(F);
% V matrix
V = [
    omega_h + mu_h, 0, 0, 0, 0;
    -omega_h*alpha, d_h + mu_h + gamma_I, 0, 0, 0;
    -omega_h*(1-alpha), 0, mu_h + gamma_A, 0, 0;
    0, 0, 0, mu_m + omega_m, 0;
    0, 0, 0, -omega_m, mu_m
];
latex(inv(V));

ngm = F*inv(V)

NGM = [0, 0 ,0 ,a ,b;
       0, 0, 0, 0, 0;
       0, 0, 0, 0, 0;
       c, d, e, 0, 0;
       0 ,0 ,0, 0, 0];

R0 = max(eig(NGM));
disp(R0);
%% 
clc
clear
syms b1 b2 b3 u
syms g1 g2 g3
syms t12 t21 t13 t31 t32 t23
syms lambda
syms A1 A2 A3

F = [b1, 0, 0;
    0, b2, 0;
    0, 0, b3];

V = [A1, -t21, -t31;
    -t12, A2, -t32;
    -t13, -t23, A3];

N = F*inv(V);

charPoly = det(N-lambda*eye(size(N)));
charPoly = collect(expand(charPoly), lambda);
latex(charPoly)
%% 

clc
clear
syms b1 b2 b3 b4 b5
syms g1 g2 g3
syms t12 t13 t14 t15
syms t21 t23 t24 t25 
syms t31 t32 t34 t35
syms t41 t42 t43 t45
syms t51 t52 t53 t54
syms lambda
syms A1 A2 A3 A4 A5 

F = [b1, 0, 0, 0 ,0;
    0, b2, 0 ,0 0;
    0, 0, b3, 0, 0;
    0 ,0 ,0 ,b4, 0;
    0, 0, 0, 0, b5];

V = [A1, -t21, -t31, -t41, -t51;
    -t12, A2, -t32, -t42, -t52;
    -t13, -t23, A3, -t43, -t53;
    -t14, -t24, -t34, A4, -t54;
    -t15, -t25, -t35, -t45, A5];

N = F*inv(V);
charEq = det(F - lambda*V);
coeff = coeffs(expand(charEq), lambda);
a4 = coeff(5)