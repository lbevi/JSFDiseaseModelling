clc
clear

% Calculate DFE
syms S1 S2 V1 V2 
syms LambdaH N1_0 N2_0 muH nu phi tS12 tS21 tV12 tV21 beta
 
Lambda1 = LambdaH*N1_0;
Lambda2 = LambdaH*N2_0;
 
eq1 = Lambda1 + phi*V1 - (muH+nu+tS12)*S1 + tS21*S2 == 0;
eq2 = Lambda2 + phi*V2 - (muH+nu+tS21)*S2 + tS12*S1 == 0;
eq3 = nu*S1 - (muH+phi+tV12)*V1 + tV21*V2 == 0;
eq4 = nu*S2 - (muH+phi+tV21)*V2 + tV12*V1 == 0;
 
sol = solve([eq1 eq2 eq3 eq4], [S1 S2 V1 V2]);
 
S1_sym = simplify(sol.S1);
S2_sym = simplify(sol.S2);
V1_sym = simplify(sol.V1);
V2_sym = simplify(sol.V2);
 
% disp('--- Symbolic equilibrium solution ---')
% disp('S1 ='); pretty(S1_sym)
% disp('S2 ='); pretty(S2_sym)
% disp('V1 ='); pretty(V1_sym)
% disp('V2 ='); pretty(V2_sym)

Q1_sym = simplify((S1_sym + beta*V1_sym)/(S1_sym+V1_sym));
Q2_sym = simplify((S2_sym + beta*V2_sym)/(S2_sym+V2_sym));

disp(' ')
disp('First fraction')
latex(Q1_sym)
disp('Second fraction')
latex(Q2_sym)
%% NGM
clc
clear

syms bsh_1 tmh_1 N1DFE N2DFE bih_1 thm_1 chi_1 bah_1 lambda
syms bsh_2 tmh_2 N1DFE N2DFE bih_2 thm_2 chi_2 bah_2
syms oh_1 uh_1 a_1 dh_1 gI_1 om_1 um_1 gA_1
syms oh_2 uh_2 a_2 dh_2 gI_2 om_2 um_2 gA_2
syms te12 te21 ti12 ti21 ta12 ta21
syms E12 E21 I12 I21 A12 A21
syms F1 F2 G1 G2

F= sym(zeros(10,10));
F(1,5) = bsh_1*tmh_1*F1;
F(6,10) = bsh_2*tmh_2*F2;
F(4,2) = bih_1*thm_1*0; %G1 = 0
F(9,7) = bih_2*thm_2*G2;
F(4,3) = bah_1*chi_1*thm_1*0;
F(9,8) = bah_2*chi_2*thm_2*G2;

V = sym(zeros(10,10));
V(1,1) = E12;
V(6,6) = E21;
V(1,6) = -te21;
V(6,1) = -te12;
V(2,1)=-oh_1*a_1;
V(7,6)=-oh_2*a_2;
V(2,2)= I12;
V(7,7)= I21;
V(2,7)=-ti21;
V(7,2)=-ti12;
V(3,1)=-oh_1*(1-a_1);
V(8,6)=-oh_2*(1-a_2);
V(3,3)=A12;
V(8,8)=A21;
V(3,8)=-ta21;
V(8,3)=-ta12;
V(4,4)=om_1+um_1;
V(9,9)=om_2+um_2;
V(5,4)=-om_1;
V(10,9)=-om_2;
V(5,5)=um_1;
V(10,10)=um_2;

charEq = det(F - lambda*V);
charEq = expand(charEq); 

c = coeffs(charEq, lambda, 'All');
degree = length(c) - 1;

fid = fopen('coefficients.txt', 'w');
for k = 1:length(c)
    fprintf(fid, 'Coefficient of lambda^%d:\n%s\n\n', degree-(k-1), char(c(k)));
end
fclose(fid);