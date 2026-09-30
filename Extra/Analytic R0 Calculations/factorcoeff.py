from sympy import symbols, sympify, factor, expand

lam10 = """A12*A21*te12*te21*ti12*ti21*um_1^2*um_2^2 - E12*E21*I12*I21*ta12*ta21*um_1^2*um_2^2 - A12*A21*I12*I21*te12*te21*um_1^2*um_2^2 + E12*E21*ta12*ta21*ti12*ti21*um_1^2*um_2^2 + I12*I21*ta12*ta21*te12*te21*um_1^2*um_2^2 + A12*A21*E12*E21*I12*I21*um_1^2*um_2^2 - ta12*ta21*te12*te21*ti12*ti21*um_1^2*um_2^2 - A12*A21*E12*E21*ti12*ti21*um_1^2*um_2^2 - A12*A21*E12*E21*om_1*ti12*ti21*um_1*um_2^2 - A12*A21*E12*E21*om_2*ti12*ti21*um_1^2*um_2 - A12*A21*I12*I21*om_1*te12*te21*um_1*um_2^2 - A12*A21*I12*I21*om_2*te12*te21*um_1^2*um_2 - E12*E21*I12*I21*om_1*ta12*ta21*um_1*um_2^2 - E12*E21*I12*I21*om_2*ta12*ta21*um_1^2*um_2 + A12*A21*om_1*te12*te21*ti12*ti21*um_1*um_2^2 + A12*A21*om_2*te12*te21*ti12*ti21*um_1^2*um_2 + E12*E21*om_1*ta12*ta21*ti12*ti21*um_1*um_2^2 + E12*E21*om_2*ta12*ta21*ti12*ti21*um_1^2*um_2 + I12*I21*om_1*ta12*ta21*te12*te21*um_1*um_2^2 + I12*I21*om_2*ta12*ta21*te12*te21*um_1^2*um_2 + A12*A21*E12*E21*I12*I21*om_1*um_1*um_2^2 + A12*A21*E12*E21*I12*I21*om_2*um_1^2*um_2 - om_1*ta12*ta21*te12*te21*ti12*ti21*um_1*um_2^2 - om_2*ta12*ta21*te12*te21*ti12*ti21*um_1^2*um_2 - A12*A21*E12*E21*om_1*om_2*ti12*ti21*um_1*um_2 - A12*A21*I12*I21*om_1*om_2*te12*te21*um_1*um_2 - E12*E21*I12*I21*om_1*om_2*ta12*ta21*um_1*um_2 + A12*A21*om_1*om_2*te12*te21*ti12*ti21*um_1*um_2 + E12*E21*om_1*om_2*ta12*ta21*ti12*ti21*um_1*um_2 + I12*I21*om_1*om_2*ta12*ta21*te12*te21*um_1*um_2 + A12*A21*E12*E21*I12*I21*om_1*om_2*um_1*um_2 - om_1*om_2*ta12*ta21*te12*te21*ti12*ti21*um_1*um_2"""

lam10_expr = sympify(lam10.replace('^', '**'))
print("l10")
print(factor(lam10_expr))
print()

lam8 = """A12*E12*F2*G2*bah_2*bsh_2*chi_2*oh_2*om_2*thm_2*ti12*ti21*tmh_2*um_1^2 - A12*A21*F2*G2*a_1*bih_2*bsh_2*oh_1*om_2*te21*thm_2*ti12*tmh_2*um_1^2 + E12*F2*G2*I12*a_2*bih_2*bsh_2*oh_2*om_2*ta12*ta21*thm_2*tmh_2*um_1^2 - F2*G2*I12*I21*bah_2*bsh_2*chi_2*oh_1*om_2*ta12*te21*thm_2*tmh_2*um_1^2 + F2*G2*a_1*bih_2*bsh_2*oh_1*om_2*ta12*ta21*te21*thm_2*ti12*tmh_2*um_1^2 + F2*G2*bah_2*bsh_2*chi_2*oh_1*om_2*ta12*te21*thm_2*ti12*ti21*tmh_2*um_1^2 - A12*A21*E12*F2*G2*I12*a_2*bih_2*bsh_2*oh_2*om_2*thm_2*tmh_2*um_1^2 - A12*E12*F2*G2*I12*I21*bah_2*bsh_2*chi_2*oh_2*om_2*thm_2*tmh_2*um_1^2 - A12*A21*F2*G2*a_1*bih_2*bsh_2*oh_1*om_1*om_2*te21*thm_2*ti12*tmh_2*um_1 + A12*E12*F2*G2*I12*I21*a_2*bah_2*bsh_2*chi_2*oh_2*om_2*thm_2*tmh_2*um_1^2 + A12*E12*F2*G2*bah_2*bsh_2*chi_2*oh_2*om_1*om_2*thm_2*ti12*ti21*tmh_2*um_1 + E12*F2*G2*I12*a_2*bih_2*bsh_2*oh_2*om_1*om_2*ta12*ta21*thm_2*tmh_2*um_1 - F2*G2*I12*I21*bah_2*bsh_2*chi_2*oh_1*om_1*om_2*ta12*te21*thm_2*tmh_2*um_1 - A12*E12*F2*G2*a_2*bah_2*bsh_2*chi_2*oh_2*om_2*thm_2*ti12*ti21*tmh_2*um_1^2 + F2*G2*I12*I21*a_1*bah_2*bsh_2*chi_2*oh_1*om_2*ta12*te21*thm_2*tmh_2*um_1^2 + F2*G2*a_1*bih_2*bsh_2*oh_1*om_1*om_2*ta12*ta21*te21*thm_2*ti12*tmh_2*um_1 + F2*G2*bah_2*bsh_2*chi_2*oh_1*om_1*om_2*ta12*te21*thm_2*ti12*ti21*tmh_2*um_1 - A12*A21*E12*F2*G2*I12*a_2*bih_2*bsh_2*oh_2*om_1*om_2*thm_2*tmh_2*um_1 - A12*E12*F2*G2*I12*I21*bah_2*bsh_2*chi_2*oh_2*om_1*om_2*thm_2*tmh_2*um_1 - F2*G2*a_1*bah_2*bsh_2*chi_2*oh_1*om_2*ta12*te21*thm_2*ti12*ti21*tmh_2*um_1^2 + A12*E12*F2*G2*I12*I21*a_2*bah_2*bsh_2*chi_2*oh_2*om_1*om_2*thm_2*tmh_2*um_1 - A12*E12*F2*G2*a_2*bah_2*bsh_2*chi_2*oh_2*om_1*om_2*thm_2*ti12*ti21*tmh_2*um_1 + F2*G2*I12*I21*a_1*bah_2*bsh_2*chi_2*oh_1*om_1*om_2*ta12*te21*thm_2*tmh_2*um_1 - F2*G2*a_1*bah_2*bsh_2*chi_2*oh_1*om_1*om_2*ta12*te21*thm_2*ti12*ti21*tmh_2*um_1"""

lam8_expr = sympify(lam8.replace('^', '**'))
print("l8")
print(factor(lam8_expr))
print()

A12, A21, E12, I12, I21 = symbols('A12 A21 E12 I12 I21')
ta12, ta21, te21, ti12, ti21 = symbols('ta12 ta21 te21 ti12 ti21')
a_1, a_2, bih_2, bah_2, chi_2, oh_1, oh_2 = symbols('a_1 a_2 bih_2 bah_2 chi_2 oh_1 oh_2')
F2, G2, bsh_2, om_1, om_2, thm_2, tmh_2, um_1 = symbols('F2 G2 bsh_2 om_1 om_2 thm_2 tmh_2 um_1')

X = A12*A21 - ta12*ta21          
Y = I12*I21 - ti12*ti21         

C = (X * bih_2 * (a_2*E12*I12*oh_2 + a_1*oh_1*te21*ti12)
     + Y * bah_2*chi_2 * (A12*E12*oh_2*(1 - a_2) + ta12*te21*oh_1*(1 - a_1)))

lam8_fac = -F2*G2*bsh_2*om_2*thm_2*tmh_2*um_1*(om_1 + um_1) * C

diff = expand(lam8_expr - lam8_fac)
assert diff == 0, "Factorisation does not match!"
print("Matches exactly!")
