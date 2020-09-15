from numba import njit
from scipy import integrate
import numpy as np


def compute_viscosity(N, rho, sigma, a, b, vm):
    xi = compute_drag_function(rho, sigma, a)
    kappa = compute_drainage_function(rho, xi, sigma, b)
    integrand1 = lambda r: rho(r)*kappa(r)**2*r**4
    integrand2 = lambda r: (1 - vm*rho(r))*xi(r)*4*np.pi*r**2
    eta = 4*np.pi**2*sigma/N*integrate.quad(integrand1, 0, np.inf) + 2.5*(vm + 1/N*integrate.quad(integrand2, 0, np.inf))


# ================= #
# DRAINAGE FUNCTION #
# ================= #

@njit
def sphere_to_cart(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# def hydro_tensor(r_vec, r0_vec, rho, b):
#     rho_sum = rho(np.linalg.norm(r0_vec)) + rho(np.linalg.norm(r_vec))
#     inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
#     dist = np.linalg.norm(np.array(r0_vec) - np.array(r_vec))
#     return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)
#
#
# def integrand_drainage(r_vec, r0_vec, rho, sigma, b):
#     return 6*np.pi*sigma * rho(r_vec)*hydro_tensor(r_vec, r0_vec, rho, b)


def compute_drainage_function(rho, xi, sigma, b, truncate=np.inf):

    def integrand2(r0_vec):
        x0, y0, z0 = r0_vec

        cartesian = False

        if cartesian:

            @njit
            def hydro_tensor(r_vec):
                rho_sum = rho(np.linalg.norm(r0_vec)) + rho(np.linalg.norm(r_vec))
                inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
                dist = np.linalg.norm(r0_vec - r_vec)
                return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)

            @njit
            def integrand(x, y, z):
                r_vec = np.array([x, y, z])
                return 6*np.pi*sigma * rho(np.linalg.norm(r_vec))*hydro_tensor(r_vec)

            # returns dictionaries of boundaries
            def create_bounds(lwr_bnd, upp_bnd):

                def y_bnd(bnd, octant):
                    if octant in [0, 1, 6, 7]:
                        sign = 1
                    else:
                        sign = -1

                    def bnd_fun(x):
                        if np.abs(x - x0) < bnd:
                            return sign*np.sqrt(bnd ** 2 - (x - x0) ** 2)
                        return y0
                    return bnd_fun

                def z_bnd(bnd, octant):
                    if octant in [0, 1, 2, 3]:
                        sign = 1
                    else:
                        sign = -1

                    def bnd_fun(x, y):
                        rsqd = (x - x0) ** 2 + (y - y0) ** 2
                        if rsqd < bnd ** 2:
                            return sign*np.sqrt(bnd ** 2 - rsqd)
                        return z0
                    return bnd_fun

                return {0: (x0, x0 + upp_bnd, y_bnd(lwr_bnd, 0), y_bnd(upp_bnd, 0), z_bnd(lwr_bnd, 0), z_bnd(upp_bnd, 0)),
                        1: (x0 - upp_bnd, x0, y_bnd(lwr_bnd, 1), y_bnd(upp_bnd, 1), z_bnd(lwr_bnd, 1), z_bnd(upp_bnd, 1)),
                        2: (x0 - upp_bnd, x0, y_bnd(upp_bnd, 2), y_bnd(lwr_bnd, 2), z_bnd(lwr_bnd, 2), z_bnd(upp_bnd, 2)),
                        3: (x0, x0 + upp_bnd, y_bnd(upp_bnd, 3), y_bnd(lwr_bnd, 3), z_bnd(lwr_bnd, 3), z_bnd(upp_bnd, 3)),
                        4: (x0, x0 + upp_bnd, y_bnd(upp_bnd, 4), y_bnd(lwr_bnd, 4), z_bnd(upp_bnd, 4), z_bnd(lwr_bnd, 4)),
                        5: (x0 - upp_bnd, x0, y_bnd(upp_bnd, 5), y_bnd(lwr_bnd, 5), z_bnd(upp_bnd, 5), z_bnd(lwr_bnd, 5)),
                        6: (x0 - upp_bnd, x0, y_bnd(lwr_bnd, 6), y_bnd(upp_bnd, 6), z_bnd(upp_bnd, 6), z_bnd(lwr_bnd, 6)),
                        7: (x0, x0 + upp_bnd, y_bnd(lwr_bnd, 7), y_bnd(upp_bnd, 7), z_bnd(upp_bnd, 7), z_bnd(lwr_bnd, 7))}

            def compute_integral(lwr_bnd, upp_bnd):
                bound_dict = create_bounds(lwr_bnd, upp_bnd)
                total = 0.0
                for bounds in bound_dict.values():
                    total += integrate.tplquad(integrand, *bounds)[0]
                return total

            # bound_list = [2*sigma, 4*sigma, 6*sigma, 10*sigma, np.inf]
            bound_list = [2*sigma, 4*sigma]
            total = 0.0
            for i in range(len(bound_list) - 1):
                sum_part = compute_integral(bound_list[i], bound_list[i+1])
                print(sum_part)
                total += sum_part

            return total

            # # boundaries for each octant
            # bound_dict = {0: (0, truncate, y_lower, truncate, z_lower, truncate),
            #               1: (-truncate, 0, y_lower, truncate, z_lower, truncate),
            #               2: (-truncate, 0, -truncate, y_upper, z_lower, truncate),
            #               3: (0, truncate, -truncate, y_upper, z_lower, truncate),
            #               4: (0, truncate, -truncate, y_upper, -truncate, z_upper),
            #               5: (-truncate, 0, -truncate, y_upper, -truncate, z_upper),
            #               6: (-truncate, 0, y_lower, truncate, -truncate, z_upper),
            #               7: (0, truncate, y_lower, truncate, -truncate, z_upper)}

            # # compute integral
            # total = 0.0
            # for bounds in bound_dict.values():
            #     total += integrate.tplquad(integrand, *bounds)[0]
            # return total

        else:

            @njit
            def rho_offset(r_vec):
                return rho(np.linalg.norm(r_vec + r0_vec))

            @njit
            def hydro_tensor(r_vec):
                rho_sum = rho_offset(np.zeros(3)) + rho_offset(np.linalg.norm(r_vec))
                inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
                dist = np.linalg.norm(r_vec)
                return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)

            @njit
            def integrand(x, y, z):
                r_vec = np.array([x, y, z])
                return 6*np.pi*sigma * rho_offset(r_vec)*hydro_tensor(r_vec)

            integrand_spherical = lambda r, phi, theta: r**2 * np.sin(theta) * integrand(*sphere_to_cart(r, phi, theta))
            # bounds = [2 * sigma, 4 * sigma, 6 * sigma, 8 * sigma, 10 * sigma, 20 * sigma, truncate]
            bounds = sigma*np.arange(2, truncate, 2)
            total = 0.0
            for i in range(len(bounds) - 1):
                sum_part, err = integrate.tplquad(integrand_spherical, bounds[i], bounds[i+1], 0, 2*np.pi, 0, np.pi)
                print(sum_part, err)
                total += sum_part
            return total

    def drainage_function(r):

        # return (1 - xi(r)) / \
        #        (1 - integrate.dblquad(lambda phi, theta: integrand2(sphere_to_cart(r, phi, theta))*np.sin(theta),
        #         0, 2*np.pi, lambda phi: 0, lambda phi: np.pi)[0])
        drag = xi(r)
        integral = integrand2(np.array([0, 0, r]))
        print("drag={}, integral={}".format(drag, integral))
        return (1 - drag) / \
               (1 + 4*np.pi*integral)

    return drainage_function


def compute_drag_function(rho, sigma, a):
    s = np.pi*(sigma + a)**2

    def drag_function(r):
        return 1 - np.exp(-s*integrate.quad(rho, r, np.inf)[0])

    return drag_function


def rho_gaussian(N, Rg2):
    @njit
    def function(r):
        if r < 10 * np.sqrt(Rg2):
            return N * (3/(2*np.pi*Rg2))**(3./2.) * np.exp(-3*r**2/(2*Rg2))
        return 0
    return function
