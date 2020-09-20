from numba import njit
from scipy import integrate
import matplotlib.pyplot as plt
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


def compute_drainage_function(rho, xi, sigma, b, truncate=np.inf):

    def integrand2(r0_vec):

        @njit
        def rho_offset(r_vec):
            return rho(np.linalg.norm(r_vec + r0_vec))

        @njit
        def integrand_spherical(r, phi, theta):
            r_vec = sphere_to_cart(r, phi, theta)
            rho_sum = rho_offset(np.zeros(3)) + rho_offset(np.linalg.norm(r_vec))
            inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
            return r * np.sin(theta) * rho_offset(r_vec) * np.exp(-r * inv_lambda_h)

        @njit
        def hydro_tensor(r_vec):
            rho_sum = rho_offset(np.zeros(3)) + rho_offset(np.linalg.norm(r_vec))
            inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
            dist = np.linalg.norm(r_vec)
            return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)

        # @njit
        # def integrand(x, y, z):
        #     r_vec = np.array([x, y, z])
        #     return 6*np.pi*sigma * rho_offset(r_vec)*hydro_tensor(r_vec)

        # def integrand_spherical(r, phi, theta):
        #     r_vec = sphere_to_cart(r, phi, theta)
        #     return r**2 * np.sin(theta) * integrand(*r_vec)

        def integrate_theta(r, phi):
            value = integrate.quad(lambda theta: integrand_spherical(r, phi, theta), 0, np.pi)[0]
            return value

        def integrate_phi(r):
            value = integrate.quad(lambda phi: integrate_theta(r, phi), 0, 2*np.pi)[0]
            return value

        if True:
            r_list = np.linspace(0, 50)
            int_list = np.empty(len(r_list))
            for i, r in enumerate(r_list):
                int_list[i] = integrate_phi(r)
            plt.figure()
            plt.plot(r_list, int_list)
            plt.show()

        # TODO: add points such that r is bounded

        bounds = sigma*np.arange(2, truncate, 50)
        total = 0.0
        for i in range(len(bounds) - 1):
            r_inner = bounds[i]
            r_outer = bounds[i+1]
            sum_part = integrate.quad(integrate_phi, r_inner, r_outer)[0]
            total += sum_part
            print("+{} = {} ({}, {})".format(sum_part, total, r_inner/sigma, r_outer/sigma))
        return total

    def drainage_function(r):
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
