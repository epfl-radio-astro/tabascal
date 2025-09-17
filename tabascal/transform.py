def affine_transform_full(x, L, mu):
    return L @ x + mu


def affine_transform_full_inv(x, L_inv, mu):
    return L_inv @ (x - mu)


def affine_transform_diag(x, sigma, mu):
    return sigma * x + mu


def affine_transform_diag_inv(x, sigma_inv, mu):
    return sigma_inv * (x - mu)
