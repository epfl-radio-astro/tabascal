import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional
from tabascal.timing import measure_runtime

import jax.numpy as jnp

import os


plt.rcParams["font.size"] = 18


def time_units(times: NDArray) -> tuple:
    """Scale the time axis to hours, minutes or seconds depending on the total range.

    Parameters
    ----------
    times : ArrayLike
        Times to consider.

    Returns
    -------
    tuple
        Rescaled times array and the scale unit as a string.
    """

    time_range = times[-1] - times[0]
    times = times - times[0]
    if time_range > 3600:
        units = "hr"
        times = times / 3600
    elif time_range > 60:
        units = "min"
        times = times / 60
    else:
        units = "s"

    return times, units


def plot_comparison(
    ax, times, mean1, mean2, std1, std2, true1, true2, rmse, diff=False
):

    times, units = time_units(times)
    for i, a in enumerate(ax):
        a[0].plot(rmse[..., i], "o")
        a[0].set_xlabel("Sample")

        if diff:
            a[1].plot(times, mean1[i] - true1[i].real, label="Estimate")
            a[1].fill_between(times, -std1[i], std1[i], color="tab:orange", alpha=0.3)
            a[1].fill_between(
                times, -2 * std1[i], 2 * std1[i], color="tab:orange", alpha=0.3
            )
            a[2].plot(times, mean2[i] - true2[i], label="Estimate")
            a[2].fill_between(times, -std2[i], std2[i], color="tab:orange", alpha=0.3)
            a[2].fill_between(
                times, -2 * std2[i], 2 * std2[i], color="tab:orange", alpha=0.3
            )
        else:
            a[1].plot(times, true1[i], label="True")
            a[1].plot(times, mean1[i], label="Estimate")
            a[1].fill_between(
                times,
                mean1[i] - std1[i],
                mean1[i] + std1[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[1].fill_between(
                times,
                mean1[i] - 2 * std1[i],
                mean1[i] + 2 * std1[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[2].plot(times, true2[i], label="True")
            a[2].plot(times, mean2[i], label="Estimate")
            a[2].fill_between(
                times,
                mean2[i] - std2[i],
                mean2[i] + std2[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[2].fill_between(
                times,
                mean2[i] - 2 * std2[i],
                mean2[i] + 2 * std2[i],
                color="tab:orange",
                alpha=0.3,
            )

        a[1].set_xlabel(f"Time [{units}]")
        a[1].legend()
        a[2].set_xlabel(f"Time [{units}]")
        a[2].legend()


def plot_complex_real_imag(
    times,
    param,
    true,
    rmse,
    name: str,
    save_name: Optional[str] = None,
    diff: bool = False,
    max_plots: int = 10,
    save_dir: str = "plots/",
):

    n_params = min(param.shape[1], max_plots)
    # idx = np.random.permutation(param.shape[1])
    # print(param.shape, true.shape)
    # param = param[:, idx]
    # true = true[idx]
    mean_r = param.real.mean(axis=0)
    mean_i = param.imag.mean(axis=0)
    std_r = param.real.std(axis=0)
    std_i = param.imag.std(axis=0)

    fig, ax = plt.subplots(n_params, 3, figsize=(18, 4.5 * n_params))

    ax[0, 0].set_title("Root Mean Squared Error")
    ax[0, 1].set_title(f"{name} Real")
    ax[0, 2].set_title(f"{name} Imag")

    plot_comparison(
        ax, times, mean_r, mean_i, std_r, std_i, true.real, true.imag, rmse, diff=diff
    )

    if save_name is not None:
        fig.savefig(
            os.path.join(save_dir, f"{save_name}_real_imag.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_complex_amp_phase(
    times,
    param,
    true,
    rmse,
    name: str,
    save_name: Optional[str] = None,
    diff: bool = False,
    max_plots: int = 10,
    save_dir: str = "plots/",
):
    n_params = min(param.shape[1], max_plots)
    # idx = np.random.permutation(param.shape[1])
    # print(param.shape, true.shape)
    # param = param[:, idx]
    # true = true[idx]
    mean_amp = jnp.abs(param).mean(axis=0)
    mean_phase = jnp.rad2deg(jnp.angle(param)).mean(axis=0)
    std_amp = jnp.abs(param).std(axis=0)
    std_phase = jnp.rad2deg(jnp.angle(param)).std(axis=0)

    fig, ax = plt.subplots(n_params, 3, figsize=(18, 4.5 * n_params))

    ax[0, 0].set_title("Root Mean Squared Error")
    ax[0, 1].set_title(f"{name} Magnitude")
    ax[0, 2].set_title(f"{name} Phase")

    plot_comparison(
        ax,
        times,
        mean_amp,
        mean_phase,
        std_amp,
        std_phase,
        jnp.abs(true),
        jnp.rad2deg(jnp.angle(true)),
        rmse,
        diff=diff,
    )

    if save_name is not None:
        fig.savefig(
            os.path.join(save_dir, f"{save_name}_amp_phase.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


@measure_runtime
def plot_predictions(
    times,
    pred,
    truth,
    type: str = "",
    model_name: str = "",
    max_plots: int = 10,
    save_dir: str = "plots/",
):

    rmse_ast = jnp.sqrt(
        jnp.mean(jnp.abs(pred["vis_ast"] - truth["vis_ast"]) ** 2, axis=(0, 1))
    )
    rmse_rfi = jnp.sqrt(
        jnp.mean(jnp.abs(pred["vis_rfi"] - truth["vis_rfi"]) ** 2, axis=(0, 1))
    )
    rmse_gains = jnp.sqrt(
        jnp.mean(jnp.abs(pred["gains"] - truth["gains"]) ** 2, axis=(0, 1))
    )

    plot_complex_real_imag(
        times=times,
        param=pred["vis_ast"][:, :, 0],
        true=truth["vis_ast"][:, 0],
        rmse=rmse_ast,
        name="Ast. Vis.",
        save_name=f"{model_name}_{type}_ast_vis",
        max_plots=max_plots,
        save_dir=save_dir,
    )

    plot_complex_amp_phase(
        times=times,
        param=pred["vis_rfi"][:, :, 0],
        true=truth["vis_rfi"][:, 0],
        rmse=rmse_rfi,
        name="RFI Vis.",
        save_name=f"{model_name}_{type}_rfi_vis",
        diff=False,  # True,
        max_plots=max_plots,
        save_dir=save_dir,
    )

    plot_complex_amp_phase(
        times=times,
        param=pred["gains"][:, :, 0],
        true=truth["gains"][:, 0],
        rmse=rmse_gains,
        name="Gains",
        save_name=f"{model_name}_{type}_gains",
        diff=False,
        max_plots=max_plots,
        save_dir=save_dir,
    )
