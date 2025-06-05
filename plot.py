# %%

import numpy as np
import matplotlib.pyplot as plt
import torch
from toy import (
    plot_brownian_bridge,
    plot_brownien,
    plot_ARp_quantization,
    tMCL,
)  # Ensure tMCL is imported
from matplotlib import rc
import argparse
import pickle
import os
import rootutils
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def main(seed):

    # Load the configuration associated with each dataset
    with open(f"{os.environ['PROJECT_ROOT']}/config/brownian_motion.yaml", "r") as f:
        config_brownian_motion = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"{os.environ['PROJECT_ROOT']}/config/brownian_bridge.yaml", "r") as f:
        config_brownian_bridge = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"{os.environ['PROJECT_ROOT']}/config/ARp.yaml", "r") as f:
        config_ar = yaml.load(f, Loader=yaml.FullLoader)

    # Parameters
    batch_size = 1000
    interval_length = config_brownian_motion["nb_step_simulation"]
    m = 2 # parameter m in the K-L decomposition of the eigenfunctions
    N_levels = [5, 2] # number of levels for the quantization of the eigenfunctions
    a = 0
    b = 1
    pred_length = config_brownian_motion["nb_step_simulation"]
    num_steps = 300
    coefficients = config_ar["coefficients"]
    p = config_ar["p"]

    nb_step_simulation = {
        "brownian_bridge": config_brownian_bridge["nb_step_simulation"],
        "brownian_motion": config_brownian_motion["nb_step_simulation"],
        "ar_quantization": config_ar["nb_step_simulation"] - p,
    }
    cond_dim = {
        "brownian_bridge": config_brownian_bridge["cond_dim"],
        "brownian_motion": config_brownian_motion["cond_dim"],
        "ar_quantization": config_ar["cond_dim"],
    }
    t_condition = {
        "brownian_bridge": 0.5,
        "brownian_motion": 0.5,
        "ar_quantization": 100,
    }
    nb_discretization_points = {
        "brownian_bridge": 500,
        "brownian_motion": 500,
        "ar_quantization": 500,
    }

    sigma = config_ar["sigma"]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_trained_model_brownian_bridge = (
        f"{os.environ['PROJECT_ROOT']}/logs/trained_timeMCL_brownian_bridge.pth"
    )
    path_trained_model_brownian_motion = (
        f"{os.environ['PROJECT_ROOT']}/logs/trained_timeMCL_brownian_motion.pth"
    )
    path_trained_model_ARp = (
        f"{os.environ['PROJECT_ROOT']}/logs/trained_timeMCL_ARp.pth"
    )

    # %%

    rc("text", usetex=True)
    rc("font", family="serif")

    # Load the trained models
    trained_model_brownian_bridge = tMCL(
        cond_dim=cond_dim["brownian_bridge"],  # Adjust as needed
        nb_step_simulation=nb_step_simulation["brownian_bridge"],
        n_hypotheses=config_brownian_bridge["n_hypotheses"],
        device=device,
        loss_type="wta",
    )
    trained_model_brownian_bridge.load_state_dict(
        torch.load(path_trained_model_brownian_bridge)
    )

    trained_model_brownian_motion = tMCL(
        cond_dim=cond_dim["brownian_motion"],
        nb_step_simulation=nb_step_simulation["brownian_motion"],
        n_hypotheses=config_brownian_motion["n_hypotheses"],
        device=device,
        loss_type="relaxed_wta",
    )
    trained_model_brownian_motion.load_state_dict(
        torch.load(path_trained_model_brownian_motion)
    )

    trained_model_ARp = tMCL(
        cond_dim=cond_dim["ar_quantization"],  # Adjust as needed
        nb_step_simulation=nb_step_simulation["ar_quantization"],
        n_hypotheses=config_ar["n_hypotheses"],
        device=device,
        loss_type="wta",
    )
    trained_model_ARp.load_state_dict(torch.load(path_trained_model_ARp))

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # Plot Brownien
    result_brownien = plot_brownien(
        T=1,  # Adjust as needed
        t_condition=t_condition["brownian_motion"],
        pred_length=pred_length,
        num_steps=num_steps,
        m=m,
        N_levels=N_levels,
        trained_model=trained_model_brownian_motion,  # Use the same model for Brownien
        ax=axs[0],  # Pass the axis to plot on
    )

    axs[0].tick_params(axis="x", labelsize=20)
    axs[0].tick_params(axis="y", labelsize=20)
    axs[0].set_title("Brownian Motion", fontsize=35)
    axs[0].set_xlabel("Time", fontsize=25)
    axs[0].set_ylabel("Amplitude", fontsize=25)
    axs[0].grid()

    handles1, labels1 = axs[0].get_legend_handles_labels()

    # Plot Brownian Bridge
    result_brownian_bridge = plot_brownian_bridge(
        interval_length=interval_length,
        nb_discretization_points=nb_discretization_points["brownian_bridge"],
        m=m,
        N_levels=N_levels,
        a=a,
        b=b,
        t_condition=t_condition["brownian_bridge"],
        trained_model=trained_model_brownian_bridge,
        ax=axs[1],  # Pass the axis to plot on
    )
    axs[1].set_title("Brownian Bridge", fontsize=35)
    axs[1].tick_params(axis="x", labelsize=20)
    axs[1].tick_params(axis="y", labelsize=20)
    axs[1].set_xlabel("Time", fontsize=25)
    axs[1].grid()

    handles2, labels2 = axs[1].get_legend_handles_labels()

    # Plot ARp Quantization
    result_ar_quantization = plot_ARp_quantization(
        batch_size=batch_size,
        nb_discretization_points=nb_discretization_points["ar_quantization"],
        interval_length=nb_step_simulation["ar_quantization"],
        coefficients=coefficients,
        sigma=sigma,
        t_condition=t_condition["ar_quantization"],
        trained_model=trained_model_ARp,
        ax=axs[2],  # Pass the axis to plot on
    )
    axs[2].set_title("AR(p)", fontsize=35)
    axs[2].tick_params(axis="x", labelsize=20)
    axs[2].tick_params(axis="y", labelsize=20)
    axs[2].set_xlabel("Time", fontsize=25)
    # axs[2].set_ylabel("Amplitude", fontsize=25)
    axs[2].grid()

    handles3, labels3 = axs[2].get_legend_handles_labels()

    # Adjust layout
    plt.tight_layout()

    all_handles = handles1
    all_labels = labels1

    unique_handles_labels = dict(zip(all_labels, all_handles))

    # Get the unique handles and labels
    unique_labels = list(unique_handles_labels.keys())
    unique_handles = list(unique_handles_labels.values())

    # Create common legend above the figure
    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fontsize=20,
    )

    # Save the figure
    plt.savefig(
        f"{os.environ['PROJECT_ROOT']}/figures/toy_figure.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    with open(
        f"{os.environ['PROJECT_ROOT']}/logs/data_to_reproduce_plot_{seed}.pkl", "wb"
    ) as f:
        pickle.dump(
            {
                "trained_model_brownian_bridge": trained_model_brownian_bridge,
                "trained_model_brownian_motion": trained_model_brownian_motion,
                "trained_model_ARp": trained_model_ARp,
            },
            f,
        )

    # %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    # set the seed
    torch.manual_seed(seed)
    main(seed)
