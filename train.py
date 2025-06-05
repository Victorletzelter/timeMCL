from toy import tMCL, train_tMCL
import torch
import yaml
import sys
import os
import time
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

config_file = sys.argv[1]
with open(f"{os.environ['PROJECT_ROOT']}/config/{config_file}", "r") as file:
    config = yaml.safe_load(file)

dataset_name = config["dataset_name"]
nb_discretization_points = config["nb_discretization_points"]
batch_size = config["batch_size"]
device = config["device"]
t_condition = config["t_condition"]
wta_mode = config["wta_mode"]
cond_dim = config["cond_dim"]
n_hypotheses = config["n_hypotheses"]
num_steps = config["num_steps"]
learning_rate = config["learning_rate"]

sigma = config["sigma"] if "sigma" in config else None
p = config["p"] if "p" in config else None
coefficients = config["coefficients"] if "coefficients" in config else None

nb_step_simulation_model = (
    config["nb_step_simulation"] - p
    if dataset_name == "ARp"
    else config["nb_step_simulation"]
)
interval_length = config["nb_step_simulation"]

model = tMCL(
    cond_dim=cond_dim,
    nb_step_simulation=nb_step_simulation_model,
    n_hypotheses=n_hypotheses,
    device=device,
    loss_type=wta_mode,
)

if dataset_name == "ARp":
    additional_params = {"p": p, "coefficients": coefficients, "sigma": sigma}
else:
    additional_params = {}

start_time = time.time()

trained_model = train_tMCL(
    model=model,
    process_type=dataset_name,
    num_steps=num_steps,
    batch_size=batch_size,
    nb_discretization_points=nb_discretization_points,
    interval_length=interval_length,
    device=device,
    learning_rate=learning_rate,
    additional_params=additional_params,
)

end_time = time.time()

print(f"Training time: {end_time - start_time} seconds")

# Check if logs/{dataset_name} exists, if not create it
if not os.path.exists(f"logs/{dataset_name}"):
    os.makedirs(f"logs/{dataset_name}")

torch.save(
    trained_model.state_dict(),
    f"{os.environ['PROJECT_ROOT']}/logs/trained_timeMCL_{dataset_name}.pth",
)
