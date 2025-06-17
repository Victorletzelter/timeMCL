import rootutils
import os, sys

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))
import torch
from fvcore.nn import FlopCountAnalysis


def count_flops_for_predictions(
    predictor, dataset_test, num_samples=10, model_name=None
):
    """
    Count FLOPs for the prediction process
    """

    # Get a single batch of data for FLOPs counting
    def get_prediction_flops_trf_tempflow():
        with torch.no_grad():
            # Get the model's forward function
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model

            model.eval()
            model.model.eval()

            batch_size = 1
            target_dim = model.model.target_dim
            history_length = model.model.history_length
            num_feat_dynamic_real = model.model.num_feat_dynamic_real
            prediction_length = model.model.prediction_length

            device = next(model.parameters()).device

            dummy_input = {
                "target_dimension_indicator": torch.zeros(
                    batch_size, target_dim, device=device, dtype=torch.long
                ),
                "past_time_feat": torch.ones(
                    batch_size, history_length, num_feat_dynamic_real, device=device
                ),
                "past_target_cdf": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_observed_values": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_is_pad": torch.ones(batch_size, history_length, device=device),
                "future_time_feat": torch.ones(
                    batch_size, prediction_length, num_feat_dynamic_real, device=device
                ),
                "future_target_cdf": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
                "future_observed_values": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }

            inputs = (
                dummy_input["target_dimension_indicator"],
                dummy_input["past_time_feat"],
                dummy_input["past_target_cdf"],
                dummy_input["past_observed_values"],
                dummy_input["past_is_pad"],
                dummy_input["future_time_feat"],
                dummy_input["future_target_cdf"],
                dummy_input["future_observed_values"],
            )

            # Count FLOPs
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    # Create a forward hook to count FLOPs
    def get_prediction_flops_tactis():
        with torch.no_grad():
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model

            batch_size = 1
            target_dim = model.model.target_dim
            context_length = model.model.context_length
            prediction_length = model.model.prediction_length

            device = next(model.parameters()).device

            dummy_input = {
                "past_target_norm": torch.ones(
                    batch_size, context_length, target_dim, device=device
                ),
                "future_target_norm": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }

            inputs = (
                dummy_input["past_target_norm"],
                dummy_input["future_target_norm"],
            )

            # Count FLOPs
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    def get_prediction_flops():
        with torch.no_grad():
            # Get the model's forward function
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model

            model.eval()
            model.model.eval()

            batch_size = 1
            target_dim = model.model.target_dim
            history_length = model.model.history_length
            num_feat_dynamic_real = model.model.num_feat_dynamic_real
            prediction_length = model.model.prediction_length

            device = next(model.parameters()).device

            dummy_input = {
                "target_dimension_indicator": torch.zeros(
                    batch_size, target_dim, device=device, dtype=torch.long
                ),
                "past_target": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_observed_values": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_target_cdf": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_is_pad": torch.ones(batch_size, history_length, device=device),
                "future_time_feat": torch.ones(
                    batch_size, prediction_length, num_feat_dynamic_real, device=device
                ),
                "past_time_feat": torch.ones(
                    batch_size, history_length, num_feat_dynamic_real, device=device
                ),
                "future_target_cdf": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
                "future_observed_values": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }

            inputs = (
                dummy_input["target_dimension_indicator"],
                dummy_input["past_target_cdf"],
                dummy_input["past_observed_values"],
                dummy_input["past_is_pad"],
                dummy_input["future_time_feat"],
                dummy_input["past_time_feat"],
                dummy_input["future_target_cdf"],
                dummy_input["future_observed_values"],
            )

            # Count FLOPs
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    if "tactis" in model_name:
        prediction_flops = get_prediction_flops_tactis()
    elif "transformer_tempflow" in model_name:
        prediction_flops = get_prediction_flops_trf_tempflow()
    else:
        prediction_flops = get_prediction_flops()

    # Calculate total FLOPs (prediction_flops * number of predictions)
    total_flops = prediction_flops * len(dataset_test)

    return prediction_flops, total_flops
