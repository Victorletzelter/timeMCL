# This code was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/transformer_tempflow/transformer_tempflow_network.py
# under MIT License

import torch
from typing import List, Optional, Tuple
from data_and_transformation import (
    MeanScaler,
    NOPScaler,
    RealNVP,
    MAF,
    FlowOutput,
    MeanStdScaler,
    CenteredMeanScaler,
)
import torch.nn as nn
from gluonts.model import Input, InputSpec


class TransformerTempFlowNetkork(nn.Module):

    def __init__(
        self,
        num_parallel_samples: int,
        d_model: int,
        num_heads: int,
        act_type: str,
        dropout_rate: float,
        dim_feedforward_scale: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        history_length: int,
        context_length: int,
        prediction_length: int,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        flow_type: str,
        n_blocks: int,
        hidden_size: int,
        n_hidden: int,
        dequantize: bool,
        num_feat_dynamic_real: int,  # added here
        embed_dim: int,
        scaling: bool = True,
        scaler_type: str = "mean",
        div_by_std: bool = False,
        minimum_std: float = 1e-3,
        minimum_std_cst: float = 1e-4,
        default_scale: bool = False,
        default_scale_cst: bool = False,
        add_minimum_std: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling
        self.embed_dim = embed_dim

        self.num_parallel_samples = num_parallel_samples  # by default.
        self.num_feat_dynamic_real = num_feat_dynamic_real  # added by me

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq
        self.shifted_lags = [l - 1 for l in self.lags_seq]

        input_size = (
            len(self.lags_seq) * self.target_dim
            + self.target_dim * self.embed_dim
            + self.num_feat_dynamic_real
        )

        self.encoder_input = nn.Linear(input_size, d_model)
        self.decoder_input = nn.Linear(input_size, d_model)

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * d_model,
            dropout=dropout_rate,
            activation=act_type,
        )

        flow_cls = {
            "RealNVP": RealNVP,
            "MAF": MAF,
        }[flow_type]
        self.flow = flow_cls(
            input_size=target_dim,
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            cond_label_size=conditioning_length,
        )
        self.dequantize = dequantize

        self.distr_output = FlowOutput(
            self.flow, input_size=target_dim, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(d_model)

        # self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            if self.scaler_type == "mean":
                self.scaler = MeanScaler(keepdim=True)
            elif self.scaler_type == "nops":
                self.scaler = NOPScaler(keepdim=True)
            elif self.scaler_type == "centered_mean":
                self.scaler = CenteredMeanScaler(keepdim=True)
            elif self.scaler_type == "mean_std":
                self.scaler = MeanStdScaler(
                    minimum_std=self.minimum_std,
                    minimum_std_cst=self.minimum_std_cst,
                    default_scale=self.default_scale,
                    default_scale_cst=self.default_scale_cst,
                    add_minimum_std=self.add_minimum_std,
                    keepdim=True,
                )
            else:
                raise ValueError(f"Invalid scaler type: {self.scaler_type}")
        else:
            self.scaler = NOPScaler(keepdim=True)

        # if self.scaling:
        #     self.scaler = MeanScaler(keepdim=True)
        # else:
        #     self.scaler = NOPScaler(keepdim=True)

        # mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "target_dimension_indicator": Input(
                    shape=(batch_size, self.target_dim),
                    dtype=torch.long,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self.history_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target_cdf": Input(
                    shape=(batch_size, self.history_length, self.target_dim),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.history_length, self.target_dim),
                    dtype=torch.float,
                ),
                "past_is_pad": Input(
                    shape=(batch_size, self.history_length), dtype=torch.float
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "future_target_cdf": Input(
                    shape=(batch_size, self.prediction_length, self.target_dim),
                    dtype=torch.float,
                ),
                "future_observed_values": Input(
                    shape=(batch_size, self.prediction_length, self.target_dim),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def create_network_input(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, -self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        if self.scaler_type == "mean":
            _, scale = self.scaler(
                past_target_cdf[:, -self.context_length :, ...],
                past_observed_values[:, -self.context_length :, ...],
            )
            scale_params = {"scale": scale}
        elif self.scaler_type == "mean_std" or self.scaler_type == "centered_mean":
            _, mean, std = self.scaler(
                past_target_cdf[:, -self.context_length :, ...],
                past_observed_values[:, -self.context_length :, ...],
            )
            scale_params = {"mean": mean, "std": std}
        else:
            raise ValueError(f"Invalid scaler type: {self.scaler_type}")

        # (batch_size, sub_seq_len, target_dim, num_lags)
        if self.scaler_type == "mean":
            lags_scaled = lags / scale_params["scale"].unsqueeze(-1)
        elif self.scaler_type == "mean_std" or self.scaler_type == "centered_mean":
            lags_scaled = (lags - scale_params["mean"].unsqueeze(-1)) / scale_params[
                "std"
            ].unsqueeze(-1)

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * self.target_dim)
        )

        if self.embed_dim > 0:

            # (batch_size, target_dim, embed_dim)
            index_embeddings = self.embed(target_dimension_indicator)
            # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

            # (batch_size, seq_len, target_dim * embed_dim)
            repeated_index_embeddings = (
                index_embeddings.unsqueeze(1)
                .expand(-1, subsequences_length, -1, -1)
                .reshape((-1, subsequences_length, self.target_dim * self.embed_dim))
            )
            # (batch_size, sub_seq_len, input_dim)
            inputs = torch.cat(
                (input_lags, repeated_index_embeddings, time_feat), dim=-1
            )

        else:
            inputs = torch.cat((input_lags, time_feat), dim=-1)
            index_embeddings = None

        return inputs, scale_params, index_embeddings

    def distr_args(self, decoder_output: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(decoder_output)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def loss(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        # seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        inputs, scale_params, _ = self.create_network_input(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        enc_inputs = inputs[:, : self.context_length, ...]
        dec_inputs = inputs[:, self.context_length :, ...]

        enc_out = self.transformer.encoder(
            self.encoder_input(enc_inputs).permute(1, 0, 2)
        )

        dec_output = self.transformer.decoder(
            self.decoder_input(dec_inputs).permute(1, 0, 2),
            enc_out,
            tgt_mask=self.tgt_mask,
        )

        if self.scaling:
            self.flow.scale = scale_params

        # we sum the last axis to have the same shape for all likelihoods
        # (batch_size, subseq_length, 1)
        if self.dequantize:
            future_target_cdf += torch.rand_like(future_target_cdf)

        distr_args = self.distr_args(decoder_output=dec_output.permute(1, 0, 2))
        # likelihoods = -self.flow.log_prob(target, distr_args).unsqueeze(-1)
        loss = -self.flow.log_prob(future_target_cdf, distr_args).unsqueeze(-1)

        return loss.mean(), distr_args

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        # scale: torch.Tensor,
        scale_params: dict,
        enc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        def repeat_dict(dict, dim=0):
            return {
                key: value.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)
                for key, value in dict.items()
            }

        # def repeat(tensor, dim=0):
        # return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        # repeated_scale = repeat(scale)
        if type(scale_params) == dict:
            repeated_scale = repeat_dict(scale_params)
        else:
            repeated_scale = scale_params
        if self.scaling:
            self.flow.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)
        repeated_enc_out = repeat(enc_out, dim=1)

        future_samples = []

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # lags_scaled = lags / repeated_scale.unsqueeze(-1)
            if self.scaler_type == "mean":
                scale = repeated_scale["scale"]
                # (batch_size, sub_seq_len, target_dim, num_lags)
                lags_scaled = lags / scale.unsqueeze(-1)
            elif self.scaler_type == "mean_std" or self.scaler_type == "centered_mean":
                mean = repeated_scale["mean"]
                std = repeated_scale["std"]
                lags_scaled = (lags - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            else:
                raise ValueError(f"Invalid scaler type: {self.scaler_type}")

            input_lags = lags_scaled.reshape(
                (-1, 1, len(self.lags_seq) * self.target_dim)
            )

            # (batch_size, target_dim, embed_dim)
            if self.embed_dim > 0:
                index_embeddings = self.embed(repeated_target_dimension_indicator)
                # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

                # (batch_size, seq_len, target_dim * embed_dim)
                repeated_index_embeddings = (
                    index_embeddings.unsqueeze(1)
                    .expand(-1, 1, -1, -1)
                    .reshape((-1, 1, self.target_dim * self.embed_dim))
                )

                # (batch_size, sub_seq_len, input_dim)
                dec_input = torch.cat(
                    (
                        input_lags,
                        repeated_index_embeddings,
                        repeated_time_feat[:, k : k + 1, ...],
                    ),
                    dim=-1,
                )

            else:
                dec_input = torch.cat(
                    (
                        input_lags,
                        repeated_time_feat[:, k : k + 1, ...],
                    ),
                    dim=-1,
                )

            dec_output = self.transformer.decoder(
                self.decoder_input(dec_input).permute(1, 0, 2), repeated_enc_out
            )

            distr_args = self.distr_args(decoder_output=dec_output.permute(1, 0, 2))

            # (batch_size, 1, target_dim)
            new_samples = self.flow.sample(cond=distr_args)

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        inputs, scale_params, static_feat = self.create_network_input(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        enc_out = self.transformer.encoder(self.encoder_input(inputs).permute(1, 0, 2))

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            # scale=scale_params,
            scale_params=scale_params,
            enc_out=enc_out,
        )
