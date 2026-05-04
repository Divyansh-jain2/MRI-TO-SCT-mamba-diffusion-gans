"""
Hybrid Gaussian Diffusion — Modified diffusion process for the hybrid
MRI encoder + cross-attention model.

Key change: Instead of concatenating MRI with noisy CT, the model receives
MRI condition separately and handles it via cross-attention internally.
"""

import torch as th
import numpy as np
from diffusion.HybridSpacedDiffusion import HybridSpacedDiffusion
from diffusion.GaussianDiffusion import _extract_into_tensor, mean_flat


class _HybridWrappedModel:
    """Wraps a model to inject mri_condition automatically."""
    def __init__(self, model, mri_condition):
        self.model = model
        self.mri_condition = mri_condition
    
    def __call__(self, x, ts, **kwargs):
        return self.model(x, ts, self.mri_condition, **kwargs)


class HybridGaussianDiffusion(HybridSpacedDiffusion):
    """
    Extends GaussianDiffusion to pass MRI condition separately to the model
    instead of concatenating it with the noisy input.

    The model's forward signature is:
        model(x_t, timesteps, mri_condition)
    where x_t is [B,1,D,H,W] (noisy CT only) and mri_condition is
    [B,1,D,H,W] (raw MRI volume).
    """

    def training_losses(self, model, x_start, condition_start=None, t=None,
                       train_target=None, model_kwargs=None, noise=None):
        """
        Compute training losses, passing MRI condition separately.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Diffuse
        x_t = self.q_sample(x_start, t, noise=noise)

        # Wrap model to inject mri_condition
        wrapped_model = _HybridWrappedModel(model, condition_start)

        terms = {}

        if self.loss_type in [self._loss_type.KL, self._loss_type.RESCALED_KL]:
            terms["loss"] = self._vb_terms_bpd(
                model=wrapped_model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                condition=condition_start,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == self._loss_type.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type in [self._loss_type.MSE, self._loss_type.RESCALED_MSE]:
            model_output = wrapped_model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [self._var_type.LEARNED, self._var_type.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    condition=condition_start,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == self._loss_type.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                self._mean_type.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                self._mean_type.START_X: x_start,
                self._mean_type.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = th.nn.L1Loss()(target, model_output)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def p_mean_variance(
        self, model, x, t, condition=None, clip_denoised=True,
        denoised_fn=None, model_kwargs=None
    ):
        """
        Override: pass MRI condition to model separately instead of concatenating.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # KEY CHANGE: No concatenation! Pass x_t and MRI condition separately
        wrapped_model = _HybridWrappedModel(model, condition)
        model_output = wrapped_model(x, self._scale_timesteps(t), **model_kwargs)

        from diffusion.GaussianDiffusion import ModelVarType, ModelMeanType

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    @property
    def _loss_type(self):
        from diffusion.GaussianDiffusion import LossType
        return LossType

    @property
    def _var_type(self):
        from diffusion.GaussianDiffusion import ModelVarType
        return ModelVarType

    @property
    def _mean_type(self):
        from diffusion.GaussianDiffusion import ModelMeanType
        return ModelMeanType
