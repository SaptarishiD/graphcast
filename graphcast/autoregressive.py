# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Predictor wrapping a one-step Predictor to make autoregressive predictions.
"""

from typing import Optional, cast

from absl import logging
from graphcast import graphcast
from graphcast import predictor_base
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import xarray




def create_india_mask(dataset: xarray.Dataset) -> xarray.DataArray:
    """Creates a boolean mask for the Indian subcontinent.
    
    Args:
        dataset: Input dataset containing lat/lon coordinates
        
    Returns:
        xarray.DataArray: Boolean mask where True indicates points within India
    """
    # Approximate geographical boundaries of India
    INDIA_BOUNDS = {
        'lat': (8.4, 37.6),  # Latitude bounds
        'lon': (68.7, 97.25)  # Longitude bounds
    }
    
    lat = dataset.coords['lat']
    lon = dataset.coords['lon']
    
    # Create boolean mask for Indian region
    mask = ((lat >= INDIA_BOUNDS['lat'][0]) & 
            (lat <= INDIA_BOUNDS['lat'][1]) & 
            (lon >= INDIA_BOUNDS['lon'][0]) & 
            (lon <= INDIA_BOUNDS['lon'][1]))
    
    return mask

def _unflatten_and_expand_time(flat_variables, tree_def, time_coords):
  variables = jax.tree_util.tree_unflatten(tree_def, flat_variables)
  return variables.expand_dims(time=time_coords, axis=0)


def _get_flat_arrays_and_single_timestep_treedef(variables):
  flat_arrays = jax.tree_util.tree_leaves(variables.transpose('time', ...))
  _, treedef = jax.tree_util.tree_flatten(variables.isel(time=0, drop=True))
  return flat_arrays, treedef


class Predictor(predictor_base.Predictor):
  """Wraps a one-step Predictor to make multi-step predictions autoregressively.

  The wrapped Predictor will be used to predict a single timestep conditional
  on the inputs passed to the outer Predictor. Its predictions are then
  passed back in as inputs at the next timestep, for as many timesteps as are
  requested in the targets_template. (When multiple timesteps of input are
  used, a rolling window of inputs is maintained with new predictions
  concatenated onto the end).

  You may ask for additional variables to be predicted as targets which aren't
  used as inputs. These will be predicted as output variables only and not fed
  back in autoregressively. All target variables must be time-dependent however.

  You may also specify static (non-time-dependent) inputs which will be passed
  in at each timestep but are not predicted.

  At present, any time-dependent inputs must also be present as targets so they
  can be passed in autoregressively.

  The loss of the wrapped one-step Predictor is averaged over all timesteps to
  give a loss for the autoregressive Predictor.
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      noise_level: Optional[float] = None,
      gradient_checkpointing: bool = False,
      ):
    """Initializes an autoregressive predictor wrapper.

    Args:
      predictor: A predictor to wrap in an auto-regressive way.
      noise_level: Optional value that multiplies the standard normal noise
        added to the time-dependent variables of the predictor inputs. In
        particular, no noise is added to the predictions that are fed back
        auto-regressively. Defaults to not adding noise.
      gradient_checkpointing: If True, gradient checkpointing will be
        used at each step of the computation to save on memory. Roughtly this
        should make the backwards pass two times more expensive, and the time
        per step counting the forward pass, should only increase by about 50%.
        Note this parameter will be ignored with a warning if the scan sequence
        length is 1.
    """
    self._predictor = predictor
    self._noise_level = noise_level
    self._gradient_checkpointing = gradient_checkpointing

  def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
    constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
    constant_inputs = constant_inputs.drop_vars(
        forcings.keys(), errors='ignore')
    for name, var in constant_inputs.items():
      if 'time' in var.dims:
        raise ValueError(
            f'Time-dependent input variable {name} must either be a forcing '
            'variable, or a target variable to allow for auto-regressive '
            'feedback.')
    return constant_inputs

  def _validate_targets_and_forcings(self, targets, forcings):
    for name, var in targets.items():
      if 'time' not in var.dims:
        raise ValueError(f'Target variable {name} must be time-dependent.')

    for name, var in forcings.items():
      if 'time' not in var.dims:
        raise ValueError(f'Forcing variable {name} must be time-dependent.')

    overlap = forcings.keys() & targets.keys()
    if overlap:
      raise ValueError('The following were specified as both targets and '
                       f'forcings, which isn\'t allowed: {overlap}')

  def _update_inputs(self, inputs, next_frame):
    num_inputs = inputs.dims['time']

    predicted_or_forced_inputs = next_frame[list(inputs.keys())]

    # Combining datasets with inputs and target time stamps aligns them.
    # Only keep the num_inputs trailing frames for use as next inputs.
    return (xarray.concat([inputs, predicted_or_forced_inputs], dim='time')
            .tail(time=num_inputs)
            # Update the time coordinate to reset the lead times for
            # next AR iteration.
            .assign_coords(time=inputs.coords['time']))

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    """Calls the Predictor.

    Args:
      inputs: input variable used to make predictions. Inputs can include both
        time-dependent and time independent variables. Any time-dependent
        input variables must also be present in the targets_template or the
        forcings.
      targets_template: A target template containing informations about which
        variables should be predicted and the time alignment of the predictions.
        All target variables must be time-dependent.
        The number of time frames is used to set the number of unroll of the AR
        predictor (e.g. multiple unroll of the inner predictor for one time step
        in the targets is not supported yet).
      forcings: Variables that will be fed to the model. The variables
        should not overlap with the target ones. The time coordinates of the
        forcing variables should match the target ones.
        Forcing variables which are also present in the inputs, will be used to
        supply ground-truth values for those inputs when they are passed to the
        underlying predictor at timesteps beyond the first timestep.
      **kwargs: Additional arguments passed along to the inner Predictor.

    Returns:
      predictions: the model predictions matching the target template.

    Raise:
      ValueError: if the time coordinates of the inputs and targets are not
        different by a constant time step.
    """

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets_template, forcings)
    self._validate_targets_and_forcings(targets_template, forcings)

    # After the above checks, the remaining inputs must be time-dependent:
    inputs = inputs.drop_vars(constant_inputs.keys())

    # A predictions template only including the next time to predict.
    target_template = targets_template.isel(time=[0])

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = flat_forcings

    def one_step_prediction(inputs, scan_variables):

      flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            target_template.coords['time'])

      # Add constant inputs:
      all_inputs = xarray.merge([constant_inputs, inputs])
      predictions: xarray.Dataset = self._predictor(
          all_inputs, target_template,
          forcings=forcings,
          **kwargs)

      next_frame = xarray.merge([predictions, forcings])
      next_inputs = self._update_inputs(inputs, next_frame)

      # Drop the length-1 time dimension, since scan will concat all the outputs
      # for different times along a new leading time dimension:
      predictions = predictions.squeeze('time', drop=True)
      # We return the prediction flattened into plain jax arrays, because the
      # extra leading dimension added by scan prevents the tree_util
      # registrations in xarray_jax from unflattening them back into an
      # xarray.Dataset automatically:
      flat_pred = jax.tree_util.tree_leaves(predictions)
      return next_inputs, flat_pred

    if self._gradient_checkpointing:
      scan_length = targets_template.dims['time']
      if scan_length <= 1:
        logging.warning(
            'Skipping gradient checkpointing for sequence length of 1')
      else:
        # Just in case we take gradients (e.g. for control), although
        # in most cases this will just be for a forward pass.
        one_step_prediction = hk.remat(one_step_prediction)

    # Loop (without unroll) with hk states in cell (jax.lax.scan won't do).
    _, flat_preds = hk.scan(one_step_prediction, inputs, scan_variables)

    # The result of scan will have an extra leading axis on all arrays,
    # corresponding to the target times in this case. We need to be prepared for
    # it when unflattening the arrays back into a Dataset:
    scan_result_template = (
        target_template.squeeze('time', drop=True)
        .expand_dims(time=targets_template.coords['time'], axis=0))
    _, scan_result_treedef = jax.tree_util.tree_flatten(scan_result_template)
    predictions = jax.tree_util.tree_unflatten(scan_result_treedef, flat_preds)
    return predictions

def loss(self,
                    inputs: xarray.Dataset,
                    targets: xarray.Dataset,
                    forcings: xarray.Dataset,
                    **kwargs
                    ) -> predictor_base.LossAndDiagnostics:
      """Computes the loss specifically over the Indian region.
      
      This method modifies the original loss computation to:
      1. Create a geographical mask for India
      2. Apply the mask when computing losses
      3. Normalize the loss by the number of grid points in the masked region
      
      Args:
          inputs: Input variables used for predictions
          targets: Target variables to predict
          forcings: Forcing variables for the model
          **kwargs: Additional arguments passed to the predictor
          
      Returns:
          Tuple containing:
          - Regional loss (masked and normalized)
          - Diagnostics dictionary
      """
      if targets.sizes['time'] == 1:
          # For single timestep, modify the underlying predictor's loss
          loss, diagnostics = self._predictor.loss(inputs, targets, forcings, **kwargs)
          india_mask = create_india_mask(targets)
          # Apply mask to loss and renormalize
          masked_loss = loss.where(india_mask, drop=True)
          return masked_loss.mean(), diagnostics

      # For multiple timesteps:
      constant_inputs = self._get_and_validate_constant_inputs(inputs, targets, forcings)
      self._validate_targets_and_forcings(targets, forcings)
      inputs = inputs.drop_vars(constant_inputs.keys())

      if self._noise_level:
          def add_noise(x):
              return x + self._noise_level * jax.random.normal(
                  hk.next_rng_key(), shape=x.shape)
          inputs = jax.tree_map(add_noise, inputs)

      flat_targets, target_treedef = _get_flat_arrays_and_single_timestep_treedef(targets)
      flat_forcings, forcings_treedef = _get_flat_arrays_and_single_timestep_treedef(forcings)
      scan_variables = (flat_targets, flat_forcings)

      def one_step_regional_loss(inputs, scan_variables):
          """Compute loss for one timestep, masked to the Indian region."""
          flat_target, flat_forcings = scan_variables
          forcings = _unflatten_and_expand_time(
              flat_forcings, forcings_treedef, targets.coords['time'][:1])
          target = _unflatten_and_expand_time(
              flat_target, target_treedef, targets.coords['time'][:1])

          all_inputs = xarray.merge([constant_inputs, inputs])
          (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
              all_inputs, target, forcings=forcings, **kwargs)

          # Create and apply India mask
          india_mask = create_india_mask(target)
          masked_loss = loss.where(india_mask, drop=True)
          
          # Normalize by number of grid points in mask
          normalized_loss = masked_loss.mean()

          # Unwrap to jax arrays
          loss, diagnostics = xarray_tree.map_structure(
              xarray_jax.unwrap_data, (normalized_loss, diagnostics))

          predictions = cast(xarray.Dataset, predictions)
          next_frame = xarray.merge([predictions, forcings])
          next_inputs = self._update_inputs(inputs, next_frame)

          return next_inputs, (loss, diagnostics)

      if self._gradient_checkpointing:
          scan_length = targets.dims['time']
          if scan_length <= 1:
              logging.warning('Skipping gradient checkpointing for sequence length of 1')
          else:
              one_step_regional_loss = hk.remat(one_step_regional_loss)

      _, (per_timestep_losses, per_timestep_diagnostics) = hk.scan(
          one_step_regional_loss, inputs, scan_variables)

      # Average losses over time
      loss, diagnostics = jax.tree_util.tree_map(
          lambda x: xarray_jax.DataArray(x, dims=('time', 'batch')).mean('time', skipna=False),
          (per_timestep_losses, per_timestep_diagnostics))

      return loss, diagnostics