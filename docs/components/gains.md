# Gain Components

Gain components calculate the complex-valued gains and then perform the computation of the observed visibilities, $V^\text{OBS}$, from the RFI visibilities, $V^\text{RFI}$, and the astronomical visibilities, $V^\text{AST}$. This is typically done in a linear manner as

$$V^\text{OBS}_{pq} = G_p \left( V_{pq}^\text{RFI} + V_{pq}^\text{AST} \right) G^*_q$$

All of these components vary over time and frequency, $(t, \nu)$, and the indices $p$ and $q$ indicate the antennas forming a baseline $pq$.

## Unitary Gains - {class}`~tabascal.components.gains.UnitaryGains`

Unitary gains are perfect gains, i.e $G(t, \nu) = 1$. Therefore,

$$V^\text{OBS}_{pq} = V_{pq}^\text{RFI} + V_{pq}^\text{AST}$$

## Constant Gains - {class}`~tabascal.components.gains.ConstGains`

One complex direction-independent gain per antenna, constant over time and frequency, and fitted:

$$V^\text{OBS}_{pq} = g_p g_q^* \left( V_{pq}^\text{RFI} + V_{pq}^\text{AST} \right)$$

This is the static gain of the array rather than a time-variable one, so it adds only $2 n_\text{ant} - 1$ parameters. It is written in the gauge the data can actually see: the amplitudes are parameterised in log space with a zero-sum constraint, so the geometric mean of $|g_p|$ is exactly 1 and the gain carries no absolute flux scale, and the phase of the reference antenna `gains.ref_ant` is pinned to 0, so it carries no absolute phase either. Both constraints remove a direction the likelihood is flat along rather than leaving the fit to drift along it.

The gain is only identifiable against a model term it cannot deform, so `ConstGains` pairs with {class}`~tabascal.components.rfi_signal.ComplexRFIConstAnt` (whose RFI amplitude has no per-antenna freedom of its own) and with a rigid sky, {class}`~tabascal.components.ast_signal.FixedDiscreteSky` and {class}`~tabascal.components.ast_vis.DiscreteSkyVis`. Combining it with {class}`~tabascal.components.rfi_signal.ComplexRFIVarAnt` raises a warning naming the flat direction. See the [gains configuration section](../config.md#a-constant-gain-per-antenna) for the configuration keys, the identifiability rules and how to initialise the fit at a previously measured gain.

