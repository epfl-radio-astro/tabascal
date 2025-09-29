# Gain Components

Gain components calculate the complex-valued gains and then perform the computation of the observed visibilities, $V^\text{OBS}$, from the RFI visibilities, $V^\text{RFI}$, and the astronomical visibilities, $V^\text{AST}$. This is typically done in a linear manner as

$$V^\text{OBS}_{pq} = G_p \left( V_{pq}^\text{RFI} + V_{pq}^\text{AST} \right) G^*_q$$

All of these components vary over time and frequency, $(t, \nu)$, and the indices $p$ and $q$ indicate the antennas forming a baseline $pq$.

## Unitary Gains - {class}`~tabascal.components.gains.UnitaryGains`

Unitary gains are perfect gains, i.e $G(t, \nu) = 1$. Therefore,

$$V^\text{OBS}_{pq} = V_{pq}^\text{RFI} + V_{pq}^\text{AST}$$

