"""EOS stage: no-op end-of-prompt marker.

The EOS token separates the fixed-length prompt from the
autoregressive decode phase but performs no graph computation.
Collision resolution lives in the RESOLVED_X / RESOLVED_Y identifiers
in the thinking phase (see ``torchwright.doom.stages.thinking_wall``),
and the post-turn angle flows through INPUT's ``new_angle`` broadcast
to the RESOLVED_ANGLE identifier step.

This module is intentionally empty; it exists so the EOS token's role
in the prompt stream is documented in a stage-sized unit.
"""
