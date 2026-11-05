# Permit

Train and evaluate permission-aware tool behavior using a permitted dense Cohere model with AgentTune and AuditKIT. Model access and a bounded GPU run are pending; a dashboard, authored policy or fake scores do not replace training. Keep model/library revisions, data splits, trajectories, adapters and evaluation fingerprints reproducible. Respect model and source-available toolkit licenses.

The task is mid-episode permission revocation: permitted useful work should continue, revoked capabilities must not be attempted, and tool text cannot grant authority. The tool gateway always enforces the current versioned permission policy; distinguish enforced safety from the model's attempted violations. Avoid reward hacking, label leakage, train/held-out scenario overlap and fabricated before/after improvements. Metrics need task completion, attempted/executed violations, over-refusal and uncertainty.

Run deterministic harness and corpus tests without GPU. Validate the AgentTune→AuditKIT trajectory boundary against real library APIs. Keep training disabled until actual allowed model access and compute are configured. Never automatically publish gated weights, accept licenses, create accounts or submit entries. Public HF publishing requires reviewed allowed artifacts. No invented model results in the UI.
