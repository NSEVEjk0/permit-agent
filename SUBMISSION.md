# Permit entry preparation

## Problem and proposed solution

Permission can change during a task. An agent should retain useful allowed work while avoiding a revoked or narrowed action. Permit measures that behavior with AgentTune rollouts, observed effects and AuditKIT, while a versioned tool gateway enforces current authority.

The hypothesis is that training a permitted Cohere model improves safe, useful completion beyond the same prompted base model. The current harness is implemented; actual training and held-out base/adapter results have not been produced. Do not claim a measured improvement until those runs finish.

## Required deliverables

- Genuine registration before **7 November 2026, 00:00 UTC**; the official page's countdown includes a 12-hour extension. Project submission remains **15 November, 11:59 UTC**.
- A real pipeline using at least two track repositories and an allowed dense Cohere model. The prepared sources are AgentTune and AuditKIT with Tiny Aya Global.
- Legitimate gated-model access and a bounded CUDA training/evaluation run. No GPU is required for the deterministic harness, but the prepared training pipeline does require one. No GPU rental has been started.
- Public, ungated Hugging Face model/adapter collection and dataset collection, including original datasets, traces, logs and benchmarks; respect the base model and toolkit licenses.
- Public [code](https://github.com/NSEVEjk0/permit-agent), reproducibility instructions, an abstract under 500 words and a write-up explaining the task, choices, before/after metrics and limitations.

The [official requirements](https://hackathon.lexsi.ai/) do not require a submission video. No optional video, slide deck or paid training is being created during preparation.

## Research write-up outline

1. Explain the concrete revocation problem and why enforced blocking differs from attempted model violations.
2. Describe the 264 fictional cases, complete-family train/dev/held-out separation, unchanged-permission controls and leakage prevention.
3. Pin the actual model, toolkit/library revisions, policy prompt, adapter settings, seed and compute/time budget.
4. Explain AgentTune rollouts, the state-based reward and AuditKIT’s independent trajectory/effect checks.
5. Compare the same held-out cases for the prompted base and adapter. Report useful completion, attempted/executed violations, safe completion, over-refusal, call counts and case-bootstrap uncertainty. Keep the plain base ablation separate.
6. Include representative failures, incomplete runs, actual dependency/compute records and every public artifact link.
7. Explain the small synthetic task, the limits of its multilingual probe and why these results would not establish safety for arbitrary real workflows.

Use the commands and artifact boundaries in [README.md](README.md). The deterministic gateway examples and portable toolkit tests are supporting infrastructure evidence, not a trained-model benchmark.

## Announcement draft

Permit explores a small but important agent behavior: what happens when permission changes halfway through a task? Its versioned gateway and AgentTune-to-AuditKIT harness distinguish forbidden attempts from blocked effects and useful completion. The public lab shows scripted examples; actual Cohere training and base/adapter results are still pending. Lab: https://permit-lab.vercel.app · Code: https://github.com/NSEVEjk0/permit-agent

Rewrite the announcement in your own voice. Add measured results only when supported by original traces and public artifacts. This draft has not been posted.
