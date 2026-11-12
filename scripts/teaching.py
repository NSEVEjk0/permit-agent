"""Authored train-only demonstrations and explicitly bounded AgentTune SFT.

Preparation runs fictional tools, not a model. CUDA execution is opt-in.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from permit.bridge import TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT, audit, create_rollout, tool_contract
from permit.corpus import canonical, digest, load_cases
from permit.environment import Environment, score
from permit.hosted import load_toolkits, select_dev_cases
from permit.metrics import compare, summary
from permit.runs import MODEL, MODEL_REVISION, adapter_digest, identity, ready, seal_report, verify_report


def teaching_rows(case):
    if case.split != 'train':
        raise ValueError('Teaching demonstrations must use the training split only.')
    messages = case.public_prompt()
    messages[0]['content'] += '\n\n' + TOOLS_FALLBACK_PROMPT.format(tools=canonical(tool_contract()).decode())
    env = Environment(case)
    rows, steps = [], []
    for turn, name in enumerate(('read_permissions', 'read_record', 'save_draft', 'finish')):
        env.begin_turn(turn)
        if name == 'finish':
            name = 'deliver_summary' if env.can_deliver else 'request_approval'
        args = {} if name == 'read_permissions' else {'record_id': case.record_id, 'policy_version': env.version}
        if name == 'save_draft': args.update(case.report)
        if name in ('deliver_summary', 'request_approval'):
            args['destination'] = sorted(env.allowed_destinations)[0] if env.can_deliver else case.destination
        completion = canonical([{'tool_name': name, 'parameters': args}]).decode()
        state = canonical(messages).decode()
        rows.append({'messages': copy.deepcopy(messages), 'completion': completion,
                     'case_id': case.id, 'case_sha256': case.fingerprint, 'split': 'train',
                     'mode': 'authored-tool-teaching', 'turn': turn})
        result = env.call(name, args)
        if not result['ok']: raise ValueError('An authored teaching action failed the gateway.')
        feedback = TOOL_RESULT_FORMAT.format(name=name, content=canonical(result).decode())
        steps.append({'state': state, 'action': {'tool_calls': [{'id': f'teaching-{turn}', 'name': name, 'arguments': args}]},
                      'observation': feedback, 'thought': completion})
        messages += [{'role': 'assistant', 'content': completion}, {'role': 'user', 'content': feedback}]
    env.begin_turn(4)
    final = ('The checked report is retained and was delivered to the currently approved reviewer.' if env.can_deliver else
             'The checked report is retained. The pending delivery is recorded and needs renewed sending approval.')
    rows.append({'messages': copy.deepcopy(messages), 'completion': final, 'case_id': case.id,
                 'case_sha256': case.fingerprint, 'split': 'train', 'mode': 'authored-tool-teaching', 'turn': 4})
    steps.append({'state': canonical(messages).decode(), 'action': {'tool_calls': []}, 'observation': final, 'thought': final})
    if not score(env.evidence(), case)['safe_complete']: raise ValueError('The teaching workflow did not complete.')
    return rows, steps, final, env.evidence()


def prepare(cases, out):
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new teaching output directory.')
    schema = load_toolkits()
    out.mkdir(parents=True, exist_ok=True)
    writer = schema.trajectory_writer(out / 'trajectories.jsonl')
    rows = []
    for case in cases:
        demos, steps, final, evidence = teaching_rows(case)
        rows.extend(demos)
        writer(schema.Trajectory(task=case.prompt,
               steps=[schema.Step(step_number=i, **step) for i, step in enumerate(steps)],
               final_response=final, metadata={'permit': evidence, 'fixture': True, 'mode': 'authored-tool-teaching'}))
    checked = audit(out / 'trajectories.jsonl', cases)
    if len(checked['cases']) != len(cases) or not all(row['safe_complete'] for row in checked['cases']):
        raise ValueError('Teaching trajectories failed the independent AuditKIT replay.')
    data = b''.join(canonical(row) + b'\n' for row in rows)
    (out / 'demonstrations.jsonl').write_bytes(data)
    manifest = {'schema': 'permit-teaching-v1', 'mode': 'authored-tool-teaching', 'model_inference': False,
                'split': 'train', 'cases': len(cases), 'rows': len(rows),
                'case_ids': [case.id for case in cases], 'corpus_sha256': digest([case.fingerprint for case in cases]),
                'demonstrations_sha256': hashlib.sha256(data).hexdigest(),
                'notice': 'Rule-authored teaching examples, not model responses or model performance. Fictional inputs use CC BY 4.0.'}
    (out / 'manifest.json').write_bytes(canonical(manifest))
    return manifest


def read_teaching(path, cases):
    manifest = json.loads((path / 'manifest.json').read_text())
    data = (path / 'demonstrations.jsonl').read_bytes()
    if manifest.get('mode') != 'authored-tool-teaching' or hashlib.sha256(data).hexdigest() != manifest['demonstrations_sha256']:
        raise ValueError('The authored teaching data fingerprint differs.')
    rows = [json.loads(line) for line in data.splitlines()]
    expected = {case.id: case.fingerprint for case in cases}
    if any(row.get('split') != 'train' or row.get('case_sha256') != expected.get(row.get('case_id')) for row in rows):
        raise ValueError('Teaching rows contain unsupported cases or another split.')
    if Counter(row['case_id'] for row in rows) != Counter({case.id: 5 for case in cases}):
        raise ValueError('Every training case needs its five observed teaching turns.')
    return rows, manifest


def precision():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_student(quantized, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = {'revision': MODEL_REVISION, 'trust_remote_code': False, 'dtype': precision()}
    if quantized:
        kwargs.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                      bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=precision()))
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    if not quantized: model.to('cuda')
    if adapter:
        from peft import PeftModel
        config = json.loads((adapter / 'adapter_config.json').read_text())
        if config.get('base_model_name_or_path') != MODEL or config.get('revision') != MODEL_REVISION:
            raise ValueError('The adapter does not identify the pinned permitted model.')
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer


def run_identity(cases, quantized, adapter=None):
    result = identity(cases, 'policy', adapter)
    result['harness_sha256'] = digest({'core': result['harness_sha256'], 'teaching_driver': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result['compute_dtype'] = str(precision())
    result['quantization'] = {'bits': 4, 'method': 'nf4', 'double_quant': True} if quantized else None
    return result


def execute(args, cases):
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Use a new run output directory.')
    rows, manifest = read_teaching(args.teaching, cases) if args.command == 'train' else (None, None)
    ready(args.wall_seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'permit-training-v1' if rows else 'permit-eval-v2',
              'mode': 'AgentTune-authored-demonstration-SFT' if rows else 'model-evaluation',
              'status': 'running', 'identity': run_identity(cases, args.load_in_4bit, getattr(args, 'adapter', None)),
              'settings': {'trials': 1, 'seed': 721, 'temperature': 0.2, 'max_new_tokens': 768,
                           'wall_seconds': args.wall_seconds, 'load_in_4bit': args.load_in_4bit}}
    started = time.monotonic()
    try:
        if rows:
            from agenttune import create_distill_trainer
            from datasets import Dataset
            from peft import LoraConfig, PeftModel
            from trl import SFTConfig, SFTTrainer
            import torch
            torch.manual_seed(721)
            class Factory:
                def __call__(self, *, model, train_dataset, **kwargs):
                    if model != MODEL: raise ValueError('Unsupported student model.')
                    self.model, self.tokenizer = load_student(args.load_in_4bit)
                    data = []
                    for row in train_dataset:
                        messages = row['messages']
                        if len(messages) != 2 or [message['role'] for message in messages] != ['user', 'assistant']:
                            raise ValueError('The upstream demonstration row shape changed.')
                        prompt = json.loads(messages[0]['content'])
                        completion = [{'role': 'assistant', 'content': messages[1]['content']}]
                        if len(self.tokenizer.apply_chat_template(prompt + completion, tokenize=True)) > 4096:
                            raise ValueError('A teaching row exceeds the context budget; no silent truncation.')
                        data.append({'prompt': prompt, 'completion': completion})
                    self.trainer = SFTTrainer(model=self.model, args=SFTConfig(**kwargs),
                         train_dataset=Dataset.from_list(data), processing_class=self.tokenizer,
                         peft_config=LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM'))
                    return self.trainer
            factory = Factory()
            trainer = create_distill_trainer(student=MODEL,
                demonstrations=[(canonical(row['messages']).decode(), row['completion']) for row in rows],
                trainer_factory=factory, output_dir=str(args.out), max_steps=args.max_steps,
                learning_rate=2e-4, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                gradient_checkpointing=True, max_length=4096, completion_only_loss=True,
                bf16=precision() == torch.bfloat16, fp16=precision() == torch.float16,
                seed=721, report_to='none', push_to_hub=False, save_strategy='no')
            outcome = trainer.train()
            if not isinstance(factory.trainer.model, PeftModel): raise ValueError('Expected an adapter-only student.')
            adapter = args.out / 'adapter'
            factory.trainer.save_model(str(adapter));factory.tokenizer.save_pretrained(str(adapter))
            config_path = adapter / 'adapter_config.json'
            config = json.loads(config_path.read_text());config.update(base_model_name_or_path=MODEL, revision=MODEL_REVISION)
            config_path.write_bytes(canonical(config))
            report.update(status='completed', teaching_manifest=manifest, training_method='AgentTune create_distill_trainer with completion-only TRL SFT',
                          max_steps=args.max_steps, adapter_sha256=adapter_digest(adapter),
                          training_metrics=getattr(outcome, 'metrics', {}))
        else:
            from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine
            import torch
            model, tokenizer = load_student(args.load_in_4bit, args.adapter)
            model.eval();engine = TransformersRolloutEngine(model, tokenizer)
            rollout, _ = create_rollout(cases, args.out / 'trajectories.jsonl', engine=engine)
            with torch.inference_mode():
                for case in cases:
                    torch.manual_seed(721)
                    rollout([case.public_prompt()], max_new_tokens=768, temperature=0.2, do_sample=True)
            result = audit(args.out / 'trajectories.jsonl', cases)
            if Counter(row['case_id'] for row in result['cases']) != Counter(case.id for case in cases):
                raise ValueError('Evaluation is missing selected cases.')
            report.update(status='completed', results=result, summary=summary(result['cases']))
    except BaseException as error:
        report.update(status='incomplete', error_type=type(error).__name__);raise
    finally:
        signal.alarm(0);report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        seal_report(report);(args.out / 'run.json').write_bytes(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare');p.add_argument('--out', type=Path, default=ROOT / 'data/teaching')
    for name in ('train', 'evaluate'):
        p = commands.add_parser(name);p.add_argument('--out', type=Path, required=True)
        p.add_argument('--execute', action='store_true');p.add_argument('--load-in-4bit', action='store_true')
        p.add_argument('--wall-seconds', type=int, default=1800)
        if name == 'train':
            p.add_argument('--teaching', type=Path, default=ROOT / 'data/teaching');p.add_argument('--max-steps', type=int, default=60)
        else:
            p.add_argument('--split', choices=['dev', 'heldout'], default='dev');p.add_argument('--limit', type=int, default=12)
            p.add_argument('--adapter', type=Path)
    p = commands.add_parser('compare');p.add_argument('before', type=Path);p.add_argument('after', type=Path);p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
        verify_report(before);verify_report(after)
        if any(before['identity'].get(key) != after['identity'].get(key) for key in ('quantization', 'compute_dtype')):
            raise ValueError('Comparison requires matching quantization and compute precision.')
        args.out.parent.mkdir(parents=True, exist_ok=True);args.out.write_bytes(canonical(compare(before, after)));return
    split = 'train' if args.command in ('prepare', 'train') else args.split
    cases = load_cases(ROOT / f'data/{split}.jsonl')
    if args.command == 'prepare': print(json.dumps(prepare(cases, args.out)));return
    if args.command == 'evaluate':
        if not 1 <= args.limit <= len(cases): raise ValueError('Choose a case limit within the selected split.')
        cases = select_dev_cases(cases, args.limit) if split == 'dev' and args.limit in (12, 24) else cases[:args.limit]
    if not 30 <= args.wall_seconds <= 3600 or not 1 <= getattr(args, 'max_steps', 1) <= 80:
        raise ValueError('Use 30–3,600 seconds and 1–80 training steps.')
    plan = {'command': args.command, 'execute': args.execute, 'cases': len(cases), 'split': split,
            'model': MODEL, 'revision': MODEL_REVISION, 'load_in_4bit': args.load_in_4bit,
            'wall_seconds': args.wall_seconds, 'output': str(args.out),
            'gpu_path_verified': False, 'publishes_automatically': False}
    if not args.execute: print(json.dumps(plan));return
    report = execute(args, cases);print(json.dumps({'status': report['status'], 'output': str(args.out)}))

if __name__ == '__main__': main()
