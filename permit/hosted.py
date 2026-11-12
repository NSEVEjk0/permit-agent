"""Bounded Cohere Chat probe. Real API output, portable AgentTune records, AuditKIT scoring.

The hosted model does not expose the pinned HF revision and is not an adapter benchmark.
No GPU dependencies, model download, account creation, retries or provisioning occur here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from .bridge import TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT, audit, tool_contract
from .corpus import Case, VERSION, canonical, digest
from .environment import Environment, MAX_TOOL_CALLS
from .metrics import summary
from .runs import AGENTTUNE_REVISION, AUDITKIT_REVISION, MODEL, seal_report

HOSTED_MODEL = 'tiny-aya-global'
CHAT_URL = 'https://api.cohere.com/v2/chat'


def select_dev_cases(all_cases: list[Case], limit: int) -> list[Case]:
    if limit not in (12, 24) or any(case.split != 'dev' for case in all_cases):
        raise ValueError('The preliminary probe uses 12 or 24 development cases only.')
    families = defaultdict(list)
    for case in all_cases:
        families[case.family].append(case)
    if not families or limit % len(families) or (limit // len(families)) % 6:
        raise ValueError('Select complete six-condition blocks balanced across development families.')
    per_family = limit // len(families)
    if any(len(group) < per_family for group in families.values()):
        raise ValueError('Not enough development cases in every family.')
    return [case for group in families.values() for case in group[:per_family]]


def probe_plan(cases: list[Case], out: Path, max_api_calls: int, max_turns: int,
               max_tokens: int, wall_seconds: int, profile: str) -> dict:
    if not cases or any(case.split != 'dev' for case in cases):
        raise ValueError('A hosted probe must use development cases, leaving held-out cases untouched.')
    if not 1 <= max_turns <= 8 or not len(cases) <= max_api_calls <= 192:
        raise ValueError('Use 1–8 turns and a request cap between the case count and 192.')
    if not 256 <= max_tokens <= 1024 or not 30 <= wall_seconds <= 1800:
        raise ValueError('Use 256–1,024 output tokens and a 30–1,800 second wall cap.')
    if profile not in ('plain', 'policy'):
        raise ValueError('Choose plain or policy prompt profile.')
    return {'command': 'probe', 'provider': 'Cohere Chat API', 'hosted_model': HOSTED_MODEL,
            'cases': len(cases), 'case_ids': [case.id for case in cases], 'split': 'dev',
            'prompt_profile': profile, 'max_api_calls': max_api_calls, 'max_turns': max_turns,
            'max_tokens': max_tokens, 'wall_seconds': wall_seconds, 'output': str(out),
            'gpu_required': False, 'training': False, 'retries': 0,
            'access': 'Use a Cohere evaluation key within its remaining free quota. The runner cannot inspect your billing plan or remaining quota.',
            'scope': 'Preliminary hosted baseline. The provider does not report a pinned HF revision; formal base/adapter comparisons use the same local backend.'}


def read_key(key_file: Path | None = None) -> str:
    if key_file is not None:
        if stat.S_IMODE(key_file.stat().st_mode) & 0o077:
            raise ValueError('The API key file must be private; use chmod 600 on it.')
        key = key_file.read_text().strip()
    else:
        key = (os.getenv('COHERE_API_KEY') or os.getenv('CO_API_KEY') or '').strip()
    if not key:
        raise ValueError('Configure COHERE_API_KEY or a private --key-file. Keep API keys out of chat and version control.')
    return key


def _pinned_source(root: Path, revision: str, name: str):
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
        clean = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'src'],
                               capture_output=True).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(f'Configure the pinned {name} source checkout; see README setup.') from None
    if head != revision or not clean:
        raise ValueError(f'{name} source must be clean at the pinned revision; see README setup.')


def load_toolkits():
    agent_root = Path(os.getenv('AGENTTUNE_SOURCE_DIR', '.local/agenttune-source')).resolve()
    audit_root = Path(os.getenv('AUDITKIT_SOURCE_DIR', '.local/auditkit-source')).resolve()
    _pinned_source(agent_root, AGENTTUNE_REVISION, 'AgentTune')
    _pinned_source(audit_root, AUDITKIT_REVISION, 'AuditKIT')
    audit_src = audit_root / 'src'
    sys.path.insert(0, str(audit_src))
    import auditkit
    if not Path(auditkit.__file__).resolve().is_relative_to(audit_src):
        raise ValueError('The imported AuditKIT differs from the pinned source checkout.')
    schema = agent_root / 'src/agenttune/agentic/trajectory/dataset.py'
    spec = importlib.util.spec_from_file_location('permit_agenttune_portable', schema)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parser_path = agent_root / 'src/agenttune/agentic/rollout_engines/tool_call_parse.py'
    parser_spec = importlib.util.spec_from_file_location('permit_agenttune_tool_parser', parser_path)
    parser_module = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser_module)
    module.permit_tool_parser = parser_module.extract_tool_calls_from_text
    module.permit_parser_sha256 = hashlib.sha256(parser_path.read_bytes()).hexdigest()
    return module


def parse_calls(text: str, extractor=None):
    if extractor is not None:
        normalized = extractor(text)
        if normalized:
            if len(normalized) > MAX_TOOL_CALLS:
                return [], 'tool_call_batch_too_large'
            calls = []
            for raw in normalized:
                function = raw['function']
                arguments = function.get('arguments', '{}')
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass  # Preserve malformed arguments; the gateway records and rejects them.
                calls.append({'tool_name': function['name'], 'parameters': arguments})
            return calls, None
        if re.search(r'"(?:tool_name|function|parameters)"\s*:|<tool_?call|<\|START_ACTION\|>', text, re.I):
            return [], 'invalid_tool_call_json'
        return [], None
    value = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', value)
    if fenced:
        value = fenced.group(1).strip()
    if not value.startswith(('[', '{')):
        return [], None  # Plain text is the final explanation, not an executed action.
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [], 'invalid_tool_call_json'
    if not isinstance(parsed, list) or not 1 <= len(parsed) <= MAX_TOOL_CALLS:
        return [], 'invalid_tool_call_shape'
    if any(not isinstance(call, dict) or set(call) != {'tool_name', 'parameters'} or
           not isinstance(call['tool_name'], str) or not isinstance(call['parameters'], dict) for call in parsed):
        return [], 'invalid_tool_call_shape'
    return parsed, None


class CohereClient:
    def __init__(self, key: str, max_calls: int, deadline: float):
        self._key = key
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.last_request = 0.0

    def chat(self, messages: list[dict], max_tokens: int):
        if self.calls >= self.max_calls:
            raise RuntimeError('The API request cap was reached; the probe is incomplete.')
        delay = max(0.0, 3.1 - (time.monotonic() - self.last_request))
        if time.monotonic() + delay >= self.deadline:
            raise TimeoutError('The hosted probe wall-clock cap was reached.')
        if delay:
            time.sleep(delay)
        payload = {'model': HOSTED_MODEL, 'messages': messages, 'max_tokens': max_tokens,
                   'temperature': 0.0, 'seed': 721}
        request = urllib.request.Request(CHAT_URL, data=canonical(payload), method='POST',
                                         headers={'Authorization': 'Bearer ' + self._key,
                                                  'Content-Type': 'application/json', 'User-Agent': 'Permit/0.2'})
        self.calls += 1
        self.last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=min(30, max(.1, self.deadline - time.monotonic()))) as response:
                result = json.loads(response.read(2 * 1024 * 1024))
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            hints = {401: 'Check the locally configured key.', 429: 'Check evaluation quota or rate limits; no automatic retry occurred.'}
            raise RuntimeError(f'Cohere returned HTTP {code}. {hints.get(code, "Inspect your provider account; no automatic retry occurred.")}') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('The Cohere request failed or timed out; no automatic retry occurred.') from None
        if not isinstance(result, dict) or not isinstance(result.get('message'), dict):
            raise ValueError('Cohere returned an unsupported response shape.')
        content = result['message'].get('content', [])
        if not isinstance(content, list):
            raise ValueError('Cohere returned unsupported message content.')
        text = '\n'.join(part['text'] for part in content if isinstance(part, dict) and
                         part.get('type') == 'text' and isinstance(part.get('text'), str))
        if not text.strip():
            raise ValueError('Cohere returned no text; this probe uses prompted JSON tools.')
        observed = {name: result.get(name) for name in ('id', 'finish_reason', 'message', 'usage')}
        return text, observed


def run_probe(cases: list[Case], out: Path, max_api_calls: int = 96, max_turns: int = 8,
              max_tokens: int = 768, wall_seconds: int = 600, profile: str = 'policy',
              key_file: Path | None = None, progress=None) -> dict:
    plan = probe_plan(cases, out, max_api_calls, max_turns, max_tokens, wall_seconds, profile)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory; a hosted probe never overwrites prior evidence.')
    key = read_key(key_file)
    schema = load_toolkits()
    contract = tool_contract()
    source = Path(__file__).parent
    identity = {'provider': 'Cohere', 'base_model': MODEL, 'hosted_model': HOSTED_MODEL,
                'model_revision': None, 'backend': 'cohere-v2-chat-json-tools', 'corpus_schema': VERSION,
                'prompt_profile': profile, 'case_ids': [case.id for case in cases],
                'corpus_sha256': digest([case.fingerprint for case in cases]),
                'tool_contract_sha256': digest(contract), 'tool_parser_sha256': schema.permit_parser_sha256,
                'agenttune_revision': AGENTTUNE_REVISION,
                'auditkit_revision': AUDITKIT_REVISION,
                'harness_sha256': digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob('*.py'))}),
                'adapter_sha256': None}
    out.mkdir(parents=True, exist_ok=True)
    writer = schema.trajectory_writer(out / 'trajectories.jsonl')
    report = {'schema': 'permit-probe-v1', 'mode': 'hosted-model-probe', 'status': 'running',
              'identity': identity, 'settings': {**plan, 'temperature': 0.0, 'seed': 721},
              'api_calls': 0, 'completed_cases': 0, 'scope': plan['scope']}
    started = time.monotonic()
    client = CohereClient(key, max_api_calls, started + wall_seconds)
    extras = {}

    def save_report():
        report['api_calls'] = client.calls
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        seal_report(report)
        (out / 'run.json').write_bytes(canonical(report))

    save_report()
    try:
        for case in cases:
            env = Environment(case)
            messages = case.public_prompt(profile)
            messages[0]['content'] += '\n' + TOOLS_FALLBACK_PROMPT.format(tools=canonical(contract).decode())
            steps, observed, final_text = [], [], ''
            parse_errors = 0
            try:
                for turn in range(max_turns):
                    env.begin_turn(turn)
                    state = canonical(messages).decode()
                    text, response = client.chat(messages, max_tokens)
                    observed.append(response)
                    messages.append({'role': 'assistant', 'content': text})
                    calls, error = parse_calls(text, schema.permit_tool_parser)
                    outputs, portable_calls = [], []
                    if error:
                        parse_errors += 1
                        feedback = 'Invalid tool JSON. Use the exact JSON list and argument fields from the tool contract.'
                        messages.append({'role': 'user', 'content': feedback})
                    elif not calls:
                        final_text = text
                        feedback = text
                    else:
                        for index, call in enumerate(calls):
                            result = env.call(call['tool_name'], call['parameters'])
                            outputs.append(TOOL_RESULT_FORMAT.format(name=call['tool_name'], content=canonical(result).decode()))
                            portable_calls.append({'id': f'call-{turn}-{index}', 'name': call['tool_name'], 'arguments': call['parameters']})
                        feedback = '\n'.join(outputs)
                        messages.append({'role': 'user', 'content': feedback})
                    steps.append(schema.Step(step_number=turn, state=state,
                                             action={'tool_calls': portable_calls}, observation=feedback, thought=text))
                    save_report()
                    if not error and not calls:
                        break
                    if len(env.attempts) >= MAX_TOOL_CALLS:
                        break
            except BaseException:
                writer(schema.Trajectory(task=case.prompt, steps=steps, final_response=final_text,
                                         metadata={'permit': env.evidence(), 'mode': 'hosted-model-probe',
                                                   'incomplete': True, 'cohere_responses': observed, 'messages': messages}))
                raise
            trajectory = schema.Trajectory(task=case.prompt, steps=steps, final_response=final_text,
                                           metadata={'permit': env.evidence(), 'mode': 'hosted-model-probe',
                                                     'cohere_responses': observed, 'messages': messages})
            writer(trajectory)
            extras[trajectory.trajectory_id] = {'parse_errors': parse_errors,
                                                'truncated_responses': sum(response.get('finish_reason') == 'MAX_TOKENS' for response in observed)}
            report['completed_cases'] += 1
            save_report()
            if progress:
                progress({'case_id': case.id, 'completed_cases': report['completed_cases'], 'api_calls': client.calls})
        result = audit(out / 'trajectories.jsonl', cases)
        for row in result['cases']:
            row.update(extras[row['trajectory_id']])
        measured = summary(result['cases'])
        failures = [row['case_id'] for row in result['cases'] if not row['safe_complete']]
        report.update(status='completed', results=result, summary=measured,
                      training_signal={'failure_cases': failures,
                                       'observation': 'Inspect these failures before choosing a training target.' if failures else
                                       'No failures in this development subset. Reconsider the training target before spending GPU credits.',
                                       'training_benefit_established': False})
    except BaseException as error:
        report.update(status='incomplete', error_type=type(error).__name__)
        raise
    finally:
        save_report()
    return report
