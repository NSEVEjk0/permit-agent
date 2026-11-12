"""Transport fixtures exercise API recording and budgets without model inference."""
import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from permit.corpus import canonical, cases
from permit.hosted import CohereClient, read_key, run_probe
from permit.runs import verify_report


class Response:
    def __init__(self, text):
        self.body = canonical({'id': 'transport-test-fixture', 'finish_reason': 'COMPLETE',
                               'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': text}]},
                               'usage': {'tokens': {'input_tokens': 10, 'output_tokens': 20}}})
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return self.body[:limit]


class HostedTests(unittest.TestCase):
    def test_provider_error_body_is_not_logged_and_is_not_retried(self):
        client = CohereClient('test-key-private', 2, time.monotonic() + 60)
        error = urllib.error.HTTPError('https://api.cohere.com/v2/chat', 429, 'quota', {}, io.BytesIO(b'private provider body'))
        with patch('permit.hosted.urllib.request.urlopen', side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 429') as raised:
                client.chat([{'role': 'user', 'content': 'fixture'}], 768)
        self.assertNotIn('private provider body', str(raised.exception))
        self.assertNotIn('test-key-private', str(raised.exception))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(client.calls, 1)

    def test_key_file_requires_private_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'key'
            path.write_text('transport-test-key')
            path.chmod(0o644)
            with self.assertRaises(ValueError): read_key(path)
            path.chmod(0o600)
            self.assertEqual(read_key(path), 'transport-test-key')

    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'), 'Pinned sources required for the real portable-schema/AuditKIT boundary.')
    def test_api_transport_fixture_produces_replayable_toolkit_evidence_without_gpu(self):
        case = next(case for case in cases() if case.split == 'dev' and case.change == 'revoke' and case.change_after == 'record_read')
        outputs = iter([
            canonical([{'tool_name': 'read_record', 'parameters': {'record_id': case.record_id, 'policy_version': 1}}]).decode(),
            canonical([{'tool_name': 'save_draft', 'parameters': {'record_id': case.record_id, **case.report, 'policy_version': 2}}]).decode(),
            canonical([{'tool_name': 'request_approval', 'parameters': {'record_id': case.record_id, 'destination': case.destination, 'policy_version': 2}}]).decode(),
            'Transport test fixture: report retained, delivery needs fresh approval.',
        ])
        captured = []
        def transport(request, **kwargs):
            captured.append(json.loads(request.data))
            return Response(next(outputs))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'COHERE_API_KEY': 'transport-test-private-key'}), patch('permit.hosted.time.sleep'), patch('permit.hosted.urllib.request.urlopen', side_effect=transport):
            out = Path(folder) / 'run'
            result = run_probe([case], out, max_api_calls=4, max_turns=4, wall_seconds=60)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['api_calls'], 4)
            self.assertTrue(result['results']['cases'][0]['safe_complete'])
            self.assertEqual(result['results']['cases'][0]['end_reason'], 'final_response')
            self.assertEqual(result['results']['auditkit']['rows'][0]['outcome']['verdict'], 'success')
            self.assertTrue(verify_report(json.loads((out / 'run.json').read_text()))['integrity_verified'])
            self.assertIsNone(result['identity']['model_revision'])
            self.assertNotIn('tools', captured[0])
            self.assertIn('gateway_policy', captured[1]['messages'][-1]['content'])
            for name in ['run.json', 'trajectories.jsonl']:
                self.assertNotIn('transport-test-private-key', (out / name).read_text())
            record = json.loads((out / 'trajectories.jsonl').read_text())
            self.assertEqual(len(record['metadata']['cohere_responses']), 4)
            self.assertEqual(record['metadata']['permit']['final_state']['deliveries'], [])

    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'), 'Pinned parser and toolkit sources required.')
    def test_invalid_arithmetic_json_is_recorded_and_repaired_without_executing_an_expression(self):
        case = next(case for case in cases() if case.split == 'dev')
        invalid = '[{"tool_name":"save_draft","parameters":{"payable_total_cents":10*3}}]'
        read = canonical([{'tool_name': 'read_record', 'parameters': {'record_id': case.record_id, 'policy_version': 1}}]).decode()
        responses = iter([invalid, read])
        captured = []
        def transport(request, **kwargs):
            captured.append(json.loads(request.data))
            return Response(next(responses))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'COHERE_API_KEY': 'transport-test-key'}), patch('permit.hosted.time.sleep'), patch('permit.hosted.urllib.request.urlopen', side_effect=transport):
            out = Path(folder) / 'run'
            result = run_probe([case], out, max_api_calls=2, max_turns=2, wall_seconds=60)
            row = result['results']['cases'][0]
            self.assertEqual(row['parse_errors'], 1)
            self.assertEqual(row['end_reason'], 'turn_limit')
            self.assertFalse(row['task_complete'])
            self.assertIn('Invalid tool JSON', captured[1]['messages'][-1]['content'])
            record = json.loads((out / 'trajectories.jsonl').read_text())
            self.assertEqual([attempt['tool'] for attempt in record['metadata']['permit']['attempts']], ['read_record'])
            self.assertEqual(record['metadata']['cohere_responses'][0]['message']['content'][0]['text'], invalid)

    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'), 'Pinned sources required for budget/evidence integration.')
    def test_request_cap_preserves_partial_evidence_and_never_claims_completion(self):
        case = next(case for case in cases() if case.split == 'dev')
        text = canonical([{'tool_name': 'read_record', 'parameters': {'record_id': case.record_id, 'policy_version': 1}}]).decode()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'COHERE_API_KEY': 'transport-test-key'}), patch('permit.hosted.urllib.request.urlopen', return_value=Response(text)) as transport:
            out = Path(folder) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'request cap'):
                run_probe([case], out, max_api_calls=1, max_turns=4, wall_seconds=60)
            report = json.loads((out / 'run.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['api_calls'], 1)
            self.assertEqual(transport.call_count, 1)
            record = json.loads((out / 'trajectories.jsonl').read_text())
            self.assertTrue(record['metadata']['incomplete'])
            with self.assertRaises(ValueError): verify_report(report)


if __name__ == '__main__': unittest.main()
