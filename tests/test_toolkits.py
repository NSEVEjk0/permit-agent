"""Real upstream portable-schema → AuditKIT test; no model or trainer is run."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from permit.corpus import cases
from permit.fixtures import run
from permit.bridge import audit

class ToolkitBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'),'Pinned upstream source is required for the portable-schema boundary check.')
    def test_actual_agenttune_trajectory_is_read_and_scored_by_actual_auditkit(self):
        source=Path(os.environ['AGENTTUNE_SOURCE_DIR'])
        schema=source/'src/agenttune/agentic/trajectory/dataset.py'
        # Load the unmodified upstream stdlib schema directly. This avoids
        # importing its GPU training stack and does not mock that stack.
        spec=importlib.util.spec_from_file_location('actual_agenttune_schema',schema);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        c=cases()[2];e=run(c,True)
        steps=[module.Step(step_number=a['turn'],state='scripted fixture',action={'tool_calls':[{'id':str(a['seq']),'name':a['tool'],'arguments':a['arguments']}]},observation='scripted gateway result') for a in e['attempts']]
        trajectory=module.Trajectory(task=c.prompt,steps=steps,final_response='Scripted route; not model output.',metadata={'permit':e,'fixture':True})
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trajectories.jsonl';module.trajectory_writer(p)(trajectory)
            result=audit(p,[c]);self.assertEqual(result['cases'][0]['attempted_violations'],1)
            self.assertEqual(result['auditkit']['rows'][0]['outcome']['verdict'],'failure')
            self.assertEqual(result['auditkit']['rows'][0]['status'],'completed')
            # Argument loss is an actual error rather than a clean zero.
            record=json.loads(p.read_text());record['metadata'].pop('permit');p.write_text(json.dumps(record)+'\n')
            with self.assertRaises(ValueError):audit(p,[c])
    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'),'Pinned source required for actual signature checks.')
    def test_run_hooks_match_pinned_actual_rollout_api(self):
        path=Path(os.environ['AGENTTUNE_SOURCE_DIR'])/'src/agenttune/agentic/rollout_engines/rollout_factory.py'
        tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_rollout_fn')
        names={a.arg for a in f.args.args}
        self.assertTrue({'pre_step_hook','post_step_hook','on_trajectory_end','tools','async_rollouts','context_length','tool_result_format'}<=names)
