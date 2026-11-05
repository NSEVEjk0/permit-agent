import copy
import tempfile
import unittest
from pathlib import Path
from permit.corpus import cases,write_corpus,load_cases,FAMILIES
from permit.environment import Environment,score,replay
from permit.fixtures import run
from permit.bridge import Hooks,permission_reward
from permit.runs import ready

class EnvironmentTests(unittest.TestCase):
    def setUp(self):self.cases=cases()
    def test_clean_task_requires_real_saved_and_delivered_state(self):
        c=self.cases[0];e=run(c);self.assertTrue(score(e,c)['safe_complete'])
        empty=Environment(c);empty.begin_turn(0)
        self.assertFalse(score(empty.evidence(),c)['task_complete'])
    def test_revocation_preserves_useful_draft_and_denies_actual_effect(self):
        c=self.cases[2];good=run(c);bad=run(c,True)
        self.assertTrue(score(good,c)['safe_complete']);self.assertTrue(score(bad,c)['task_complete'])
        self.assertEqual(score(bad,c)['attempted_violations'],1);self.assertEqual(score(bad,c)['executed_violations'],0)
        self.assertFalse(score(bad,c)['safe_complete']);self.assertEqual(bad['final_state']['deliveries'],[])
    def test_destination_narrowing_and_older_version_do_not_restore_authority(self):
        c=self.cases[3];e=Environment(c);e.begin_turn(2)
        r=e.call('deliver_summary',{'destination':c.destination,'total_cents':c.total,'policy_version':1})
        self.assertEqual(r['error'],'revoked_or_out_of_scope')
        self.assertFalse(r['ok']);self.assertTrue(e.attempts[-1]['attempted_violation'])
    def test_gateway_announces_next_turn_permission_in_real_tool_output(self):
        c=self.cases[2];e=Environment(c);e.begin_turn(1)
        r=e.call('read_record',{'record_id':c.record_id,'policy_version':1})
        self.assertTrue(r['ok']);self.assertFalse(r['PERMIT_CONTROL']['policy']['deliver_summary'])
        self.assertEqual(e.version,1);e.begin_turn(2);self.assertEqual(e.version,2)
    def test_stale_version_rejects_allowed_draft_without_unsafe_label(self):
        c=self.cases[2];e=Environment(c);e.begin_turn(1);e.call('read_record',{'record_id':c.record_id,'policy_version':1});e.begin_turn(2)
        e.call('save_draft',{'record_id':c.record_id,'total_cents':c.total,'policy_version':1})
        self.assertEqual(e.attempts[-1]['reason'],'stale_policy_version');self.assertFalse(e.attempts[-1]['attempted_violation'])
    def test_unknown_permission_tool_and_wrong_record_are_denied(self):
        c=self.cases[0];e=Environment(c);e.begin_turn(0)
        e.call('grant_permission',{'deliver':True});e.call('read_record',{'record_id':'other','policy_version':1})
        self.assertEqual(score(e.evidence(),c)['attempted_violations'],2);self.assertFalse(e.read)
    def test_invalid_totals_and_delivery_before_saved_draft_cannot_pass(self):
        c=self.cases[0];e=Environment(c);e.begin_turn(0)
        e.call('read_record',{'record_id':c.record_id,'policy_version':1})
        e.call('save_draft',{'record_id':c.record_id,'total_cents':True,'policy_version':1})
        e.call('deliver_summary',{'destination':c.destination,'total_cents':c.total,'policy_version':1})
        self.assertIsNone(e.draft);self.assertEqual(e.deliveries,[])
    def test_evidence_replays_and_tampered_final_state_fails(self):
        c=self.cases[2];e=run(c);self.assertTrue(replay(e,c)['safe_complete'])
        e['final_state']['draft']['total_cents']+=1
        with self.assertRaises(ValueError):replay(e,c)
    def test_family_splits_and_prompt_projection_do_not_leak_oracle(self):
        cs=self.cases
        self.assertEqual(len({c.prompt for c in cs}),len(cs));self.assertEqual(len({c.record_id for c in cs}),len(cs))
        groups=[{c.family for c in cs if c.split==s} for s in FAMILIES]
        for i,g in enumerate(groups):
            for h in groups[i+1:]:self.assertFalse(g&h)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);m=write_corpus(p);self.assertEqual(m['counts'],{'train':144,'dev':48,'heldout':72})
            self.assertEqual(load_cases(p/'heldout.jsonl'),[c for c in cs if c.split=='heldout'])
            prompts=(p/'heldout-prompts.jsonl').read_text()
            for key in ('revoke_turn','items','change','note','total_cents'):self.assertNotIn('"'+key+'":',prompts)
    def test_new_rollouts_reset_and_hooks_do_not_inject_unmasked_tokens(self):
        hooks=Hooks(self.cases[:2]);conversation=self.cases[0].public_prompt();original=copy.deepcopy(conversation)
        hooks.before(0,conversation,[]);self.assertEqual(conversation,original)
        hooks.active.call('read_record',{'record_id':self.cases[0].record_id,'policy_version':1})
        hooks.before(0,self.cases[1].public_prompt(),[]);self.assertFalse(hooks.active.read)
    def test_no_trajectory_rewards_and_missing_access_fail_loudly(self):
        with self.assertRaises(ValueError):permission_reward(['I finished.'])
        with self.assertRaises(ValueError):ready(5)

if __name__=='__main__':unittest.main()
