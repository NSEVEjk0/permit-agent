"""Disclosed scripted harness demonstrations; never model benchmarks."""
from __future__ import annotations
from .corpus import cases,digest,asdict
from .environment import Environment,score

def run(case,attempt_delivery=False):
    env=Environment(case)
    env.begin_turn(0);env.call('read_permissions',{})
    env.begin_turn(1);env.call('read_record',{'record_id':case.record_id,'policy_version':1})
    env.begin_turn(2);env.call('save_draft',{'record_id':case.record_id,'total_cents':case.total,'policy_version':env.version})
    if attempt_delivery or case.change=='none':
        env.begin_turn(3);env.call('deliver_summary',{'destination':case.destination,'total_cents':case.total,'policy_version':env.version})
    return env.evidence()

def examples():
    cs=cases();selected=[cs[0],cs[2],cs[3]]
    result={'schema':'permit-fixtures-v1','mode':'scripted-harness-no-model-no-training','notice':'These are authored tool routes executed by the real fictional gateway. They are not base or trained model scores.',
            'examples':[]}
    for case in selected:
        routes=[{'label':'Useful work under current permission','evidence':run(case)}]
        if case.change!='none':routes.append({'label':'Scripted forbidden delivery attempt','evidence':run(case,True)})
        result['examples'].append({'case':asdict(case),'routes':routes})
    result['payload_sha256']=digest(result);return result
