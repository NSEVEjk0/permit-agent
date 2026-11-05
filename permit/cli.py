from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .corpus import write_corpus,load_cases,canonical,digest
from .runs import MODEL,MODEL_REVISION,AGENTTUNE_REVISION,AUDITKIT_REVISION

def main():
    parser=argparse.ArgumentParser(description='Permit reproducible training and current-authority evaluation.')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('corpus');p.add_argument('--out',type=Path,default=Path('data'))
    p=commands.add_parser('fixtures');p.add_argument('--out',type=Path,default=Path('web/examples/harness.json'))
    p=commands.add_parser('doctor')
    for name in ('train','evaluate'):
        p=commands.add_parser(name);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,required=True)
        p.add_argument('--execute',action='store_true');p.add_argument('--wall-seconds',type=int,default=1800)
        if name=='train':p.add_argument('--max-steps',type=int,default=20)
        else:
            p.add_argument('--split',choices=['dev','heldout'],default='heldout');p.add_argument('--limit',type=int,default=12)
            p.add_argument('--adapter',type=Path);p.add_argument('--trials',type=int,default=1);p.add_argument('--profile',choices=['plain','policy'],default='policy')
    p=commands.add_parser('audit');p.add_argument('file',type=Path);p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=commands.add_parser('verify');p.add_argument('file',type=Path)
    p=commands.add_parser('compare');p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='corpus':print(json.dumps(write_corpus(args.out)));return
    if args.command=='fixtures':
        from .fixtures import examples
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(canonical(examples()));print('Wrote disclosed scripted harness examples; no model inference or training occurred.');return
    if args.command=='doctor':
        print(json.dumps({'model':MODEL,'revision':MODEL_REVISION,'license_confirmed':os.getenv('PERMIT_MODEL_LICENSE_ACCEPTED')=='true',
              'model_access_token_present':bool(os.getenv('HF_TOKEN')),'training_automatic':False,'gpu_run':'Requires the configured CUDA machine and an explicit bounded --execute command.'}));return
    if args.command in ('train','evaluate'):
        split='train' if args.command=='train' else args.split;cases=load_cases(args.data/(split+'.jsonl'))
        if args.command=='evaluate':
            if not 1<=args.limit<=144:raise SystemExit('Choose a bounded case limit of 1–144.')
            cases=cases[:args.limit]
        plan={'command':args.command,'execute':args.execute,'model':MODEL,'revision':MODEL_REVISION,'agenttune_revision':AGENTTUNE_REVISION,'auditkit_revision':AUDITKIT_REVISION,
              'cases':len(cases),'split':split,'case_ids':[c.id for c in cases],'wall_seconds':args.wall_seconds,'output':str(args.out)}
        if not args.execute:print(json.dumps(plan));return
        from .runs import train,evaluate
        result=train(cases,args.out,args.max_steps,args.wall_seconds) if args.command=='train' else evaluate(cases,args.out,args.adapter,args.profile,args.trials,args.wall_seconds)
        print(json.dumps({'completed':result['status']=='completed','output':str(args.out)}));return
    if args.command=='verify':
        from .runs import verify_report
        print(json.dumps(verify_report(json.loads(args.file.read_text()))));return
    if args.command=='audit':
        from .bridge import audit
        result=audit(args.file,load_cases(args.cases))
    else:
        from .metrics import compare
        from .runs import verify_report
        before=json.loads(args.before.read_text());after=json.loads(args.after.read_text());verify_report(before);verify_report(after);result=compare(before,after)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(canonical(result));print(json.dumps({'output':str(args.out)}))

if __name__=='__main__':main()
