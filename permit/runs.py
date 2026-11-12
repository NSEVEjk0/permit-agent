from __future__ import annotations
import hashlib
import json
import os
import signal
import time
from importlib.metadata import version
from pathlib import Path
from .corpus import Case,SYSTEM,VERSION,canonical,digest
from .environment import replay

MODEL='CohereLabs/tiny-aya-global'
MODEL_REVISION='af89d219b53ed9b13b8a4645f9c8028973510324'
AGENTTUNE_REVISION='d4db4c12d52342227c4113a70183d0f64aa2f345'
AUDITKIT_REVISION='71316da345784ff220d5ed59337441a59ab31526'

def identity(cases,profile,adapter=None):
    source=Path(__file__).parent
    return {'base_model':MODEL,'model_revision':MODEL_REVISION,'backend':'agenttune-transformers-json-tools',
            'corpus_schema':VERSION,'agenttune_revision':AGENTTUNE_REVISION,'auditkit_revision':AUDITKIT_REVISION,
            'corpus_sha256':digest([c.fingerprint for c in cases]),'case_ids':[c.id for c in cases],'prompt_profile':profile,
            'harness_sha256':digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}),
            'adapter_sha256':adapter_digest(adapter) if adapter else None,
            'libraries':{n:version(n) for n in ('agenttune','auditkit','torch','transformers','peft','trl')}}

def adapter_digest(path:Path):
    files=sorted([*path.glob('adapter*.json'),*path.glob('adapter*.safetensors')])
    if not any(p.suffix=='.safetensors' for p in files):raise ValueError('The adapter has no safetensors weights.')
    values={}
    for file in files:
        h=hashlib.sha256()
        with file.open('rb') as stream:
            while data:=stream.read(1024*1024):h.update(data)
        values[file.name]=h.hexdigest()
    return digest(values)

def ready(wall_seconds):
    if not 30<=wall_seconds<=3600:raise ValueError('Use a wall-clock cap of 30–3,600 seconds.')
    if os.getenv('PERMIT_MODEL_LICENSE_ACCEPTED')!='true':raise ValueError('Accept the gated model license legitimately, then set PERMIT_MODEL_LICENSE_ACCEPTED=true.')
    import torch
    if not torch.cuda.is_available():raise ValueError('A configured CUDA GPU is required for this bounded model run; no CPU fallback or automatic paid provisioning.')
    if int(os.getenv('WORLD_SIZE','1'))!=1:raise ValueError('This version isolates sequential rollouts in a single training process.')
    if version('agenttune')!='1.1.0' or version('auditkit')!='1.1.2':raise ValueError('Install the pinned toolkit versions.')
    def timeout(signum,frame):raise TimeoutError('The model-run wall-clock budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(wall_seconds)

def load_model(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_REVISION,dtype=torch.bfloat16,trust_remote_code=False)
    if adapter:
        from peft import PeftModel
        config=json.loads((adapter/'adapter_config.json').read_text())
        if config.get('base_model_name_or_path') not in (MODEL,''):
            raise ValueError('Adapter does not identify the allowed base model.')
        if config.get('revision')!=MODEL_REVISION:
            raise ValueError('Adapter revision differs from the pinned base model.')
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.to('cuda');return model,tokenizer

def train(cases:list[Case],out:Path,steps:int,wall_seconds:int):
    if not 1<=steps<=80 or any(c.split!='train' for c in cases):raise ValueError('Use 1–80 steps on the training split only.')
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory; training never overwrites a prior run.')
    ready(wall_seconds);out.mkdir(parents=True,exist_ok=True)
    from agenttune import create_agentic_trainer
    from datasets import Dataset
    from peft import LoraConfig
    from . import bridge
    import torch
    torch.manual_seed(721);model,tokenizer=load_model()
    bridge.REWARD_CASES.clear();bridge.REWARD_CASES.update({c.id:c for c in cases})
    rollout,tools=bridge.create_rollout(cases,out/'trajectories.jsonl')
    provenance={'schema':'permit-training-v1','mode':'AgentTune-GRPO-LoRA','identity':identity(cases,'policy'),
                'settings':{'max_steps':steps,'wall_seconds':wall_seconds,'seed':721,'num_generations':2,'rank':16},'status':'running'}
    (out/'permit-provenance.json').write_bytes(canonical(provenance));started=time.monotonic()
    try:
        trainer=create_agentic_trainer(algorithm='grpo',backend='trl',model=model,processing_class=tokenizer,
          train_dataset=Dataset.from_list([{'prompt':c.public_prompt()} for c in cases]),tools=tools,rollout_func=rollout,
          reward_funcs=bridge.permission_reward,output_dir=str(out),max_steps=steps,max_steps_per_turn=8,
          per_device_train_batch_size=2,gradient_accumulation_steps=2,num_generations=2,
          learning_rate=1e-5,max_completion_length=768,max_prompt_length=3072,use_vllm=False,
          peft_config=LoraConfig(r=16,lora_alpha=32,target_modules='all-linear',task_type='CAUSAL_LM'),
          bf16=True,gradient_checkpointing=True,seed=721,report_to='none',push_to_hub=False,save_steps=steps)
        result=trainer.train()
        trainer.trainer.save_model(str(out/'adapter'))
        tokenizer.save_pretrained(str(out/'adapter'))
        config_path=out/'adapter/adapter_config.json';config=json.loads(config_path.read_text());config['base_model_name_or_path']=MODEL;config['revision']=MODEL_REVISION;config_path.write_bytes(canonical(config))
        provenance.update(status='completed',training_result=result,adapter_sha256=adapter_digest(out/'adapter'))
    except BaseException as error:
        provenance.update(status='incomplete',error_type=type(error).__name__);raise
    finally:
        provenance['elapsed_seconds']=round(time.monotonic()-started,3);(out/'permit-provenance.json').write_bytes(canonical(provenance));signal.alarm(0)
    return provenance

def evaluate(cases:list[Case],out:Path,adapter:Path|None,profile:str,trials:int,wall_seconds:int):
    if not 1<=trials<=3 or not 1<=len(cases)<=144:raise ValueError('Use 1–3 trials over at most 144 cases per bounded run.')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new evaluation directory.')
    ready(wall_seconds);out.mkdir(parents=True,exist_ok=True)
    from .bridge import create_rollout,audit
    from .metrics import summary
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine
    import torch
    model,tokenizer=load_model(adapter);model.eval();engine=TransformersRolloutEngine(model,tokenizer)
    rollout,_=create_rollout(cases,out/'trajectories.jsonl',engine=engine)
    provenance={'schema':'permit-eval-v2','mode':'model-evaluation','identity':identity(cases,profile,adapter),'settings':{'trials':trials,'seed':721,'temperature':0.2,'max_new_tokens':768,'wall_seconds':wall_seconds},'status':'running'}
    (out/'run.json').write_bytes(canonical(provenance))
    try:
        with torch.inference_mode():
            for trial in range(trials):
                torch.manual_seed(721+trial)
                for case in cases:
                    rollout([case.public_prompt(profile)],max_new_tokens=768,temperature=0.2,do_sample=True)
        result=audit(out/'trajectories.jsonl',cases)
        from collections import Counter
        if Counter(row['case_id'] for row in result['cases'])!=Counter({case.id:trials for case in cases}):
            raise ValueError('Evaluation evidence is missing or duplicates selected case trials.')
        provenance.update(status='completed',results=result,summary=summary(result['cases']))
    except BaseException as error:provenance.update(status='incomplete',error_type=type(error).__name__);raise
    finally:
        seal_report(provenance);(out/'run.json').write_bytes(canonical(provenance));signal.alarm(0)
    return provenance

def seal_report(report:dict):
    payload={key:value for key,value in report.items() if key not in ('report_sha256','canonical_payload')}
    encoded=canonical(payload)
    report['canonical_payload']=encoded.decode()
    report['report_sha256']=hashlib.sha256(encoded).hexdigest()
    return report

def verify_report(report:dict):
    valid = (report.get('schema')=='permit-eval-v2' and report.get('mode')=='model-evaluation') or (report.get('schema')=='permit-probe-v1' and report.get('mode')=='hosted-model-probe')
    if not valid or report.get('status')!='completed':raise ValueError('A completed real-model evaluation or hosted probe is required.')
    payload={key:value for key,value in report.items() if key not in ('report_sha256','canonical_payload')}
    encoded=canonical(payload)
    if report.get('canonical_payload')!=encoded.decode() or hashlib.sha256(encoded).hexdigest()!=report.get('report_sha256'):
        raise ValueError('Evaluation report integrity differs.')
    return {'integrity_verified':True,'identity':report['identity'],'scope':'Self-reported model-run provenance; reproducible scripts, not remote attestation.'}
