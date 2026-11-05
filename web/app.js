let bundle,index=0,route=0;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const stable=v=>v===null||typeof v!=='object'?JSON.stringify(v):Array.isArray(v)?'['+v.map(stable).join(',')+']':'{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+stable(v[k])).join(',')+'}';
const hash=async v=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(v)))].map(b=>b.toString(16).padStart(2,'0')).join('');
function render(){
 const example=bundle.examples[index],e=example.routes[route].evidence,c=example.case;
 document.querySelector('#case-list').innerHTML=bundle.examples.map((x,i)=>`<button class="case-button ${index===i?'active':''}" data-case="${i}" aria-pressed="${index===i}"><strong>${i===0?'Permission remains':i===1?'Delivery permission revoked':'Destination scope changes'}</strong>${i===0?'Complete the whole task':i===1?'Keep the draft; stop delivery':'Respect the newly allowed destination'}</button>`).join('');
 document.querySelector('#case-title').textContent=c.family.replaceAll('-',' ');
 document.querySelector('#task').textContent=c.prompt;
 document.querySelector('#route-list').innerHTML=example.routes.map((x,i)=>`<button data-route="${i}" aria-pressed="${i===route}">${esc(x.label)}</button>`).join('');
 const delivery=e.final_policy.deliver_summary&&e.final_policy.destinations.includes(c.destination);
 document.querySelector('#delivery-state').textContent=delivery?'Allowed':c.change==='narrow'?'Destination outside current scope':'Revoked';
 let prior=-1;
 document.querySelector('#trace').innerHTML=e.attempts.map((a,i)=>{
  const control=e.controls.find(x=>x.turn>prior&&x.turn<=a.turn);prior=a.turn;
  const note=a.tool==='read_record'?`<small>Untrusted record note: ${esc(c.note)}</small>`:'';
  return `${control?`<div class="control-row"><strong>Gateway policy changed: version ${control.snapshot.policy_version}.</strong><br>${control.kind==='revoke'?'Delivery permission has been revoked. The draft remains allowed.':'Delivery is restricted to a different destination.'}</div>`:''}<div class="trace-row ${a.attempted_violation?'bad':''}"><span class="step-num">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(a.tool)}</strong><small>Turn ${a.turn+1} · policy version ${a.policy_version}${a.reason?' · '+esc(a.reason.replaceAll('_',' ')):''}</small>${note}</div><b>${a.executed?'Executed':'Blocked'}</b></div>`;
 }).join('');
 const violations=e.attempts.filter(a=>a.attempted_violation).length;
 document.querySelector('#outcome').innerHTML=`<div class="outcome-card ${violations?'warn':''}"><strong>${violations?'The forbidden attempt was blocked.':'Useful work stayed within current authority.'}</strong>${e.final_state.draft?'The checked draft exists.':'No checked draft exists.'} ${e.final_state.deliveries.length?'The authorized delivery exists.':'No delivery occurred.'}<br>${violations} attempted permission violation${violations===1?'':'s'} · ${e.attempts.filter(a=>a.executed_violation).length} executed violations. This is a scripted harness result.</div>`;
}
document.addEventListener('click',event=>{
 if(!(event.target instanceof Element))return;
 const caseButton=event.target.closest('[data-case]'),routeButton=event.target.closest('[data-route]');
 if(caseButton){index=Number(caseButton.dataset.case);route=0;render();}
 if(routeButton){route=Number(routeButton.dataset.route);render();}
});
document.querySelector('#download').addEventListener('click',()=>{
 const url=URL.createObjectURL(new Blob([JSON.stringify(bundle,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='permit-scripted-harness-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.querySelector('#report-file').addEventListener('change',async event=>{
 const file=event.target.files?.[0],result=document.querySelector('#report-result');if(!file)return;
 try{
  if(file.size>16*1024*1024)throw new Error('Use a report smaller than 16 MiB.');
  const r=JSON.parse(await file.text());
  if(r.schema!=='permit-eval-v1'||r.mode!=='model-evaluation'||r.status!=='completed')throw new Error('Choose a completed real-model Permit evaluation report; scripted harness files are not model results.');
  const {report_sha256,...payload}=r;if(await hash(stable(payload))!==report_sha256)throw new Error('The report fingerprint does not match its contents.');
  const m=r.summary?.metrics;
  if(!m||!Number.isFinite(m.safe_complete?.mean)||!Number.isInteger(r.summary.episodes)||r.summary.episodes<1)throw new Error('The report is missing measured outcomes.');
  result.innerHTML=`<strong>Local report fingerprint agrees.</strong><br>${esc(r.identity.base_model)} · ${r.summary.episodes} recorded episodes · ${(m.safe_complete.mean*100).toFixed(1)}% safe completion on this subset.<br>Model identity is supplied by the run artifact. Integrity is not remote attestation. Inspect the original trajectories and reproduce the run to establish the model claim.`;
 }catch(error){result.textContent=error.message;}
});
async function start(){
 try{const response=await fetch('/examples/harness.json');if(!response.ok)throw new Error('The harness evidence could not be loaded.');bundle=await response.json();const {payload_sha256,...payload}=bundle;
  if(bundle.mode!=='scripted-harness-no-model-no-training'||await hash(stable(payload))!==payload_sha256)throw new Error('The harness bundle does not match its recorded provenance.');render();
 }catch(error){document.querySelector('#task').textContent=error.message;}
}
void start();
