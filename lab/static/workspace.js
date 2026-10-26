const $=s=>document.querySelector(s);
const routes={
 overview:{module:'overview',path:'/',title:'Overview',detail:'Your experiments, models and evidence in one place.',domain:'RESEARCH OVERVIEW'},
 pet:{module:'pet',path:'/pet',section:'design',title:'Enzyme discovery',detail:'Compare PET enzymes under defined assay conditions.',domain:'PET · TPA CONCENTRATION'},
 conditions:{module:'analysis',path:'/analysis',section:'conditions',title:'Environmental analysis',detail:'Explore the response to temperature, pH and reaction time.',domain:'PET · CONDITION RESPONSE'},
 images:{module:'analysis',path:'/analysis',section:'images',title:'Protein images',detail:'Generate reproducible views from sequence-matched coordinates.',domain:'STRUCTURAL REPRESENTATIONS'},
 structure:{module:'pet',path:'/pet',section:'structure',title:'Sequence & structure',detail:'Inspect molecules, substitutions and source evidence.',domain:'PROTEIN INSPECTION'},
 evolution:{module:'evolution',path:'/evolution',section:'studio',title:'Directed evolution',detail:'Train fitness predictors and follow the sequence search.',domain:'avGFP · DATASET FITNESS'},
 training:{module:'training',path:'/training',title:'Training laboratory',detail:'Follow training, inspect predictions and compare generalization.',domain:'MODEL DEVELOPMENT'},
 evaluation:{module:'analysis',path:'/analysis',section:'evaluation',title:'Model comparison',detail:'Compare methods using the same outcomes and splits.',domain:'EVALUATION EVIDENCE'},
 experiments:{module:'overview',path:'/',section:'experiments',title:'Experiment registry',detail:'Return to saved experiments and export their provenance.',domain:'SAVED RESEARCH'},
 evidence:{module:'overview',path:'/',section:'evidence',title:'Evidence & readiness',detail:'Check what the current models can support.',domain:'MODEL READINESS'},
 roadmap:{module:'overview',path:'/',section:'roadmap',title:'Research roadmap',detail:'Track the engineering and scientific work still required.',domain:'RESEARCH PLAN'}
};
const frames=new Map();let current='';let transitionId=0;
function routeName(){let key=location.hash.slice(1).replace(/^\//,'');if(routes[key])return key;return {'/analysis':'conditions','/pet':'pet','/evolution':'evolution','/training':'training'}[location.pathname]||'overview';}
function activate(name){
 const route=routes[name]||routes.overview;current=name;const generation=++transitionId;
 $('#workspace-title').textContent=route.title;$('#workspace-description').textContent=route.detail;$('#workspace-domain').textContent=route.domain;document.title=`${route.title} · EnzAIme × Helix`;
 document.querySelectorAll('[data-route]').forEach(a=>{a.classList.toggle('active',a.dataset.route===name);if(a.dataset.route===name)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
 $('#workspace-error').hidden=true;
 for(const [key,frame] of frames){frame.hidden=key!==route.module;frame.contentWindow?.postMessage({type:'workspace-active',active:key===route.module},location.origin);}
 let frame=frames.get(route.module);
 const send=()=>frame.contentWindow.postMessage({type:'workspace-section',section:route.section||'overview'},location.origin);
 if(!frame){
  frame=document.createElement('iframe');frame.className='workspace-view';frame.title=route.title;frame.dataset.module=route.module;
  frame.src=route.path+'?embedded=1'+(route.section?'#'+route.section:'');frame.hidden=false;
  frames.set(route.module,frame);$('#workspace-views').append(frame);$('#workspace-loading').hidden=false;
  frame.onload=()=>{frame.dataset.ready='true';if(routes[current].module===route.module){$('#workspace-loading').hidden=true;frame.hidden=false;frame.contentWindow.postMessage({type:'workspace-section',section:routes[current].section||'overview'},location.origin);}};
  frame.onerror=()=>{if(generation===transitionId){$('#workspace-loading').hidden=true;$('#workspace-error').textContent='This workspace could not load. Check that Start-Research.cmd is running.';$('#workspace-error').hidden=false;}};
 }else{frame.hidden=false;$('#workspace-loading').hidden=frame.dataset.ready==='true';send();}
 if(!matchMedia('(prefers-reduced-motion: reduce)').matches)frame.animate([{opacity:.3,transform:'translateY(7px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'cubic-bezier(.2,.7,.2,1)'});
}
window.addEventListener('hashchange',()=>activate(routeName()));
window.addEventListener('message',event=>{
 if(event.origin!==location.origin||![...frames.values()].some(frame=>frame.contentWindow===event.source))return;
 if(event.data?.type==='workspace-route'&&routes[event.data.route]&&event.data.route!==current)location.hash=event.data.route;
});
async function health(){
 try{const response=await fetch('/api/research/overview');if(!response.ok)throw Error();const d=await response.json();$('#service-status').textContent=d.pet_service==='online'?'Both engines connected':'PET engine offline';$('#service-dot').classList.toggle('online',d.pet_service==='online');}
 catch{$('#service-status').textContent='Connection unavailable';$('#service-dot').classList.remove('online');}
 try{const response=await fetch('/api/training/status');if(!response.ok)return;const data=await response.json();const running=data.status==='running';$('#training-indicator').hidden=!running;$('#training-chip').hidden=!running;if(running)$('#training-chip').textContent=`Training · ${Math.round(data.progress||0)}%`;}
 catch{}
}
activate(routeName());health();setInterval(health,15000);
