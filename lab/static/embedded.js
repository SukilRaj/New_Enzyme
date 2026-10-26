(() => {
 const module=document.body.dataset.workspaceModule;window.workspaceActive=true;
 const notify=route=>parent.postMessage({type:'workspace-route',route},location.origin);
 const links={'/':'overview','/analysis':'conditions','/pet':'pet','/evolution':'evolution','/training':'training'};
 document.addEventListener('click',event=>{
  const link=event.target.closest('a[href]');if(!link||link.hasAttribute('download')||link.target==='_blank')return;
  const url=new URL(link.href,location.href);if(url.origin!==location.origin)return;
  if(url.pathname.startsWith('/api/')||!Object.hasOwn(links,url.pathname))return;
  if(url.hash&&module==='overview'&&['#evidence','#experiments','#roadmap'].includes(url.hash)){event.preventDefault();notify(url.hash.slice(1));return;}
  if(link.getAttribute('href')==='#')return;
  event.preventDefault();notify(url.hash.slice(1)||links[url.pathname]);
 });
 function subnav(){
  const buttons=module==='pet'?[...document.querySelectorAll('.research-nav nav button')]:module==='evolution'?[...document.querySelectorAll('aside nav button')]:[];
  const main=document.querySelector(module==='pet'?'.research-main':'main');
  if(!main||!buttons.length||document.querySelector('.workspace-subnav'))return;
  const nav=document.createElement('nav');nav.className='workspace-subnav';nav.setAttribute('aria-label',module==='pet'?'Discovery sections':'Evolution sections');
  buttons.forEach((b,index)=>{const copy=document.createElement('button');copy.textContent=b.textContent.replace(/^\s*(0[1-4]|[◉▦◷◇])\s*/,'').trim();copy.onclick=()=>{b.click();nav.querySelectorAll('button').forEach(x=>x.classList.toggle('selected',x===copy));if(module==='pet'&&index<2)notify(index===1?'structure':'pet');};copy.classList.toggle('selected',index===0);nav.append(copy);});main.prepend(nav);
 }
 function navigate(section){
  if(module==='analysis'){
   const button=document.querySelector(`[data-tab="${section}"]`);if(button)button.click();window.scrollTo({top:0,behavior:'instant'});
  }else if(module==='overview'){
   if(section==='overview'){window.scrollTo({top:0,behavior:'instant'});return;}
   document.getElementById(section)?.scrollIntoView({block:'start'});if(section==='experiments')document.getElementById('refresh')?.click();
  }else if(module==='pet'){
   const index={design:0,structure:1,evaluation:2,data:3}[section]??0;
   document.querySelectorAll('.research-nav nav button')[index]?.click();
   document.querySelectorAll('.workspace-subnav button').forEach((button,i)=>button.classList.toggle('selected',i===index));
   window.scrollTo({top:0,behavior:'instant'});
  }else if(module==='evolution'){document.querySelector(`[data-page="${section}"]`)?.click();}
 }
 let pending=location.hash.slice(1);
 window.addEventListener('message',event=>{if(event.origin!==location.origin||event.source!==parent)return;if(event.data?.type==='workspace-active')window.workspaceActive=event.data.active;if(event.data?.type==='workspace-section'){pending=event.data.section;subnav();navigate(pending);}});
 subnav();if(pending)navigate(pending);
 if(module==='pet'&&!document.querySelector('.workspace-subnav')){
  const observer=new MutationObserver(()=>{subnav();if(document.querySelector('.workspace-subnav')){navigate(pending||'design');observer.disconnect();}});observer.observe(document.body,{childList:true,subtree:true});
 }
})();
