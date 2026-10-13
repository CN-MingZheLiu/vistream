const $ = (s) => document.querySelector(s);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const canvas = $('#trajectory');
const ctx = canvas.getContext('2d');
let datasets = [], scene, progress = 1, playing = !reducedMotion.matches, display = 'compare', loopMode = 'loopOn';
let visible = false, previous = 0, elapsedHold = 0, speed = 1;
const colors = {visual:'#4b91ff',current:'#ff983f',gt:'#c8d3df'};

const filmDialog = $('#film-dialog'), film = $('#film');
document.querySelectorAll('[data-open-film]').forEach(b => b.addEventListener('click', () => {
  filmDialog.showModal(); film.play().catch(() => {});
}));
$('#pipeline-open').addEventListener('click', () => $('#pipeline-dialog').showModal());
document.querySelectorAll('dialog').forEach(d => {
  d.querySelector('.dialog-close').addEventListener('click', () => d.close());
  d.addEventListener('click', e => { if(e.target === d && (e.clientY < d.getBoundingClientRect().top || e.clientY > d.getBoundingClientRect().bottom || e.clientX < d.getBoundingClientRect().left || e.clientX > d.getBoundingClientRect().right)) d.close(); });
});
filmDialog.addEventListener('close', () => film.pause());
const hero = $('#hero-video'), heroMotion = $('#hero-motion');
fetch('assets/hero-loop.mp4', {method:'HEAD'}).then(r => {
  if(!r.ok)return;
  hero.src='assets/hero-loop.mp4'; heroMotion.hidden=false;
  if(!reducedMotion.matches)hero.play().catch(() => {});
}).catch(() => {});
function heroState(){heroMotion.textContent=hero.paused?'▶':'Ⅱ';heroMotion.setAttribute('aria-label',hero.paused?'Play background video':'Pause background video');}
heroMotion.addEventListener('click',()=>{if(hero.paused)hero.play().catch(()=>{});else hero.pause();});
hero.addEventListener('play',heroState);hero.addEventListener('pause',heroState);
document.addEventListener('visibilitychange',()=>{if(document.hidden){hero.pause();previous=0;}else if(!reducedMotion.matches){hero.play().catch(()=>{});}});

function playState(){const b=$('#replay-toggle');b.textContent=playing?'Ⅱ':'▶';b.setAttribute('aria-label',playing?'Pause trajectory replay':'Play trajectory replay');}
function selectScene(id){
  scene=datasets.find(d=>d.id===id); if(!scene)return;
  progress=1;elapsedHold=0;previous=0;
  document.querySelectorAll('.scene-tab').forEach(b=>{const active=b.dataset.scene===id;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
  $('#trajectory-panel').setAttribute('aria-labelledby','tab-'+id);
  $('#scene-title').textContent=scene.title;
  $('#scene-caption').textContent=id==='kitti06'?'New KITTI run · offline repair configuration':'Paper experiment · first recorded repetition';
  $('#case-protocol').textContent=id==='kitti06'?'Sim(3) ATE · fresh KITTI run · offline whole-sequence repair configuration · fixed full-support alignment. Distinct from the paper’s EuRoC examples.':'SE(3) ATE · first recorded repetition · fixed rigid alignment over shared support · no scale fitting. Display paths are subsampled; scores use all supported 3D poses.';
  $('#loop-selector').hidden=id!=='kitti06';
  updateScores();draw();
}
function updateScores(){
  if(!scene)return;
  const a=scene.ate.visual,b=scene.id==='kitti06'?scene.ate[loopMode]:scene.ate.current;
  $('#visual-ate').innerHTML=a.toFixed(2)+' <em>m</em>';
  $('#current-ate').innerHTML=b.toFixed(2)+' <em>m</em>';
  $('#case-gain').textContent=((1-b/a)*100).toFixed(1)+'%';
  $('#plot-labels').lastElementChild.innerHTML='<i class="dot orange"></i>VIStream'+(scene.id==='kitti06'?(loopMode==='loopOn'?' · loop on':' · loop off'):'');
}
document.querySelectorAll('.scene-tab').forEach(b=>{
  b.addEventListener('click',()=>selectScene(b.dataset.scene));
  b.addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll('.scene-tab:not([hidden])')];let i=tabs.indexOf(b);i=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[i].focus();selectScene(tabs[i].dataset.scene);});
});
const loopLabel=document.createElement('label');loopLabel.id='loop-selector';loopLabel.hidden=true;loopLabel.innerHTML='<span class="sr-only">VIStream loop closure</span><select aria-label="VIStream loop closure"><option value="loopOn">Loop on</option><option value="loopOff">Loop off</option></select>';
$('#case-protocol').before(loopLabel);
loopLabel.querySelector('select').addEventListener('change',e=>{loopMode=e.target.value;updateScores();draw();});
$('#replay-toggle').addEventListener('click',()=>{playing=!playing;if(playing&&progress>=1){progress=0;elapsedHold=0;}playState();draw();});
$('#restart').addEventListener('click',()=>{progress=0;elapsedHold=0;playing=true;playState();draw();});
$('#scrubber').addEventListener('input',e=>{playing=false;progress=Number(e.target.value)/1000;elapsedHold=0;playState();draw();});
$('#speed').addEventListener('change',e=>{speed=Number(e.target.value);});
for(const type of ['compare','overlay'])$('#'+type).addEventListener('click',()=>{
  display=type;
  for(const x of ['compare','overlay']){$('#'+x).classList.toggle('selected',x===type);$('#'+x).setAttribute('aria-pressed',String(x===type));}
  draw();
});
function draw(){
  if(!scene)return;
  const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(window.devicePixelRatio||1,2);
  if(!w||!h)return;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const panels=display==='compare'?[[0,w/2,['visual']],[w/2,w/2,['current']]]:[[0,w,['visual','current']]];
  if(display==='compare'){ctx.strokeStyle='#ffffff0c';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(w/2,18);ctx.lineTo(w/2,h-25);ctx.stroke();}
  const lo=scene.bounds.min,hi=scene.bounds.max,cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2;
  const target=progress*scene.durationSeconds;
  for(const [left,width,methods] of panels){
    const pad=width<300?22:44,pw=width-pad*2,ph=h-64,scale=Math.min(pw/(hi[0]-lo[0]||1),ph/(hi[1]-lo[1]||1))*.88;
    const point=p=>[left+width/2+(p[0]-cx)*scale,h/2-6-(p[1]-cy)*scale];
    ctx.save();ctx.beginPath();ctx.rect(left+8,9,width-16,h-18);ctx.clip();
    ctx.strokeStyle='#ffffff06';ctx.lineWidth=1;ctx.setLineDash([]);
    for(let j=1;j<5;j++){ctx.beginPath();ctx.moveTo(left+pad+pw*j/5,20);ctx.lineTo(left+pad+pw*j/5,h-30);ctx.stroke();ctx.beginPath();ctx.moveTo(left+pad,20+(h-50)*j/5);ctx.lineTo(left+width-pad,20+(h-50)*j/5);ctx.stroke();}
    for(const key of [...methods,'gt']){
      ctx.strokeStyle=colors[key];ctx.lineWidth=key==='gt'?1.5:2.35;ctx.setLineDash(key==='gt'?[5,5]:[]);ctx.lineJoin='round';ctx.lineCap='round';
      let finalPoint;
      for(const segment of scene.segments){
        const sourceKey=key==='current'&&scene.id==='kitti06'?loopMode:key;
        const pts=segment[sourceKey];ctx.beginPath();let started=false;
        for(let j=0;j<pts.length;j++){
          if(key!=='gt'&&segment.timeSeconds[j]>target)break;
          const p=point(pts[j]);if(!started){ctx.moveTo(...p);started=true;}else ctx.lineTo(...p);if(key!=='gt')finalPoint=p;
        }
        ctx.stroke();
      }
      if(key!=='gt'&&finalPoint){ctx.setLineDash([]);ctx.fillStyle=colors[key];ctx.shadowColor=colors[key];ctx.shadowBlur=10;ctx.beginPath();ctx.arc(...finalPoint,3.6,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;}
    }
    ctx.setLineDash([]);const metre=scene.id==='kitti06'?50:scene.id==='v203'?1:5;
    ctx.strokeStyle='#536276';ctx.lineWidth=1;const bx=left+pad,by=h-21,bw=metre*scale;
    ctx.beginPath();ctx.moveTo(bx,by-3);ctx.lineTo(bx,by);ctx.lineTo(bx+bw,by);ctx.lineTo(bx+bw,by-3);ctx.stroke();
    ctx.fillStyle='#68798f';ctx.font='9px Arial';ctx.textAlign='left';ctx.fillText(metre+' m',bx,by-8);ctx.restore();
  }
  $('#scrubber').value=Math.round(progress*1000);$('#replay-time').textContent=target.toFixed(1)+' s';
}
const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;previous=0;},{threshold:.15});observer.observe(canvas);
new ResizeObserver(draw).observe(canvas);
function animate(now){
  const dt=previous?Math.min((now-previous)/1000,.1):0;previous=now;
  if(visible&&playing&&!document.hidden&&scene){
    if(progress>=1){elapsedHold+=dt;if(elapsedHold>2.2){progress=0;elapsedHold=0;}}
    else progress=Math.min(1,progress+dt*speed/15);
    draw();
  }
  requestAnimationFrame(animate);
}
fetch('assets/trajectories.json').then(r=>{if(!r.ok)throw Error('data');return r.json();}).then(j=>{
  datasets=j.datasets;datasets.forEach(d=>{const tab=$('#tab-'+d.id);if(tab)tab.hidden=false;});selectScene('mh05');playState();requestAnimationFrame(animate);
}).catch(()=>{$('#data-error').hidden=false;});
fetch('assets/kitti-results.json').then(r=>r.json()).then(data=>{
  const body=$('#kitti-table');
  for(const row of [...data.rows,{sequence:'Mean',...data.mean}]){
    const tr=document.createElement('tr');if(row.sequence==='Mean')tr.className='mean';
    const values=[row.visual,row.loop_off,row.loop_on],best=Math.min(...values);
    const title=document.createElement('th');title.scope='row';title.textContent=row.sequence;tr.append(title);
    for(const value of values){const td=document.createElement('td');const text=value.toFixed(2);if(Math.abs(value-best)<1e-9){const strong=document.createElement('strong');strong.textContent=text;td.append(strong);}else td.textContent=text;tr.append(td);}body.append(tr);
  }
}).catch(()=>{$('#kitti-table').innerHTML='<tr><td colspan="4">Results could not load. <a href="assets/kitti-results.json">Download the data</a>.</td></tr>';});
