/* Native Plotly tools: move the actual modebar into a toggleable row. */
(function(){
 'use strict';
 function enhance(){
  document.querySelectorAll('.main-content .dash-graph').forEach(function(graph){
   if (!graph.parentElement) return;
   var holder=graph.previousElementSibling;
   if (!holder || !holder.classList.contains('chart-tools-holder')) {
    holder=document.createElement('div'); holder.className='chart-tools-holder';
    var btn=document.createElement('button'); btn.type='button'; btn.className='chart-tools-toggle';
    btn.textContent='⚙ Chart Tools ▾'; btn.setAttribute('aria-expanded','false');
    var tray=document.createElement('div'); tray.className='plotly-tools-tray';
    holder.appendChild(btn); holder.appendChild(tray); graph.parentElement.insertBefore(holder,graph);
    btn.addEventListener('click',function(){
     var open=holder.classList.toggle('tools-open');
     btn.textContent=open?'⚙ Hide Tools ▴':'⚙ Chart Tools ▾';
     btn.setAttribute('aria-expanded',String(open));
     sync(graph,holder);
    });
   }
   sync(graph,holder);
  });
 }
 function sync(graph,holder){
  var plot=graph.querySelector('.js-plotly-plot'); if(!plot) return;
  var tray=holder.querySelector('.plotly-tools-tray');
  var bar=plot.querySelector('.modebar-container');
  if(bar && bar.parentElement!==tray) tray.appendChild(bar);
  /* If Plotly redraws its toolbar, it will be moved back to this row. */
  var btn=holder.querySelector('.chart-tools-toggle');
  if(btn) btn.disabled=!tray.querySelector('.modebar-container');
 }
 var queued=false;
 function schedule(){if(queued)return; queued=true; requestAnimationFrame(function(){queued=false; enhance();});}
 function start(){
  schedule();
  var root=document.getElementById('page-content');
  if(root) new MutationObserver(schedule).observe(root,{childList:true,subtree:true});
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start); else start();
})();
