/* Progressive enhancement: native Plotly toolbar, shown on demand. */
(function () {
  'use strict';
  function enhance() {
    document.querySelectorAll('.main-content .dash-graph').forEach(function (graph) {
      if (graph.dataset.toolsReady) return;
      graph.dataset.toolsReady = '1';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'chart-tools-toggle';
      button.textContent = '⚙ Chart Tools ▾';
      button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-label', 'Show or hide chart tools');
      graph.parentNode.insertBefore(button, graph);
      button.addEventListener('click', function () {
        const active = graph.classList.toggle('chart-tools-open');
        button.textContent = active ? '⚙ Hide Tools ▴' : '⚙ Chart Tools ▾';
        button.setAttribute('aria-expanded', String(active));
        window.dispatchEvent(new Event('resize'));
      });
    });
  }
  function start() {
    enhance();
    const root = document.getElementById('page-content');
    if (root) new MutationObserver(enhance).observe(root, {childList:true,subtree:true});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
