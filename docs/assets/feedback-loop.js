(function () {
  'use strict';

  var DURATION = 12000;
  var PHASES = [
    { name: 'compose', label: 'Compose', start: 0, end: 3000, nodes: ['agents', 'response'] },
    { name: 'respond', label: 'Feedback', start: 3000, end: 6000, nodes: ['response', 'toolkit', 'user'] },
    { name: 'route', label: 'Route', start: 6000, end: 11000, nodes: ['feedback', 'agents'] },
    { name: 'hold', label: 'Route', start: 11000, end: DURATION, nodes: [] }
  ];

  function initializeFeedbackLoops() {
    var figures = document.querySelectorAll('figure.docs-rollout[data-feedback-loop]');
    if (!figures.length) return;
    var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    figures.forEach(function (figure) {
      var svg = figure.querySelector('.docs-rollout-stage > svg');
      var playButton = figure.querySelector('[data-rollout-play]');
      var replayButton = figure.querySelector('[data-rollout-replay]');
      var status = figure.querySelector('[data-rollout-status]');
      var progress = figure.querySelector('[data-rollout-progress]');
      if (!svg || !playButton || !replayButton || !status || !progress || figure.dataset.enhanced) return;

      // The path and dot share a parent coordinate system; no SVG ID lookup is needed.
      var edges = Array.from(svg.querySelectorAll('[data-flow-edge]')).map(function (group) {
        var path = group.querySelector('path');
        var dot = group.querySelector('circle');
        var length = 0;
        if (path && dot) {
          try { length = path.getTotalLength(); } catch (_) { /* Leave malformed edges static. */ }
        }
        var delay = Number(group.dataset.flowDelay) || 0;
        return {
          group: group, path: path, dot: dot, phase: group.dataset.flowEdge,
          length: length, delay: Math.max(0, Math.min(0.7, delay))
        };
      }).filter(function (edge) { return Number.isFinite(edge.length) && edge.length > 0; });
      if (!edges.length) return;

      var phaseElements = Array.from(svg.querySelectorAll('[data-flow-phase]'));
      var nodes = Array.from(svg.querySelectorAll('[data-flow-node]'));
      var playLabel = playButton.querySelector('[data-rollout-play-label]');
      var elapsed = 0;
      var previousTime = null;
      var frame = 0;
      var started = false;
      var manuallyPaused = false;
      var inViewport = false;
      var pageHidden = false;
      var printing = false;
      var state = 'ready';
      var renderedPhase = null;

      function renderTimeline(staticView) {
        var phase = PHASES.find(function (item) { return elapsed < item.end; }) || PHASES[0];
        var phaseName = staticView ? 'static' : phase.name;
        if (renderedPhase !== phaseName) {
          renderedPhase = phaseName;
          figure.dataset.phase = phaseName;
          phaseElements.forEach(function (element) {
            var phases = element.dataset.flowPhase.split(/\s+/);
            element.classList.toggle('is-active', staticView || phases.indexOf(phase.name) !== -1);
          });
          nodes.forEach(function (node) {
            node.classList.toggle('is-active', !staticView && phase.nodes.indexOf(node.dataset.flowNode) !== -1);
          });
          edges.forEach(function (edge) {
            edge.group.classList.toggle('is-active', !staticView && edge.phase === phase.name);
          });
          status.textContent = staticView ? 'Static' : phase.label;
          progress.setAttribute('aria-valuetext', staticView ? 'Static diagram' : phase.label);
        }
        edges.forEach(function (edge) {
          var active = !staticView && edge.phase === phase.name;
          if (!active) {
            edge.dot.style.opacity = '0';
            return;
          }
          var phaseProgress = (elapsed - phase.start) / (phase.end - phase.start);
          var amount = Math.max(0, Math.min(1, (phaseProgress - edge.delay) / (1 - edge.delay)));
          var point = edge.path.getPointAtLength(edge.length * amount);
          edge.dot.setAttribute('transform', 'translate(' + point.x + ' ' + point.y + ')');
          edge.dot.style.opacity = String(Math.min(1, amount * 8, (1 - amount) * 8));
        });
        progress.max = DURATION;
        progress.value = staticView ? DURATION : elapsed;
      }

      function renderControls() {
        figure.dataset.state = state;
        figure.dataset.motion = reducedMotion.matches ? 'reduced' : 'full';
        var playing = state === 'playing';
        var label = playing ? 'Pause' : 'Play';
        var accessibleLabel = playing ? 'Pause feedback animation' : 'Play feedback animation';
        if (playLabel) playLabel.textContent = label;
        playButton.setAttribute('aria-label', accessibleLabel);
        playButton.title = accessibleLabel;
        playButton.disabled = reducedMotion.matches || printing;
        replayButton.disabled = reducedMotion.matches || printing;
      }

      function stopFrame() {
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        previousTime = null;
      }

      function tick(timestamp) {
        frame = 0;
        if (state !== 'playing') return;
        if (previousTime !== null) elapsed = (elapsed + timestamp - previousTime) % DURATION;
        previousTime = timestamp;
        renderTimeline(false);
        frame = window.requestAnimationFrame(tick);
      }

      function synchronize() {
        var staticView = reducedMotion.matches || printing;
        if (staticView) {
          stopFrame();
          state = 'static';
        } else {
          var visible = inViewport && !document.hidden && !pageHidden;
          if (visible) started = true;
          if (started && !manuallyPaused && visible) {
            state = 'playing';
            if (!frame) frame = window.requestAnimationFrame(tick);
          } else {
            stopFrame();
            state = started ? 'paused' : 'ready';
          }
        }
        renderTimeline(staticView);
        renderControls();
      }

      playButton.addEventListener('click', function () {
        if (reducedMotion.matches || printing) return;
        manuallyPaused = state === 'playing';
        synchronize();
      });
      replayButton.addEventListener('click', function () {
        if (reducedMotion.matches || printing) return;
        stopFrame();
        elapsed = 0;
        manuallyPaused = false;
        synchronize();
      });

      figure.dataset.enhanced = 'true';
      var controls = figure.querySelector('.docs-rollout-controls');
      if (controls) controls.hidden = false;
      synchronize();

      if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            inViewport = entry.isIntersecting && entry.intersectionRatio >= 0.15;
            synchronize();
          });
        }, { threshold: [0, 0.15] });
        observer.observe(figure);
      } else {
        var updateViewport = function () {
          var rect = figure.getBoundingClientRect();
          inViewport = rect.bottom > 0 && rect.top < window.innerHeight && rect.right > 0 && rect.left < window.innerWidth;
          synchronize();
        };
        window.addEventListener('scroll', updateViewport, { passive: true });
        window.addEventListener('resize', updateViewport);
        updateViewport();
      }

      document.addEventListener('visibilitychange', synchronize);
      window.addEventListener('pagehide', function () { pageHidden = true; synchronize(); });
      window.addEventListener('pageshow', function () { pageHidden = false; synchronize(); });
      window.addEventListener('beforeprint', function () { printing = true; synchronize(); });
      window.addEventListener('afterprint', function () { printing = false; synchronize(); });
      if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', synchronize);
      else reducedMotion.addListener(synchronize);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeFeedbackLoops, { once: true });
  } else {
    initializeFeedbackLoops();
  }
})();
