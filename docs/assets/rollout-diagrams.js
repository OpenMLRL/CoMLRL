(function () {
  'use strict';

  var STEP_DURATION = 1800;
  var EDGE_DURATION = 900;
  var NODE_DURATION = 450;

  function initializeRollouts() {
    var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    document.querySelectorAll('figure.docs-rollout[data-rollout]').forEach(function (figure) {
      var svg = figure.querySelector('svg');
      var playButton = figure.querySelector('[data-rollout-play]');
      var replayButton = figure.querySelector('[data-rollout-replay]');
      var status = figure.querySelector('[data-rollout-status]');
      var progress = figure.querySelector('[data-rollout-progress]');
      if (!svg || !playButton || !replayButton || !status || !progress || figure.dataset.enhanced) return;

      function originalProperty(element, name) {
        return {
          value: element.style.getPropertyValue(name),
          priority: element.style.getPropertyPriority(name)
        };
      }

      function restoreProperty(element, name, original) {
        if (original.value) {
          element.style.setProperty(name, original.value, original.priority);
        } else {
          element.style.removeProperty(name);
        }
      }

      var elements = Array.from(svg.querySelectorAll('[data-rollout-stage]')).map(function (element) {
        var edge = element.dataset.rolloutKind === 'edge';
        var paths = edge ? (element.matches('path') ? [element] : Array.from(element.querySelectorAll('path'))) : [];
        return {
          element: element,
          stage: Number(element.dataset.rolloutStage),
          edge: edge,
          opacity: originalProperty(element, 'opacity'),
          paths: paths.map(function (path) {
            var length = 0;
            try { length = path.getTotalLength(); } catch (_) { /* Fall back to a whole-edge fade. */ }
            return {
              element: path,
              length: Number.isFinite(length) ? length : 0,
              dasharray: originalProperty(path, 'stroke-dasharray'),
              dashoffset: originalProperty(path, 'stroke-dashoffset')
            };
          })
        };
      }).filter(function (item) { return Number.isFinite(item.stage); });
      var stages = Array.from(new Set(elements.map(function (item) { return item.stage; }))).sort(function (a, b) { return a - b; });
      if (!stages.length) return;
      elements.forEach(function (item) { item.index = stages.indexOf(item.stage); });

      var duration = (stages.length - 1) * STEP_DURATION;
      var elapsed = 0;
      var previousTime = null;
      var frame = 0;
      var started = false;
      var manuallyPaused = false;
      var inViewport = false;
      var printing = false;
      var pageHidden = false;
      var state = 'ready';
      var playLabel = playButton.querySelector('[data-rollout-play-label]');

      function clamp(value) { return Math.max(0, Math.min(1, value)); }

      function restoreEdge(item) {
        item.paths.forEach(function (path) {
          restoreProperty(path.element, 'stroke-dasharray', path.dasharray);
          restoreProperty(path.element, 'stroke-dashoffset', path.dashoffset);
        });
      }

      function reveal(item, amount) {
        if (amount >= 1) {
          restoreProperty(item.element, 'opacity', item.opacity);
        } else {
          var originalOpacity = item.opacity.value === '' ? 1 : Number(item.opacity.value);
          item.element.style.opacity = String(amount * (Number.isFinite(originalOpacity) ? originalOpacity : 1));
        }
      }

      function renderTimeline(time) {
        elements.forEach(function (item) {
          if (item.index === 0 || time >= duration) {
            restoreEdge(item);
            reveal(item, 1);
            return;
          }
          var localTime = time - (item.index - 1) * STEP_DURATION;
          if (!item.edge) {
            // Formula glyphs and their surrounding nodes appear together, never character by character.
            reveal(item, clamp((localTime - EDGE_DURATION) / NODE_DURATION));
            return;
          }
          var amount = clamp(localTime / EDGE_DURATION);
          if (amount === 0 || amount === 1) {
            restoreEdge(item);
            reveal(item, amount);
            return;
          }
          var drawablePaths = item.paths.filter(function (path) { return path.length > 0; });
          reveal(item, drawablePaths.length ? 1 : amount);
          drawablePaths.forEach(function (path) {
            // Only solid connectors are marked as edges. Dashed connectors retain their
            // original pattern and are revealed as whole elements alongside their nodes.
            path.element.style.strokeDasharray = path.length + ' ' + path.length;
            path.element.style.strokeDashoffset = String(path.length * (1 - amount));
          });
        });
        progress.max = Math.max(1, duration);
        progress.value = duration ? time : 1;
      }

      function renderControls() {
        figure.dataset.state = state;
        figure.dataset.motion = reducedMotion.matches ? 'reduced' : 'full';
        var label = state === 'playing' ? 'Pause' : (state === 'complete' ? 'Play again' : 'Play');
        var accessibleLabel = state === 'playing' ? 'Pause animation' : (state === 'complete' ? 'Play animation again' : 'Play animation');
        if (playLabel) playLabel.textContent = label;
        else playButton.textContent = label;
        playButton.setAttribute('aria-label', accessibleLabel);
        playButton.title = accessibleLabel;
        playButton.disabled = reducedMotion.matches || duration === 0;
        replayButton.disabled = reducedMotion.matches || duration === 0;

        var step = !started || elapsed === 0 ? 1 : Math.min(stages.length, Math.floor(elapsed / STEP_DURATION) + 2);
        var message = reducedMotion.matches ? 'Static diagram · reduced motion' : 'Step ' + step + ' of ' + stages.length;
        if (!reducedMotion.matches && state === 'complete') message += ' · Complete';
        else if (!reducedMotion.matches && state === 'paused') message += ' · Paused';
        // Avoid repeatedly mutating the live region on every animation frame.
        if (status.textContent !== message) status.textContent = message;
      }

      function stopFrame() {
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        previousTime = null;
      }

      function tick(timestamp) {
        frame = 0;
        if (state !== 'playing') return;
        if (previousTime !== null) elapsed = Math.min(duration, elapsed + timestamp - previousTime);
        previousTime = timestamp;
        renderTimeline(elapsed);
        if (elapsed >= duration) {
          state = 'complete';
          previousTime = null;
        }
        renderControls();
        if (state === 'playing') frame = window.requestAnimationFrame(tick);
      }

      function synchronize() {
        if (reducedMotion.matches) {
          stopFrame();
          elapsed = duration;
          state = 'complete';
          renderTimeline(duration);
        } else if (elapsed >= duration) {
          stopFrame();
          state = 'complete';
        } else if (started && !manuallyPaused && inViewport && !document.hidden && !pageHidden && !printing) {
          state = 'playing';
          if (!frame) frame = window.requestAnimationFrame(tick);
        } else {
          stopFrame();
          state = started ? 'paused' : 'ready';
        }
        renderControls();
      }

      function restart() {
        stopFrame();
        elapsed = 0;
        started = true;
        manuallyPaused = false;
        renderTimeline(elapsed);
        synchronize();
      }

      playButton.addEventListener('click', function () {
        if (reducedMotion.matches) return;
        if (elapsed >= duration) {
          restart();
          return;
        }
        if (state === 'playing') manuallyPaused = true;
        else {
          started = true;
          manuallyPaused = false;
        }
        synchronize();
      });
      replayButton.addEventListener('click', function () {
        if (!reducedMotion.matches) restart();
      });

      figure.dataset.enhanced = 'true';
      var controls = figure.querySelector('.docs-rollout-controls');
      if (controls) controls.hidden = false;
      renderTimeline(0);
      synchronize();

      if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            inViewport = entry.isIntersecting && entry.intersectionRatio >= 0.3;
            if (inViewport && !started && !reducedMotion.matches) started = true;
            synchronize();
          });
        }, { threshold: [0, 0.3] });
        observer.observe(figure);
      } else {
        inViewport = true;
        started = true;
        synchronize();
      }

      document.addEventListener('visibilitychange', synchronize);
      window.addEventListener('pagehide', function () { pageHidden = true; synchronize(); });
      window.addEventListener('pageshow', function () { pageHidden = false; synchronize(); });
      window.addEventListener('beforeprint', function () {
        printing = true;
        synchronize();
        renderTimeline(duration);
      });
      window.addEventListener('afterprint', function () {
        printing = false;
        renderTimeline(elapsed);
        synchronize();
      });
      if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', synchronize);
      else reducedMotion.addListener(synchronize);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeRollouts, { once: true });
  } else {
    initializeRollouts();
  }
})();
