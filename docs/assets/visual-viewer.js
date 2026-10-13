(function () {
  'use strict';

  function initializeViewers() {
    if (document.documentElement.hasAttribute('data-visual-viewer-ready')) return;
    document.documentElement.setAttribute('data-visual-viewer-ready', '');
    var active = null;
    var focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    var expandIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/></svg>';
    var closeIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>';
    var zoomInIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5M7.5 10.5h6m-3-3v6"/></svg>';
    var zoomOutIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5M7.5 10.5h6"/></svg>';

    function focusableElements(figure) {
      return Array.from(figure.querySelectorAll(focusableSelector)).filter(function (element) {
        return !element.closest('[inert]') && element.getClientRects().length > 0 && window.getComputedStyle(element).visibility !== 'hidden';
      });
    }

    function setButton(button, expanded) {
      var label = expanded ? 'Close expanded animation' : 'Expand animation';
      button.innerHTML = expanded ? closeIcon : expandIcon;
      button.setAttribute('aria-label', label);
      button.setAttribute('aria-expanded', String(expanded));
      button.title = label;
    }

    function setZoom(figure, button, zoomed) {
      figure.classList.toggle('is-zoomed', zoomed);
      var label = zoomed ? 'Zoom out' : 'Zoom in';
      button.innerHTML = zoomed ? zoomOutIcon : zoomInIcon;
      button.setAttribute('aria-label', label);
      button.setAttribute('aria-pressed', String(zoomed));
      button.title = label;
    }

    function closeViewer() {
      if (!active) return;
      var previous = active;
      active = null;
      previous.figure.classList.remove('is-expanded');
      setZoom(previous.figure, previous.zoomButton, false);
      previous.zoomButton.hidden = true;
      Object.keys(previous.attributes).forEach(function (name) {
        var value = previous.attributes[name];
        if (value === null) previous.figure.removeAttribute(name);
        else previous.figure.setAttribute(name, value);
      });
      previous.inertElements.forEach(function (item) {
        if (item.value === null) item.element.removeAttribute('inert');
        else item.element.setAttribute('inert', item.value);
      });
      previous.backdrop.remove();
      previous.placeholder.remove();
      if (!previous.bodyWasOpen) document.body.classList.remove('docs-visual-open');
      setButton(previous.button, false);
      if (previous.focus && previous.focus.isConnected) previous.focus.focus({ preventScroll: true });
      window.scrollTo({ left: previous.scrollX, top: previous.scrollY, behavior: 'instant' });
    }

    function openViewer(figure, button, zoomButton) {
      if (active) closeViewer();
      var bounds = figure.getBoundingClientRect();
      var styles = window.getComputedStyle(figure);
      var placeholder = document.createElement('div');
      placeholder.className = 'docs-visual-placeholder';
      placeholder.setAttribute('aria-hidden', 'true');
      // Keep the original layout occupied while the same figure becomes fixed.
      placeholder.style.cssText = 'display:block;box-sizing:border-box;visibility:hidden;pointer-events:none;';
      placeholder.style.width = bounds.width + 'px';
      placeholder.style.height = bounds.height + 'px';
      placeholder.style.margin = styles.marginTop + ' ' + styles.marginRight + ' ' + styles.marginBottom + ' ' + styles.marginLeft;

      var backdrop = document.createElement('div');
      backdrop.className = 'docs-visual-backdrop';
      backdrop.setAttribute('aria-hidden', 'true');
      backdrop.addEventListener('click', closeViewer);

      var attributes = {};
      ['role', 'aria-modal', 'aria-label', 'aria-labelledby'].forEach(function (name) {
        attributes[name] = figure.getAttribute(name);
      });
      active = {
        figure: figure,
        button: button,
        zoomButton: zoomButton,
        placeholder: placeholder,
        backdrop: backdrop,
        attributes: attributes,
        inertElements: [],
        focus: document.activeElement,
        scrollX: window.scrollX,
        scrollY: window.scrollY,
        bodyWasOpen: document.body.classList.contains('docs-visual-open')
      };
      figure.before(placeholder);
      document.body.appendChild(backdrop);
      figure.classList.add('is-expanded');
      document.body.classList.add('docs-visual-open');
      figure.setAttribute('role', 'dialog');
      figure.setAttribute('aria-modal', 'true');
      figure.removeAttribute('aria-labelledby');
      var caption = figure.querySelector('.docs-rollout-title, figcaption');
      figure.setAttribute('aria-label', caption ? caption.textContent.trim() : 'Animated demo');
      setButton(button, true);
      setZoom(figure, zoomButton, false);
      zoomButton.hidden = false;

      // Disable each branch outside the figure, never an ancestor containing it.
      var branch = figure;
      while (branch.parentElement) {
        Array.from(branch.parentElement.children).forEach(function (sibling) {
          if (sibling === branch || sibling === backdrop || sibling === placeholder) return;
          active.inertElements.push({ element: sibling, value: sibling.getAttribute('inert') });
          sibling.setAttribute('inert', '');
        });
        branch = branch.parentElement;
        if (branch === document.body) break;
      }
      // No image, SVG, animation state, or frame timing is changed by the viewer.
      button.focus({ preventScroll: true });
    }

    document.querySelectorAll('figure.docs-demo, figure.docs-rollout').forEach(function (figure) {
      if (figure.dataset.visualViewer) return;
      var image = figure.querySelector('img');
      if (image) {
        var recordNativeSize = function () {
          // A 2× asset retains its declared display size when zoomed on Retina screens.
          var width = Number(image.getAttribute('width')) || image.naturalWidth;
          var height = Number(image.getAttribute('height')) || image.naturalHeight;
          if (width > 0) figure.style.setProperty('--docs-visual-native-width', width + 'px');
          if (height > 0) figure.style.setProperty('--docs-visual-native-height', height + 'px');
        };
        recordNativeSize();
        if (!image.complete) image.addEventListener('load', recordNativeSize, { once: true });
      }
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'docs-visual-expand';
      button.setAttribute('aria-haspopup', 'dialog');
      setButton(button, false);
      var zoomButton = document.createElement('button');
      zoomButton.type = 'button';
      zoomButton.className = 'docs-visual-zoom';
      zoomButton.hidden = true;
      setZoom(figure, zoomButton, false);
      zoomButton.addEventListener('click', function () {
        if (active && active.figure === figure) {
          setZoom(figure, zoomButton, !figure.classList.contains('is-zoomed'));
        }
      });
      button.addEventListener('click', function () {
        if (active && active.figure === figure) closeViewer();
        else openViewer(figure, button, zoomButton);
      });
      figure.appendChild(button);
      figure.appendChild(zoomButton);
      figure.dataset.visualViewer = 'true';
    });

    document.addEventListener('keydown', function (event) {
      if (!active) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        closeViewer();
      } else if (event.key === 'Tab') {
        var elements = focusableElements(active.figure);
        var first = elements[0] || active.button;
        var last = elements[elements.length - 1] || active.button;
        if (event.shiftKey && (document.activeElement === first || !active.figure.contains(document.activeElement))) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !active.figure.contains(document.activeElement))) {
          event.preventDefault();
          first.focus();
        }
      }
    });
    document.addEventListener('focusin', function (event) {
      if (active && !active.figure.contains(event.target)) active.button.focus({ preventScroll: true });
    });
    window.addEventListener('beforeprint', closeViewer);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeViewers, { once: true });
  } else {
    initializeViewers();
  }
})();
