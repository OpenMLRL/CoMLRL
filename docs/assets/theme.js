(function () {
  'use strict';

  var root = document.documentElement;
  if (root.hasAttribute('data-theme-initialized')) return;
  root.setAttribute('data-theme-initialized', 'true');

  var storageKey = 'comlrl-theme';
  var systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  var preference = null;

  function isTheme(value) {
    return value === 'light' || value === 'dark' || value === 'reading';
  }

  try {
    var savedTheme = window.localStorage.getItem(storageKey);
    if (isTheme(savedTheme)) preference = savedTheme;
  } catch (error) {
    // Private browsing or storage restrictions should not prevent switching.
  }

  function updateButtons() {
    var theme = root.getAttribute('data-theme');
    document.querySelectorAll('[data-theme-option]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.getAttribute('data-theme-option') === theme));
    });
  }

  function applyTheme() {
    var theme = preference || (systemTheme.matches ? 'dark' : 'light');
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme === 'dark' ? 'dark' : 'light';
    updateButtons();
  }

  // This script runs synchronously in the head so the first paint uses the
  // saved theme, or the operating system preference on a first visit.
  applyTheme();

  function initializeSwitcher() {
    document.querySelectorAll('[data-theme-option]').forEach(function (button) {
      button.addEventListener('click', function () {
        var theme = button.getAttribute('data-theme-option');
        if (!isTheme(theme)) return;
        preference = theme;
        try {
          window.localStorage.setItem(storageKey, preference);
        } catch (error) {
          // Keep the explicit choice for this page when storage is unavailable.
        }
        applyTheme();
      });
    });
    updateButtons();
    document.querySelectorAll('[data-theme-switcher]').forEach(function (switcher) {
      switcher.hidden = false;
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSwitcher, { once: true });
  } else {
    initializeSwitcher();
  }

  function followSystemTheme() {
    if (!preference) applyTheme();
  }
  if (systemTheme.addEventListener) {
    systemTheme.addEventListener('change', followSystemTheme);
  } else {
    systemTheme.addListener(followSystemTheme);
  }

  window.addEventListener('storage', function (event) {
    if (event.key !== storageKey && event.key !== null) return;
    preference = isTheme(event.newValue) ? event.newValue : null;
    applyTheme();
  });
})();
