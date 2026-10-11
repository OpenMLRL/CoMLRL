(function () {
  'use strict';

  var cacheLifetime = 60 * 60 * 1000;
  var repositories = new Map();

  function isCount(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }

  function render(badges, repository, count, fetchedAt) {
    var formatted = count.toLocaleString('en-US');
    var label = repository + ' on GitHub (' + formatted + (count === 1 ? ' star)' : ' stars)');
    badges.forEach(function (badge) {
      badge.querySelector('[data-star-count]').textContent = formatted;
      badge.setAttribute('aria-label', label);
      badge.title = label + ' · Updated ' + new Date(fetchedAt).toISOString();
    });
  }

  async function refresh(repository, badges) {
    var cacheKey = 'comlrl-github-stars:' + repository.toLowerCase();
    var now = Date.now();
    try {
      var cached = JSON.parse(window.localStorage.getItem(cacheKey));
      if (cached && isCount(cached.count) && Number.isSafeInteger(cached.fetchedAt) &&
          cached.fetchedAt >= 0 && cached.fetchedAt <= now) {
        render(badges, repository, cached.count, cached.fetchedAt);
        if (now - cached.fetchedAt < cacheLifetime) return;
      }
    } catch (error) {
      // Storage restrictions or malformed entries leave the server snapshot usable.
    }

    if (typeof fetch !== 'function' || typeof AbortController !== 'function') return;
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 5000);
    try {
      var path = repository.split('/').map(encodeURIComponent).join('/');
      var response = await fetch('https://api.github.com/repos/' + path, {
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
        headers: { Accept: 'application/vnd.github+json' },
        signal: controller.signal
      });
      if (!response.ok) return;
      var data = await response.json();
      if (!data || typeof data.full_name !== 'string' ||
          data.full_name.toLowerCase() !== repository.toLowerCase() ||
          !isCount(data.stargazers_count)) return;

      var fetchedAt = Date.now();
      render(badges, repository, data.stargazers_count, fetchedAt);
      try {
        window.localStorage.setItem(cacheKey, JSON.stringify({
          count: data.stargazers_count,
          fetchedAt: fetchedAt
        }));
      } catch (error) {
        // The fresh count still displays when the browser cannot persist it.
      }
    } catch (error) {
      // Offline requests, timeouts and API failures keep the last available count.
    } finally {
      clearTimeout(timeout);
    }
  }

  document.querySelectorAll('[data-github-stars]').forEach(function (badge) {
    var repository = badge.getAttribute('data-repository');
    if (!repository || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository) ||
        !badge.querySelector('[data-star-count]')) return;
    var key = repository.toLowerCase();
    if (!repositories.has(key)) repositories.set(key, { name: repository, badges: [] });
    repositories.get(key).badges.push(badge);
  });
  repositories.forEach(function (entry) { refresh(entry.name, entry.badges); });
})();
