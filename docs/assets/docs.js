(function () {
  'use strict';

  function initializeDocs() {
    var article = document.querySelector('article.book-article');
    var search = document.getElementById('book-search-input');
    var platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    var isMac = /Mac|iPhone|iPad|iPod/i.test(platform);
    var pageLanguage = document.documentElement.lang.toLowerCase();
    var labelLanguage = pageLanguage.startsWith('zh') ? 'zh-cn' : pageLanguage.split('-')[0];
    var copyLabelsByLanguage = {
      en: {
        default: 'Copy code to clipboard', copied: 'Code copied to clipboard',
        error: 'Copy failed. Select the code and copy it manually.'
      },
      'zh-cn': {
        default: '复制代码', copied: '代码已复制', error: '复制失败，请选中代码手动复制。'
      },
      ja: {
        default: 'コードをコピー', copied: 'コードをコピーしました',
        error: 'コピーできませんでした。コードを選択して手動でコピーしてください。'
      },
      ko: {
        default: '코드 복사', copied: '코드를 복사했습니다',
        error: '복사하지 못했습니다. 코드를 선택한 뒤 직접 복사하세요.'
      },
      es: {
        default: 'Copiar código', copied: 'Código copiado',
        error: 'No se pudo copiar. Selecciona el código y cópialo manualmente.'
      },
      fr: {
        default: 'Copier le code', copied: 'Code copié',
        error: 'Échec de la copie. Sélectionnez le code et copiez-le manuellement.' // codespell:ignore copie
      },
      de: {
        default: 'Code kopieren', copied: 'Code kopiert',
        error: 'Kopieren fehlgeschlagen. Wählen Sie den Code aus und kopieren Sie ihn manuell.' // codespell:ignore sie,manuell
      }
    };
    var copyLabels = copyLabelsByLanguage[labelLanguage] || copyLabelsByLanguage.en;

    // Headings keep their English anchors across generated translations.
    function updateLanguageLinks() {
      document.querySelectorAll('[data-language-link]').forEach(function (link) {
        var url = new URL(link.href, window.location.href);
        url.hash = window.location.hash;
        link.href = url.href;
      });
    }
    updateLanguageLinks();
    window.addEventListener('hashchange', updateLanguageLinks);

    var languageMenus = Array.from(document.querySelectorAll('[data-language-menu]'));

    function closeLanguageMenus(restoreFocus) {
      var wasOpen = false;
      languageMenus.forEach(function (menu) {
        if (!menu.open) return;
        wasOpen = true;
        var hadFocus = menu.contains(document.activeElement);
        menu.open = false;
        if (restoreFocus && hadFocus) menu.querySelector('summary').focus({ preventScroll: true });
      });
      return wasOpen;
    }

    languageMenus.forEach(function (menu) {
      menu.addEventListener('toggle', function () {
        if (!menu.open) return;
        closePanels(false);
        languageMenus.forEach(function (other) {
          if (other !== menu) other.open = false;
        });
      });
      menu.addEventListener('click', function (event) {
        if (event.target instanceof Element && event.target.closest('[data-language-link]')) {
          closeLanguageMenus(false);
        }
      });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && closeLanguageMenus(true)) event.preventDefault();
    });
    ['click', 'focusin'].forEach(function (eventName) {
      document.addEventListener(eventName, function (event) {
        languageMenus.forEach(function (menu) {
          if (menu.open && !menu.contains(event.target)) menu.open = false;
        });
      });
    });

    document.querySelectorAll('[data-search-shortcut]').forEach(function (hint) {
      hint.textContent = isMac ? '⌘ K' : 'Ctrl K';
    });

    // Keep the theme's checkbox controls working without JavaScript as well.
    var panelControls = ['menu-control', 'toc-control'].map(function (id) {
      return document.getElementById(id);
    }).filter(Boolean);
    var panelButtons = Array.from(document.querySelectorAll('[data-toggle="menu-control"], [data-toggle="toc-control"]'));
    var lastPanelButton = null;

    function synchronizePanels() {
      panelButtons.forEach(function (button) {
        var control = document.getElementById(button.dataset.toggle);
        button.setAttribute('aria-expanded', String(Boolean(control && control.checked)));
      });
    }

    function closePanels(restoreFocus) {
      var wasOpen = panelControls.some(function (control) { return control.checked; });
      panelControls.forEach(function (control) { control.checked = false; });
      synchronizePanels();
      if (restoreFocus && wasOpen && lastPanelButton && document.contains(lastPanelButton)) {
        lastPanelButton.focus({ preventScroll: true });
      }
      return wasOpen;
    }

    panelButtons.forEach(function (button) {
      button.addEventListener('click', function (event) {
        var control = document.getElementById(button.dataset.toggle);
        if (!control) return;
        event.preventDefault();
        var shouldOpen = !control.checked;
        closePanels(false);
        control.checked = shouldOpen;
        lastPanelButton = button;
        synchronizePanels();
      });
    });

    panelControls.forEach(function (control) {
      control.addEventListener('change', function () {
        if (control.checked) {
          panelControls.forEach(function (other) {
            if (other !== control) other.checked = false;
          });
        }
        synchronizePanels();
      });
    });

    document.addEventListener('keydown', function (event) {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === 'k' && search) {
        event.preventDefault();
        closePanels(false);
        search.focus();
        search.select();
      } else if (event.key === 'Escape' && closePanels(true)) {
        event.preventDefault();
      }
    });

    document.addEventListener('click', function (event) {
      if (event.target instanceof Element && event.target.closest('.book-menu a[href], .book-toc a[href], .book-header aside a[href], .docs-mobile-toc a[href]')) {
        closePanels(false);
      }
    });
    synchronizePanels();

    if (!article) return;

    function copyText(text) {
      if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text);
      }

      // Local static previews may run in a context without the Clipboard API.
      return new Promise(function (resolve, reject) {
        var field = document.createElement('textarea');
        var focused = document.activeElement;
        var selection = window.getSelection();
        var ranges = [];
        if (selection) {
          for (var index = 0; index < selection.rangeCount; index += 1) {
            ranges.push(selection.getRangeAt(index).cloneRange());
          }
        }
        field.value = text;
        field.setAttribute('readonly', '');
        field.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0;';
        document.body.appendChild(field);
        field.select();
        try {
          if (!document.execCommand('copy')) throw new Error('Copy was unavailable.');
          resolve();
        } catch (error) {
          reject(error);
        } finally {
          field.remove();
          if (selection) {
            selection.removeAllRanges();
            ranges.forEach(function (range) { selection.addRange(range); });
          }
          if (focused && typeof focused.focus === 'function') focused.focus({ preventScroll: true });
        }
      });
    }

    var copyIcons = {
      default: '<rect x="6" y="6" width="8" height="8" rx="1.5"/><path d="M10 6V3.5A1.5 1.5 0 0 0 8.5 2h-5A1.5 1.5 0 0 0 2 3.5v5A1.5 1.5 0 0 0 3.5 10H6"/>',
      copied: '<path d="m3 8 3 3 7-7"/>',
      error: '<circle cx="8" cy="8" r="6"/><path d="M8 4.5v4M8 11h.01"/>'
    };

    article.querySelectorAll('pre').forEach(function (pre) {
      var code = pre.querySelector('code');
      if (!code || pre.querySelector('.copy-code-button')) return;

      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy-code-button';
      var status = document.createElement('span');
      status.className = 'docs-sr-only copy-code-status';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      status.setAttribute('aria-atomic', 'true');
      var resetTimer;
      var copying = false;

      function setCopyState(state, label) {
        button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' + copyIcons[state] + '</svg>';
        button.setAttribute('aria-label', label);
        button.title = label;
        button.classList.toggle('is-copied', state === 'copied');
        button.classList.toggle('is-error', state === 'error');
        status.textContent = state === 'default' ? '' : label;
      }
      setCopyState('default', copyLabels.default);

      button.addEventListener('click', function (event) {
        // The theme focuses the pre on click; retain focus on this button.
        event.stopPropagation();
        if (copying) return;
        copying = true;
        window.clearTimeout(resetTimer);
        status.textContent = '';
        copyText(code.textContent).then(function () {
          setCopyState('copied', copyLabels.copied);
        }).catch(function () {
          setCopyState('error', copyLabels.error);
        }).finally(function () {
          copying = false;
          resetTimer = window.setTimeout(function () {
            setCopyState('default', copyLabels.default);
          }, 3000);
        });
      });

      // Override Hugo Book's pre.textContent copy handler before it runs, so
      // copy-button labels never become part of manually copied source code.
      pre.addEventListener('copy', function (event) {
        var selection = window.getSelection();
        var text = '';
        if (selection && !selection.isCollapsed) {
          for (var index = 0; index < selection.rangeCount; index += 1) {
            var fragment = selection.getRangeAt(index).cloneContents();
            fragment.querySelectorAll('.copy-code-button, .copy-code-status').forEach(function (copyControl) {
              copyControl.remove();
            });
            text += fragment.textContent;
          }
        }
        text = text || code.textContent;
        if (event.clipboardData) {
          event.preventDefault();
          event.stopImmediatePropagation();
          event.clipboardData.setData('text/plain', text);
        }
      }, true);

      pre.appendChild(button);
      pre.appendChild(status);
    });

    var tocLinks = Array.from(document.querySelectorAll('.book-toc nav a[href*="#"], .book-header aside nav a[href*="#"], .docs-mobile-toc nav a[href*="#"]'));
    var articleTargets = new Map(Array.from(article.querySelectorAll('[id]')).map(function (element) {
      return [element.id, element];
    }));
    var targets = [];
    var targetLinks = new Map();

    tocLinks.forEach(function (link) {
      link.title = link.textContent.trim();
      var url = new URL(link.href, window.location.href);
      if (url.origin !== window.location.origin || url.pathname !== window.location.pathname) return;
      var id;
      try { id = decodeURIComponent(url.hash.slice(1)); } catch (_) { return; }
      var target = articleTargets.get(id);
      if (!target) return;
      if (!targetLinks.has(target)) {
        targets.push(target);
        targetLinks.set(target, []);
      }
      targetLinks.get(target).push(link);
    });

    if (!targets.length) return;
    targets.sort(function (left, right) {
      return left.compareDocumentPosition(right) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
    });
    var activeTarget = null;
    var scheduled = false;

    function updateActiveSection() {
      scheduled = false;
      var scrollMargin = parseFloat(window.getComputedStyle(targets[0]).scrollMarginTop) || 0;
      var threshold = Math.max(100, scrollMargin + 16);
      var current = targets[0];
      targets.forEach(function (target) {
        if (target.getBoundingClientRect().top <= threshold) current = target;
      });
      var maximumScroll = document.documentElement.scrollHeight - window.innerHeight;
      if (maximumScroll > 1 && window.scrollY >= maximumScroll - 3) {
        current = targets[targets.length - 1];
      }
      if (current === activeTarget) return;
      activeTarget = current;
      targetLinks.forEach(function (links, target) {
        links.forEach(function (link) {
          var active = target === current;
          link.classList.toggle('is-active', active);
          if (active) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }

    function scheduleSectionUpdate() {
      if (scheduled) return;
      scheduled = true;
      window.requestAnimationFrame(updateActiveSection);
    }

    window.addEventListener('scroll', scheduleSectionUpdate, { passive: true });
    window.addEventListener('resize', scheduleSectionUpdate, { passive: true });
    window.addEventListener('hashchange', scheduleSectionUpdate);
    window.addEventListener('pageshow', scheduleSectionUpdate);
    window.addEventListener('load', scheduleSectionUpdate);
    article.addEventListener('load', scheduleSectionUpdate, true);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleSectionUpdate);
    scheduleSectionUpdate();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDocs, { once: true });
  } else {
    initializeDocs();
  }
})();
