/**
 * Scrollytelling scroll triggers — vanilla IntersectionObserver.
 * Activates the matching .visual-state when a .step enters the trigger zone.
 * Falls back to instant swaps when prefers-reduced-motion is set (handled in CSS).
 */
(function () {
  'use strict';

  const steps = document.querySelectorAll('.step');
  const visualStates = document.querySelectorAll('.visual-state');
  const navItems = document.querySelectorAll('.section-nav__item');
  const progressBar = document.getElementById('scroll-progress');
  const sectionIndicator = document.getElementById('section-indicator');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!steps.length || !visualStates.length) {
    return;
  }

  /** @type {string|null} */
  let activeKey = null;
  var progressRaf = null;

  function setActive(sectionId, substep) {
    var sub = substep || '';
    var key = sectionId + ':' + sub;
    if (activeKey === key) {
      return;
    }
    activeKey = key;

    visualStates.forEach(function (visual) {
      var matchesSection = visual.dataset.section === sectionId;
      var visualSub = visual.dataset.substep || '';
      var matchesSubstep = visualSub === sub;
      visual.classList.toggle('is-active', matchesSection && matchesSubstep);
    });

    navItems.forEach(function (item) {
      item.classList.toggle('is-active', item.dataset.section === sectionId);
      item.setAttribute('aria-current', item.dataset.section === sectionId ? 'true' : 'false');
    });

    if (sectionIndicator) {
      sectionIndicator.textContent = 'Section ' + sectionId + ' of 12';
    }
  }

  function updateProgressNow() {
    if (!progressBar) {
      return;
    }
    var scrollTop = window.scrollY;
    var docHeight = document.documentElement.scrollHeight - window.innerHeight;
    var pct = docHeight > 0 ? Math.min(100, (scrollTop / docHeight) * 100) : 0;
    progressBar.style.width = pct + '%';
  }

  function updateProgress() {
    if (progressRaf) {
      return;
    }
    progressRaf = window.requestAnimationFrame(function () {
      progressRaf = null;
      updateProgressNow();
    });
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var sectionId = entry.target.dataset.section;
          var substep = entry.target.dataset.substep || '';
          setActive(sectionId, substep);
          entry.target.classList.add('is-visible');
        }
      });
    },
    {
      threshold: 0.55,
      rootMargin: '-18% 0px -18% 0px',
    }
  );

  steps.forEach(function (step) {
    observer.observe(step);
  });

  navItems.forEach(function (item) {
    item.addEventListener('click', function () {
      var target = document.getElementById('section-' + item.dataset.section);
      if (target) {
        target.scrollIntoView({
          behavior: prefersReducedMotion ? 'auto' : 'smooth',
          block: 'center',
        });
      }
    });
  });

  window.addEventListener('scroll', updateProgress, { passive: true });
  updateProgressNow();

  /* Activate first section on load */
  var firstStep = steps[0];
  if (firstStep) {
    setActive(firstStep.dataset.section, firstStep.dataset.substep || '');
    firstStep.classList.add('is-visible');
  }
})();
