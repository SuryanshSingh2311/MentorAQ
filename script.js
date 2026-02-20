// Minimal interactions & theme handler for MentorAQ
(function() {
  // Nav active link highlight
  const nav = document.querySelector('.nav-links');
  if (nav) {
    nav.addEventListener('click', (e) => {
      const a = e.target.closest('a');
      if (!a) return;
      [...nav.querySelectorAll('a')].forEach(n => n.classList.remove('active'));
      a.classList.add('active');
    });
  }

  // Count-up animation for stats
  function countUp(el, to, suffix = '') {
    const duration = 1200; // ms
    const start = performance.now();
    const from = 0;
    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const val = Math.floor(from + (to - from) * eased);
      el.textContent = `${val}${suffix}`;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function initCounts() {
    document.querySelectorAll('.stat-card').forEach(card => {
      const numEl = card.querySelector('.stat-number');
      const target = parseInt(card.getAttribute('data-count') || '0', 10);
      const suffix = card.getAttribute('data-suffix') || '';
      if (numEl && target) countUp(numEl, target, suffix);
    });
  }

  // Drag & drop behavior hookup (existing IDs)
  const uploadArea = document.getElementById('uploadArea');
  if (uploadArea) {
    ['dragenter','dragover','dragleave','drop'].forEach(ev => uploadArea.addEventListener(ev, (e)=>{e.preventDefault(); e.stopPropagation();}));
    ['dragenter','dragover'].forEach(ev => uploadArea.addEventListener(ev, ()=>uploadArea.classList.add('dragover')));
    ['dragleave','drop'].forEach(ev => uploadArea.addEventListener(ev, ()=>uploadArea.classList.remove('dragover')));
    uploadArea.addEventListener('drop', (e)=>{
      const dt = e.dataTransfer; const files = dt?.files; if (!files?.length) return;
      const input = document.getElementById('resumeFile'); if (input) input.files = files;
      if (files[0]) {
        const h3 = uploadArea.querySelector('h3'); if (h3) h3.textContent = `Selected: ${files[0].name}`;
      }
    });
  }

  // Public API to get checked search type (kept for compatibility)
  window.getSearchType = () => document.querySelector('input[name=searchType]:checked')?.value;

  function updateRoadmapUI() {
    const has = localStorage.getItem('hasRoadmap') === '1';
    const navLabel = document.getElementById('navRoadmapLabel');
    const bodyLabel = document.getElementById('bodyRoadmapLabel');
    const homeCta = document.getElementById('homeRoadmapCta');
    if (navLabel) navLabel.textContent = has ? 'Roadmap' : 'Make Roadmap';
    if (bodyLabel) bodyLabel.textContent = has ? 'View Roadmap' : 'Make Roadmap';
    if (homeCta) homeCta.textContent = has ? 'View Roadmap' : 'Make Roadmap';
  }

  window.goToRoadmapOrResume = function() {
    const has = localStorage.getItem('hasRoadmap') === '1';
    showSection('roadmap');
    if (!has) {
      const jd = document.getElementById('jobDescription');
      const goalEl = document.getElementById('roadmapGoal');
      if (jd && goalEl && !goalEl.value) {
        goalEl.value = `Align resume to: ${jd.value.slice(0, 120)}${jd.value.length > 120 ? '…' : ''}`;
      }
    }
  }

  function segmentWeeks(totalWeeks) {
    // Returns array of segments with label and week range
    const segments = [];
    const chunk = totalWeeks <= 4 ? 1 : totalWeeks <= 6 ? 2 : totalWeeks <= 8 ? 2 : 3; // adaptive chunk size
    let start = 1;
    while (start <= totalWeeks) {
      const end = Math.min(totalWeeks, start + chunk - 1);
      segments.push({ label: end - start + 1 > 1 ? `Weeks ${start}–${end}` : `Week ${start}`, start, end });
      start = end + 1;
    }
    return segments;
  }

  function deriveFocusAreas(goal, skills) {
    const g = (goal || '').toLowerCase();
    const s = (skills || '').toLowerCase();
    const areas = new Set();
    if (/front\s*end|react|web/i.test(g + ' ' + s)) {
      areas.add('HTML/CSS'); areas.add('JavaScript'); areas.add('React'); areas.add('UI/UX');
    }
    if (/back\s*end|api|node|django|spring/i.test(g + ' ' + s)) {
      areas.add('APIs'); areas.add('Databases'); areas.add('Authentication');
    }
    if (/data|ml|ai|analytics|python/i.test(g + ' ' + s)) {
      areas.add('Python'); areas.add('Pandas'); areas.add('SQL'); areas.add('ML Basics');
    }
    if (/devops|cloud|aws|docker|kubernetes/i.test(g + ' ' + s)) {
      areas.add('GitHub Actions'); areas.add('Docker'); areas.add('Cloud Basics');
    }
    if (areas.size === 0) { areas.add('Core Fundamentals'); areas.add('Problem Solving'); }
    return [...areas];
  }

  function buildRoadmapPhases(totalWeeks, goal, skills) {
    const segments = segmentWeeks(totalWeeks);
    const areas = deriveFocusAreas(goal, skills);
    const phases = [];

    segments.forEach((seg, idx) => {
      const focus = areas[idx % areas.length];
      const tasks = [
        `Learn/Revise: ${focus}`,
        `Mini-project: Apply ${focus} to a small task`,
        `Assessment: Create notes and summarize learnings`,
      ];
      if (idx === 0) tasks.unshift('Refine resume and LinkedIn summary');
      if (idx === segments.length - 1) tasks.push('Apply to roles/courses and seek feedback from mentors');
      phases.push({ title: seg.label, tasks });
    });

    return phases;
  }

  function renderRoadmap(phases) {
    const container = document.getElementById('roadmapContent');
    if (!container) return;
    container.innerHTML = phases.map(p => `
      <div class="card p-4">
        <h3>${p.title}</h3>
        <ul class="check-list">
          ${p.tasks.map(t => `<li>${t}</li>`).join('')}
        </ul>
      </div>
    `).join('');
  }

  window.generateRoadmap = function() {
    const goal = document.getElementById('roadmapGoal')?.value || '';
    const skills = document.getElementById('roadmapSkills')?.value || '';
    const duration = parseInt(document.getElementById('roadmapDuration')?.value || '8', 10);

    const loading = document.getElementById('loadingRoadmap');
    const results = document.getElementById('roadmapResults');

    if (loading) loading.style.display = 'block';
    if (results) results.style.display = 'none';

    // Simulate async generation for UX consistency
    setTimeout(() => {
      const phases = buildRoadmapPhases(duration, goal, skills);
      renderRoadmap(phases);
      const intro = document.getElementById('roadmapIntro');
      if (intro) {
        const skillsText = (skills || '').trim();
        intro.textContent = `Personalized plan for: ${goal || 'your target'} • Duration: ${duration} weeks${skillsText ? ' • Focus: ' + skillsText : ''}.`;
      }
      localStorage.setItem('hasRoadmap', '1');
      updateRoadmapUI();
      if (loading) loading.style.display = 'none';
      if (results) results.style.display = 'block';
    }, 600);
  }

  window.makeRoadmap = function() {
    // Backwards-compatible: generate with defaults if user clicks older entry points
    if (!document.getElementById('roadmapResults')) {
      showSection('roadmap');
      return;
    }
    generateRoadmap();
    showSection('roadmap');
  }

  // Kick-off
  window.addEventListener('load', () => { initCounts(); updateRoadmapUI(); });
})();
