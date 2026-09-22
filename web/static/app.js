/**
 * The Film Archive - Client Application
 * Clean, accessible film catalog interface with sequential click-to-explore recommendations
 * and interactive Discovery Journey Graph.
 */

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    selectedUserId: 50,
    personas: [],
    selectedAnchorMovie: null,
    discoveryMode: 'balanced',
    broadenVariety: false,
    selectedAgeDesc: '25-34',
    journeyTrail: [],
  };

  // DOM Elements
  const personaSelect = document.getElementById('persona-select');
  const ageSelect = document.getElementById('age-select');
  const demographicIndicator = document.getElementById('demographic-indicator');
  const profileBio = document.getElementById('profile-bio');
  const favoritesList = document.getElementById('favorites-list');

  const filmSearchInput = document.getElementById('film-search-input');
  const filmSearchResults = document.getElementById('film-search-results');
  const activeAnchorBox = document.getElementById('active-anchor-box');
  const activeAnchorTitle = document.getElementById('active-anchor-title');
  const clearAnchorBtn = document.getElementById('clear-anchor-btn');

  const curationStyleSelect = document.getElementById('curation-style-select');
  const varietyCheckbox = document.getElementById('variety-checkbox');
  const submitBtn = document.getElementById('submit-recommendations-btn');

  const filmsList = document.getElementById('films-list');
  const loadingIndicator = document.getElementById('loading-indicator');
  const catalogSubtitle = document.getElementById('catalog-subtitle');
  const resultsCountBadge = document.getElementById('results-count-badge');

  const discoveryJourneySection = document.getElementById('discovery-journey-section');
  const journeyGraphContainer = document.getElementById('journey-graph-container');
  const resetJourneyBtn = document.getElementById('reset-journey-btn');

  function updateDemographicIndicator() {
    if (!demographicIndicator) return;
    const ageVal = state.selectedAgeDesc;
    let ageStr = ageVal ? `Age ${ageVal}` : 'All Ages';
    demographicIndicator.textContent = `Demographic Focus: ${ageStr}`;
  }

  // 1. Fetch Viewer Profiles
  async function loadProfiles() {
    try {
      const res = await fetch('/api/personas');
      state.personas = await res.json();
      personaSelect.innerHTML = state.personas
        .map(p => `<option value="${p.user_id}">${p.label}</option>`)
        .join('');

      updateProfileDetails(state.personas[0]);
    } catch (err) {
      console.error('Failed to load profiles:', err);
    }
  }

  // 2. Update Profile Details
  async function updateProfileDetails(profile) {
    state.selectedUserId = profile.user_id;
    state.selectedAgeDesc = profile.age_desc || '';

    if (ageSelect) ageSelect.value = state.selectedAgeDesc;
    updateDemographicIndicator();

    profileBio.textContent = profile.bio;

    if (profile.type === 'cold_start') {
      favoritesList.innerHTML = `<span class="favorites-label" style="border-top:none; padding-top:0;">New viewer profile. Recommendations will draw from popular consensus, chosen demographic, or any reference film.</span>`;
      return;
    }

    try {
      const res = await fetch(`/api/user_history/${profile.user_id}`);
      const history = await res.json();
      if (!history || history.length === 0) {
        favoritesList.innerHTML = `<span class="favorites-label" style="border-top:none;">No prior ratings listed.</span>`;
        return;
      }
      favoritesList.innerHTML = history.slice(0, 4).map(item => `
        <div class="favorite-item">
          <span>${item.title}</span>
          <span class="favorite-rating">${item.rating.toFixed(1)} / 5</span>
        </div>
      `).join('');
    } catch (err) {
      console.error('Failed to load favorites:', err);
    }
  }

  personaSelect.addEventListener('change', (e) => {
    const uid = parseInt(e.target.value, 10);
    const profile = state.personas.find(p => p.user_id === uid);
    if (profile) {
      state.selectedAnchorMovie = null;
      state.journeyTrail = [];
      activeAnchorBox.classList.add('hidden');
      renderJourneyGraph();
      updateProfileDetails(profile);
      fetchRecommendations();
    }
  });

  if (ageSelect) {
    ageSelect.addEventListener('change', (e) => {
      state.selectedAgeDesc = e.target.value;
      updateDemographicIndicator();
      fetchRecommendations();
    });
  }

  // 3. Search Reference Film
  let searchTimer = null;
  filmSearchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    clearTimeout(searchTimer);
    if (query.length < 2) {
      filmSearchResults.classList.add('hidden');
      return;
    }

    searchTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/movies?q=${encodeURIComponent(query)}`);
        const movies = await res.json();
        if (movies.length === 0) {
          filmSearchResults.innerHTML = `<div class="search-item">No matching titles found</div>`;
        } else {
          filmSearchResults.innerHTML = movies.slice(0, 10).map(m => `
            <div class="search-item" data-id="${m.movie_id}" data-title="${m.title}">
              <div>${m.title}</div>
              <div class="search-item-meta">${m.genres.join(', ')} (${m.bayesian_rating.toFixed(1)} / 5)</div>
            </div>
          `).join('');
        }
        filmSearchResults.classList.remove('hidden');

        document.querySelectorAll('.search-item[data-id]').forEach(el => {
          el.addEventListener('click', () => {
            const mid = parseInt(el.getAttribute('data-id'), 10);
            const mtitle = el.getAttribute('data-title');
            const foundObj = movies.find(m => m.movie_id === mid);
            const myear = foundObj ? foundObj.year : null;
            const mgenres = foundObj ? foundObj.genres : [];

            setAnchorFilm(mid, mtitle, myear, mgenres);
            // Initialize journey trail with searched movie
            state.journeyTrail = [{
              id: mid,
              title: mtitle,
              year: myear,
              genres: mgenres,
            }];

            filmSearchResults.classList.add('hidden');
            filmSearchInput.value = '';
            renderJourneyGraph();
            fetchRecommendations();
          });
        });
      } catch (err) {
        console.error('Search error:', err);
      }
    }, 250);
  });

  document.addEventListener('click', (e) => {
    if (!filmSearchInput.contains(e.target) && !filmSearchResults.contains(e.target)) {
      filmSearchResults.classList.add('hidden');
    }
  });

  function setAnchorFilm(id, title, year = null, genres = []) {
    state.selectedAnchorMovie = { id, title, year, genres };
    activeAnchorTitle.textContent = title;
    activeAnchorBox.classList.remove('hidden');
  }

  function clearAnchorAndJourney() {
    state.selectedAnchorMovie = null;
    state.journeyTrail = [];
    activeAnchorBox.classList.add('hidden');
    renderJourneyGraph();
    fetchRecommendations();
  }

  clearAnchorBtn.addEventListener('click', clearAnchorAndJourney);
  if (resetJourneyBtn) {
    resetJourneyBtn.addEventListener('click', clearAnchorAndJourney);
  }

  // 4. Discovery Mode & Options
  curationStyleSelect.addEventListener('change', (e) => {
    state.discoveryMode = e.target.value;
  });

  varietyCheckbox.addEventListener('change', (e) => {
    state.broadenVariety = e.target.checked;
  });

  // 5. Generate Recommendations
  async function fetchRecommendations() {
    loadingIndicator.classList.remove('hidden');
    filmsList.innerHTML = '';
    submitBtn.disabled = true;

    try {
      const activeProfile = state.personas.find(p => p.user_id === state.selectedUserId) || {};

      // Map friendly discovery mode to model parameters
      let alpha = 0.60;
      let strategy = 'rrf';
      if (state.discoveryMode === 'story_match') {
        alpha = 0.25;
        strategy = 'weighted';
      } else if (state.discoveryMode === 'crowd_favorites') {
        alpha = 0.85;
        strategy = 'rrf';
      }

      // Gather all movie IDs currently in the exploration trail to prevent recommendation loops
      const trailIds = (state.journeyTrail || []).map(t => t.id).filter(Boolean);
      if (state.selectedAnchorMovie && !trailIds.includes(state.selectedAnchorMovie.id)) {
        trailIds.push(state.selectedAnchorMovie.id);
      }

      const payload = {
        user_id: state.selectedUserId,
        seed_movie_id: state.selectedAnchorMovie ? state.selectedAnchorMovie.id : null,
        exclude_movie_ids: trailIds,
        alpha: alpha,
        strategy: strategy,
        apply_diversity: state.broadenVariety,
        age_desc: state.selectedAgeDesc || null,
        top_n: 10,
      };

      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      // If backend returned active seed details, synchronize with current anchor and journey node
      if (data.active_seed && state.selectedAnchorMovie) {
        state.selectedAnchorMovie.year = data.active_seed.year;
        state.selectedAnchorMovie.genres = data.active_seed.genres;
        const trailNode = state.journeyTrail.find(t => t.id === data.active_seed.movie_id);
        if (trailNode) {
          trailNode.year = data.active_seed.year;
          trailNode.genres = data.active_seed.genres;
        }
      }

      renderFilmList(data.recommendations);
      renderJourneyGraph();

      const demoText = state.selectedAgeDesc ? `viewers aged ${state.selectedAgeDesc}` : 'all audiences';
      if (state.selectedAnchorMovie) {
        catalogSubtitle.textContent = `Curated for preferences branching from "${state.selectedAnchorMovie.title}" (${demoText})`;
      } else {
        catalogSubtitle.textContent = `Curated selections for ${activeProfile.label || 'current profile'} (${demoText})`;
      }
      resultsCountBadge.textContent = `${data.recommendations.length} Titles Selected`;

    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
      filmsList.innerHTML = `<div class="film-entry" style="grid-template-columns: 1fr;">Failed to load recommendations. Please verify the archive service is online.</div>`;
    } finally {
      loadingIndicator.classList.add('hidden');
      submitBtn.disabled = false;
    }
  }

  // 6. Render Film List with Click-to-Explore Interaction
  function renderFilmList(items) {
    if (!items || items.length === 0) {
      filmsList.innerHTML = `<div class="film-entry" style="grid-template-columns: 1fr;">No titles matched the selected criteria.</div>`;
      return;
    }

    filmsList.innerHTML = items.map((item, index) => {
      const padRank = item.rank < 10 ? `0${item.rank}` : `${item.rank}`;
      return `
        <article class="film-entry clickable" data-index="${index}" title="Click to explore recommendations based on ${item.title}">
          <div class="film-number">${padRank}</div>
          <div class="film-body">
            <div class="film-title-row">
              <h3 class="film-title">${item.title}</h3>
              ${item.year ? `<span class="film-year">${item.year}</span>` : ''}
            </div>
            <div class="film-genres">${item.genres.join(' / ')}</div>
            <p class="film-curator-note">${item.reason}</p>
            <div class="film-explore-action">
              <span>Explore from this film</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </div>
          </div>
          <div class="film-side">
            <span class="curator-tag">${item.badge}</span>
            <span class="audience-score"><span class="score-highlight">${item.bayesian_rating.toFixed(1)}</span> / 5 (${item.rating_count.toLocaleString()} reviews)</span>
          </div>
        </article>
      `;
    }).join('');

    // Attach click listeners to dynamically refresh recommendations on clicked film
    document.querySelectorAll('.film-entry.clickable').forEach(entry => {
      entry.addEventListener('click', () => {
        const idx = parseInt(entry.getAttribute('data-index'), 10);
        const item = items[idx];
        if (!item) return;

        // If journey trail was empty, add previous anchor first if present
        if (state.journeyTrail.length === 0 && state.selectedAnchorMovie) {
          state.journeyTrail.push(state.selectedAnchorMovie);
        }

        // Check if item already exists in trail
        const existingIdx = state.journeyTrail.findIndex(t => t.id === item.movie_id);
        if (existingIdx !== -1) {
          state.journeyTrail = state.journeyTrail.slice(0, existingIdx + 1);
        } else {
          state.journeyTrail.push({
            id: item.movie_id,
            title: item.title,
            year: item.year,
            genres: item.genres,
          });
        }

        setAnchorFilm(item.movie_id, item.title, item.year, item.genres);
        renderJourneyGraph();
        fetchRecommendations();

        // Smooth scroll to catalog header
        const catalogHeader = document.querySelector('.catalog-header');
        if (catalogHeader) {
          catalogHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  // 7. Render Discovery Journey Graph
  function renderJourneyGraph() {
    if (!discoveryJourneySection || !journeyGraphContainer) return;

    if (!state.journeyTrail || state.journeyTrail.length === 0) {
      discoveryJourneySection.classList.add('hidden');
      journeyGraphContainer.innerHTML = '';
      return;
    }

    discoveryJourneySection.classList.remove('hidden');

    const htmlParts = [];
    state.journeyTrail.forEach((node, index) => {
      const stepNum = index < 9 ? `0${index + 1}` : `${index + 1}`;
      const isActive = state.selectedAnchorMovie && state.selectedAnchorMovie.id === node.id;
      const genresSummary = (node.genres && node.genres.length > 0)
        ? node.genres.slice(0, 2).join(' / ')
        : (node.year ? `${node.year}` : 'Archive Film');

      htmlParts.push(`
        <div class="journey-node ${isActive ? 'active' : ''}" data-step="${index}" data-id="${node.id}" title="Click to branch recommendations from ${node.title}">
          <div class="journey-node-step">
            <span>Step ${stepNum}</span>
            ${isActive ? '<span class="journey-node-status">Active</span>' : ''}
          </div>
          <div class="journey-node-title">${node.title}</div>
          <div class="journey-node-meta">${node.year ? `${node.year} · ` : ''}${genresSummary}</div>
        </div>
      `);

      if (index < state.journeyTrail.length - 1) {
        htmlParts.push(`
          <div class="journey-connector" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>
        `);
      }
    });

    journeyGraphContainer.innerHTML = htmlParts.join('');

    // Attach click events on journey nodes for jump-back navigation
    journeyGraphContainer.querySelectorAll('.journey-node').forEach(nodeEl => {
      nodeEl.addEventListener('click', () => {
        const step = parseInt(nodeEl.getAttribute('data-step'), 10);
        const targetNode = state.journeyTrail[step];
        if (!targetNode) return;

        // If clicking already active anchor, do not redundant fetch
        if (state.selectedAnchorMovie && state.selectedAnchorMovie.id === targetNode.id) return;

        // Jump back to this step in the trail
        state.journeyTrail = state.journeyTrail.slice(0, step + 1);
        setAnchorFilm(targetNode.id, targetNode.title, targetNode.year, targetNode.genres);
        renderJourneyGraph();
        fetchRecommendations();

        const catalogHeader = document.querySelector('.catalog-header');
        if (catalogHeader) {
          catalogHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // Auto-scroll graph container to show the newest node
    journeyGraphContainer.scrollLeft = journeyGraphContainer.scrollWidth;
  }

  submitBtn.addEventListener('click', fetchRecommendations);

  async function loadStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      const metaVal = document.querySelector('.meta-value');
      if (metaVal) {
        metaVal.textContent = `${data.total_movies.toLocaleString()} Films / ${data.total_ratings.toLocaleString()} Community Reviews`;
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  }

  // Initialize
  loadStats();
  loadProfiles().then(() => {
    fetchRecommendations();
  });
});
