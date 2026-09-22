/**
 * The Film Archive - Client Application
 * Clean, accessible film catalog interface.
 */

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    selectedUserId: 50,
    personas: [],
    selectedAnchorMovie: null,
    discoveryMode: 'balanced',
    broadenVariety: false,
    selectedAgeDesc: '25-34',
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
            setAnchorFilm(mid, mtitle);
            filmSearchResults.classList.add('hidden');
            filmSearchInput.value = '';
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

  function setAnchorFilm(id, title) {
    state.selectedAnchorMovie = { id, title };
    activeAnchorTitle.textContent = title;
    activeAnchorBox.classList.remove('hidden');
  }

  clearAnchorBtn.addEventListener('click', () => {
    state.selectedAnchorMovie = null;
    activeAnchorBox.classList.add('hidden');
  });

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

      const payload = {
        user_id: state.selectedUserId,
        seed_movie_id: state.selectedAnchorMovie ? state.selectedAnchorMovie.id : null,
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
      renderFilmList(data.recommendations);

      const demoText = state.selectedAgeDesc ? `viewers aged ${state.selectedAgeDesc}` : 'all audiences';
      if (state.selectedAnchorMovie) {
        catalogSubtitle.textContent = `Curated for preferences anchored on "${state.selectedAnchorMovie.title}" (${demoText})`;
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

  function renderFilmList(items) {
    if (!items || items.length === 0) {
      filmsList.innerHTML = `<div class="film-entry" style="grid-template-columns: 1fr;">No titles matched the selected criteria.</div>`;
      return;
    }

    filmsList.innerHTML = items.map(item => {
      const padRank = item.rank < 10 ? `0${item.rank}` : `${item.rank}`;
      return `
        <article class="film-entry">
          <div class="film-number">${padRank}</div>
          <div class="film-body">
            <div class="film-title-row">
              <h3 class="film-title">${item.title}</h3>
              ${item.year ? `<span class="film-year">${item.year}</span>` : ''}
            </div>
            <div class="film-genres">${item.genres.join(' / ')}</div>
            <p class="film-curator-note">${item.reason}</p>
          </div>
          <div class="film-side">
            <span class="curator-tag">${item.badge}</span>
            <span class="audience-score"><span class="score-highlight">${item.bayesian_rating.toFixed(1)}</span> / 5 (${item.rating_count.toLocaleString()} reviews)</span>
          </div>
        </article>
      `;
    }).join('');
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
