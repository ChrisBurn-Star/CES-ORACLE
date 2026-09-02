(function () {
  const data = window.PROFILE_DATA;
  const root = document.documentElement;

  function setTextFields() {
    document.querySelectorAll("[data-field]").forEach((element) => {
      const key = element.dataset.field;
      if (key === "emailLink") return;
      if (data[key] !== undefined) element.textContent = data[key];
    });
    document.title = `${data.name} — AI Researcher`;
    document.querySelector('[data-field="emailLink"]').href = `mailto:${data.email}`;
  }

  function renderProfile() {
    document.getElementById("profile-list").innerHTML = data.basicInfo
      .map(
        (item) => `
          <div>
            <dt>${item.label}</dt>
            <dd>${item.value}</dd>
          </div>`,
      )
      .join("");
  }

  function renderAbout() {
    document.getElementById("about-paragraphs").innerHTML = data.aboutParagraphs
      .map((text) => `<p>${text}</p>`)
      .join("");
  }

  function renderMarquee() {
    const words = [...data.researchKeywords, ...data.researchKeywords];
    document.getElementById("marquee-track").innerHTML = words
      .map((word) => `<span>${word}</span><i aria-hidden="true">✳</i>`)
      .join("");
  }

  function renderResearch() {
    document.getElementById("research-grid").innerHTML = data.research
      .map(
        (item) => `
          <article class="research-card reveal">
            <div class="research-card-top"><span>${item.code}</span><span aria-hidden="true">↗</span></div>
            <h3>${item.title}</h3>
            <p class="research-en">${item.english}</p>
            <p class="research-description">${item.description}</p>
            <ul>${item.tags.map((tag) => `<li>${tag}</li>`).join("")}</ul>
          </article>`,
      )
      .join("");
  }

  function renderArticles() {
    document.getElementById("articles-list").innerHTML = data.articles
      .map(
        (item) => `
          <a class="article reveal" href="${item.url}" ${item.url !== "#" ? 'target="_blank" rel="noreferrer"' : ""}>
            <span class="article-number">${item.number}</span>
            <div class="article-main">
              <div class="article-meta"><span>${item.category}</span><span>${item.meta}</span></div>
              <h3>${item.title}</h3>
              <p>${item.summary}</p>
            </div>
            <span class="article-arrow" aria-hidden="true">↗</span>
          </a>`,
      )
      .join("");
  }

  function renderEducation() {
    document.getElementById("education-list").innerHTML = data.education
      .map(
        (item) => `
          <article class="timeline-item reveal">
            <div class="timeline-year">${item.period}</div>
            <div class="timeline-dot" aria-hidden="true"></div>
            <div class="timeline-content">
              <h3>${item.school}</h3>
              <p class="degree">${item.degree}</p>
              <p>${item.description}</p>
            </div>
          </article>`,
      )
      .join("");
  }

  function updateClock() {
    const time = new Intl.DateTimeFormat("en-GB", {
      timeZone: data.timezone,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date());
    document.getElementById("local-time").textContent = time;
  }

  function setupTheme() {
    const button = document.getElementById("theme-toggle");
    const label = button.querySelector(".theme-label");
    const stored = localStorage.getItem("homepage-theme");
    const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const initial = stored || preferred;

    function apply(theme) {
      root.dataset.theme = theme;
      label.textContent = theme === "dark" ? "Light" : "Dark";
      root.style.colorScheme = theme;
    }

    apply(initial);
    button.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      apply(next);
      localStorage.setItem("homepage-theme", next);
    });
  }

  function setupInteractions() {
    document.querySelectorAll('.article[href="#"]').forEach((link) => {
      link.addEventListener("click", (event) => event.preventDefault());
    });

    const toast = document.getElementById("toast");
    document.querySelector(".copy-email").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(data.email);
      } catch (_) {
        const field = document.createElement("textarea");
        field.value = data.email;
        document.body.appendChild(field);
        field.select();
        document.execCommand("copy");
        field.remove();
      }
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 1800);
    });

    const progress = document.getElementById("progress-bar");
    window.addEventListener(
      "scroll",
      () => {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
      },
      { passive: true },
    );

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 },
    );
    document.querySelectorAll(".reveal").forEach((item) => observer.observe(item));
  }

  setTextFields();
  renderProfile();
  renderAbout();
  renderMarquee();
  renderResearch();
  renderArticles();
  renderEducation();
  setupTheme();
  setupInteractions();
  updateClock();
  setInterval(updateClock, 30000);
  document.getElementById("current-year").textContent = new Date().getFullYear();
})();
