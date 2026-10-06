import { useEffect, useRef, useState } from "react";
import { getModelInfo, getSchema, predict } from "./api.js";
import Form from "./Form.jsx";
import Result from "./Result.jsx";
import NotFound from "./NotFound.jsx";

const MODEL_SUMMARY = {
  classification: {
    name: "1. Logistic Regression",
    metrics: [
      ["Accuracy", "0.7356"],
      ["Precision", "0.7825"],
      ["Recall", "0.8120"],
      ["F1 score", "0.7970"],
      ["ROC AUC", "0.7981"],
      ["PR AUC", "0.8769"],
      ["Brier score", "0.1735"],
      ["Log loss", "0.5156"],
    ],
  },
  regression: {
    name: "3. XGBoost",
    metrics: [
      ["MAE", "2.61 LPA"],
      ["RMSE", "3.83 LPA"],
      ["R²", "0.6558"],
    ],
  },
  uncertainty: {
    coverage: "0.7916",
    note: "80% salary interval coverage",
  },
  endToEnd: {
    metrics: [
      ["MAE", "4.16 LPA"],
      ["RMSE", "5.16 LPA"],
      ["R²", "0.5887"],
    ],
  },
};

const NAV_ITEMS = [
  { label: "Home", href: "#home" },
  { label: "Predictor", href: "#predictor" },
  { label: "About / Methodology", href: "#about" },
  { label: "Models / Results", href: "#models" },
  { label: "Dataset", href: "#dataset" },
  { label: "GitHub", href: "https://github.com/kishlay-bit/PlacementPredictor", external: true },
];

function currentRoute() {
  if (typeof window === "undefined") return "/";
  return window.location.pathname || "/";
}

export default function App() {
  const [fields, setFields] = useState(null);
  const [info, setInfo] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [result, setResult] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const resultRef = useRef(null);

  const route = currentRoute();
  const is404 = route !== "/" && route !== "/index.html";

  useEffect(() => {
    getSchema()
      .then((d) => setFields(d.fields))
      .catch((e) => setLoadError(e.message));
    getModelInfo()
      .then(setInfo)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  useEffect(() => {
    const onResize = () => setMenuOpen(false);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  if (is404) {
    return <NotFound />;
  }

  const rows = info?.training_rows ? info.training_rows.toLocaleString("en-IN") : null;

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="container nav-wrap">
          <a href="#home" className="brand" aria-label="Go to homepage">
            <span className="brand-mark">P</span>
            <span className="brand-text">Placement Predictor</span>
          </a>

          <button
            type="button"
            className="nav-toggle"
            aria-expanded={menuOpen}
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span />
            <span />
            <span />
          </button>

          <nav className={`site-nav ${menuOpen ? "open" : ""}`} aria-label="Primary navigation">
            {NAV_ITEMS.map((item) =>
              item.external ? (
                <a
                  key={item.label}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => setMenuOpen(false)}
                >
                  {item.label}
                </a>
              ) : (
                <a key={item.label} href={item.href} onClick={() => setMenuOpen(false)}>
                  {item.label}
                </a>
              )
            )}
          </nav>
        </div>
      </header>

      <main className="page-shell">
        <section id="home" className="hero container">
          <div className="hero-copy">
            <span className="eyebrow">Student Placement & Salary Predictor</span>
            <h1>Predict Your Placement. Estimate Your Salary.</h1>
            <p>
              This application uses machine learning to estimate placement probability and expected salary
              from a student&apos;s academic, technical, internship, and career profile information.
            </p>
            <div className="hero-actions">
              <a href="#predictor" className="button primary">
                Predict My Placement
              </a>
              <a href="#models" className="button secondary">
                View Model Results
              </a>
            </div>
            <ul className="hero-points" aria-label="Key system capabilities">
              <li>Placement probability</li>
              <li>Expected LPA</li>
              <li>Salary range estimate</li>
            </ul>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="visual-card large-card">
              <div className="card-label">Placement outlook</div>
              <div className="card-score">82.4%</div>
              <div className="mini-progress">
                <span style={{ width: "82%" }} />
              </div>
              <div className="card-meta">Model-estimated probability</div>
            </div>

            <div className="visual-stack">
              <div className="visual-card small-card">
                <span>Expected salary</span>
                <strong>₹8.4 LPA</strong>
              </div>
              <div className="visual-card small-card">
                <span>Range</span>
                <strong>₹6.3 LPA – ₹11.5 LPA</strong>
              </div>
            </div>
          </div>
        </section>

        <section id="predictor" className="section container">
          <div className="section-heading">
            <span className="eyebrow">Prediction Engine</span>
            <h2>Student profile analysis</h2>
          </div>

          {loadError && <p className="banner error">{loadError}</p>}
          {!fields && !loadError && <p className="muted empty-state">Loading profile form…</p>}

          {fields && (
            <div className="predictor-layout">
              <Form fields={fields} onSubmit={predict} onResult={setResult} onClear={() => setResult(null)} />

              {result && (
                <div ref={resultRef} className="result-panel">
                  <Result result={result} note={info?.reliability_note} onReset={() => setResult(null)} />
                </div>
              )}
            </div>
          )}
        </section>

        <section id="about" className="section container">
          <div className="section-heading">
            <span className="eyebrow">Methodology</span>
            <h2>How the prediction works</h2>
          </div>

          <div className="methodology-flow" aria-label="Placement prediction workflow">
            <div className="flow-step">
              <span>1</span>
              <strong>Student Profile</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>2</span>
              <strong>Feature Processing</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>3</span>
              <strong>Placement Classification</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>4</span>
              <strong>Placement Probability</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>5</span>
              <strong>Salary Regression</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>6</span>
              <strong>Expected LPA</strong>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="flow-step">
              <span>7</span>
              <strong>Salary Range</strong>
            </div>
          </div>

          <div className="methodology-copy">
            <p>
              The system combines academic information, technical skills, project experience, internships,
              coding capability, communication and interview performance, extracurricular activity, and career
              preparation signals into a single profile scoring model.
            </p>
          </div>
        </section>

        <section id="models" className="section container">
          <div className="section-heading">
            <span className="eyebrow">Model performance</span>
            <h2>Models & Results</h2>
          </div>

          <div className="metrics-grid">
            <article className="metric-card">
              <h3>Placement classification</h3>
              <p className="metric-model">{MODEL_SUMMARY.classification.name}</p>
              <ul>
                {MODEL_SUMMARY.classification.metrics.map(([label, value]) => (
                  <li key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </li>
                ))}
              </ul>
            </article>

            <article className="metric-card">
              <h3>Salary regression</h3>
              <p className="metric-model">{MODEL_SUMMARY.regression.name}</p>
              <ul>
                {MODEL_SUMMARY.regression.metrics.map(([label, value]) => (
                  <li key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </li>
                ))}
              </ul>
            </article>

            <article className="metric-card">
              <h3>Salary uncertainty</h3>
              <p className="metric-model">q10 / q90 salary models</p>
              <ul>
                <li>
                  <span>{MODEL_SUMMARY.uncertainty.note}</span>
                  <strong>{MODEL_SUMMARY.uncertainty.coverage}</strong>
                </li>
              </ul>
            </article>

            <article className="metric-card">
              <h3>End-to-end expected LPA</h3>
              <ul>
                {MODEL_SUMMARY.endToEnd.metrics.map(([label, value]) => (
                  <li key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </li>
                ))}
              </ul>
            </article>
          </div>

          <div className="dataset-note">
            {rows ? `Training data: ${rows} student records.` : "Training data: model evaluation from the project repository."}
          </div>
        </section>

        <section id="dataset" className="section container">
          <div className="dataset-box">
            <div>
              <span className="eyebrow">Dataset</span>
              <h2>Student Placement Prediction Dataset</h2>
            </div>
            <a
              className="button primary"
              href="https://www.kaggle.com/datasets/kishlaytejeswi/student-placement-prediction-dataset"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Dataset on Kaggle
            </a>
          </div>
        </section>

        <section id="github" className="section container">
          <div className="dataset-box github-box">
            <div>
              <span className="eyebrow">Project source</span>
              <h2>Placement Predictor repository</h2>
            </div>
            <a
              className="button secondary"
              href="https://github.com/kishlay-bit/PlacementPredictor"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Source Code
            </a>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="container footer-grid">
          <div>
            <h3>Placement Predictor</h3>
            <p>
              Student placement and salary prediction using academic, technical, internship, and career profile data.
            </p>
          </div>

          <div>
            <h4>Navigation</h4>
            <ul>
              {NAV_ITEMS.filter((item) => item.label !== "GitHub").map((item) => (
                <li key={item.label}>
                  {item.external ? (
                    <a href={item.href} target="_blank" rel="noopener noreferrer">
                      {item.label}
                    </a>
                  ) : (
                    <a href={item.href}>{item.label}</a>
                  )}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4>External resources</h4>
            <ul>
              <li>
                <a href="https://www.kaggle.com/datasets/kishlaytejeswi/student-placement-prediction-dataset" target="_blank" rel="noopener noreferrer">
                  Kaggle Dataset
                </a>
              </li>
              <li>
                <a href="https://github.com/kishlay-bit/PlacementPredictor" target="_blank" rel="noopener noreferrer">
                  GitHub Repository
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="container footer-bottom">
          <span>© 2026 Placement Predictor. All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
}
