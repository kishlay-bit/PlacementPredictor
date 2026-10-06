import { lpa, lpaRange, percent, rupees } from "./format.js";

export default function Result({ result, note, onReset }) {
  const placementProbability = Number(result.placement_percent ?? result.placement_probability * 100 ?? 0);
  const salaryEstimate = Number(result.salary_if_placed_lpa ?? 0);
  const expectedSalary = Number(result.expected_lpa ?? 0);
  const salaryRange = Array.isArray(result.salary_range_lpa) ? result.salary_range_lpa : [0, 0];
  const [lowLpa, highLpa] = salaryRange;
  const predictedStatus = result.predicted_placed ? "High likelihood of placement" : "Lower likelihood of placement";

  return (
    <section className="result-dashboard" aria-live="polite">
      <div className="result-header">
        <div>
          <span className="eyebrow">Prediction result</span>
          <h3>Profile analyzed successfully.</h3>
        </div>
      </div>

      <div className="result-grid">
        <article className="result-card accent-card">
          <div className="label">Placement Probability</div>
          <div className="big-value">{percent(placementProbability)}</div>
          <div className="track" role="img" aria-label={`${placementProbability} percent probability`}>
            <div className="fill" style={{ width: `${Math.min(100, Math.max(0, placementProbability))}%` }} />
          </div>
          <p className="meta-label">Model-estimated placement probability</p>
        </article>

        <article className="result-card">
          <div className="label">Placement Prediction</div>
          <div className="big-value small">{predictedStatus}</div>
          <p className="meta-label">{result.predicted_placed ? "Likely to receive an offer" : "Probability is below the placement threshold"}</p>
        </article>

        <article className="result-card">
          <div className="label">Expected Salary</div>
          <div className="big-value small">{lpa(salaryEstimate)}</div>
          <p className="meta-label">Model-estimated expected salary</p>
          <p className="subtle">{rupees(result.salary_if_placed_inr ?? salaryEstimate * 100000)} per year</p>
        </article>

        <article className="result-card">
          <div className="label">Salary Range</div>
          <div className="big-value small">{lpaRange(lowLpa, highLpa)}</div>
          <p className="meta-label">Estimated salary range</p>
          <p className="subtle">
            {rupees(result.salary_range_inr?.[0] ?? lowLpa * 100000)} – {rupees(result.salary_range_inr?.[1] ?? highLpa * 100000)} per year
          </p>
        </article>
      </div>

      {result.profile_percentile != null && (
        <p className="result-footnote">
          This profile is above approximately {Math.round(result.profile_percentile)}% of the training data.
        </p>
      )}

      {expectedSalary > 0 && (
        <p className="result-footnote">
          Expected LPA: {lpa(expectedSalary)}
        </p>
      )}

      {note && <div className="result-note">{note}</div>}

      <div className="result-actions">
        <button type="button" className="button primary" onClick={onReset}>
          Predict Again
        </button>
      </div>
    </section>
  );
}
