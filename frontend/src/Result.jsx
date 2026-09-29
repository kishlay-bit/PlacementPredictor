import { lpa, rupees } from "./format.js";

export default function Result({ result, note }) {
  const chance = Math.round(result.placement_percent);
  const [low, high] = result.salary_range_lpa;
  const [lowInr, highInr] = result.salary_range_inr;

  return (
    <section className="result" aria-live="polite">
      <h2>Result</h2>

      <div className="block">
        <div className="label">Chance of placement</div>
        <div className="big">{chance}%</div>
        <div className="track" role="img" aria-label={`${chance} percent`}>
          <div className="fill" style={{ width: `${chance}%` }} />
        </div>
        {result.profile_percentile != null && (
          <p className="muted">
            This profile scores higher than about {Math.round(result.profile_percentile)}% of the students in the training data.
          </p>
        )}
      </div>

      <div className="block">
        <div className="label">Estimated package if placed</div>
        <div className="big">{lpa(result.salary_if_placed_lpa)}</div>
        <p>{rupees(result.salary_if_placed_inr)} per year</p>
        <p className="muted">
          Likely range: {lpa(low)} to {lpa(high)} ({rupees(lowInr)} to {rupees(highInr)} per year)
        </p>
      </div>

      {note && <p className="note">{note}</p>}
    </section>
  );
}
