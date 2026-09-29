import { useState } from "react";

const GROUPS = [
  ["Academics", ["cgpa", "backlogs"]],
  ["Experience", ["internships_count", "projects_count", "certifications_count", "hackathons_participated", "github_repos"]],
  ["Skill scores", ["coding_skill_score", "aptitude_score", "communication_skill_score", "logical_reasoning_score", "mock_interview_score"]],
  ["Activities", ["leadership_score", "extracurricular_score"]],
];

const g = (n) => String(Number(n));

function groupFields(fields) {
  const used = new Set();
  const groups = GROUPS.map(([title, names]) => {
    const items = names.map((n) => fields.find((f) => f.name === n)).filter(Boolean);
    items.forEach((f) => used.add(f.name));
    return [title, items];
  });
  const rest = fields.filter((f) => !used.has(f.name));
  if (rest.length) groups.push(["Other details", rest]);
  return groups.filter(([, items]) => items.length);
}

function check(field, raw) {
  if (raw === undefined || raw === "") return "This field is required.";
  if (field.type === "categorical") return "";
  const v = Number(raw);
  if (Number.isNaN(v)) return "Enter a number.";
  if (v < field.min || v > field.max) return `Must be between ${g(field.min)} and ${g(field.max)}.`;
  if (field.type === "integer" && !Number.isInteger(v)) return "Enter a whole number.";
  return "";
}

export default function Form({ fields, onSubmit, onResult, onClear }) {
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");

  const setValue = (name, v) => {
    setValues((prev) => ({ ...prev, [name]: v }));
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: "" }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError("");
    const found = {};
    fields.forEach((f) => {
      const msg = check(f, values[f.name]);
      if (msg) found[f.name] = msg;
    });
    setErrors(found);
    if (Object.keys(found).length) {
      setFormError("Please correct the highlighted fields.");
      const first = fields.find((f) => found[f.name]);
      document.getElementById(first.name)?.focus();
      return;
    }
    setBusy(true);
    try {
      onResult(await onSubmit(values));
    } catch (err) {
      setFormError(err.message);
      if (err.fieldErrors) setErrors(err.fieldErrors);
    } finally {
      setBusy(false);
    }
  };

  const handleClear = () => {
    setValues({});
    setErrors({});
    setFormError("");
    onClear();
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      {groupFields(fields).map(([title, items]) => (
        <fieldset key={title}>
          <legend>{title}</legend>
          {items.map((f) => (
            <div className="field" key={f.name}>
              <label htmlFor={f.name}>{f.label}</label>
              {f.type === "categorical" ? (
                <select
                  id={f.name}
                  value={values[f.name] ?? ""}
                  onChange={(e) => setValue(f.name, e.target.value)}
                  className={errors[f.name] ? "invalid" : ""}
                >
                  <option value="" disabled>
                    Select
                  </option>
                  {f.options.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id={f.name}
                  type="number"
                  inputMode="decimal"
                  step={f.type === "integer" ? 1 : "any"}
                  min={f.min}
                  max={f.max}
                  value={values[f.name] ?? ""}
                  onChange={(e) => setValue(f.name, e.target.value)}
                  onWheel={(e) => e.currentTarget.blur()}
                  className={errors[f.name] ? "invalid" : ""}
                />
              )}
              {errors[f.name] ? (
                <span className="msg error">{errors[f.name]}</span>
              ) : (
                f.type !== "categorical" && (
                  <span className="msg muted">
                    {g(f.min)} to {g(f.max)}
                  </span>
                )
              )}
            </div>
          ))}
        </fieldset>
      ))}

      {formError && <p className="banner error">{formError}</p>}

      <div className="actions">
        <button type="submit" disabled={busy}>
          {busy ? "Calculating..." : "Get estimate"}
        </button>
        <button type="button" className="secondary" onClick={handleClear} disabled={busy}>
          Clear
        </button>
      </div>
    </form>
  );
}
