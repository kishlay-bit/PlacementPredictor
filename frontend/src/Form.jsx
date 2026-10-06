import { useState } from "react";

const GROUPS = [
  {
    title: "Academic Profile",
    names: ["cgpa", "attendance_percentage", "backlogs", "academic_consistency", "core_subject_score"],
  },
  {
    title: "Technical Profile",
    names: [
      "technical_skills_count",
      "coding_skill_score",
      "dsa_score",
      "github_projects",
      "coding_contest_rating",
    ],
  },
  {
    title: "Experience & Projects",
    names: ["internship_count", "internship_months", "projects_count", "work_experience_months", "hackathon_count", "certifications_count"],
  },
  {
    title: "Communication & Career Preparation",
    names: ["communication_score", "aptitude_score", "resume_score", "interview_score", "placement_preparation_hours_per_week"],
  },
  {
    title: "Profile Preferences",
    names: ["college_tier", "degree_type", "specialization", "preferred_job_role", "location_preference"],
  },
];

const fmtNumber = (n) => String(Number(n));

function groupFields(fields) {
  const used = new Set();
  const groups = GROUPS.map(({ title, names }) => {
    const items = names
      .map((name) => fields.find((field) => field.name === name))
      .filter(Boolean);
    items.forEach((field) => used.add(field.name));
    return [title, items];
  });

  const rest = fields.filter((field) => !used.has(field.name));
  if (rest.length) groups.push(["Other details", rest]);
  return groups.filter(([, items]) => items.length);
}

function check(field, raw) {
  if (raw === undefined || raw === "") return "This field is required.";
  if (field.type === "categorical") return "";

  const value = Number(raw);
  if (Number.isNaN(value)) return "Please enter a valid number.";
  if (value < field.min || value > field.max) {
    const label = field.label.toLowerCase();
    return `Please enter a valid ${label} between ${fmtNumber(field.min)} and ${fmtNumber(field.max)}.`;
  }
  if (field.type === "integer" && !Number.isInteger(value)) {
    return "Please enter a whole number.";
  }
  return "";
}

export default function Form({ fields, onSubmit, onResult, onClear }) {
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const setValue = (name, v) => {
    setValues((prev) => ({ ...prev, [name]: v }));
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: "" }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError("");
    setSuccessMessage("");

    const found = {};
    fields.forEach((field) => {
      const msg = check(field, values[field.name]);
      if (msg) found[field.name] = msg;
    });

    setErrors(found);
    if (Object.keys(found).length) {
      setFormError("Please correct the highlighted fields.");
      const firstField = fields.find((field) => found[field.name]);
      if (firstField) {
        document.getElementById(firstField.name)?.focus();
      }
      return;
    }

    setBusy(true);
    try {
      const data = await onSubmit(values);
      onResult(data);
      setSuccessMessage("Prediction completed successfully.");
    } catch (err) {
      setFormError(err.message || "Prediction failed. Please try again.");
      if (err.fieldErrors) setErrors(err.fieldErrors);
      setSuccessMessage("");
    } finally {
      setBusy(false);
    }
  };

  const handleClear = () => {
    setValues({});
    setErrors({});
    setFormError("");
    setSuccessMessage("");
    onClear();
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit} noValidate>
      {groupFields(fields).map(([title, items]) => (
        <fieldset key={title}>
          <legend>{title}</legend>
          <div className="field-grid">
            {items.map((field) => (
              <div className="field" key={field.name}>
                <label htmlFor={field.name}>{field.label}</label>
                {field.type === "categorical" ? (
                  <select
                    id={field.name}
                    value={values[field.name] ?? ""}
                    onChange={(e) => setValue(field.name, e.target.value)}
                    className={errors[field.name] ? "invalid" : ""}
                    aria-invalid={Boolean(errors[field.name])}
                  >
                    <option value="" disabled>
                      Select
                    </option>
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={field.name}
                    type="number"
                    inputMode="decimal"
                    step={field.type === "integer" ? 1 : "any"}
                    min={field.min}
                    max={field.max}
                    value={values[field.name] ?? ""}
                    onChange={(e) => setValue(field.name, e.target.value)}
                    onWheel={(e) => e.currentTarget.blur()}
                    className={errors[field.name] ? "invalid" : ""}
                    aria-invalid={Boolean(errors[field.name])}
                  />
                )}

                {errors[field.name] ? (
                  <span className="msg error">{errors[field.name]}</span>
                ) : (
                  field.type !== "categorical" && (
                    <span className="msg muted">
                      {fmtNumber(field.min)} to {fmtNumber(field.max)}
                    </span>
                  )
                )}
              </div>
            ))}
          </div>
        </fieldset>
      ))}

      {formError && <p className="banner error">{formError}</p>}
      {successMessage && <p className="banner success">{successMessage}</p>}

      <div className="actions">
        <button type="submit" disabled={busy} className="button primary submit-button">
          {busy ? "Analyzing Profile..." : "Predict Placement"}
        </button>
        <button type="button" className="button secondary submit-button" onClick={handleClear} disabled={busy}>
          Clear
        </button>
      </div>
    </form>
  );
}
