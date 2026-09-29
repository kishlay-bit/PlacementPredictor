import { useEffect, useRef, useState } from "react";
import { getModelInfo, getSchema, predict } from "./api.js";
import Form from "./Form.jsx";
import Result from "./Result.jsx";

export default function App() {
  const [fields, setFields] = useState(null);
  const [info, setInfo] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [result, setResult] = useState(null);
  const resultRef = useRef(null);

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

  const rows = info && info.training_rows ? info.training_rows.toLocaleString("en-IN") : null;

  return (
    <main className="page">
      <header>
        <h1>Student Placement Estimator</h1>
        <p className="lead">
          Fill in every field below to get an estimated chance of placement and the expected salary package.
        </p>
      </header>

      {loadError && <p className="banner error">{loadError}</p>}
      {!fields && !loadError && <p className="muted">Loading form...</p>}

      {fields && <Form fields={fields} onSubmit={predict} onResult={setResult} onClear={() => setResult(null)} />}

      {result && (
        <div ref={resultRef}>
          <Result result={result} note={info && info.reliability_note} />
        </div>
      )}

      <footer className="muted">
        {rows
          ? `Estimates come from a model trained on a dataset of ${rows} students. `
          : "Estimates come from a model trained on past student data. "}
        They are a rough guide, not a guarantee.
      </footer>
    </main>
  );
}
