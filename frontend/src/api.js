async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error("Could not reach the prediction server. Make sure the backend is running on port 8000.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.errors ? "Please correct the highlighted fields." : "The server returned an error.");
    err.fieldErrors = data.errors || {};
    throw err;
  }
  return data;
}

export const getSchema = () => request("/api/schema");
export const getModelInfo = () => request("/api/model-info");
export const predict = (values) =>
  request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
