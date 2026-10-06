const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function request(url, options) {
  const fullUrl = `${API_BASE_URL}${url}`;
  let res;
  try {
    res = await fetch(fullUrl, options);
  } catch {
    throw new Error("Could not reach the prediction server. Make sure the backend is running on port 8000.");
  }

  let data = {};
  try {
    data = await res.json();
  } catch {
    data = {};
  }

  if (!res.ok) {
    const err = new Error(
      data.errors ? "Please correct the highlighted fields." : "The prediction request failed. Please try again."
    );
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
