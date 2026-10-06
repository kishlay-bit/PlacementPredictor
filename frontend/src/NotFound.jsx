export default function NotFound() {
  return (
    <div className="not-found-page">
      <div className="not-found-card">
        <span className="eyebrow">404</span>
        <h1>Page not found</h1>
        <p>The page you requested could not be found.</p>
        <button type="button" className="button primary" onClick={() => (window.location.href = "/")}>
          Return home
        </button>
      </div>
    </div>
  );
}
