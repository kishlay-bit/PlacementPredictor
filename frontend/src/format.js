const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

export const rupees = (amount) => "\u20B9" + inr.format(Math.round(amount));

export const lpa = (value) => {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return "—";
  return `₹${number.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1")} LPA`;
};

export const lpaRange = (low, high) => `${lpa(low)} – ${lpa(high)}`;

export const percent = (value) => {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return "—";
  return `${number.toFixed(1).replace(/\.0$/, "")}%`;
};
