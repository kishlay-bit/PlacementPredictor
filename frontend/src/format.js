const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

export const rupees = (amount) => "\u20B9" + inr.format(Math.round(amount));

export const lpa = (value) => value.toFixed(2) + " LPA";
