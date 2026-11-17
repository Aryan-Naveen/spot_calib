// Shared helpers for the calibration GUIs.
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}
const post = (path, data) => api(path, { method: "POST", body: JSON.stringify(data || {}) });

function renderBadges(el, badges) {
  el.innerHTML = badges.map((b) =>
    `<div class="badge ${b.ok ? "ok" : ""}"><div class="t">${esc(b.label)}</div>` +
    `<div class="v">${esc(b.value)} <span class="t">${esc(b.unit)}</span></div><div class="t">target ${esc(b.threshold)}</div></div>`
  ).join("");
}

function renderDone(el, badges) {
  const ok = badges.length && badges.every((b) => b.ok);
  el.className = "alldone " + (ok ? "ok" : "no");
  el.textContent = ok ? "All targets met — ready to export" : `${badges.filter((b) => b.ok).length}/${badges.length} targets met`;
}

function renderHeat(grid) {
  // grid: 2D array of counts
  const rows = grid.length, cols = grid[0].length;
  const max = Math.max(1, ...grid.flat());
  const cells = grid.flat().map((v) => {
    const a = v === 0 ? 0.08 : 0.25 + 0.75 * (v / max);
    return `<div title="${v}" style="background: rgba(42,91,215,${a})"></div>`;
  }).join("");
  return `<div class="heat" style="grid-template-columns: repeat(${cols}, 14px)">${cells}</div>`;
}

function showMsg(el, text, isErr) {
  el.innerHTML = text ? `<div class="msg ${isErr ? "err" : ""}">${esc(text)}</div>` : "";
}

async function copyText(textarea) {
  try { await navigator.clipboard.writeText(textarea.value); }
  catch { textarea.select(); document.execCommand("copy"); }
}

function fmt(v, d = 3) { return v === null || v === undefined ? "—" : Number(v).toFixed(d); }

function sortableTable(tableEl, rows, columns, onRender) {
  // columns: [{key,label,fmt?}], state kept on element
  const st = tableEl._sort || (tableEl._sort = { key: columns[0].key, dir: 1 });
  const sorted = [...rows].sort((a, b) => {
    const x = a[st.key], y = b[st.key];
    if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (x < y ? -1 : 1) * st.dir;
  });
  tableEl.innerHTML = `<thead><tr>${columns.map((c) => `<th data-k="${c.key}">${esc(c.label)}${st.key === c.key ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}<th></th></tr></thead>` +
    `<tbody>${sorted.map((r) => `<tr class="${r.status || ""}">${columns.map((c) => `<td>${c.fmt ? c.fmt(r[c.key], r) : esc(r[c.key] ?? "—")}</td>`).join("")}<td>${onRender(r)}</td></tr>`).join("")}</tbody>`;
  tableEl.querySelectorAll("th[data-k]").forEach((th) => th.onclick = () => {
    const k = th.dataset.k; st.dir = st.key === k ? -st.dir : 1; st.key = k; sortableTable(tableEl, rows, columns, onRender);
  });
}
