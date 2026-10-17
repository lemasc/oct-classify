// Behaviour for paper_diff pages: history picker, contents scroll-spy, copy tools.

const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// --- sticky header height, so anchors and the sidebar clear it even when it wraps -----------------
const legend = document.querySelector(".legend");
if (legend) {
  const measure = () =>
    document.documentElement.style.setProperty("--legend-h", legend.offsetHeight + "px");
  new ResizeObserver(measure).observe(legend);
  measure();
}

// --- history picker: one 'from' and one 'to', oldest-first ordering enforced --------------------
for (const form of $$("form.history")) {
  const radios = (name) => $$(`input[name=${name}]`, form);
  const at = (name) => radios(name).findIndex((x) => x.checked);
  const sync = () => {
    const a = at("from"), b = at("to");
    radios("from").forEach((x, i) => (x.disabled = x.hasAttribute("data-never") || (b >= 0 && i <= b)));
    radios("to").forEach((x, i) => (x.disabled = a >= 0 && i >= a));
  };
  form.addEventListener("change", sync);
  sync();
}

// --- contents: open on wide screens, highlight the section being read ---------------------------
const wide = matchMedia("(min-width:1100px)");
const lists = $$("details.toc");
const place = () => lists.forEach((d) => (d.open = wide.matches));
wide.addEventListener("change", place);
place();

const toc = document.querySelector("details.contents");
if (toc) {
  const links = new Map($$("a", toc).map((a) => [a.hash.slice(1), a]));
  const heads = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  const spy = () => {
    let current = heads[0];
    for (const h of heads) if (h.getBoundingClientRect().top <= (legend?.offsetHeight ?? 0) + 24) current = h;
    links.forEach((a, id) => a.classList.toggle("on", id === current?.id));
  };
  addEventListener("scroll", spy, { passive: true });
  spy();
}

// --- copy tools ---------------------------------------------------------------------------------
function flash(button, text) {
  const old = button.textContent;
  button.textContent = text;
  button.classList.add("ok");
  setTimeout(() => {
    button.textContent = old;
    button.classList.remove("ok");
  }, 1200);
}

function tool(label, title, action) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.title = title;
  b.addEventListener("click", async () => {
    try {
      flash(b, (await action()) || "copied");
    } catch (err) {
      console.error(err);
      flash(b, "failed");
    }
  });
  return b;
}

function host(el, inline) {
  const wrap = document.createElement("div");
  wrap.className = "tool-host" + (inline ? " inline" : "");
  el.replaceWith(wrap);
  wrap.append(el);
  const bar = document.createElement("div");
  bar.className = "tools";
  wrap.append(bar);
  return bar;
}

// Render an <img> (PNG or the SVGs Typst embeds) onto a white canvas. SVGs are drawn large so the
// copy stays sharp; the white fill keeps transparent figures readable once pasted elsewhere.
function rasterize(img, type, quality) {
  const svg = img.src.startsWith("data:image/svg");
  const w0 = img.naturalWidth || img.width, h0 = img.naturalHeight || img.height;
  const k = svg ? Math.max(1, 1600 / w0) : 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w0 * k);
  canvas.height = Math.round(h0 * k);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return new Promise((ok, no) =>
    canvas.toBlob((blob) => (blob ? ok(blob) : no(new Error("toBlob failed"))), type, quality));
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

for (const img of $$("img")) {
  const id = img.closest("figure[id]")?.id || "figure";
  const bar = host(img, true);
  // The async clipboard only accepts PNG across browsers, hence JPEG is a download.
  bar.append(
    tool("copy PNG", "Copy image to the clipboard as PNG", async () => {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": rasterize(img, "image/png") })]);
    }),
    tool("save JPEG", "Download image as JPEG", async () => {
      download(await rasterize(img, "image/jpeg", 0.92), id + ".jpg");
      return "saved";
    }),
  );
}

const cellText = (c) => c.innerText.replace(/\s+/g, " ").trim();
const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// What Excel and LibreOffice Calc put on the clipboard: a plain HTML table (no page styling, every
// cell a <td>) next to tab-separated text. Canva pastes the HTML flavour as an editable table.
function tableClip(table) {
  const rows = $$("tr", table).map((tr) => $$("th,td", tr));
  const html = rows
    .map((cells) => "<tr>" + cells.map((c) => {
      const span = ["colspan", "rowspan"].filter((a) => c.hasAttribute(a))
        .map((a) => ` ${a}="${c.getAttribute(a)}"`).join("");
      return `<td${span}>${esc(cellText(c))}</td>`;
    }).join("") + "</tr>")
    .join("");
  const text = rows.map((cells) => cells.map(cellText).join("\t")).join("\n");
  return { html: `<table><tbody>${html}</tbody></table>`, text };
}

async function writeTable({ html, text }) {
  if (window.ClipboardItem && navigator.clipboard?.write) {
    const blob = (type, data) => new Blob([data], { type });
    await navigator.clipboard.write([new ClipboardItem({
      "text/html": blob("text/html", html),
      "text/plain": blob("text/plain", text),
    })]);
    return;
  }
  // The async clipboard needs a secure context; fall back to a copy event for LAN hosts.
  const onCopy = (e) => {
    e.clipboardData.setData("text/html", html);
    e.clipboardData.setData("text/plain", text);
    e.preventDefault();
  };
  document.addEventListener("copy", onCopy, { once: true });
  document.execCommand("copy");
}

for (const wrap of $$(".tbl")) {
  const table = wrap.querySelector("table");
  if (!table) continue;
  host(wrap, false).append(
    tool("copy table", "Copy as a spreadsheet-style table (paste into Canva, Excel or Calc)",
      () => writeTable(tableClip(table))),
  );
}

document.getElementById("print")?.addEventListener("click", () => print());
