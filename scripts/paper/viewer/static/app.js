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

let spy = () => {};
function watchContents() {
  const toc = document.querySelector("details.contents");
  if (!toc) return (spy = () => {});
  const links = new Map($$("a", toc).map((a) => [a.hash.slice(1), a]));
  const heads = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  spy = () => {
    let current = heads[0];
    for (const h of heads) if (h.getBoundingClientRect().top <= (legend?.offsetHeight ?? 0) + 24) current = h;
    links.forEach((a, id) => a.classList.toggle("on", id === current?.id));
  };
  spy();
}
addEventListener("scroll", () => spy(), { passive: true });
watchContents();

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

// An inline <svg> (Typst html.frame, e.g. the flowchart) as a loaded <img>, sized from its viewBox
// because the page sizes it in em.
function svgImage(svg) {
  const copy = svg.cloneNode(true);
  const [, , w, h] = svg.getAttribute("viewBox").split(/\s+/);
  copy.setAttribute("width", w);
  copy.setAttribute("height", h);
  copy.removeAttribute("style");
  const img = new Image();
  return new Promise((ok, no) => {
    img.onload = () => ok(img);
    img.onerror = () => no(new Error("svg failed to load"));
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(copy));
  });
}

// Render an <img> or inline <svg> (PNG or the SVGs Typst embeds) onto a white canvas. SVGs are drawn large so the
// copy stays sharp; the white fill keeps transparent figures readable once pasted elsewhere.
async function rasterize(source, type, quality) {
  const img = source instanceof SVGElement ? await svgImage(source) : source;
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

function addImageTools(root) {
  for (const img of $$("img, figure svg", root)) {
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

}

function addTableTools(root) {
  for (const wrap of $$(".tbl", root)) {
    const table = wrap.querySelector("table");
    if (!table) continue;
    host(wrap, false).append(
      tool("copy table", "Copy as a spreadsheet-style table (paste into Canva, Excel or Calc)",
        () => writeTable(tableClip(table))),
    );
  }
}

const addTools = (root) => (addImageTools(root), addTableTools(root));
addTools(document);

document.getElementById("print")?.addEventListener("click", () => print());

// --- live reload: the working tree changed on disk ------------------------------------------------
// The server streams a fingerprint of the paper sources. The page either offers a reload or, with
// auto-reload on, fetches the new page and swaps it in, putting the reader back at the same text.
const BLOCKS = "h2,h3,h4,p,li,figure";
const auto = document.getElementById("auto");
const reload = document.getElementById("reload");
let version = document.body.dataset.version;

const stored = (key, value) => {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {}
};

// The block at the top of the viewport, named by its section, its text and its place in the section.
function anchor() {
  const edge = (legend?.offsetHeight ?? 0) + 8;
  let head = "", n = 0;
  for (const el of $$(`main ${BLOCKS}`)) {
    if (/^H[2-4]$/.test(el.tagName)) [head, n] = [el.id, -1];
    n++;
    const box = el.getBoundingClientRect();
    if (box.bottom > edge) return { head, n, text: el.textContent.trim().slice(0, 40), shift: edge - box.top };
  }
}

function restore(at) {
  if (!at) return;
  const edge = (legend?.offsetHeight ?? 0) + 8;
  let head = "", n = 0;
  const section = [];
  for (const el of $$(`main ${BLOCKS}`)) {
    if (/^H[2-4]$/.test(el.tagName)) [head, n] = [el.id, -1];
    n++;
    if (head === at.head) section.push([n, el]);
  }
  // Prefer the same text (it survives paragraphs added above), else the same position in the section.
  const hit = section.find(([, el]) => at.text && el.textContent.trim().startsWith(at.text))
    ?? section.find(([k]) => k === at.n) ?? section.at(-1);
  if (hit) scrollBy({ top: hit[1].getBoundingClientRect().top + at.shift - edge, behavior: "instant" });
}

function complain(message) {
  reload.hidden = false;
  reload.classList.add("fail");
  reload.textContent = "build failed · retry";
  reload.title = message;
}

async function refresh() {
  const res = await fetch(location.href);
  const text = await res.text();
  if (!res.ok) return complain(new DOMParser().parseFromString(text, "text/html").body.textContent);
  const next = new DOMParser().parseFromString(text, "text/html");
  const at = anchor();
  document.title = next.title;
  for (const sel of ["nav.side", "main", ".legend .label"]) document.querySelector(sel).replaceWith(next.querySelector(sel));
  addTools(document.querySelector("main"));
  place();
  watchContents();
  restore(at);
  version = next.body.dataset.version;
  reload.hidden = true;
  reload.classList.remove("fail");
  reload.textContent = "changed · reload";
  listen();
}

let source;
function listen() {
  source?.close();
  source = new EventSource(`/_events?v=${version}`);
  source.onmessage = (e) => {
    if (e.data === version) return;
    if (auto.checked) refresh().catch((err) => complain(String(err)));
    else reload.hidden = false;
  };
}

if (version) {
  auto.checked = stored("paper-auto") === "1";
  auto.addEventListener("change", () => {
    stored("paper-auto", auto.checked ? "1" : "0");
    if (auto.checked && !reload.hidden) reload.click();
  });
  reload.addEventListener("click", () => refresh().catch((err) => complain(String(err))));
  listen();
}

// --- page marks toggle ----------------------------------------------------------------------------
const marks = document.getElementById("marks");
marks.checked = stored("paper-marks") !== "0";
const showMarks = () => document.body.classList.toggle("marks", marks.checked);
marks.addEventListener("change", () => (stored("paper-marks", marks.checked ? "1" : "0"), showMarks()));
showMarks();
