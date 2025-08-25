// ADD THIS WHOLE FUNCTION
function init() {
  console.log("[DEBUG] DOMContentLoaded - initializing UI bindings");

  // Re-query after DOM is ready to avoid nulls
  window.formContainer = document.getElementById('formContainer');
  window.uploadForm    = document.getElementById('uploadForm');
  window.statusDiv     = document.getElementById('status');
  window.exportBtn     = document.getElementById('exportBtn');

  if (!formContainer || !uploadForm || !statusDiv || !exportBtn) {
    console.error("[ERROR] One or more required elements not found.", {
      hasFormContainer: !!formContainer,
      hasUploadForm: !!uploadForm,
      hasStatusDiv: !!statusDiv,
      hasExportBtn: !!exportBtn
    });
    alert("Required elements missing in HTML. Check IDs: formContainer, uploadForm, status, exportBtn.");
    return;
  }

  // Attach listeners safely
  uploadForm.addEventListener('submit', onUploadSubmit);
  exportBtn.addEventListener('click', onExportClick);

  // Initial fetch of config/fields
  fetchConfig().catch(err => {
    console.error("[ERROR] fetchConfig failed:", err);
    statusDiv.textContent = "Failed to load fields.";
  });
}


const formContainer = document.getElementById('formContainer');
const uploadForm = document.getElementById('uploadForm');
const statusDiv = document.getElementById('status');
const exportBtn = document.getElementById('exportBtn');

let fields = [];
let currentResults = [];

function rowTemplate(field) {
  const options = ["Satisfactory", "Unsatisfactory", "Information not present"];
  const status = field.status || "";
  const page = field.page || "";
  const section = field.section || "";
  const information_found = field.information_found || "";
  return `
    <div class="row" data-id="${field.field_id}">
      <div class="cell name">${field.field_name}</div>
      <div class="cell">
        <select class="status">
          <option value="">-- choose --</option>
          ${options.map(o => `<option value="${o}" ${o===status?'selected':''}>${o}</option>`).join('')}
        </select>
      </div>
      <div class="cell">
        <input type="number" class="page" min="1" step="1" value="${page}"/>
      </div>
      <div class="cell">
        <input type="text" class="section" placeholder="Heading / section" value="${section}"/>
      </div>
      <div class="cell">
        <textarea class="information-found" placeholder="Information found..." rows="2">${information_found}</textarea>
      </div>
    </div>
  `;
}

function renderForm(resultsOrFields) {
  formContainer.innerHTML = `
    <div class="row header">
      <div>Requirement</div>
      <div>Status</div>
      <div>Page #</div>
      <div>Section</div>
      <div>Information Found</div>
    </div>
    ${resultsOrFields.map(rowTemplate).join('')}
  `;

  // Toggle page/section enabled only when status != "Information not present"
  document.querySelectorAll('.row').forEach(r => {
    const sel = r.querySelector('.status');
    const page = r.querySelector('.page');
    const section = r.querySelector('.section');
    function toggle() {
      const val = sel.value;
      const enable = (val === 'Satisfactory' || val === 'Unsatisfactory');
      page.disabled = !enable;
      section.disabled = !enable;
      if (!enable) { page.value = ''; section.value=''; }
    }
    sel.addEventListener('change', toggle);
    toggle();
  });
}

// REPLACE fetchConfig WITH THIS WHOLE FUNCTION
async function fetchConfig() {
  console.log("[DEBUG] Fetching /api/config ...");
  try {
    const res = await fetch('/api/config');
    if (!res.ok) {
      console.error("[ERROR] /api/config HTTP error", res.status);
      throw new Error("HTTP " + res.status);
    }
    const data = await res.json();
    console.log("[DEBUG] /api/config data", data);

    fields = Array.isArray(data.fields) ? data.fields : [];
    if (!fields.length) {
      console.warn("[WARN] No fields returned from /api/config. Rendering empty table.");
    }

    currentResults = fields.map(f => ({
      field_id: f.id,
      field_name: f.name,
      status: "",
      page: "",
      section: "",
      information_found: ""
    }));

    renderForm(currentResults);
    statusDiv.textContent = fields.length ? `Loaded ${fields.length} fields.` : "No fields loaded.";
  } catch (err) {
    console.error("[ERROR] fetchConfig failed:", err);
    // Render just the header so the page doesn't look 'blank'
    formContainer.innerHTML = `
      <div class="row header">
        <div>Requirement</div><div>Status</div><div>Page #</div><div>Section</div><div>Information Found</div>
      </div>
    `;
    statusDiv.textContent = "Could not load config. Try again after fixing server/static paths.";
  }
}


// REPLACE inline submit-handler with this WHOLE FUNCTION
async function onUploadSubmit(e) {
  try {
    e.preventDefault();
    console.log("[DEBUG] Upload form submitted");

    const pdf = document.getElementById('pdf')?.files?.[0];
    const use_ocr = document.getElementById('use_ocr')?.checked;

    if (!pdf) {
      alert('Choose a PDF first.');
      return;
    }

    statusDiv.textContent = 'Analyzing...';
    const fd = new FormData();
    fd.append('pdf', pdf);
    fd.append('use_ocr', use_ocr ? 'true' : 'false');

    console.log("[DEBUG] Sending /api/analyze request", { use_ocr });
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });

    if (!res.ok) {
      console.error("[ERROR] /api/analyze HTTP error", res.status);
      statusDiv.textContent = 'Error analyzing the PDF.';
      return;
    }

    const data = await res.json();
    console.log("[DEBUG] /api/analyze response", data);

    currentResults = data.results || [];
    if (!Array.isArray(currentResults)) {
      console.warn("[WARN] results is not an array. Resetting to []");
      currentResults = [];
    }

    renderForm(currentResults);
    statusDiv.textContent = '✅ Analyzed! Review and adjust if needed, then export.';
  } catch (err) {
    console.error("[ERROR] Exception in onUploadSubmit:", err);
    statusDiv.textContent = 'Unexpected error while analyzing.';
  }
}


// REPLACE inline export click-handler with this WHOLE FUNCTION
async function onExportClick() {
  try {
    console.log("[DEBUG] Export button clicked");

    const rows = Array.from(document.querySelectorAll('.row[data-id]'));
    const payload = {
      results: rows.map(r => ({
        field_id: r.dataset.id,
        field_name: r.querySelector('.name')?.textContent?.trim() || "",
        status: r.querySelector('.status')?.value || "",
        page: (() => {
          const v = r.querySelector('.page')?.value;
          return v ? parseInt(v, 10) : null;
        })(),
        section: r.querySelector('.section')?.value || null,
        information_found: r.querySelector('.information-found')?.value || null
      }))
    };

    console.log("[DEBUG] Sending /api/export payload", payload);
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    console.log("[DEBUG] /api/export response", data);

    if (data.ok && data.download) {
      window.location.href = data.download;
    } else {
      alert("Export failed.");
    }
  } catch (err) {
    console.error("[ERROR] Exception in onExportClick:", err);
    alert("Unexpected error during export.");
  }
}


// REPLACE the one-liner init with this:
document.addEventListener('DOMContentLoaded', init);

