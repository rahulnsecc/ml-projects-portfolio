// ===== Utilities =====
function debounce(fn, delay = 300) { let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), delay); }; }
function qs(id){ return document.getElementById(id); }

// ===== State =====
let currentFilters = { status: 'all', anomaly: 'all', time: 'all', event: '', from: '', to: '' };
let certFilters = { 
  type: 'all', 
  status: 'all', 
  name: '', 
  expiration_date: '',
  owner_team: '',
  vendor_name: ''
};
let currentPage = 1;
const pageSize = 50;
let currentTab = 'alerts';
let allCertsData = []; // Store full certificate data for the modal

// ===== Persistence (localStorage) =====
function saveState(){
  const data = { 
    currentFilters, 
    certFilters,
    page: currentPage, 
    tab: currentTab 
  };
  localStorage.setItem("alertMonitorState", JSON.stringify(data));
}
function loadState(){
  const raw = localStorage.getItem("alertMonitorState");
  if(!raw) return;
  try {
    const saved = JSON.parse(raw);
    currentFilters.status = saved.currentFilters?.status || 'all';
    currentFilters.anomaly = saved.currentFilters?.anomaly || 'all';
    currentFilters.time = saved.currentFilters?.time || 'all';
    currentFilters.event = saved.currentFilters?.event || '';
    currentFilters.from = saved.currentFilters?.from || '';
    currentFilters.to = saved.currentFilters?.to || '';
    certFilters.type = saved.certFilters?.type || 'all';
    certFilters.status = saved.certFilters?.status || 'all';
    certFilters.name = saved.certFilters?.name || '';
    certFilters.expiration_date = saved.certFilters?.expiration_date || '';
    certFilters.owner_team = saved.certFilters?.owner_team || '';
    certFilters.vendor_name = saved.certFilters?.vendor_name || '';
    currentPage = saved.page || 1;
    currentTab = saved.tab || 'alerts';
  } catch(e){ console.warn("Failed to parse saved state", e); }
}

// ===== Fetchers =====
async function fetchSummary(){
  const params = new URLSearchParams({
    time: currentFilters.time,
    event: currentFilters.event,
    from_filter: currentFilters.from,
    to_filter: currentFilters.to
  });
  const res = await fetch(`/summary?${params.toString()}`);
  return res.json();
}
async function fetchTable(){
  const params = new URLSearchParams({
    status: currentFilters.status,
    anomaly: currentFilters.anomaly,
    time: currentFilters.time,
    event: currentFilters.event,
    from_filter: currentFilters.from,
    to_filter: currentFilters.to,
    page: currentPage,
    page_size: pageSize
  });
  const res = await fetch(`/data?${params.toString()}`);
  return res.json();
}
async function fetchTodayGlance(){
    const res = await fetch("/today_at_a_glance");
    return res.json();
}
async function fetchEventHistory(event_name){
    const params = new URLSearchParams({
        page: currentPage,
        page_size: pageSize,
        time: currentFilters.time,
        status: currentFilters.status,
    });
    const res = await fetch(`/event_history_details/${event_name}?${params.toString()}`);
    return res.json();
}
async function fetchCerts(){
    const params = new URLSearchParams({
        type: certFilters.type,
        status: certFilters.status,
        name: certFilters.name,
        expiration_date: certFilters.expiration_date,
        owner_team: certFilters.owner_team,
        vendor_name: certFilters.vendor_name
    });
    const res = await fetch(`/certs?${params.toString()}`);
    return res.json();
}

async function loadData(){
  saveState();
  qs("lastUpdated").textContent = "Updating...";
  if(currentTab === 'alerts'){
    const [summary, table, todayGlance] = await Promise.all([fetchSummary(), fetchTable(), fetchTodayGlance()]);
    renderSummary(summary);
    renderTable(table.rows);
    renderPagination(table.total_pages, table.page);
    renderTodayGlance(todayGlance);
    qs("lastUpdated").textContent = "Updated " + (summary.generated_at || new Date().toLocaleTimeString());
  } else if (currentTab === 'certs'){
    allCertsData = await fetchCerts();
    renderCertsTable(allCertsData);
    qs("lastUpdated").textContent = "Updated " + new Date().toLocaleTimeString();
  }
}

// ===== Renderers =====
function renderSummary(s){
  qs("statTotal").textContent = s.total ?? 0;
  qs("statOk").textContent = s.success ?? 0;
  qs("statFail").textContent = s.failed ?? 0;
  qs("statAnom").textContent = s.anomalies ?? 0;
}

function renderTable(rows){
  const tbody = document.querySelector("#alerts tbody");
  tbody.innerHTML = "";
  if(!rows || rows.length === 0){
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">No results</td>`;
    tbody.appendChild(tr);
    return;
  }
  const frag = document.createDocumentFragment();
  rows.forEach(r => {
    const statusOk = String(r.status).toUpperCase() === "SUCCESS";
    const isAnom = (r.anomaly === true) || (String(r.anomaly).toLowerCase() === "true");
    const tr = document.createElement("tr");
    const badge = statusOk ? '<span class="badge ok">SUCCESS</span>' : '<span class="badge fail">FAILED</span>';
    tr.innerHTML = `
      <td class="time">${r.received_time || ""}</td>
      <td class="event">${r.event_name || ""}</td>
      <td>${badge}</td>
      <td>${isAnom ? '<span class="badge warn">Anomaly</span>' : '<span class="badge ok">Normal</span>'}</td>
      <td class="next">${r.next_time || ""}</td>
    `;
    frag.appendChild(tr);
  });
  tbody.appendChild(frag);
}

function renderPagination(totalPages, page){
  const el = qs("pagination");
  el.innerHTML = "";
  if(!totalPages || totalPages <= 1){ return; }

  const btn = (label, disabled, handler) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.disabled = !!disabled;
    b.className = "page-btn";
    b.addEventListener("click", handler);
    return b;
  };

  // Prev
  el.appendChild(btn("« Prev", page <= 1, () => { currentPage = Math.max(1, page - 1); loadData(); } ));

  // Windowed page numbers around current
  const windowSize = 2;
  const start = Math.max(1, page - windowSize);
  const end = Math.min(totalPages, page + windowSize);
  if(start > 1){
    el.appendChild(btn("1", false, () => { currentPage = 1; loadData(); }));
    if(start > 2){
      const dots = document.createElement("span"); dots.textContent = "…"; dots.className = "dots"; el.appendChild(dots);
    }
  }
  for(let p = start; p <= end; p++){
    const b = btn(String(p), p === page, () => { currentPage = p; loadData(); });
    if(p === page) b.classList.add("active");
    el.appendChild(b);
  }
  if(end < totalPages){
    if(end < totalPages - 1){
      const dots = document.createElement("span"); dots.textContent = "…"; dots.className = "dots"; el.appendChild(dots);
    }
    el.appendChild(btn(String(totalPages), false, () => { currentPage = totalPages; loadData(); }));
  }

  // Next
  el.appendChild(btn("Next »", page >= totalPages, () => { currentPage = Math.min(totalPages, page + 1); loadData(); } ));
}

function renderTodayGlance(rows){
    const tbody = qs("todayGlance").querySelector("tbody");
    tbody.innerHTML = "";
    if(!rows || rows.length === 0){
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="4">No alerts found for today.</td>`;
        tbody.appendChild(tr);
        return;
    }
    const frag = document.createDocumentFragment();
    rows.forEach(r => {
        const isFail = String(r.last_run_status).toUpperCase() === "FAILED";
        const tr = document.createElement("tr");
        const statusBadge = isFail ? `<span class="badge fail">FAILED</span>` : `<span class="badge ok">SUCCESS</span>`;
        tr.innerHTML = `
            <td class="event-link" data-event-name="${r.event_name}">${r.event_name}</td>
            <td>${r.last_run || ""}</td>
            <td>${statusBadge}</td>
            <td>${r.next_run_forecast || ""}</td>
        `;
        frag.appendChild(tr);
    });
    tbody.appendChild(frag);
    // Add click listeners to event names
    tbody.querySelectorAll('.event-link').forEach(el => {
        el.addEventListener('click', (e) => {
            const eventName = e.target.getAttribute('data-event-name');
            showEventHistoryModal(eventName);
        });
    });
}
function renderEventHistory(rows){
    const tbody = qs("eventHistoryTable").querySelector("tbody");
    tbody.innerHTML = "";
    if(!rows || rows.length === 0){
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="7">No history found for this event.</td>`;
        tbody.appendChild(tr);
        return;
    }
    const frag = document.createDocumentFragment();
    rows.forEach(r => {
        const statusOk = String(r.status).toUpperCase() === "SUCCESS";
        const tr = document.createElement("tr");
        const badge = statusOk ? '<span class="badge ok">SUCCESS</span>' : '<span class="badge fail">FAILED</span>';
        tr.innerHTML = `
            <td>${r.received_time || ""}</td>
            <td>${badge}</td>
            <td>${r.from || "N/A"}</td>
            <td>${r.to || "N/A"}</td>
            <td>${r.subject || "N/A"}</td>
            <td>${r.source_path || "N/A"}</td>
            <td>${r.destination_path || "N/A"}</td>
        `;
        frag.appendChild(tr);
    });
    tbody.appendChild(frag);
}
function renderEventPagination(totalPages, page, eventName) {
    const el = qs("modalPagination");
    el.innerHTML = "";
    if (totalPages <= 1) return;

    const btn = (label, disabled, handler) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.disabled = !!disabled;
        b.className = "page-btn";
        b.addEventListener("click", handler);
        return b;
    };
    el.appendChild(btn("« Prev", page <= 1, () => { currentPage = Math.max(1, page - 1); showEventHistoryModal(eventName); }));
    el.appendChild(btn("Next »", page >= totalPages, () => { currentPage = Math.min(totalPages, page + 1); showEventHistoryModal(eventName); }));
}

function renderCertsTable(certs){
    const certsBody = qs("certsTable").querySelector("tbody");
    certsBody.innerHTML = "";
    if(!certs || certs.length === 0){
        certsBody.innerHTML = `<tr><td colspan="6">No certificates found.</td></tr>`;
        return;
    }
    
    certs.forEach((c, index) => {
        const statusBadge = getCertStatusBadge(c.status);
        const detailsRow = document.createElement("tr");
        detailsRow.classList.add('cert-row-link');
        detailsRow.setAttribute('data-cert-index', index);
        detailsRow.innerHTML = `
            <td>${c.name || "N/A"}</td>
            <td>${c.type || "N/A"}</td>
            <td>${c.platform || "N/A"}</td>
            <td>${c.expiration_date || "N/A"}</td>
            <td>${statusBadge}</td>
            <td>${c.next_action || "N/A"}</td>
        `;
        certsBody.appendChild(detailsRow);
    });
    // Add click listeners to rows after they are rendered
    certsBody.querySelectorAll('.cert-row-link').forEach(row => {
        row.addEventListener('click', (e) => {
            const index = e.currentTarget.getAttribute('data-cert-index');
            showCertDetailsModal(allCertsData[index]);
        });
    });
}

function getCertStatusBadge(status){
    switch(status.toLowerCase()){
        case 'expired':
            return '<span class="badge expired">EXPIRED</span>';
        case 'expiring soon':
            return '<span class="badge expiring">EXPIRING SOON</span>';
        case 'healthy':
            return '<span class="badge healthy">HEALTHY</span>';
        default:
            return '';
    }
}

// ===== Modals =====
async function showEventHistoryModal(eventName){
    qs("modalEventName").textContent = eventName;
    qs("alertModal").style.display = "flex";
    
    qs('modalTimeFilter').value = currentFilters.time;
    qs('modalStatusFilter').value = currentFilters.status;

    const historyData = await fetchEventHistory(eventName);
    renderEventHistory(historyData.rows);
    renderEventPagination(historyData.total_pages, historyData.page, eventName);
}

function closeAlertModal(){
    qs("alertModal").style.display = "none";
}

function showCertDetailsModal(cert) {
    const teamInfoBody = qs("certTeamInfoTable").querySelector("tbody");
    const processBody = qs("certProcessTable").querySelector("tbody");

    // Clear old data
    teamInfoBody.innerHTML = "";
    processBody.innerHTML = "";
    
    // Set modal title
    qs("certModalName").textContent = `Details for ${cert.name || 'N/A'}`;

    // Populate Team & Vendor Info table
    const teamInfoRow = document.createElement("tr");
    teamInfoRow.innerHTML = `
        <td>${cert.name || "N/A"}</td>
        <td>${cert.owner_team || "N/A"}</td>
        <td>${cert.subscriber || "N/A"}</td>
        <td>${cert.authorizer || "N/A"}</td>
        <td>${cert.spoc_name || "N/A"}</td>
        <td>${cert.spoc_email || "N/A"}</td>
        <td>${cert.spoc_phone || "N/A"}</td>
        <td>${cert.vendor_name || "N/A"}</td>
        <td>${cert.signed_by || "N/A"}</td>
    `;
    teamInfoBody.appendChild(teamInfoRow);

    // Populate Process & Tracking table
    const processRow = document.createElement("tr");
    processRow.innerHTML = `
        <td>${cert.name || "N/A"}</td>
        <td>${cert.sop_created || "N/A"}</td>
        <td>${cert.is_updated_in_ca_portal || "N/A"}</td>
        <td>${cert.remedy_queue || "N/A"}</td>
        <td>${cert.incident_crq_number || "N/A"}</td>
        <td>${cert.remarks || "N/A"}</td>
    `;
    processBody.appendChild(processRow);

    qs("certDetailsModal").style.display = 'flex';
}

function closeCertDetailsModal(){
    qs("certDetailsModal").style.display = 'none';
}


// ===== Event wiring =====
function wireFilters(){
  // Summary cards quick-filters
  document.querySelectorAll('.stat-item').forEach(item => {
    item.addEventListener('click', (ev) => {
      const status = ev.currentTarget.getAttribute('data-filter-status') || 'all';
      const anomaly = ev.currentTarget.getAttribute('data-filter-anomaly') || 'all';
      currentFilters = { status, anomaly, time: 'all', event: '', from: '', to: '' };
      currentPage = 1;
      // sync controls
      if (qs('alertStatusFilter')) qs('alertStatusFilter').value = status;
      if (qs('alertTimeFilter')) qs('alertTimeFilter').value = 'all';
      if (qs('alertEventFilter')) qs('alertEventFilter').value = '';
      if (qs('fromFilter')) qs('fromFilter').value = '';
      if (qs('toFilter')) qs('toFilter').value = '';
      loadData();
    });
  });

  // Alert table filters
  if (qs('alertStatusFilter')) qs('alertStatusFilter').addEventListener('change', e => { currentFilters.status = e.target.value; currentPage = 1; loadData(); });
  if (qs('alertTimeFilter')) qs('alertTimeFilter').addEventListener('change', e => { currentFilters.time = e.target.value; currentPage = 1; loadData(); });
  if (qs('alertEventFilter')) qs('alertEventFilter').addEventListener('input', debounce(e => { currentFilters.event = e.target.value; currentPage = 1; loadData(); }, 300));
  if (qs('fromFilter')) qs('fromFilter').addEventListener('input', debounce(e => { currentFilters.from = e.target.value; currentPage = 1; loadData(); }, 300));
  if (qs('toFilter')) qs('toFilter').addEventListener('input', debounce(e => { currentFilters.to = e.target.value; currentPage = 1; loadData(); }, 300));

  // Modal filters
  if (qs("modalTimeFilter")) qs("modalTimeFilter").addEventListener('change', e => {
      currentFilters.time = e.target.value;
      currentPage = 1;
      const eventName = qs("modalEventName").textContent;
      showEventHistoryModal(eventName);
  });
  if (qs("modalStatusFilter")) qs("modalStatusFilter").addEventListener('change', e => {
      currentFilters.status = e.target.value;
      currentPage = 1;
      const eventName = qs("modalEventName").textContent;
      showEventHistoryModal(eventName);
  });
  if (qs("closeAlertModalBtn")) qs("closeAlertModalBtn").addEventListener('click', closeAlertModal);
  
  // Tab wiring
  document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
          document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
          
          btn.classList.add('active');
          const tabName = btn.getAttribute('data-tab');
          qs(tabName + '-tab').classList.add('active');
          currentTab = tabName;
          loadData();
      });
  });

  // Certs table filters
  if (qs('certTypeFilter')) qs('certTypeFilter').addEventListener('change', e => { certFilters.type = e.target.value; loadData(); });
  if (qs('certStatusFilter')) qs('certStatusFilter').addEventListener('change', e => { certFilters.status = e.target.value; loadData(); });
  if (qs('certNameFilter')) qs('certNameFilter').addEventListener('input', debounce(e => { certFilters.name = e.target.value; loadData(); }, 300));
  if (qs('certExpirationFilter')) qs('certExpirationFilter').addEventListener('change', e => { certFilters.expiration_date = e.target.value; loadData(); });
  if (qs('certOwnerFilter')) qs('certOwnerFilter').addEventListener('input', debounce(e => { certFilters.owner_team = e.target.value; loadData(); }, 300));
  if (qs('certVendorFilter')) qs('certVendorFilter').addEventListener('input', debounce(e => { certFilters.vendor_name = e.target.value; loadData(); }, 300));

  if(qs("closeCertDetailsModalBtn")) qs("closeCertDetailsModalBtn").addEventListener('click', closeCertDetailsModal);
}

// ===== Init & Auto-Refresh =====
window.addEventListener("DOMContentLoaded", () => {
  loadState();
  // set controls to saved values
  if (qs('alertStatusFilter')) qs('alertStatusFilter').value = currentFilters.status;
  if (qs('alertTimeFilter')) qs('alertTimeFilter').value = currentFilters.time;
  if (qs('alertEventFilter')) qs('alertEventFilter').value = currentFilters.event;
  if (qs('fromFilter')) qs('fromFilter').value = currentFilters.from;
  if (qs('toFilter')) qs('toFilter').value = currentFilters.to;

  if (qs('certTypeFilter')) qs('certTypeFilter').value = certFilters.type;
  if (qs('certStatusFilter')) qs('certStatusFilter').value = certFilters.status;
  if (qs('certNameFilter')) qs('certNameFilter').value = certFilters.name;
  if (qs('certExpirationFilter')) qs('certExpirationFilter').value = certFilters.expiration_date;
  if (qs('certOwnerFilter')) qs('certOwnerFilter').value = certFilters.owner_team;
  if (qs('certVendorFilter')) qs('certVendorFilter').value = certFilters.vendor_name;
  
  // Activate the correct tab based on saved state
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  
  const activeTabBtn = document.querySelector(`.tab-btn[data-tab="${currentTab}"]`);
  if (activeTabBtn) {
    activeTabBtn.classList.add('active');
  }
  const activeTabContent = qs(currentTab + '-tab');
  if (activeTabContent) {
    activeTabContent.classList.add('active');
  }

  wireFilters();
  loadData();
  // light polling without re-render storms
  setInterval(loadData, 10000);
});
