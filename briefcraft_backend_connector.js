(function BriefCraftBackendConnector() {
  if (window.__BRIEFCRAFT_BACKEND_CONNECTOR__) return;
  window.__BRIEFCRAFT_BACKEND_CONNECTOR__ = true;

  var API_BASE = String(
    window.API_BASE ||
      window.BRIEFCRAFT_API_BASE ||
      window.SC_API ||
      "https://briefcraft-ai.onrender.com"
  ).replace(/\/+$/, "");

  var state = {
    agents: [],
    account: null,
    packages: [],
    balance: 0,
    unit: "tokens",
  };

  function token() {
    try {
      return (
        (window.bcReadToken && window.bcReadToken()) ||
        localStorage.getItem("briefcraft_access_token") ||
        localStorage.getItem("access_token") ||
        localStorage.getItem("token") ||
        localStorage.getItem("briefcraft_token") ||
        sessionStorage.getItem("briefcraft_access_token") ||
        sessionStorage.getItem("access_token") ||
        sessionStorage.getItem("token") ||
        sessionStorage.getItem("briefcraft_token") ||
        ""
      ).replace(/^Bearer\s+/i, "");
    } catch (_) {
      return "";
    }
  }

  function userId() {
    try {
      var raw = localStorage.getItem("current_user") || localStorage.getItem("user");
      if (raw) {
        var u = JSON.parse(raw);
        return u.id || u.user_id || u.email || "";
      }
    } catch (_) {}
    return "";
  }

  function headers(json) {
    var h = { Accept: "application/json" };
    if (json !== false) h["Content-Type"] = "application/json";
    var t = token();
    if (t) h.Authorization = "Bearer " + t;
    var uid = userId();
    if (uid) h["X-User-Id"] = uid;
    return h;
  }

  async function api(path, options) {
    var res = await fetch(API_BASE + path, Object.assign({ headers: headers(true) }, options || {}));
    var text = await res.text();
    var data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { raw: text };
    }
    if (!res.ok || data.ok === false) {
      var msg = (data.detail && (data.detail.message || data.detail)) || data.message || data.error || text || "Request failed";
      var err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function money(p) {
    if (p == null) return "Custom";
    try {
      return "INR " + Number(p).toLocaleString("en-IN");
    } catch (_) {
      return "INR " + p;
    }
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toast(msg, type) {
    if (typeof window.showToast === "function") window.showToast(msg, type || "");
    else console.log(msg);
  }

  function ensureStyles() {
    if (document.getElementById("bc-account-agent-style")) return;
    var style = document.createElement("style");
    style.id = "bc-account-agent-style";
    style.textContent =
      ".bc-credit-chip{display:inline-flex;align-items:center;gap:8px;height:34px;padding:0 12px;border-radius:999px;border:1px solid var(--gold-mid,#c9a84c55);background:var(--gold-dim,#c9a84c1a);color:var(--gold2,#e8c97a);font-size:11px;font-weight:800;letter-spacing:.04em;white-space:nowrap;cursor:pointer}" +
      ".bc-credit-chip b{color:var(--text,#fff);font-variant-numeric:tabular-nums}" +
      ".bc-agent-panel{background:var(--surface,#0d0f18);border:1px solid var(--border2,#262840);border-radius:10px;padding:14px;margin-bottom:14px}" +
      ".bc-agent-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}" +
      ".bc-agent-card{background:var(--surface2,#12141f);border:1px solid var(--border2,#262840);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}" +
      ".bc-agent-card h4{margin:0;font-size:12px;color:var(--text,#fff);font-family:Syne,system-ui,sans-serif}" +
      ".bc-agent-card p{margin:0;color:var(--text2,#c9c6d2);font-size:10.5px;line-height:1.5}" +
      ".bc-agent-meta{display:flex;align-items:center;justify-content:space-between;gap:8px;color:var(--text3,#8e8ba0);font-size:10px}" +
      ".bc-agent-btn,.bc-plan-btn{border:1px solid var(--gold-mid,#c9a84c55);background:var(--gold-dim,#c9a84c1a);color:var(--gold2,#e8c97a);border-radius:8px;padding:8px 10px;font-size:11px;font-weight:800;cursor:pointer}" +
      ".bc-agent-btn:hover,.bc-plan-btn:hover{border-color:var(--gold,#c9a84c)}" +
      ".bc-plan-modal{position:fixed;inset:0;background:rgba(0,0,0,.74);z-index:99999;display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(8px)}" +
      ".bc-plan-box{width:min(1120px,100%);max-height:92vh;overflow:auto;background:var(--surface,#0d0f18);border:1px solid var(--border2,#262840);border-radius:16px;padding:20px;color:var(--text,#fff);box-shadow:0 30px 90px rgba(0,0,0,.55)}" +
      ".bc-plan-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:16px}.bc-plan-head h2{margin:0;font-family:Syne,system-ui,sans-serif;font-size:24px}.bc-plan-close{width:34px;height:34px;border-radius:8px;border:1px solid var(--border2,#262840);background:transparent;color:var(--text,#fff);cursor:pointer}" +
      ".bc-plan-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.bc-plan-tabs button{border:1px solid var(--border2,#262840);background:var(--surface2,#12141f);color:var(--text2,#c9c6d2);border-radius:999px;padding:8px 12px;cursor:pointer}.bc-plan-tabs button.active{border-color:var(--gold,#c9a84c);color:var(--gold2,#e8c97a);background:var(--gold-dim,#c9a84c1a)}" +
      ".bc-plan-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.bc-plan-card{position:relative;border:1px solid var(--border2,#262840);background:var(--surface2,#12141f);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px}.bc-plan-card.recommended{border-color:var(--gold,#c9a84c);box-shadow:0 0 0 2px var(--gold-dim,#c9a84c1a)}.bc-plan-card h3{margin:0;font-family:Syne,system-ui,sans-serif;font-size:15px}.bc-plan-price{font-size:24px;font-weight:900;color:var(--gold2,#e8c97a)}.bc-plan-card ul{padding-left:18px;margin:0;color:var(--text2,#c9c6d2);font-size:11px;line-height:1.6}.bc-plan-badge{position:absolute;right:12px;top:12px;border-radius:999px;background:var(--gold,#c9a84c);color:#07080d;font-size:9px;font-weight:900;padding:4px 8px;text-transform:uppercase}";
    document.head.appendChild(style);
  }

  function ensureCreditChip() {
    ensureStyles();
    var chip = document.getElementById("bcCreditChip");
    if (!chip) {
      chip = document.createElement("button");
      chip.id = "bcCreditChip";
      chip.className = "bc-credit-chip";
      chip.type = "button";
      chip.onclick = openPlans;
      var actions = document.querySelector(".tb-actions") || document.querySelector(".topbar") || document.body;
      actions.insertBefore(chip, actions.firstChild);
    }
    chip.innerHTML = 'Credits <b>' + esc(state.balance || 0) + '</b> ' + esc(state.unit || "tokens");
  }

  function renderAgentsPanel() {
    ensureStyles();
    var grid = document.getElementById("connGrid");
    if (!grid) return;
    var panel = document.getElementById("bcAgentsPanel");
    if (!panel) {
      panel = document.createElement("div");
      panel.id = "bcAgentsPanel";
      panel.className = "bc-agent-panel";
      grid.parentNode.insertBefore(panel, grid);
    }
    panel.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px">' +
      '<div><div style="font-family:Syne,system-ui,sans-serif;font-size:14px;font-weight:800">Backend Agents</div>' +
      '<div style="font-size:11px;color:var(--text2,#c9c6d2);margin-top:3px">Auto-discovered from /agents. Add a backend agent and it appears here automatically.</div></div>' +
      '<button class="bc-agent-btn" onclick="BriefCraftAccount.refresh()">Refresh</button></div>' +
      '<div class="bc-agent-grid">' +
      state.agents
        .map(function (a) {
          return (
            '<div class="bc-agent-card"><h4>' +
            esc(a.name || a.id) +
            "</h4><p>" +
            esc(a.description || "") +
            '</p><div class="bc-agent-meta"><span>' +
            esc(a.category || "agent") +
            "</span><span>" +
            esc(a.credit_cost || 0) +
            " tokens/run</span></div>" +
            '<button class="bc-agent-btn" onclick="BriefCraftAccount.runAgent(\'' +
            esc(a.id) +
            "')\">Connect / Test</button></div>"
          );
        })
        .join("") +
      "</div>";
  }

  function openPlans(audience) {
    ensureStyles();
    var old = document.getElementById("bcPlanModal");
    if (old) old.remove();
    var modal = document.createElement("div");
    modal.id = "bcPlanModal";
    modal.className = "bc-plan-modal";
    modal.innerHTML =
      '<div class="bc-plan-box"><div class="bc-plan-head"><div><h2>Credits and Packages</h2>' +
      '<div style="font-size:12px;color:var(--text2,#c9c6d2);margin-top:6px">Balance: <b style="color:var(--gold2,#e8c97a)">' +
      esc(state.balance || 0) +
      " " +
      esc(state.unit || "tokens") +
      "</b></div></div>" +
      '<button class="bc-plan-close" onclick="document.getElementById(\'bcPlanModal\').remove()">x</button></div>' +
      '<div class="bc-plan-tabs"><button data-audience="all">All</button><button data-audience="individual">Individuals</button><button data-audience="institution">Institutions</button></div>' +
      '<div class="bc-plan-grid" id="bcPlanGrid"></div></div>';
    document.body.appendChild(modal);
    modal.addEventListener("click", function (e) {
      if (e.target === modal) modal.remove();
    });
    modal.querySelectorAll("[data-audience]").forEach(function (btn) {
      btn.onclick = function () {
        renderPlans(btn.getAttribute("data-audience"));
      };
    });
    renderPlans(audience || "all");
  }

  function renderPlans(audience) {
    audience = audience || "all";
    document.querySelectorAll("#bcPlanModal [data-audience]").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-audience") === audience);
    });
    var grid = document.getElementById("bcPlanGrid");
    if (!grid) return;
    var plans = state.packages.filter(function (p) {
      return audience === "all" || p.audience === audience;
    });
    grid.innerHTML = plans
      .map(function (p) {
        return (
          '<div class="bc-plan-card ' +
          (p.recommended ? "recommended" : "") +
          '">' +
          (p.recommended ? '<span class="bc-plan-badge">Recommended</span>' : "") +
          "<h3>" +
          esc(p.name) +
          '</h3><div class="bc-plan-price">' +
          esc(money(p.price_inr)) +
          '</div><div style="font-size:12px;color:var(--text2,#c9c6d2)">' +
          esc((p.credits || 0).toLocaleString ? p.credits.toLocaleString("en-IN") : p.credits) +
          " tokens / " +
          esc(p.billing) +
          "</div><ul>" +
          (p.features || [])
            .map(function (f) {
              return "<li>" + esc(f) + "</li>";
            })
            .join("") +
          '</ul><button class="bc-plan-btn" onclick="BriefCraftAccount.checkout(\'' +
          esc(p.id) +
          "')\">" +
          (p.price_inr == null ? "Contact / Request" : "Upgrade and Pay") +
          "</button></div>"
        );
      })
      .join("");
  }

  async function refreshAccount() {
    try {
      var data = await api("/account/balance", { method: "GET", headers: headers(false) });
      state.account = data.account || null;
      state.balance = data.balance || (data.account && data.account.credit_balance) || 0;
      state.unit = data.unit || "tokens";
    } catch (e) {
      console.warn("[BriefCraftAccount] balance failed", e);
    }
    ensureCreditChip();
  }

  async function refreshPackages() {
    try {
      var data = await api("/account/packages", { method: "GET", headers: headers(false) });
      state.packages = data.packages || [];
    } catch (e) {
      console.warn("[BriefCraftAccount] packages failed", e);
    }
  }

  async function refreshAgents() {
    try {
      var data = await api("/agents", { method: "GET", headers: headers(false) });
      state.agents = data.agents || [];
    } catch (e) {
      console.warn("[BriefCraftAccount] agents failed", e);
    }
    renderAgentsPanel();
  }

  async function checkout(packageId) {
    try {
      var data = await api("/account/checkout", {
        method: "POST",
        body: JSON.stringify({
          package_id: packageId,
          success_url: location.href,
          cancel_url: location.href,
        }),
      });
      toast(data.message || "Checkout ready", "success");
      if (data.checkout_url && /^https?:\/\//i.test(data.checkout_url)) window.open(data.checkout_url, "_blank", "noopener");
      await refreshAccount();
      ensureCreditChip();
      openPlans();
    } catch (e) {
      toast(e.message || "Checkout failed", "error");
    }
  }

  async function runAgent(agentId, message, context) {
    try {
      var data = await api("/agents/run", {
        method: "POST",
        body: JSON.stringify({
          agent_id: agentId,
          message: message || "Connect and report status",
          project_id: (window.S && window.S.currentProjectId) || localStorage.getItem("currentProjectId") || null,
          session_id: (window.S && window.S.session_id) || null,
          context: context || {},
        }),
      });
      state.account = data.account || state.account;
      state.balance = (data.account && data.account.credit_balance) || state.balance;
      ensureCreditChip();
      var reply =
        (data.result && (data.result.message || data.result.detail)) ||
        (data.agent && data.agent.name ? data.agent.name + " connected." : "Agent connected.");
      if (typeof window.addChat === "function") window.addChat("ai", reply);
      else toast(reply, "success");
      return data;
    } catch (e) {
      if (e.status === 402) openPlans();
      toast(e.message || "Agent failed", "error");
      throw e;
    }
  }

  async function refreshAll() {
    await Promise.all([refreshAccount(), refreshPackages(), refreshAgents()]);
    ensureCreditChip();
    renderAgentsPanel();
  }

  window.BriefCraftAccount = {
    state: state,
    refresh: refreshAll,
    refreshAccount: refreshAccount,
    refreshAgents: refreshAgents,
    openPlans: openPlans,
    checkout: checkout,
    runAgent: runAgent,
    api: api,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", refreshAll);
  } else {
    setTimeout(refreshAll, 60);
  }
})();
