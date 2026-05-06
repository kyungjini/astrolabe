let d3;
let vegaEmbed;
const EXPORT_SCALE = 6;
const COLORBAR_EXPORT_SCALE = 4;

async function loadD3() {
  if (window.d3) return window.d3;
  try {
    const mod = await import("https://cdn.jsdelivr.net/npm/d3@7/+esm");
    return mod;
  } catch (err) {
    console.error("Failed to load D3", err);
    return null;
  }
}

async function loadVegaEmbed() {
  if (window.vegaEmbed) return window.vegaEmbed;
  try {
    const mod = await import("https://cdn.jsdelivr.net/npm/vega-embed@6/+esm");
    return mod.default || mod;
  } catch (err) {
    console.error("Failed to load vega-embed", err);
    return null;
  }
}

const CONFIG = {
  thresholds: {
    node: {
      default: 0.05,
      min: 0,
      max: 0.1,
      step: 0.0005
    },
    link: {
      default: 0.05,
      min: 0,
      max: 0.1,
      step: 0.005
    }
  },
  
  edge: {
    correlation: {
      redundant: 0.9,
      nonLinearMax: 0.4
    },
    mutualInformation: {
      nonLinearMin: 0.75
    },
    interaction: {
      modelDriven: 0.97,
      redundancy: 0.2
    }
  },
  
  node: {
    normalization: {
      linearCoeffMax: 0.5
    }
  },
  
  visual: {
    colors: {
      highlight: "#ffcc00",
      positive: "#3b82f6",
      negative: "#ef4444",
      conflict: "#f97316",
      neutral: "#999",
      redundant: "#ef4444",
      nonLinear: "#9333ea",
      modelDriven: "#f59e0b",
      hover: "#888",
      default: "#fff",
      direction: {
        strongNegative: "#1e40af",
        weakNegative: "#60a5fa",
        neutral: "#9ca3af",
        weakPositive: "#f87171",
        strongPositive: "#dc2626"
      }
    },
    directionThresholds: {
      strongNegative: -0.7,
      weakNegative: -0.2,
      weakPositive: 0.2,
      strongPositive: 0.7
    },
    sizes: {
      node: {
        default: 8,
        focused: 15,
        neighbor: 10,
        hidden: 6,
        min: 6,
        max: 18
      },
      stroke: {
        default: 1.5,
        hover: 2,
        selected: 2.5,
        redundancy: 2
      },
      marker: {
        min: 10,
        max: 20,
        multiplier: 3
      },
      badge: {
        fontSize: 16
      }
    },
    opacities: {
      default: 1,
      selected: 1,
      neighbor: 0.6,
      hidden: 0.12,
      focusedHidden: 0.02,
      linkDefault: 0.5,
      linkHover: 0.9,
      linkSelected: 1,
      linkHidden: 0.1,
      linkFocusedHidden: 0.01,
      circle: 0.4
    },
    transitions: {
      fast: 300,
      normal: 500,
      slow: 750
    }
  },
  
  layout: {
    sidebar: {
      width: "30%"
    },
    content: {
      width: "70%"
    },
    focus: {
      radiusScale: {
        1: 120,
        2: 100,
        default: 80
      },
      hiddenOffset: 2000,
      zoomDelay: 100
    },
    simulation: {
      linkDistance: 60,
      linkStrength: 0.1,
      chargeStrength: -150,
      collisionPadding: 3,
      alphaTarget: 0.3
    },
    zoom: {
      min: 0.1,
      max: 8
    }
  },
  
  ui: {
    sidebar: {
      maxHeight: 300,
      gap: 8,
      padding: 16
    },
    tooltip: {
      offsetX: 15,
      offsetY: -28
    }
  }
};

// Helper: stable link key
const linkKey = (d) => {
  const s = typeof d.source === "object" ? d.source.id : d.source;
  const t = typeof d.target === "object" ? d.target.id : d.target;
  return `${s}__${t}`;
};

// Helper: extract node/link IDs (handles object/string/null formats safely)
const getNodeId = (node) => {
  if (node === null || node === undefined) return null;
  if (typeof node === "object") return node.id ?? null;
  return node;
};

// Helper: update selection (used by both node clicks and sidebar clicks)
function updateSelection(nodeId, currentSelection) {
  const i = currentSelection.indexOf(nodeId);
  const wasSelected = i !== -1;
  
  let newSelection;
  if (wasSelected) {
    // If already selected, remove it
    newSelection = currentSelection.filter((x) => x !== nodeId);
  } else if (currentSelection.length < 2) {
    // Add to selection
    newSelection = [...currentSelection, nodeId];
  } else {
    // Replace second selection
    newSelection = [currentSelection[1], nodeId];
  }
  
  return { newSelection, wasSelected };
}

// Helper: apply selection safely so UI interactions don't die on highlight errors
function applySelection(model, selection, options = {}) {
  const { emit = true, highlight = true } = options;
  let safeSelection = Array.isArray(selection) ? selection.slice(0, 2) : [];
  // In ablation view, the excluded feature must never be selectable for plotting.
  if (typeof state !== "undefined" && state?.currentAblationTarget) {
    safeSelection = safeSelection.filter((id) => id !== state.currentAblationTarget);
  }

  model.set("selected_node_ids", safeSelection);
  model.save_changes();
  if (emit) {
    model.send({ type: "selection", payload: safeSelection });
  }

  if (highlight && typeof updateNodeHighlights === "function") {
    try {
      updateNodeHighlights(safeSelection);
    } catch (e) {
      console.warn("Selection highlight failed, but selection state was updated:", e);
    }
  }
}

function analyzeEdgeRelationship(metrics, miTop20Threshold = 0) {
  const { 
    correlation,
    corr_pearson = null,
    mutual_information, 
    model_coefficient, 
    interaction_score,
    corr_percentile = null,
    corr_pearson_percentile = null,
    mi_percentile = null,
    interaction_percentile = null
  } = metrics;
  const cfg = CONFIG.edge;
  
  const pctOk = (v) => v != null && Number.isFinite(v);
  const usePercentiles =
    pctOk(corr_percentile) && pctOk(mi_percentile) && pctOk(interaction_percentile);
  
  let badgeSymbol, badgeColor;
  
  if (usePercentiles) {
    const pearsonR = Math.abs(corr_pearson ?? 0);
    const redundantByPearson = pearsonR >= 0.7;
    const redundantByPercentile =
      pctOk(corr_pearson_percentile) &&
      corr_pearson_percentile >= cfg.correlation.redundant;
    
    if (redundantByPearson || redundantByPercentile) {
      badgeSymbol = "!!";
      badgeColor = CONFIG.visual.colors.redundant;
    } else if (mi_percentile >= cfg.mutualInformation.nonLinearMin && corr_percentile < cfg.correlation.nonLinearMax) {
      badgeSymbol = "!?";
      badgeColor = CONFIG.visual.colors.nonLinear;
    } else if (interaction_percentile >= cfg.interaction.modelDriven) {
      badgeSymbol = "💡";
      badgeColor = CONFIG.visual.colors.modelDriven;
    } else {
      badgeSymbol = "";
      badgeColor = CONFIG.visual.colors.neutral;
    }
  } else {
    if (Math.abs(correlation) > 0.8) {
      badgeSymbol = "!!";
      badgeColor = CONFIG.visual.colors.redundant;
    } else if (miTop20Threshold > 0 && 
                mutual_information >= miTop20Threshold &&
                Math.abs(correlation) < 0.3) {
      badgeSymbol = "!?";
      badgeColor = CONFIG.visual.colors.nonLinear;
    } else if (interaction_score >= 0.2) {
      badgeSymbol = "💡";
      badgeColor = CONFIG.visual.colors.modelDriven;
    } else {
      badgeSymbol = "";
      badgeColor = CONFIG.visual.colors.neutral;
    }
  }
  
  let edgeColor;
  if (correlation > 0) {
    edgeColor = CONFIG.visual.colors.positive;
  } else if (correlation < 0) {
    edgeColor = CONFIG.visual.colors.negative;
  } else {
    edgeColor = CONFIG.visual.colors.neutral;
  }
  
  // C. Interaction Type (Edge Style)
  const strokeWidth = CONFIG.visual.sizes.stroke.redundancy;
  const strokeDasharray = null;
  
  // Synergy glow removed — always false to preserve return shape
  const hasGlow = false;

  return { badgeSymbol, badgeColor, edgeColor, strokeWidth, strokeDasharray, hasGlow };
}

// Analyze node importance and return visual config
function analyzeNodeImportance(metrics) {
  // Hidden-gem node logic removed. Keep API stable for callers.
  return {
    isHiddenGem: false,
    showLightbulb: false,
    tooltip: null
  };
}

// ===== MAIN RENDER FUNCTION =====

export async function render({ model, el }) {
  // Prevent stacked DOM / duplicate listeners when the notebook re-runs the cell or
  // the output is rehydrated: anywidget may reuse `el` without clearing children first.
  el.replaceChildren();
  el.classList.add("astrolabe-root");

  d3 = await loadD3();
  if (!d3) {
    el.textContent = "Error: D3 not available";
    return;
  }
  
  vegaEmbed = await loadVegaEmbed();
  if (!vegaEmbed) {
    console.warn("vega-embed not available, plots will not render");
  }

  // Load edge config from Python (if available)
  const edgeConfig = model.get("edge_config") || {};
  if (edgeConfig.redundant !== undefined) {
    CONFIG.edge.correlation.redundant = edgeConfig.redundant;
  }
  if (edgeConfig.model_driven !== undefined) {
    CONFIG.edge.interaction.modelDriven = edgeConfig.model_driven;
  }
  if (edgeConfig.nonlinear_mi !== undefined) {
    CONFIG.edge.mutualInformation.nonLinearMin = edgeConfig.nonlinear_mi;
  }
  if (edgeConfig.nonlinear_corr !== undefined) {
    CONFIG.edge.correlation.nonLinearMax = edgeConfig.nonlinear_corr;
  }

  // Load ablation study data from Python (if available)
  const ablationCandidates = model.get("ablation_candidates") || [];
  const ablationResults = model.get("ablation_results") || {};
  console.log("Ablation candidates:", ablationCandidates);
  console.log("Ablation results keys:", Object.keys(ablationResults));

  // Theme management - auto-detect IDE theme
  const detectIDETheme = () => {
    // Check VS Code theme via CSS variables
    const bgColor = getComputedStyle(document.body).getPropertyValue('--vscode-editor-background').trim();
    if (bgColor) {
      // Parse RGB to determine if dark
      const rgb = bgColor.match(/\d+/g);
      if (rgb && rgb.length >= 3) {
        const brightness = (parseInt(rgb[0]) + parseInt(rgb[1]) + parseInt(rgb[2])) / 3;
        return brightness < 128 ? 'dark' : 'light';
      }
    }
    // Fallback: check if common dark theme class exists
    if (document.body.classList.contains('vscode-dark') || 
        document.documentElement.getAttribute('data-vscode-theme-kind') === 'vscode-dark') {
      return 'dark';
    }
    return 'light';
  };
  
  let currentTheme = detectIDETheme(); // Auto-detect IDE theme
  let themeToggleBtn = null; // Will be set when button is created
  const plotOnly = !!model.get("plot_only");
  const applyTheme = (theme) => {
    currentTheme = theme;
    el.setAttribute("data-theme", theme);
    // Send theme to Python side for plot data updates
    model.send({ type: "theme", payload: theme });
    // Update theme button appearance
    if (themeToggleBtn) {
      themeToggleBtn.textContent = theme === "dark" ? "🌙" : "☀️";
      themeToggleBtn.title = theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
    }
  };
  applyTheme(currentTheme);

  // Layout root
  const wrapper = document.createElement("div");
  wrapper.className = "astrolabe-wrapper";
  if (plotOnly) {
    wrapper.classList.add("astrolabe-plot-only");
  }
  el.appendChild(wrapper);

  // Left sidebar - Important features and suggestions
  const leftSidebar = document.createElement("div");
  leftSidebar.className = "astrolabe-sidebar";
  wrapper.appendChild(leftSidebar);

  // Sidebar toggle button (fixed position at top of graph area)
  let sidebarCollapsed = false;
  const sidebarToggleBtn = document.createElement("button");
  sidebarToggleBtn.className = "astrolabe-sidebar-toggle-btn";
  sidebarToggleBtn.innerHTML = "◀"; // Left arrow when expanded
  sidebarToggleBtn.title = "Collapse Sidebar";
  sidebarToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    sidebarCollapsed = !sidebarCollapsed;
    leftSidebar.classList.toggle("collapsed", sidebarCollapsed);
    wrapper.classList.toggle("sidebar-collapsed", sidebarCollapsed);
    sidebarToggleBtn.innerHTML = sidebarCollapsed ? "▶" : "◀";
    sidebarToggleBtn.title = sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar";
    // Redraw graph and plot after sidebar animation completes
    setTimeout(() => {
      if (state.filteredNodes && state.filteredNodes.length > 0) {
        fullDraw();
      }
      // Force re-render the plot to adjust to new container size
      if (lastPlotData) {
        renderPlot(lastPlotData, true);
      }
    }, 320); // Wait for CSS transition (300ms) + small buffer
  });
  // Append to wrapper instead of sidebar so it stays visible when sidebar is collapsed
  wrapper.appendChild(sidebarToggleBtn);

  // Sidebar: Important Features Section
  const sidebarTitle = document.createElement("div");
  sidebarTitle.className = "astrolabe-sidebar-title";
  sidebarTitle.textContent = "Important Variables";
  leftSidebar.appendChild(sidebarTitle);

  const importantFeaturesList = document.createElement("div");
  importantFeaturesList.className = "astrolabe-features-list";
  leftSidebar.appendChild(importantFeaturesList);

  // Right content area
  const rightContent = document.createElement("div");
  rightContent.className = "astrolabe-content";
  wrapper.appendChild(rightContent);
  
  // Graph container (left side) - added first
  const container = document.createElement("div");
  container.className = "astrolabe-graph-container";
  rightContent.appendChild(container);
  
  // Store original node rankings (before ablation) for comparison
  let originalRankings = new Map(); // nodeId -> rank
  
  function updateOriginalRankings(nodes) {
    originalRankings.clear();
    const sortedNodes = [...nodes].sort((a, b) => (b.importance || 0) - (a.importance || 0));
    sortedNodes.forEach((node, index) => {
      originalRankings.set(node.id, index + 1);
    });
  }
  
  // Theme toggle button (top-right of network graph)
  themeToggleBtn = document.createElement("button");
  themeToggleBtn.className = "astrolabe-theme-toggle-btn astrolabe-sidebar-mini-btn";
  themeToggleBtn.textContent = currentTheme === "dark" ? "🌙" : "☀️";
  themeToggleBtn.title = currentTheme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
  themeToggleBtn.addEventListener("click", () => {
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(newTheme);
  });
  
  // Network floating controls (align with scatter panel style)
  const networkFloatingControls = document.createElement("div");
  networkFloatingControls.className = "astrolabe-network-floating-controls";

  // Reset button
  const resetBtn = document.createElement("button");
  resetBtn.className = "astrolabe-reset-btn astrolabe-chart-icon-btn";
  resetBtn.innerHTML = "🔄"; // Rotating arrows emoji
  resetBtn.title = "Reset selection and layout";
  resetBtn.addEventListener("click", () => {
    // Exit ablation view if active
    if (state.currentAblationTarget) {
      exitAblationView();
      state.shouldResetZoom = true;  // Reset zoom when exiting ablation
      return;
    }
    
    // Clear selection
    model.set("selected_node_ids", []);
    model.save_changes();
    model.send({ type: "selection", payload: [] });
    // Release all manually pinned nodes only when reset is explicitly pressed.
    state.shouldUnfixAll = true;
    // Request zoom reset
    state.shouldResetZoom = true;
    // Regenerate network (this will unfix all nodes)
    fullDraw();
  });
  networkFloatingControls.appendChild(resetBtn);

  // Save network graph PNG
  const networkSaveBtn = document.createElement("button");
  networkSaveBtn.className = "astrolabe-network-save-btn astrolabe-chart-icon-btn";
  networkSaveBtn.textContent = "💾";
  networkSaveBtn.title = "Save network graph as PNG";
  networkSaveBtn.type = "button";
  const sanitizeToken = (value) => {
    const text = String(value ?? "")
      .trim()
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_.-]/g, "-")
      .replace(/_+/g, "_")
      .replace(/-+/g, "-")
      .replace(/^[_-]+|[_-]+$/g, "");
    return text || "none";
  };
  const formatTimestampMMDDHHMM = () => {
    const d = new Date();
    const z = (n) => String(n).padStart(2, "0");
    return `${z(d.getMonth() + 1)}${z(d.getDate())}${z(d.getHours())}${z(d.getMinutes())}`;
  };
  const getSelectedVarTokens = (fallbackMain = null, fallbackSecond = null) => {
    const selected = (model.get("selected_node_ids") || []).slice(0, 2);
    const v1 = sanitizeToken(selected[0] ?? fallbackMain ?? "none");
    const v2Raw = selected[1] ?? fallbackSecond ?? null;
    const v2 = v2Raw ? sanitizeToken(v2Raw) : null;
    return [v1, v2];
  };
  const toDownload = (dataUrl, filename) => {
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  networkSaveBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    e.preventDefault();
    try {
      const svgNode =
        (state.svg && typeof state.svg.node === "function" ? state.svg.node() : null) ||
        container.querySelector("svg");
      if (!svgNode) return;

      const clone = svgNode.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
      const width = parseInt(svgNode.getAttribute("width") || `${container.clientWidth || 1200}`, 10);
      const height = parseInt(svgNode.getAttribute("height") || `${container.clientHeight || 900}`, 10);

      const svgText = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = url;
      });

      const scale = EXPORT_SCALE;
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        return;
      }
      // Paper-friendly export background
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);

      const dataUrl = canvas.toDataURL("image/png");
      const [v1, v2] = getSelectedVarTokens("none", null);
      const stamp = formatTimestampMMDDHHMM();
      const fname = `network_${v1}${v2 ? `_${v2}` : ""}_${stamp}.png`;
      toDownload(dataUrl, fname);
    } catch (err) {
      console.error("Failed to export network image:", err);
    }
  });
  
  networkFloatingControls.appendChild(networkSaveBtn);
  container.appendChild(networkFloatingControls);

  // Network graph help button
  const networkHelpBtn = document.createElement("button");
  networkHelpBtn.className = "astrolabe-network-help-btn astrolabe-sidebar-mini-btn";
  networkHelpBtn.innerHTML = "?";
  networkHelpBtn.title = "Network graph legend";
  
  // Help panel (initially hidden)
  const helpPanel = document.createElement("div");
  helpPanel.className = "astrolabe-help-panel";
  helpPanel.style.display = "none";
  helpPanel.innerHTML = `
    <div class="astrolabe-help-header">
      <strong>Network Graph Legend</strong>
      <button class="astrolabe-help-close">×</button>
    </div>
    <div class="astrolabe-help-content">
      <div class="astrolabe-help-item">
        <strong>Node Size:</strong> Feature importance (SHAP value)
      </div>
      <div class="astrolabe-help-item">
        <strong>Node Color:</strong> Direction (correlation with SHAP)
        <ul>
          <li><span style="color: #1e40af;">●</span> Strong negative (&lt; -0.7)</li>
          <li><span style="color: #60a5fa;">●</span> Weak negative (-0.7 to -0.2)</li>
          <li><span style="color: #9ca3af;">●</span> Neutral (-0.2 to 0.2)</li>
          <li><span style="color: #f87171;">●</span> Weak positive (0.2 to 0.7)</li>
          <li><span style="color: #dc2626;">●</span> Strong positive (&gt; 0.7)</li>
        </ul>
      </div>
      <div class="astrolabe-help-item">
        <strong>Edge Badges:</strong>
        <ul>
          <li><span style="color: #ef4444;">!!</span> Redundant (high correlation)</li>
          <li><span style="color: #9333ea;">!?</span> Non-linear relationship</li>
          <li><span style="color: #f59e0b;">💡</span> Model-driven interaction</li>
        </ul>
      </div>
      <div class="astrolabe-help-item">
        <strong>Edge Color:</strong>
        <ul>
          <li><span style="color: #3b82f6;">─</span> Positive correlation</li>
          <li><span style="color: #ef4444;">─</span> Negative correlation</li>
          <li><span style="color: #f97316;">─</span> Conflict (direction mismatch)</li>
        </ul>
      </div>
      <div class="astrolabe-help-item">
        <strong>Edge Thickness:</strong> Interaction score (thicker = stronger interaction)
      </div>
    </div>
  `;
  container.appendChild(helpPanel);

  // Tooltip inside graph container (not document.body) so it scrolls away with the widget
  // and does not break the flex row (wrapper) or leave stray hit-targets after re-run.
  const tooltip = d3
    .select(container)
    .append("div")
    .attr("class", "astrolabe-network-tooltip astrolabe-tooltip");
  
  // Help button toggle
  networkHelpBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isVisible = helpPanel.style.display !== "none";
    helpPanel.style.display = isVisible ? "none" : "block";
  });
  
  // Close button
  helpPanel.querySelector(".astrolabe-help-close").addEventListener("click", (e) => {
    e.stopPropagation();
    helpPanel.style.display = "none";
  });
  
  // Plot container for vega-embed charts (right side)
  const plotContainer = document.createElement("div");
  plotContainer.className = "astrolabe-plot-container";
  rightContent.appendChild(plotContainer);

  const plotFloatingControls = document.createElement("div");
  plotFloatingControls.className = "astrolabe-plot-floating-controls";

  // Plot mode toggle button (only shown when 2 nodes are selected)
  const plotModeToggleBtn = document.createElement("button");
  plotModeToggleBtn.type = "button";
  plotModeToggleBtn.className = "astrolabe-chart-icon-btn astrolabe-plot-mode-toggle-btn";
  plotModeToggleBtn.style.display = plotOnly ? "none" : "none"; // Hidden by default; always hidden in plot-only mode
  plotModeToggleBtn.title = "Toggle plot mode";
  plotFloatingControls.appendChild(plotModeToggleBtn);

  // LOWESS smooth toggle button
  const smoothToggleBtn = document.createElement("button");
  smoothToggleBtn.type = "button";
  smoothToggleBtn.className = "astrolabe-chart-icon-btn astrolabe-smooth-toggle-btn";
  smoothToggleBtn.style.display = "none";
  smoothToggleBtn.title = "Toggle LOWESS trendline";
  plotFloatingControls.appendChild(smoothToggleBtn);

  // Export button for high-resolution PNG
  const exportBtn = document.createElement("button");
  exportBtn.className = "astrolabe-chart-icon-btn astrolabe-export-btn";
  exportBtn.style.display = "none";
  exportBtn.title = "Download high-resolution PNG";
  exportBtn.textContent = "💾";
  exportBtn.type = "button";
  exportBtn.setAttribute("aria-label", "Save plot as PNG");
  plotFloatingControls.appendChild(exportBtn);

  // Manual axis range toggle + hover popover
  const rangeToggleBtn = document.createElement("button");
  rangeToggleBtn.className = "astrolabe-chart-icon-btn astrolabe-range-toggle-btn";
  rangeToggleBtn.style.display = "none";
  rangeToggleBtn.title = "Toggle manual axis range";
  rangeToggleBtn.textContent = "🔒";
  rangeToggleBtn.type = "button";
  rangeToggleBtn.setAttribute("aria-label", "Toggle manual axis range");
  rangeToggleBtn.setAttribute("aria-pressed", "false");
  plotFloatingControls.appendChild(rangeToggleBtn);

  const rangePopover = document.createElement("div");
  rangePopover.className = "astrolabe-range-popover";
  rangePopover.style.display = "none";
  rangePopover.innerHTML = `
    <div class="astrolabe-range-row">
      <label>x min</label><input data-field="xMin" type="number" step="any" />
      <label>x max</label><input data-field="xMax" type="number" step="any" />
    </div>
    <div class="astrolabe-range-row">
      <label>y min</label><input data-field="yMin" type="number" step="any" />
      <label>y max</label><input data-field="yMax" type="number" step="any" />
    </div>
    <div class="astrolabe-range-actions">
      <button type="button" data-action="apply">Apply</button>
      <button type="button" data-action="reset">Reset</button>
    </div>
  `;
  plotContainer.appendChild(rangePopover);

  const vegaWrapper = document.createElement("div");
  vegaWrapper.className = "astrolabe-vega-wrapper";
  vegaWrapper.style.cssText = "width: 100%; height: 100%; position: relative;";
  plotContainer.appendChild(vegaWrapper);

  plotContainer.appendChild(plotFloatingControls);
  
  const updatePlotModeButton = () => {
    if (plotOnly) {
      plotModeToggleBtn.style.display = "none";
      return;
    }
    const mode = model.get("plot_mode") || "shap_dependence";
    if (mode === "shap_dependence") {
      plotModeToggleBtn.textContent = "ƒ";
      plotModeToggleBtn.setAttribute(
        "aria-label",
        "SHAP dependence view (f) — click to switch to feature scatter (y)"
      );
      plotModeToggleBtn.dataset.plotMode = "shap_dependence";
    } else {
      plotModeToggleBtn.textContent = "y";
      plotModeToggleBtn.setAttribute(
        "aria-label",
        "Feature scatter view (y) — click to switch to SHAP dependence (f)"
      );
      plotModeToggleBtn.dataset.plotMode = "feature_scatter";
    }
  };
  updatePlotModeButton();

  const updateSmoothToggleButton = () => {
    const smoothOn = !!model.get("plot_smooth");
    smoothToggleBtn.setAttribute("aria-pressed", smoothOn ? "true" : "false");
    smoothToggleBtn.textContent = "\u{1F4C8}"; // 📈 fixed icon; state via style/aria-pressed
    smoothToggleBtn.setAttribute(
      "aria-label",
      smoothOn ? "LOWESS trend line on — click to turn off" : "LOWESS trend line off — click to turn on"
    );
  };
  updateSmoothToggleButton();
  
  plotModeToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (plotOnly) {
      return;
    }
    const currentMode = model.get("plot_mode") || "shap_dependence";
    const newMode = currentMode === "shap_dependence" ? "feature_scatter" : "shap_dependence";
    model.set("plot_mode", newMode);
    model.save_changes();
    updatePlotModeButton();
  });

  smoothToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    e.preventDefault();
    const smoothOn = !!model.get("plot_smooth");
    model.set("plot_smooth", !smoothOn);
    model.save_changes();
    updateSmoothToggleButton();
  });

  const formatRangeToken = (values) => {
    const fmt = (n) => Number(n).toFixed(3).replace(/\.?0+$/, "");
    return `${fmt(values.xMin)}-${fmt(values.xMax)}-${fmt(values.yMin)}-${fmt(values.yMax)}`;
  };

  const formatColorbarToken = ({ vmin, lower, upper, vmax }) => {
    const fmt = (n) => Number(n).toFixed(3).replace(/\.?0+$/, "");
    return `${fmt(vmin)}-${fmt(lower)}-${fmt(upper)}-${fmt(vmax)}`;
  };

  const buildColorbarPngDataUrl = (
    { vmin, lower, upper, vmax },
    width = 56,
    height = 360,
    scale = COLORBAR_EXPORT_SCALE
  ) => {
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.scale(scale, scale);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const barX = 14;
    const barW = 20;
    const barY = 10;
    const barH = height - 20;
    const radius = 4;
    const toPx = (val) => {
      const t = (val - vmin) / (vmax - vmin);
      return barY + (1 - t) * barH;
    };
    const yUpper = Math.max(barY, Math.min(barY + barH, toPx(upper)));
    const yLower = Math.max(barY, Math.min(barY + barH, toPx(lower)));

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(barX + radius, barY);
    ctx.lineTo(barX + barW - radius, barY);
    ctx.quadraticCurveTo(barX + barW, barY, barX + barW, barY + radius);
    ctx.lineTo(barX + barW, barY + barH - radius);
    ctx.quadraticCurveTo(barX + barW, barY + barH, barX + barW - radius, barY + barH);
    ctx.lineTo(barX + radius, barY + barH);
    ctx.quadraticCurveTo(barX, barY + barH, barX, barY + barH - radius);
    ctx.lineTo(barX, barY + radius);
    ctx.quadraticCurveTo(barX, barY, barX + radius, barY);
    ctx.closePath();
    ctx.clip();

    ctx.fillStyle = "#fde725";
    ctx.fillRect(barX, barY, barW, Math.max(0, yUpper - barY));
    if (yLower > yUpper) {
      const grad = ctx.createLinearGradient(0, yUpper, 0, yLower);
      grad.addColorStop(0.0, "#fde725");
      grad.addColorStop(0.17, "#3fbf73");
      grad.addColorStop(0.34, "#1f9e89");
      grad.addColorStop(0.50, "#277f8e");
      grad.addColorStop(0.67, "#365c8d");
      grad.addColorStop(0.84, "#46327e");
      grad.addColorStop(1.0, "#440154");
      ctx.fillStyle = grad;
      ctx.fillRect(barX, yUpper, barW, yLower - yUpper);
    }
    ctx.fillStyle = "#440154";
    ctx.fillRect(barX, yLower, barW, Math.max(0, barY + barH - yLower));
    ctx.restore();
    ctx.strokeStyle = "rgba(120, 130, 145, 0.55)";
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barW, barH);
    return canvas.toDataURL("image/png");
  };

  let JSZipLib = null;
  async function getJSZip() {
    if (JSZipLib) return JSZipLib;
    try {
      const mod = await import("https://cdn.jsdelivr.net/npm/jszip@3/+esm");
      JSZipLib = mod.default || mod;
      return JSZipLib;
    } catch (err) {
      console.error("Failed to load JSZip:", err);
      return null;
    }
  }

  const dataUrlToBlob = async (dataUrl) => {
    const res = await fetch(dataUrl);
    if (!res.ok) throw new Error(`dataUrl fetch failed: ${res.status}`);
    return await res.blob();
  };

  exportBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    e.preventDefault();
    if (!currentView) return;
    try {
      const dataUrl = await currentView.toImageURL("png", EXPORT_SCALE);
      const [v1, v2] = getSelectedVarTokens(lastPlotData?.main_feature ?? "none", null);
      const rangeToken =
        manualRangeEnabled &&
        [manualRangeValues.xMin, manualRangeValues.xMax, manualRangeValues.yMin, manualRangeValues.yMax].every(
          Number.isFinite
        )
          ? `_${formatRangeToken(manualRangeValues)}`
          : "";
      const stamp = formatTimestampMMDDHHMM();
      const plotName = `plot_${v1}${v2 ? `_${v2}` : ""}${rangeToken}_${stamp}.png`;
      const zipName = `plot_bundle_${v1}${v2 ? `_${v2}` : ""}_${stamp}.zip`;
      const zipCtor = await getJSZip();
      if (!zipCtor) {
        // Graceful fallback if zip library fails to load.
        toDownload(dataUrl, plotName);
        return;
      }
      const zip = new zipCtor();
      zip.file(plotName, await dataUrlToBlob(dataUrl));
      if (
        currentColorbarRange &&
        [currentColorbarRange.vmin, currentColorbarRange.lower, currentColorbarRange.upper, currentColorbarRange.vmax].every(
          Number.isFinite
        )
      ) {
        const cbDataUrl = buildColorbarPngDataUrl(currentColorbarRange);
        if (cbDataUrl) {
          const cbToken = formatColorbarToken(currentColorbarRange);
          const cbName = `colorbar_${cbToken}_${stamp}.png`;
          zip.file(cbName, await dataUrlToBlob(cbDataUrl));
        }
      }
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const zipUrl = URL.createObjectURL(zipBlob);
      toDownload(zipUrl, zipName);
      setTimeout(() => URL.revokeObjectURL(zipUrl), 1000);
    } catch (err) {
      console.error("Failed to export plot image:", err);
    }
  });
  
  let currentView = null; // Store current vega view for cleanup
  let lastPlotSpec = null; // Track last rendered spec to avoid unnecessary re-renders
  let lastPlotData = null; // Track last plot data for re-rendering on resize
  let plotRenderTimeout = null; // Debounce plot rendering
  let currentColorbarRange = null; // { vmin, lower, upper, vmax } for colorbar export
  let manualRangeEnabled = false;
  let manualRangeValues = { xMin: null, xMax: null, yMin: null, yMax: null };

  const rangeInputs = {
    xMin: rangePopover.querySelector('input[data-field="xMin"]'),
    xMax: rangePopover.querySelector('input[data-field="xMax"]'),
    yMin: rangePopover.querySelector('input[data-field="yMin"]'),
    yMax: rangePopover.querySelector('input[data-field="yMax"]'),
  };
  const rangeApplyBtn = rangePopover.querySelector('button[data-action="apply"]');
  const rangeResetBtn = rangePopover.querySelector('button[data-action="reset"]');

  function syncRangeInputsFromState() {
    Object.entries(rangeInputs).forEach(([k, el]) => {
      const v = manualRangeValues[k];
      el.value = Number.isFinite(v) ? String(v) : "";
    });
  }

  function updateRangeToggleUI() {
    rangeToggleBtn.setAttribute("aria-pressed", manualRangeEnabled ? "true" : "false");
  }

  function parseRangeFromInputs() {
    const xMin = parseFloat(rangeInputs.xMin.value);
    const xMax = parseFloat(rangeInputs.xMax.value);
    const yMin = parseFloat(rangeInputs.yMin.value);
    const yMax = parseFloat(rangeInputs.yMax.value);
    if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) return null;
    if (!(xMin < xMax && yMin < yMax)) return null;
    return { xMin, xMax, yMin, yMax };
  }

  function sendManualRange(enabled, rangeObj = null) {
    const payload = {
      enabled: !!enabled,
      range: rangeObj ? [[rangeObj.xMin, rangeObj.xMax], [rangeObj.yMin, rangeObj.yMax]] : null,
    };
    model.send({ type: "manual_range", payload });
  }

  function maybeShowRangePopover(show) {
    if (!manualRangeEnabled) {
      rangePopover.style.display = "none";
      return;
    }
    rangePopover.style.display = show ? "block" : "none";
    if (show) syncRangeInputsFromState();
  }

  rangeToggleBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    manualRangeEnabled = !manualRangeEnabled;
    updateRangeToggleUI();
    if (!manualRangeEnabled) {
      rangePopover.style.display = "none";
      // Keep values for reuse, but disable fixed domain.
      sendManualRange(false, null);
      return;
    }
    syncRangeInputsFromState();
    sendManualRange(true, parseRangeFromInputs());
  });

  rangeToggleBtn.addEventListener("mouseenter", () => maybeShowRangePopover(true));
  rangeToggleBtn.addEventListener("mouseleave", () => {
    setTimeout(() => {
      if (!rangePopover.matches(":hover")) maybeShowRangePopover(false);
    }, 120);
  });
  rangePopover.addEventListener("mouseleave", () => maybeShowRangePopover(false));
  rangePopover.addEventListener("mouseenter", () => maybeShowRangePopover(true));
  rangePopover.addEventListener("mousedown", (e) => e.stopPropagation());
  rangePopover.addEventListener("click", (e) => e.stopPropagation());

  rangeApplyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const parsed = parseRangeFromInputs();
    if (!parsed) {
      console.warn("Invalid manual range: must satisfy x_min < x_max and y_min < y_max");
      return;
    }
    manualRangeValues = parsed;
    manualRangeEnabled = true;
    updateRangeToggleUI();
    sendManualRange(true, parsed);
  });

  rangeResetBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    manualRangeValues = { xMin: null, xMax: null, yMin: null, yMax: null };
    syncRangeInputsFromState();
    sendManualRange(manualRangeEnabled, null);
  });

  // Threshold controls section
  const thresholdSection = document.createElement("div");
  thresholdSection.className = "astrolabe-threshold-section";
  
  const thresholdTitle = document.createElement("div");
  thresholdTitle.className = "astrolabe-threshold-title";
  thresholdTitle.textContent = "Filter Thresholds";
  thresholdSection.appendChild(thresholdTitle);
  
  // Prevent clicks on threshold section from propagating (prevents accidental node selection)
  thresholdSection.addEventListener("mousedown", (e) => e.stopPropagation());
  thresholdSection.addEventListener("mouseup", (e) => e.stopPropagation());
  thresholdSection.addEventListener("click", (e) => e.stopPropagation());
  thresholdSection.addEventListener("contextmenu", (e) => e.stopPropagation());
  thresholdTitle.addEventListener("mousedown", (e) => e.stopPropagation());
  thresholdTitle.addEventListener("click", (e) => e.stopPropagation());

  // Helper function to create histogram overlay
  function createHistogramOverlay(values, currentValue, min, max, label) {
    const overlay = document.createElement("div");
    overlay.className = "astrolabe-histogram-overlay";
    
    // Histogram configuration
    const width = 200;
    const height = 120;
    const margin = { top: 10, right: 10, bottom: 20, left: 10 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const numBins = 20;
    
    // Create bins
    const binSize = (max - min) / numBins;
    const bins = Array(numBins).fill(0);
    
    values.forEach(v => {
      const binIndex = Math.min(Math.floor((v - min) / binSize), numBins - 1);
      if (binIndex >= 0) bins[binIndex]++;
    });
    
    const maxCount = Math.max(...bins, 1);
    
    // Create SVG
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.style.display = "block";
    
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("transform", `translate(${margin.left},${margin.top})`);
    svg.appendChild(g);
    
    // Draw bars
    const barWidth = innerWidth / numBins;
    bins.forEach((count, i) => {
      const barHeight = (count / maxCount) * innerHeight;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", i * barWidth);
      rect.setAttribute("y", innerHeight - barHeight);
      rect.setAttribute("width", Math.max(barWidth - 1, 0));
      rect.setAttribute("height", barHeight);
      rect.setAttribute("fill", "#888");
      rect.setAttribute("opacity", "0.6");
      rect.classList.add("histogram-bar");
      g.appendChild(rect);
    });
    
    // Draw threshold indicator line
    const thresholdX = ((currentValue - min) / (max - min)) * innerWidth;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", thresholdX);
    line.setAttribute("y1", 0);
    line.setAttribute("x2", thresholdX);
    line.setAttribute("y2", innerHeight);
    line.setAttribute("stroke", "#ef4444");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-dasharray", "3,3");
    line.classList.add("threshold-indicator");
    g.appendChild(line);
    
    // Add count label
    const visibleCount = values.filter(v => v >= currentValue).length;
    const label_text = document.createElement("div");
    label_text.className = "astrolabe-histogram-label";
    label_text.textContent = `${visibleCount} ${label} visible`;
    
    overlay.appendChild(svg);
    overlay.appendChild(label_text);
    
    // Function to update threshold line position
    overlay.updateThreshold = (newValue) => {
      const newX = ((newValue - min) / (max - min)) * innerWidth;
      line.setAttribute("x1", newX);
      line.setAttribute("x2", newX);
      
      const newCount = values.filter(v => v >= newValue).length;
      label_text.textContent = `${newCount} ${label} visible`;
    };
    
    return overlay;
  }

  // Function to calculate dynamic thresholds based on data
  function calculateDynamicThresholds(networkData) {
    const nodes = networkData.nodes || [];
    const links = networkData.links || [];
    
    const thresholds = {
      node: { min: 0, max: 0.1, default: 0.05 },
      link: { min: 0, max: 0.1, default: 0.05 }
    };
    
    // Node threshold: max = max feature importance
    if (nodes.length > 0) {
      const importances = nodes.map(n => n.importance || 0);
      const maxImportance = Math.max(...importances);
      thresholds.node.max = maxImportance > 0 ? maxImportance : 0.1;
      thresholds.node.default = thresholds.node.max * 0.05; // Start at 5% of max
    }
    
    // Link threshold: min = top 30% interaction threshold (for performance)
    // max = max interaction score
    if (links.length > 0) {
      const interactions = links.map(l => l.importance || 0).filter(v => v > 0);
      if (interactions.length > 0) {
        const sortedInteractions = [...interactions].sort((a, b) => b - a);
        const maxInteraction = sortedInteractions[0];
        
        // Calculate top 30% threshold
        const top30Index = Math.floor(sortedInteractions.length * 0.3);
        const top30Threshold = sortedInteractions[top30Index] || 0;
        
        thresholds.link.min = top30Threshold;
        thresholds.link.max = maxInteraction > 0 ? maxInteraction : 0.1;
        thresholds.link.default = top30Threshold; // Start at 30% threshold
      }
    }
    
    return thresholds;
  }

  // Get initial network data and calculate thresholds
  const initialNetworkData = model.get("network_data") || {};
  const dynamicThresholds = calculateDynamicThresholds(initialNetworkData);

  // Node threshold
  const nodeThresholdContainer = document.createElement("div");
  nodeThresholdContainer.className = "astrolabe-threshold-container";
  nodeThresholdContainer.style.position = "relative"; // For overlay positioning
  
  const nodeThresholdLabel = document.createElement("label");
  nodeThresholdLabel.className = "astrolabe-threshold-label";
  
  const nodeThresholdText = document.createElement("span");
  nodeThresholdText.className = "astrolabe-threshold-text";
  nodeThresholdText.textContent = "Feature importance";
  
  const nodeThresholdSlider = document.createElement("input");
  nodeThresholdSlider.type = "range";
  nodeThresholdSlider.className = "astrolabe-threshold-slider";
  nodeThresholdSlider.min = String(dynamicThresholds.node.min);
  nodeThresholdSlider.max = String(dynamicThresholds.node.max);
  nodeThresholdSlider.step = String(CONFIG.thresholds.node.step);
  const nodeThrDefault = dynamicThresholds.node.default;
  nodeThresholdSlider.value = model.get("node_threshold") || nodeThrDefault;
  
  // Create histogram overlay for node threshold
  let nodeHistogramOverlay = null;
  const nodeImportances = (initialNetworkData.nodes || []).map(n => n.importance || 0);
  
  nodeThresholdSlider.addEventListener("mouseenter", () => {
    if (nodeImportances.length > 0) {
      nodeHistogramOverlay = createHistogramOverlay(
        nodeImportances,
        parseFloat(nodeThresholdSlider.value),
        parseFloat(nodeThresholdSlider.min),
        parseFloat(nodeThresholdSlider.max),
        "nodes"
      );
      nodeThresholdContainer.appendChild(nodeHistogramOverlay);
    }
  });
  
  nodeThresholdSlider.addEventListener("mouseleave", () => {
    if (nodeHistogramOverlay) {
      nodeHistogramOverlay.remove();
      nodeHistogramOverlay = null;
    }
  });
  
  nodeThresholdSlider.addEventListener("input", (e) => {
    e.stopPropagation();
    const v = parseFloat(e.target.value);
    model.set("node_threshold", v);
    model.save_changes();
    
    // Update histogram overlay if visible
    if (nodeHistogramOverlay) {
      nodeHistogramOverlay.updateThreshold(v);
    }
  });
  
  // Prevent all mouse/touch events on slider from propagating to graph
  nodeThresholdSlider.addEventListener("mousedown", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("mouseup", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("mousemove", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("click", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("change", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("touchstart", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("touchmove", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("touchend", (e) => e.stopPropagation());
  nodeThresholdSlider.addEventListener("contextmenu", (e) => e.stopPropagation());
  
  // Also prevent events on the container and label from propagating to graph
  nodeThresholdContainer.addEventListener("mousedown", (e) => e.stopPropagation());
  nodeThresholdContainer.addEventListener("mouseup", (e) => e.stopPropagation());
  nodeThresholdContainer.addEventListener("click", (e) => e.stopPropagation());
  nodeThresholdContainer.addEventListener("contextmenu", (e) => e.stopPropagation());
  nodeThresholdLabel.addEventListener("mousedown", (e) => e.stopPropagation());
  nodeThresholdLabel.addEventListener("mouseup", (e) => e.stopPropagation());
  nodeThresholdLabel.addEventListener("click", (e) => e.stopPropagation());
  nodeThresholdLabel.addEventListener("contextmenu", (e) => e.stopPropagation());
  nodeThresholdText.addEventListener("mousedown", (e) => e.stopPropagation());
  nodeThresholdText.addEventListener("click", (e) => e.stopPropagation());
  
  nodeThresholdLabel.appendChild(nodeThresholdText);
  nodeThresholdLabel.appendChild(nodeThresholdSlider);
  nodeThresholdContainer.appendChild(nodeThresholdLabel);
  thresholdSection.appendChild(nodeThresholdContainer);

  // Link threshold
  const linkThresholdContainer = document.createElement("div");
  linkThresholdContainer.className = "astrolabe-threshold-container";
  linkThresholdContainer.style.position = "relative"; // For overlay positioning
  
  const linkThresholdLabel = document.createElement("label");
  linkThresholdLabel.className = "astrolabe-threshold-label";
  
  const linkThresholdText = document.createElement("span");
  linkThresholdText.className = "astrolabe-threshold-text";
  linkThresholdText.textContent = "Interaction score";
  
  const linkThresholdSlider = document.createElement("input");
  linkThresholdSlider.type = "range";
  linkThresholdSlider.className = "astrolabe-threshold-slider";
  linkThresholdSlider.min = String(dynamicThresholds.link.min);
  linkThresholdSlider.max = String(dynamicThresholds.link.max);
  linkThresholdSlider.step = String(CONFIG.thresholds.link.step);
  const linkThrDefault = dynamicThresholds.link.default;
  linkThresholdSlider.value = model.get("link_threshold") || linkThrDefault;
  
  // Create histogram overlay for link threshold
  let linkHistogramOverlay = null;
  const linkImportances = (initialNetworkData.links || []).map(l => l.importance || 0);
  
  linkThresholdSlider.addEventListener("mouseenter", () => {
    if (linkImportances.length > 0) {
      linkHistogramOverlay = createHistogramOverlay(
        linkImportances,
        parseFloat(linkThresholdSlider.value),
        parseFloat(linkThresholdSlider.min),
        parseFloat(linkThresholdSlider.max),
        "edges"
      );
      linkThresholdContainer.appendChild(linkHistogramOverlay);
    }
  });
  
  linkThresholdSlider.addEventListener("mouseleave", () => {
    if (linkHistogramOverlay) {
      linkHistogramOverlay.remove();
      linkHistogramOverlay = null;
    }
  });
  
  // Prevent all events from propagating to prevent accidental node selection
  linkThresholdSlider.addEventListener("input", (e) => {
    e.stopPropagation();
    const v = parseFloat(e.target.value);
    model.set("link_threshold", v);
    model.save_changes();
    
    // Update histogram overlay if visible
    if (linkHistogramOverlay) {
      linkHistogramOverlay.updateThreshold(v);
    }
  });
  
  // Prevent all mouse/touch events on slider from propagating to graph
  linkThresholdSlider.addEventListener("mousedown", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("mouseup", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("mousemove", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("click", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("change", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("touchstart", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("touchmove", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("touchend", (e) => e.stopPropagation());
  linkThresholdSlider.addEventListener("contextmenu", (e) => e.stopPropagation());
  
  // Also prevent events on the container and label from propagating to graph
  linkThresholdContainer.addEventListener("mousedown", (e) => e.stopPropagation());
  linkThresholdContainer.addEventListener("mouseup", (e) => e.stopPropagation());
  linkThresholdContainer.addEventListener("click", (e) => e.stopPropagation());
  linkThresholdContainer.addEventListener("contextmenu", (e) => e.stopPropagation());
  linkThresholdLabel.addEventListener("mousedown", (e) => e.stopPropagation());
  linkThresholdLabel.addEventListener("mouseup", (e) => e.stopPropagation());
  linkThresholdLabel.addEventListener("click", (e) => e.stopPropagation());
  linkThresholdLabel.addEventListener("contextmenu", (e) => e.stopPropagation());
  linkThresholdText.addEventListener("mousedown", (e) => e.stopPropagation());
  linkThresholdText.addEventListener("click", (e) => e.stopPropagation());
  
  linkThresholdLabel.appendChild(linkThresholdText);
  linkThresholdLabel.appendChild(linkThresholdSlider);
  linkThresholdContainer.appendChild(linkThresholdLabel);
  thresholdSection.appendChild(linkThresholdContainer);
  
  leftSidebar.appendChild(thresholdSection);

  // Sidebar utility controls (placed naturally under thresholds)
  const sidebarMiniControls = document.createElement("div");
  sidebarMiniControls.className = "astrolabe-sidebar-mini-controls";
  sidebarMiniControls.appendChild(networkHelpBtn);
  sidebarMiniControls.appendChild(themeToggleBtn);
  leftSidebar.appendChild(sidebarMiniControls);

  // Function to update important features sidebar
  function updateImportantFeatures(nodes) {
    importantFeaturesList.innerHTML = "";
    
    // Get current selection to highlight selected nodes
    const selectedIds = model.get("selected_node_ids") || [];
    
    // Sort by importance (SHAP) and show all features (scrollable)
    const sortedNodes = [...nodes]
      .sort((a, b) => (b.importance || 0) - (a.importance || 0));
    
    sortedNodes.forEach((node, index) => {
      const isSelected = selectedIds.includes(node.id);
      const featureItem = document.createElement("div");
      featureItem.className = "astrolabe-feature-item";
      if (isSelected) {
        featureItem.classList.add("selected");
      }
      // hidden-gem styling removed
      featureItem.setAttribute("data-node-id", node.id);
      
      const nameContainer = document.createElement("div");
      nameContainer.className = "astrolabe-feature-name-container";
      
      // Ranking number with change indicator (if in ablation mode)
      const currentRank = index + 1;
      const rankSpan = document.createElement("span");
      rankSpan.className = "astrolabe-feature-rank";
      
      let rankText = `#${currentRank}`;
      if (state.currentAblationTarget && originalRankings.has(node.id)) {
        const originalRank = originalRankings.get(node.id);
        const rankChange = originalRank - currentRank; // positive = moved up, negative = moved down
        
        if (rankChange > 0) {
          // Moved up in ranking (better)
          rankText += ` <span style="color: #10b981; font-size: 11px;">(↑${rankChange})</span>`;
        } else if (rankChange < 0) {
          // Moved down in ranking (worse)
          rankText += ` <span style="color: #ef4444; font-size: 11px;">(↓${Math.abs(rankChange)})</span>`;
        }
      }
      
      rankSpan.innerHTML = rankText;
      nameContainer.appendChild(rankSpan);
      
      const nameSpan = document.createElement("span");
      nameSpan.className = "astrolabe-feature-name";
      nameSpan.textContent = node.id;
      
      // hidden-gem lightbulb removed
      
      nameContainer.appendChild(nameSpan);
      
      // Removed importance value display - only show ranking
      featureItem.appendChild(nameContainer);
      
      featureItem.addEventListener("click", () => {
        if (node.excluded) return;
        // Sidebar click: select for plotting
        const currentSel = model.get("selected_node_ids") || [];
        const { newSelection: sel } = updateSelection(node.id, currentSel);

        applySelection(model, sel);
        // Zoom will be handled by updateNodeHighlights
        // Sidebar will be updated by updateNodeHighlights
      });
      
      featureItem.addEventListener("mouseenter", () => {
        // CSS handles hover styles
        // Highlight corresponding node in graph
        highlightNodeOnHover(node.id, true);
      });
      
      featureItem.addEventListener("mouseleave", () => {
        // CSS handles hover styles
        // Remove highlight from corresponding node in graph
        highlightNodeOnHover(node.id, false);
      });
      
      importantFeaturesList.appendChild(featureItem);
    });
  }

  // State
  const state = {
    svg: null,
    g: null,
    simulation: null,
    nodeSel: null,
    linkSel: null,
    zoom: null,
    adjList: {},
    allNodes: [],
    allLinks: [],
    filteredNodes: [],
    filteredLinks: [],
    // Ablation study state
    ablationCandidates: [],
    ablationResults: {},
    currentAblationTarget: null,  // Feature currently excluded in ablation view
    originalNetworkData: null,    // Store original data for restoration
    selectionBeforeAblation: null, // Preserve current plot selection across ablation toggle
    pinnedBeforeAblation: null,   // Preserve manually pinned node positions across ablation round-trip
    shouldUnfixAll: false,        // Explicit reset-only unpin behavior
    shouldResetZoom: true,        // Only reset zoom on initial load or explicit reset
  };
  
  // Store ablation data in state (after state is defined)
  state.ablationCandidates = ablationCandidates;
  state.ablationResults = ablationResults;
  state.originalNetworkData = initialNetworkData;  // Save for restoration

  // Filter nodes and links based on thresholds
  const getFiltered = (networkData) => {
    const nodeThr = model.get("node_threshold") ?? CONFIG.thresholds.node.default;
    const linkThr = model.get("link_threshold") ?? CONFIG.thresholds.link.default;
    // Return nodes as-is (don't copy) to preserve x, y, fx, fy properties
    const nodes = (networkData?.nodes || []).filter((n) => (n.importance || 0) >= nodeThr);
    const idSet = new Set(nodes.map((n) => n.id));
    const links = (networkData?.links || []).filter((l) => {
      const s = getNodeId(l.source);
      const t = getNodeId(l.target);
      if (!idSet.has(s) || !idSet.has(t)) return false;
      const imp = l.importance || 0;
      // Always keep high-collinearity pairs visible so redundant (!!) badges can appear;
      // the link slider is interaction-based and would otherwise hide them.
      const highPearson = Math.abs(l.corr_pearson ?? 0) >= 0.7;
      return imp >= linkThr || highPearson;
    });
    return { nodes, links };
  };

  // Build adjacency list for subgraph navigation
  function buildAdjacencyList(nodes, links) {
    state.adjList = {};
    nodes.forEach(n => state.adjList[n.id] = new Set());
    links.forEach(l => {
      const s = getNodeId(l.source);
      const t = getNodeId(l.target);
      if (!s || !t) return;
      if (!state.adjList[s] || !state.adjList[t]) return;
      state.adjList[s].add(t);
      state.adjList[t].add(s);
    });
  }

  // Helper function to highlight a node on hover (used by both node hover and sidebar hover)
  function highlightNodeOnHover(nodeId, isHovering) {
    if (!state.nodeSel) return;
    
    state.nodeSel.each(function(d) {
      if (d.id === nodeId) {
        const circ = d3.select(this).select("circle");
        if (isHovering) {
          if (circ.attr("data-selected") !== "true") {
            circ.attr("stroke", CONFIG.visual.colors.hover)
                .attr("stroke-width", CONFIG.visual.sizes.stroke.hover);
          }
          d3.select(this).raise();
        } else {
          if (circ.attr("data-selected") !== "true") {
            circ.attr("stroke", CONFIG.visual.colors.default)
                .attr("stroke-width", CONFIG.visual.sizes.stroke.default);
          }
        }
      }
    });
  }

  // Helper: build neighbor map from links
  function buildNeighborMap() {
    const neighborMap = new Map();
    state.linkSel.data().forEach((l) => {
      const sid = getNodeId(l.source);
      const tid = getNodeId(l.target);
      if (!sid || !tid) return;
      if (!neighborMap.has(sid)) neighborMap.set(sid, new Set());
      if (!neighborMap.has(tid)) neighborMap.set(tid, new Set());
      neighborMap.get(sid).add(tid);
      neighborMap.get(tid).add(sid);
    });
    return neighborMap;
  }

  // Helper: get visible nodes based on selection
  function getVisibleNodes(selectedIds, neighborMap) {
    const visible = new Set();
    if (selectedIds.length === 1) {
      const a = selectedIds[0];
      visible.add(a);
      (neighborMap.get(a) || new Set()).forEach((n) => visible.add(n));
    } else if (selectedIds.length === 2) {
      const [a, b] = selectedIds;
      visible.add(a);
      visible.add(b);
      const A = neighborMap.get(a) || new Set();
      const B = neighborMap.get(b) || new Set();
      A.forEach((n) => { if (B.has(n)) visible.add(n); });
    }
    return visible;
  }

  // Helper: zoom to selected node(s)
  function zoomToSelection(selectedIds) {
    if (!state.nodeSel || !state.svg || !state.zoom || selectedIds.length === 0) return;
    
    const nodes = state.nodeSel.data();
    const selectedNodes = nodes.filter(d => selectedIds.includes(d.id));
    
    if (selectedNodes.length === 0) return;
    
    // Calculate bounding box of selected nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    selectedNodes.forEach(d => {
      if (d.x !== undefined && d.y !== undefined) {
        minX = Math.min(minX, d.x);
        minY = Math.min(minY, d.y);
        maxX = Math.max(maxX, d.x);
        maxY = Math.max(maxY, d.y);
      }
    });
    
    if (minX === Infinity) return; // No valid positions
    
    const width = state.width || container.clientWidth || 800;
    const height = state.height || container.clientHeight || 600;
    
    // Add padding around the bounding box
    const padding = 100;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;
    
    const dx = maxX - minX;
    const dy = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Calculate scale to fit the bounding box
    const scale = Math.min(width / dx, height / dy, 1.5); // Max zoom 1.5x
    const clampedScale = Math.max(CONFIG.layout.zoom.min, Math.min(CONFIG.layout.zoom.max, scale));
    
    // Calculate translation to center the selection
    const translateX = width / 2 - centerX * clampedScale;
    const translateY = height / 2 - centerY * clampedScale;
    
    // Apply zoom transform
    state.svg.transition()
      .duration(CONFIG.visual.transitions.normal)
      .call(
        state.zoom.transform,
        d3.zoomIdentity.translate(translateX, translateY).scale(clampedScale)
      );
  }

  // Update node highlights based on selection
  function updateNodeHighlights(selectedIds) {
    if (!state.nodeSel || !state.linkSel) return;
    const HIGHLIGHT_COLOR = CONFIG.visual.colors.highlight;
    selectedIds = (selectedIds || []).slice(0, 2);

    const nodesSel = state.nodeSel;
    const linkLines = state.linkSel.selectAll("line");
    const linkBadges = state.linkSel.selectAll("text.link-badge");

    const present = new Set(nodesSel.data().map((d) => d.id));
    selectedIds = selectedIds.filter((id) => present.has(id));
    
    // Zoom to selection if nodes are selected
    if (selectedIds.length > 0) {
      // Wait a bit for simulation to settle, then zoom
      setTimeout(() => {
        zoomToSelection(selectedIds);
      }, 100);
    }

    // If selection becomes empty, reset highlights
    if (!selectedIds.length) {
      const transitionDuration = CONFIG.visual.transitions.fast;
      nodesSel
        .transition()
        .duration(transitionDuration)
        .style("opacity", CONFIG.visual.opacities.default);
      nodesSel
        .select("circle")
        .transition()
        .duration(transitionDuration)
        .attr("stroke", CONFIG.visual.colors.default)
        .attr("stroke-width", CONFIG.visual.sizes.stroke.default)
        .attr("data-selected", null);
      linkLines
        .transition()
        .duration(transitionDuration)
        .style("opacity", CONFIG.visual.opacities.linkDefault)
        .attr("stroke", function () {
          return this.getAttribute("data-original-stroke") || d3.select(this).attr("stroke");
        })
        .attr("stroke-width", function () {
          return +this.getAttribute("data-base-width") || parseFloat(d3.select(this).attr("stroke-width")) || 1;
        })
        .attr("marker-end", null)
        .attr("marker-start", null);
      linkBadges
        .transition()
        .duration(transitionDuration)
        .style("opacity", CONFIG.visual.opacities.linkDefault);
      return;
    }

    // Build neighbor map and get visible nodes
    const neighborMap = buildNeighborMap();
    const visible = getVisibleNodes(selectedIds, neighborMap);

    const transitionDuration = CONFIG.visual.transitions.fast;
    nodesSel
      .transition()
      .duration(transitionDuration)
      .style("opacity", (d) => {
        // Keep excluded node always visible in ablation view.
        if (d.excluded) return CONFIG.visual.opacities.default;
        return visible.has(d.id) ? CONFIG.visual.opacities.default : CONFIG.visual.opacities.hidden;
      });
    nodesSel
      .select("circle")
      .transition()
      .duration(transitionDuration)
      .attr("stroke", (d) => (selectedIds.includes(d.id) ? HIGHLIGHT_COLOR : CONFIG.visual.colors.default))
      .attr("stroke-width", (d) => (selectedIds.includes(d.id) ? CONFIG.visual.sizes.stroke.selected : CONFIG.visual.sizes.stroke.default))
      .attr("data-selected", (d) => (selectedIds.includes(d.id) ? "true" : null));

    const betweenSelected = (l) => {
      if (selectedIds.length !== 2) return false;
      const sid = getNodeId(l.source);
      const tid = getNodeId(l.target);
      if (!sid || !tid) return false;
      return (sid === selectedIds[0] && tid === selectedIds[1]) ||
             (sid === selectedIds[1] && tid === selectedIds[0]);
    };

    // Get arrow direction for bivariate view (selectedIds[1] -> selectedIds[0])
    const getArrowDirection = (l) => {
      if (selectedIds.length !== 2) return null;
      const sid = getNodeId(l.source);
      const tid = getNodeId(l.target);
      if (!sid || !tid) return null;
      
      // Arrow indicates: selectedIds[1] -> selectedIds[0] (second feature colors first feature)
      if (sid === selectedIds[0] && tid === selectedIds[1]) {
        return "reverse";  // target -> source direction
      }
      if (tid === selectedIds[0] && sid === selectedIds[1]) {
        return "forward";  // source -> target direction
      }
      return null;
    };

    const linkOpacity = (d) => {
      const sid = getNodeId(d.source);
      const tid = getNodeId(d.target);
      if (!sid || !tid) return CONFIG.visual.opacities.linkHidden;
      if (selectedIds.includes(sid) && selectedIds.includes(tid)) {
        return CONFIG.visual.opacities.linkSelected;
      }
      if (selectedIds.length === 1) {
        return selectedIds.includes(sid) || selectedIds.includes(tid)
          ? CONFIG.visual.opacities.neighbor
          : CONFIG.visual.opacities.linkHidden;
      }
      return visible.has(sid) && visible.has(tid)
        ? CONFIG.visual.opacities.neighbor
        : CONFIG.visual.opacities.linkHidden;
    };

    linkLines
      .transition()
      .duration(transitionDuration)
      .style("opacity", linkOpacity)
      .attr("stroke", function (d) {
        return betweenSelected(d) ? HIGHLIGHT_COLOR : (this.getAttribute("data-original-stroke") || d3.select(this).attr("stroke"));
      })
      .attr("stroke-width", function (d) {
        const base = +this.getAttribute("data-base-width") || parseFloat(d3.select(this).attr("stroke-width")) || 1;
        return betweenSelected(d) ? Math.max(2, base * 2) : base;
      })
      .attr("marker-end", function(d) {
        const arrowDir = getArrowDirection(d);
        return arrowDir === "forward" ? `url(#arrow-end-${model.cid})` : null;
      })
      .attr("marker-start", function(d) {
        const arrowDir = getArrowDirection(d);
        return arrowDir === "reverse" ? `url(#arrow-start-${model.cid})` : null;
      });

    // Dim badge glyphs with the same visibility logic as their parent links.
    linkBadges
      .transition()
      .duration(transitionDuration)
      .style("opacity", linkOpacity);
    
    // Update sidebar highlighting after graph highlights are updated
    if (state.allNodes.length > 0) {
      updateImportantFeatures(state.allNodes);
    }
  }

  // Make scales for visualization
  function makeScales(nodes, links, allLinksForScale = null) {
    const nodeMax = d3.max(nodes, d => d.importance) || 1;
    const linkScaleSource = (Array.isArray(allLinksForScale) && allLinksForScale.length > 0)
      ? allLinksForScale
      : links;
    const linkMax = d3.max(linkScaleSource, d => d.importance) || 1;
    const nodeSize = d3.scaleSqrt()
      .domain([0, nodeMax])
      .range([CONFIG.visual.sizes.node.min, CONFIG.visual.sizes.node.max])
      .clamp(true);
    const linkWidth = d3.scaleLinear()
      .domain([0, linkMax])
      .range([1, 6])
      .clamp(true);
    return { nodeSize, linkWidth };
  }

  // Full draw function
  function fullDraw() {
    // Re-read every draw: Jupyter/anywidget can sync ablation traits after the first render,
    // so a one-time capture would leave state.ablationCandidates empty forever (no "?" markers).
    state.ablationCandidates = model.get("ablation_candidates") || [];
    state.ablationResults = model.get("ablation_results") || {};

    // In ablation view, use state data instead of model data
    let networkData;
    if (state.currentAblationTarget) {
      networkData = {
        nodes: state.allNodes || [],
        links: state.allLinks || []
      };
    } else {
      networkData = model.get("network_data") || {};
    }
    
    const allNodes = networkData.nodes || [];
    const allLinks = networkData.links || [];
    
    if (!allNodes.length && !allLinks.length) {
      container.textContent = "Waiting for network data...";
      return;
    }

    // Canonical graph for exitAblationView (must not point at synthetic ablation networkData)
    if (!state.currentAblationTarget) {
      state.originalNetworkData = model.get("network_data") || networkData;
    }
    
    // Store all nodes and links (for sidebar and adjacency list)
    // Only update if not in ablation view (to preserve ablation state)
    if (!state.currentAblationTarget) {
      state.allNodes = allNodes.map(d => ({...d}));
      state.allLinks = allLinks.map(d => ({...d}));
      
      // Also store original nodes for sidebar (never filtered, never modified)
      if (!state.originalAllNodes) {
        state.originalAllNodes = allNodes.map(d => ({...d}));
      }
    }
    
    // Filter nodes and links based on thresholds
    const { nodes, links } = getFiltered(networkData);
    
    // Store filtered nodes/links in state - DON'T copy, use references to preserve fx, fy
    state.filteredNodes = nodes;
    state.filteredLinks = links;
    
    if (!nodes.length && !links.length) {
      container.textContent = "No nodes/links above threshold. Adjust thresholds to see data.";
      return;
    }
    
    // Build adjacency list from filtered data
    buildAdjacencyList(nodes, links);
    
    // Calculate MI top 20% threshold for non-linearity detection
    const miValues = links.map(d => d.mutual_information || 0).filter(v => v > 0);
    let miTop20Threshold = 0;
    if (miValues.length > 0) {
      const sortedMI = [...miValues].sort((a, b) => b - a);
      const top20Index = Math.floor(sortedMI.length * 0.2);
      miTop20Threshold = sortedMI[top20Index] || 0;
    }
    
    // Store original rankings on first load (not in ablation mode)
    if (!state.currentAblationTarget && originalRankings.size === 0) {
      updateOriginalRankings(state.allNodes);
    }
    
    // Update sidebar (show ALL original nodes always, regardless of threshold or ablation)
    // In ablation view, we need to update node properties from current state but show all nodes
    let sidebarNodes;
    if (state.currentAblationTarget) {
      // In ablation view: merge original nodes with updated properties from state.allNodes
      sidebarNodes = state.originalAllNodes.map(origNode => {
        const updatedNode = state.allNodes.find(n => n.id === origNode.id);
        if (updatedNode) {
          // Use updated importance/direction from ablation, but keep original node in list
          return updatedNode;
        } else {
          // Node was excluded from ablation results, keep original
          return origNode;
        }
      }).filter(n => !n.excluded); // Exclude the ablated node
    } else {
      // Normal view: show all original nodes
      sidebarNodes = state.originalAllNodes || state.allNodes;
    }
    updateImportantFeatures(sidebarNodes);
    
    // Save current zoom transform before clearing
    let currentTransform = d3.zoomIdentity;
    const existingSvg = d3.select(container).select("svg");
    if (!existingSvg.empty() && state.zoom && !state.shouldResetZoom) {
      const currentZoomNode = existingSvg.node();
      if (currentZoomNode && currentZoomNode.__zoom) {
        currentTransform = currentZoomNode.__zoom;
      }
    }
    
    // Stop and clear if re-drawing
    if (state.simulation) state.simulation.stop();
    d3.select(container).select("svg").remove();
    
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    const { nodeSize, linkWidth } = makeScales(nodes, links, allLinks);
    
    // Store dimensions in state
    state.width = width;
    state.height = height;
    
    const svg = d3
      .select(container)
      .append("svg")
      .attr("viewBox", [0, 0, width, height])
      .attr("width", width)
      .attr("height", height)
      .attr("style", "max-width: 100%; height: auto; display: block;");
    
    const defs = svg.append("defs");
    state.g = svg.append("g");
    
    const zoom = d3.zoom()
      .scaleExtent([CONFIG.layout.zoom.min, CONFIG.layout.zoom.max])
      .on("zoom", (event) => {
        state.g.attr("transform", event.transform);
      });
    svg.call(zoom);
    state.zoom = zoom;
    
    // Links
    const linkG = state.g.append("g").attr("class", "links");

    // Create shared arrow markers once per draw (more robust than per-link defs).
    // markerUnits=strokeWidth ties marker geometry to line width.
    // base = 2*arrowWidth ~= 2x edge thickness, length ~= 2x edge thickness.
    const markerSize = 4;
    const arrowWidth = 1;
    const arrowLength = 2;
    const arrowColor = CONFIG.visual.colors.highlight;

    defs.append("marker")
      .attr("id", `arrow-end-${model.cid}`)
      .attr("viewBox", `0 -${arrowWidth} ${arrowLength} ${arrowWidth * 2}`)
      .attr("refX", arrowLength)
      .attr("refY", 0)
      .attr("markerWidth", markerSize)
      .attr("markerHeight", markerSize)
      .attr("orient", "auto")
      .attr("markerUnits", "strokeWidth")
      .append("path")
      .attr("d", `M0,-${arrowWidth}L${arrowLength},0L0,${arrowWidth}`)
      .attr("fill", arrowColor);

    defs.append("marker")
      .attr("id", `arrow-start-${model.cid}`)
      .attr("viewBox", `0 -${arrowWidth} ${arrowLength} ${arrowWidth * 2}`)
      .attr("refX", 0)
      .attr("refY", 0)
      .attr("markerWidth", markerSize)
      .attr("markerHeight", markerSize)
      .attr("orient", "auto")
      .attr("markerUnits", "strokeWidth")
      .append("path")
      .attr("d", `M${arrowLength},-${arrowWidth}L0,0L${arrowLength},${arrowWidth}`)
      .attr("fill", arrowColor);

    state.linkSel = linkG
      .selectAll("g.link-group")
      .data(links, linkKey)
      .join("g")
      .attr("class", "link-group");
    
    // Link line
    state.linkSel.append("line")
      .attr("class", "link")
      .each(function(d, i) {
        const edgeAnalysis = analyzeEdgeRelationship({
          correlation: d.corr_spearman || 0,
          corr_pearson: d.corr_pearson || 0,
          mutual_information: d.mutual_information || 0,
          model_coefficient: d.model_coefficient || 0,
          interaction_score: d.importance || 0,
          corr_percentile: d.corr_percentile,
          corr_pearson_percentile: d.corr_pearson_percentile,
          mi_percentile: d.mi_percentile,
          interaction_percentile: d.interaction_percentile
        }, miTop20Threshold);
        
        // Calculate edge thickness based on interaction score
        const interactionWidth = linkWidth(d.importance || 0);
        
        // No arrows by default (only show when 2 nodes are selected)
        // Use interaction_score for edge thickness
        d3.select(this)
          .attr("stroke", edgeAnalysis.edgeColor)
          .attr("stroke-width", interactionWidth)
          .attr("stroke-dasharray", edgeAnalysis.strokeDasharray)
          .attr("data-original-stroke", edgeAnalysis.edgeColor)
          .attr("data-base-width", interactionWidth)
          .attr("data-link-index", i)
          .attr("marker-end", null)
          .attr("marker-start", null)
          .style("stroke-opacity", CONFIG.visual.opacities.linkDefault)
          .style("cursor", "pointer");
        
        // Synergy glow removed
      })
      .on("click", function(event, d) {
        event.preventDefault();
        event.stopPropagation();
        const sid = getNodeId(d.source);
        const tid = getNodeId(d.target);
        if (!sid || !tid) return;
        let sel = model.get("selected_node_ids") || [];
        // If both nodes are already selected, swap order; otherwise select both
        if (sel.length === 2 && sel.includes(sid) && sel.includes(tid)) {
          sel = [sel[1], sel[0]]; // Swap order
        } else {
          sel = [sid, tid]; // Select both nodes
        }
        applySelection(model, sel);
        // Zoom will be handled by updateNodeHighlights
      })
      .on("mouseover", function(event, d) {
        d3.select(this).style("stroke-opacity", CONFIG.visual.opacities.linkHover);
        const edgeAnalysis = analyzeEdgeRelationship({
          correlation: d.corr_spearman || 0,
          corr_pearson: d.corr_pearson || 0,
          mutual_information: d.mutual_information || 0,
          model_coefficient: d.model_coefficient || 0,
          interaction_score: d.importance || 0,
          corr_percentile: d.corr_percentile,
          corr_pearson_percentile: d.corr_pearson_percentile,
          mi_percentile: d.mi_percentile,
          interaction_percentile: d.interaction_percentile
        }, miTop20Threshold);
        const s = getNodeId(d.source);
        const t = getNodeId(d.target);
        const badgeLabel = edgeAnalysis.badgeSymbol === "!!" ? "Redundant" : 
                          edgeAnalysis.badgeSymbol === "!?" ? "Non-Linear" : 
                          edgeAnalysis.badgeSymbol === "💡" ? "Model-Driven" : "";
        tooltip.transition()
          .duration(CONFIG.visual.transitions.fast)
          .style("opacity", 0.95);
        tooltip
          .html(`<strong>${s} ↔ ${t}</strong><br/>
                 Spearman r: ${(d.corr_spearman || 0).toFixed(3)}<br/>
                 Pearson r: ${(d.corr_pearson || 0).toFixed(3)}<br/>
                 MI: ${(d.mutual_information || 0).toFixed(3)}<br/>
                 Interaction: ${(d.importance || 0).toFixed(3)}<br/>
                 ${edgeAnalysis.badgeSymbol ? `<span style="color: ${edgeAnalysis.badgeColor}">${edgeAnalysis.badgeSymbol}</span> ${badgeLabel}` : ''}`)
          .style("left", `${event.pageX + CONFIG.ui.tooltip.offsetX}px`)
          .style("top", `${event.pageY + CONFIG.ui.tooltip.offsetY}px`);
      })
      .on("mouseout", function() {
        d3.select(this).style("stroke-opacity", CONFIG.visual.opacities.linkDefault);
        tooltip.transition()
          .duration(CONFIG.visual.transitions.fast)
          .style("opacity", 0);
      });
    
    // Link badge (center symbol) - only show if badge exists
    state.linkSel.append("text")
      .attr("class", "link-badge")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", `${CONFIG.visual.sizes.badge.fontSize}px`)
      // Latin "!!" / "!?" use a different font path than emoji (💡); set an explicit
      // stack so badges render in Linux/notebook SVG, not as invisible/blank glyphs.
      .style("font-family", "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans', 'Liberation Sans', sans-serif")
      .attr("pointer-events", "none")
      .style("user-select", "none")
      .each(function(d) {
        const edgeAnalysis = analyzeEdgeRelationship({
          correlation: d.corr_spearman || 0,
          corr_pearson: d.corr_pearson || 0,
          mutual_information: d.mutual_information || 0,
          model_coefficient: d.model_coefficient || 0,
          interaction_score: d.importance || 0,
          corr_percentile: d.corr_percentile,
          corr_pearson_percentile: d.corr_pearson_percentile,
          mi_percentile: d.mi_percentile,
          interaction_percentile: d.interaction_percentile
        }, miTop20Threshold);
        if (edgeAnalysis.badgeSymbol) {
          d3.select(this)
            .text(edgeAnalysis.badgeSymbol)
            .attr("fill", edgeAnalysis.badgeColor);
        } else {
          d3.select(this).text("");
        }
      });
    
    // Nodes
    const nodeG = state.g.append("g").attr("class", "nodes");
    state.nodeSel = nodeG
      .selectAll("g.node-group")
      .data(nodes, d => d.id)
      .join("g")
      .attr("class", "node-group")
      .call(nodeDrag)
      .on("contextmenu", function(event, d) {
        event.preventDefault();
        event.stopPropagation();
        
        // Hide tooltip immediately on right-click
        tooltip.transition()
          .duration(0)
          .style("opacity", 0);
        
        // Excluded node: allow right-click to exit ablation, but never treat as plot selection.
        if (d.excluded) {
          if (state.currentAblationTarget === d.id) {
            exitAblationView();
          }
          return;
        }

        // Check if this is an ablation candidate
        if (state.ablationCandidates && state.ablationCandidates.includes(d.id)) {
          // Toggle ablation view
          if (state.currentAblationTarget === d.id) {
            // Exit ablation view - restore original data
            exitAblationView();
          } else {
            // Enter ablation view for this feature
            enterAblationView(d.id);
          }
        } else {
          // Not an ablation candidate - normal right-click behavior (select for plotting)
          const currentSel = model.get("selected_node_ids") || [];
          const { newSelection: sel } = updateSelection(d.id, currentSel);

          applySelection(model, sel);
        }
      })
      .on("click", function(event, d) {
        event.stopPropagation();
        if (d.excluded) return;
        // Left-click: select for plotting
        const currentSel = model.get("selected_node_ids") || [];
        const { newSelection: sel } = updateSelection(d.id, currentSel);

        applySelection(model, sel);
        // Zoom will be handled by updateNodeHighlights
      });
    
    // Helper function to get discrete color for direction value
    function getDirectionColor(dir) {
      const thresh = CONFIG.visual.directionThresholds;
      const colors = CONFIG.visual.colors.direction;
      
      if (dir < thresh.strongNegative) {
        return colors.strongNegative; // Blue
      } else if (dir < thresh.weakNegative) {
        return colors.weakNegative; // Light blue
      } else if (dir <= thresh.weakPositive) {
        return colors.neutral; // Gray
      } else if (dir <= thresh.strongPositive) {
        return colors.weakPositive; // Light red
      } else {
        return colors.strongPositive; // Red
      }
    }

    // Node circle
    state.nodeSel.append("circle")
      .attr("class", "node-circle")
      .attr("r", d => nodeSize(d.importance))
      .attr("fill", d => {
        // Color based on direction (correlation with SHAP) - discrete 5 categories
        const dir = d.direction || 0;
        return getDirectionColor(dir);
      })
      .attr("stroke", CONFIG.visual.colors.default)
      .attr("stroke-width", CONFIG.visual.sizes.stroke.default)
      .on("mouseover", function(event, d) {
        // Skip hover effects for excluded nodes
        if (d.excluded) {
          tooltip.transition()
            .duration(CONFIG.visual.transitions.fast)
            .style("opacity", 0.95);
          tooltip
            .html(`<strong>${d.id}</strong><br/>
                   <span style="color: #6b7280;">⊘ EXCLUDED FROM ANALYSIS</span><br/>
                   <em>Right-click to restore</em>`)
            .style("left", `${event.pageX + CONFIG.ui.tooltip.offsetX}px`)
            .style("top", `${event.pageY + CONFIG.ui.tooltip.offsetY}px`);
          return;
        }
        
        highlightNodeOnHover(d.id, true);
        tooltip.transition()
          .duration(CONFIG.visual.transitions.fast)
          .style("opacity", 0.95);
        tooltip
          .html(`<strong>${d.id}</strong><br/>
                 Importance (SHAP): ${(d.importance || 0).toFixed(3)}<br/>
                 Linear Coef: ${(d.linear_coefficient || 0).toFixed(3)}<br/>
                 Direction: ${(d.direction || 0).toFixed(3)}`)
          .style("left", `${event.pageX + CONFIG.ui.tooltip.offsetX}px`)
          .style("top", `${event.pageY + CONFIG.ui.tooltip.offsetY}px`);
      })
      .on("mouseout", function(event, d) {
        // Don't highlight excluded nodes
        if (!d.excluded) {
          highlightNodeOnHover(d.id, false);
        }
        tooltip.transition()
          .duration(CONFIG.visual.transitions.fast)
          .style("opacity", 0);
      });
    
    // Node label with optional lightbulb
    // Light mode: solid black text, Dark mode: white text with dark stroke for contrast
    const labelFill = currentTheme === "dark" ? "#ffffff" : "#000000";
    const labelStroke = currentTheme === "dark" ? "#1e1e1e" : "none";
    const labelStrokeWidth = currentTheme === "dark" ? "0.7px" : "0";
    
    state.nodeSel.append("text")
      .attr("class", "node-label")
      .attr("x", d => nodeSize(d.importance) + 4)
      .attr("y", 3)
      .style("pointer-events", "none")
      .style("fill", labelFill)
      .style("paint-order", "stroke")
      .style("stroke", labelStroke)
      .style("stroke-width", labelStrokeWidth)
      .style("font-weight", "500") // Ensure text is readable
      .each(function(d) {
        d3.select(this).text(d.id);
      });
    
    // Add ablation indicator (?) for ablation candidate nodes
    // Place it at the center of the node
    if (state.ablationCandidates && state.ablationCandidates.length > 0) {
      state.nodeSel.each(function(d) {
        if (state.ablationCandidates.includes(d.id)) {
          d3.select(this).append("text")
            .attr("class", "ablation-indicator")
            .attr("x", 0)
            .attr("y", 5)  // Slightly below center for better vertical alignment
            .style("text-anchor", "middle")
            .style("pointer-events", "none")
            .style("font-size", "16px")
            .style("font-family", "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans', sans-serif")
            .style("font-weight", "bold")
            .style("fill", "#f59e0b")  // Amber color
            .style("paint-order", "stroke")
            .style("stroke", currentTheme === "dark" ? "#1e1e1e" : "#fff")
            .style("stroke-width", "2px")
            .text("?");
        }
      });
    }
    
    // Unfix only on explicit reset. Otherwise preserve user-pinned nodes across redraws.
    if (state.shouldUnfixAll) {
      nodes.forEach(d => {
        d.fx = null;
        d.fy = null;
      });
      state.shouldUnfixAll = false;
    }
    
    // Adaptive force parameters based on edge density
    const edgeDensity = links.length / (nodes.length * (nodes.length - 1) / 2);
    const avgDegree = links.length > 0 ? (2 * links.length) / nodes.length : 0;
    
    // Adjust forces based on connectivity
    // When there are few edges, increase link strength and reduce charge to keep nodes closer
    let linkStrength = CONFIG.layout.simulation.linkStrength;
    let chargeStrength = CONFIG.layout.simulation.chargeStrength;
    let linkDistance = CONFIG.layout.simulation.linkDistance;
    
    if (avgDegree < 2) {
      // Very sparse graph - strengthen links, reduce repulsion
      linkStrength = 0.3;
      chargeStrength = -80;
      linkDistance = 50;
    } else if (avgDegree < 5) {
      // Sparse graph - moderate adjustment
      linkStrength = 0.2;
      chargeStrength = -100;
      linkDistance = 55;
    }
    // else: use default values for dense graphs
    
    // CRITICAL: Preserve node positions from previous simulation before creating new one
    if (state.simulation && state.simulation.nodes()) {
      const oldNodes = state.simulation.nodes();
      const oldPosMap = new Map();
      oldNodes.forEach(n => {
        if (n.x !== undefined && n.y !== undefined) {
          oldPosMap.set(n.id, { x: n.x, y: n.y, vx: n.vx, vy: n.vy, fx: n.fx, fy: n.fy });
        }
      });
      
      // Apply old positions to new nodes array
      nodes.forEach(n => {
        const oldPos = oldPosMap.get(n.id);
        if (oldPos) {
          n.x = oldPos.x;
          n.y = oldPos.y;
          n.vx = oldPos.vx || 0;
          n.vy = oldPos.vy || 0;
          // Preserve user-pinned nodes unless reset explicitly requested.
          if (!state.shouldUnfixAll && Number.isFinite(oldPos.fx) && Number.isFinite(oldPos.fy)) {
            n.fx = oldPos.fx;
            n.fy = oldPos.fy;
          } else if (state.currentAblationTarget) {
            // In ablation view, freeze all nodes at their current position
            n.fx = n.x;
            n.fy = n.y;
          }
        }
      });
    }
    
    // Simulation
    const simCfg = CONFIG.layout.simulation;
    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance(linkDistance)
        .strength(linkStrength))
      .force("charge", d3.forceManyBody().strength(chargeStrength))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide()
        .radius(d => nodeSize(d.importance) + simCfg.collisionPadding))
      .force("x", d3.forceX(width / 2).strength(0.05))  // Gentle pull toward center X
      .force("y", d3.forceY(height / 2).strength(0.05)); // Gentle pull toward center Y
    
    state.simulation = simulation;
    state.svg = svg;
    
    // In ablation view, stop simulation immediately (nodes are already frozen)
    if (state.currentAblationTarget) {
      simulation.alpha(0).stop();
      
      // CRITICAL FIX: Manually update DOM positions since tick won't run
      state.nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
      
      // Update link positions manually
      state.linkSel.selectAll("line").each(function(d) {
        const sourceId = getNodeId(d.source);
        const targetId = getNodeId(d.target);
        const source = typeof d.source === "object" ? d.source : nodes.find(n => n.id === sourceId);
        const target = typeof d.target === "object" ? d.target : nodes.find(n => n.id === targetId);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            const sourceRadius = nodeSize(source.importance);
            const targetRadius = nodeSize(target.importance);
            const unitX = dx / dist;
            const unitY = dy / dist;
            const x1 = source.x + unitX * (sourceRadius + 2);
            const y1 = source.y + unitY * (sourceRadius + 2);
            const x2 = target.x - unitX * (targetRadius + 2);
            const y2 = target.y - unitY * (targetRadius + 2);
            
            d3.select(this)
              .attr("x1", x1)
              .attr("y1", y1)
              .attr("x2", x2)
              .attr("y2", y2);
            
            // Update badge position
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;
            d3.select(this.parentNode)
              .select("text.link-badge")
              .attr("x", midX)
              .attr("y", midY);
          }
        }
      });
    }
    
    // Apply zoom transform: reset if requested, otherwise restore previous transform
    if (state.zoom) {
      if (state.shouldResetZoom) {
        state.svg.transition()
          .duration(CONFIG.visual.transitions.fast)
          .call(state.zoom.transform, d3.zoomIdentity);
        state.shouldResetZoom = false;  // Don't reset again unless explicitly requested
      } else {
        // Restore previous zoom transform without animation
        state.svg.call(state.zoom.transform, currentTransform);
      }
    }
    
    // Sync current highlight state
    updateNodeHighlights(model.get("selected_node_ids") || []);
    
    // If in ablation view, re-apply excluded node styling
    if (state.currentAblationTarget) {
      styleExcludedNode(state.currentAblationTarget);
    }
    
    // Update positions on tick
    simulation.on("tick", () => {
      // Update link positions
      state.linkSel.selectAll("line").each(function(d) {
        const sourceId = getNodeId(d.source);
        const targetId = getNodeId(d.target);
        const source = typeof d.source === "object" ? d.source : nodes.find(n => n.id === sourceId);
        const target = typeof d.target === "object" ? d.target : nodes.find(n => n.id === targetId);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            const sourceRadius = nodeSize(source.importance);
            const targetRadius = nodeSize(target.importance);
            const unitX = dx / dist;
            const unitY = dy / dist;
            const x1 = source.x + unitX * (sourceRadius + 2);
            const y1 = source.y + unitY * (sourceRadius + 2);
            const x2 = target.x - unitX * (targetRadius + 2);
            const y2 = target.y - unitY * (targetRadius + 2);
            
            d3.select(this)
              .attr("x1", x1)
              .attr("y1", y1)
              .attr("x2", x2)
              .attr("y2", y2);
            
            // Update badge position (center of link)
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;
            d3.select(this.parentNode)
              .select("text.link-badge")
              .attr("x", midX)
              .attr("y", midY);
          }
        }
      });
      
      // Update node positions
      state.nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
    });
    
    function nodeDrag(sel) {
      sel.call(
        d3.drag()
          .on("start", (event, d) => {
            if (!event.active && state.simulation) {
              state.simulation.alphaTarget(CONFIG.layout.simulation.alphaTarget).restart();
            }
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active && state.simulation) state.simulation.alphaTarget(0);
          })
      );
    }
  }

  function safeFullDraw(reason = "unknown") {
    try {
      fullDraw();
    } catch (e) {
      console.error(`fullDraw failed (${reason}):`, e);
    }
  }
  
  // Ablation view functions (defined after fullDraw to avoid hoisting issues)
  function enterAblationView(featureName) {
    console.log(`Entering ablation view for: ${featureName}`);

    state.ablationResults = model.get("ablation_results") || {};
    state.ablationCandidates = model.get("ablation_candidates") || [];
    
    // Check if ablation results exist for this feature
    if (!state.ablationResults[featureName]) {
      console.warn(`No ablation results found for ${featureName}`);
      return;
    }
    
    // CRITICAL FIX: Get nodes from running simulation, NOT from state.allNodes
    const simulationNodes = state.simulation ? state.simulation.nodes() : state.allNodes;
    
    // Set current ablation target FIRST
    state.currentAblationTarget = featureName;
    
    // Get ablated network data (from Python)
    const ablatedData = state.ablationResults[featureName];
    
    // Clone nodes from SIMULATION (which has current positions) and FREEZE them
    // Also UPDATE ALL PROPERTIES from ablation results
    const ablatedNodes = simulationNodes.map(node => {
      // Find node data from ablation results
      const ablationEntry = ablatedData.nodes.find(n => n.id === node.id);
      
      if (ablationEntry) {
        // Use ablation result data (importance, direction, linear_coefficient updated)
        return {
          id: node.id,
          importance: ablationEntry.importance,
          linear_coefficient: ablationEntry.linear_coefficient,
          direction: ablationEntry.direction,
          excluded: node.id === featureName,  // Mark excluded node
          x: node.x,
          y: node.y,
          vx: node.vx || 0,
          vy: node.vy || 0,
          fx: node.x,  // FREEZE at current position
          fy: node.y
        };
      } else {
        // Node not in ablation results (should not happen, but fallback to original)
        return {
          id: node.id,
          importance: node.importance || 0,
          linear_coefficient: node.linear_coefficient || 0,
          direction: node.direction || 0,
          excluded: node.id === featureName,
          x: node.x,
          y: node.y,
          vx: node.vx || 0,
          vy: node.vy || 0,
          fx: node.x,
          fy: node.y
        };
      }
    });
    
    // Use ablated links (edges removed)
    const ablatedLinks = JSON.parse(JSON.stringify(ablatedData.links));
    
    // Preserve current selection so exiting ablation can restore the same plot context.
    const currentSelection = (model.get("selected_node_ids") || []).slice(0, 2);
    state.selectionBeforeAblation = currentSelection;
    // Preserve user-pinned node positions for post-ablation restore.
    const pinnedMap = new Map();
    simulationNodes.forEach((n) => {
      if (Number.isFinite(n.fx) && Number.isFinite(n.fy)) {
        pinnedMap.set(n.id, { fx: n.fx, fy: n.fy });
      }
    });
    state.pinnedBeforeAblation = pinnedMap;

    // Keep only features present in ablation plot arrays.
    const availableFeatures = new Set(ablatedData.feature_names || ablatedNodes.map((n) => n.id));
    const ablationSelection = currentSelection.filter((f) => availableFeatures.has(f));

    // Update state
    state.allNodes = ablatedNodes;
    state.allLinks = ablatedLinks;

    // Switch plot arrays first, then ask Python to redraw with preserved valid selection.
    model.send({ type: "ablation_context", payload: featureName });
    model.set("selected_node_ids", ablationSelection);
    model.save_changes();
    model.send({ type: "selection", payload: ablationSelection });
    
    // Redraw - fullDraw will:
    // 1. Preserve positions from previous simulation
    // 2. Freeze all nodes in ablation view
    // 3. Stop simulation immediately
    fullDraw();
    
    // Style the excluded node
    styleExcludedNode(featureName);
  }
  
  function exitAblationView() {
    console.log(`Exiting ablation view`);
    
    if (!state.currentAblationTarget) return;
    
    // Clear ablation target
    state.currentAblationTarget = null;
    model.send({ type: "ablation_context", payload: "" });

    const restoreSelectionRaw = Array.isArray(state.selectionBeforeAblation)
      ? state.selectionBeforeAblation.slice(0, 2)
      : (model.get("selected_node_ids") || []).slice(0, 2);
    const pinnedMap = state.pinnedBeforeAblation instanceof Map ? state.pinnedBeforeAblation : new Map();

    if (!state.originalNetworkData || !Array.isArray(state.originalNetworkData.nodes)) {
      console.warn("exitAblationView: missing originalNetworkData; restoring from model");
      state.allNodes = (model.get("network_data")?.nodes || []).map((d) => ({ ...d }));
      state.allLinks = (model.get("network_data")?.links || []).map((d) => ({ ...d }));
      const nodeIdSet = new Set(state.allNodes.map((n) => n.id));
      const restoreSelection = restoreSelectionRaw.filter((f) => nodeIdSet.has(f));
      model.set("selected_node_ids", restoreSelection);
      model.save_changes();
      model.send({ type: "selection", payload: restoreSelection });
      state.selectionBeforeAblation = null;
      state.pinnedBeforeAblation = null;
      fullDraw();
      return;
    }
    
    // Restore original network data
    // Clone nodes, remove excluded flag, and recover pre-ablation pinned nodes.
    const restoredNodes = state.originalNetworkData.nodes.map(node => {
      const cloned = {...node};
      // Remove excluded flag if it exists
      delete cloned.excluded;
      const pinned = pinnedMap.get(cloned.id);
      if (pinned && Number.isFinite(pinned.fx) && Number.isFinite(pinned.fy)) {
        cloned.fx = pinned.fx;
        cloned.fy = pinned.fy;
      } else {
        cloned.fx = null;
        cloned.fy = null;
      }
      return cloned;
    });
    
    const restoredLinks = JSON.parse(JSON.stringify(state.originalNetworkData.links));
    
    state.allNodes = restoredNodes;
    state.allLinks = restoredLinks;

    // Restore pre-ablation selection (if still present) so plot updates instead of disappearing.
    const restoredIdSet = new Set(restoredNodes.map((n) => n.id));
    const restoreSelection = restoreSelectionRaw.filter((f) => restoredIdSet.has(f));
    model.set("selected_node_ids", restoreSelection);
    model.save_changes();
    model.send({ type: "selection", payload: restoreSelection });
    state.selectionBeforeAblation = null;
    state.pinnedBeforeAblation = null;
    
    // Redraw with original data (simulation will run normally)
    fullDraw();
  }
  
  function styleExcludedNode(featureName) {
    // The excluded node is already in the graph (added in enterAblationView)
    // Just need to style it specially
    if (!state.nodeSel) return;
    
    // Find and style the excluded node
    state.nodeSel.each(function(d) {
      if (d.id === featureName && d.excluded) {
        const nodeGroup = d3.select(this);
        
        // Add a larger invisible circle for easier clicking
        nodeGroup.insert("circle", ":first-child")
          .attr("class", "excluded-clickable-area")
          .attr("r", d => {
            const baseRadius = state.nodeSize ? state.nodeSize(d.importance || 0) : 10;
            return baseRadius + 8;  // Extra padding for easier clicking
          })
          .style("fill", "transparent")
          .style("cursor", "pointer");
        
        // Remove fill and use only gray dashed stroke for the visible circle
        nodeGroup.select("circle:not(.excluded-clickable-area)")
          .style("fill", "none")  // Remove fill color
          .style("opacity", 1)    // Full opacity for stroke
          .style("stroke", "#9ca3af")  // Gray stroke
          .style("stroke-width", 2)
          .style("stroke-dasharray", "5,5");  // Dashed line
        
        // Add "EXCLUDED" text
        nodeGroup.append("text")
          .attr("class", "excluded-label")
          .attr("dy", -20)
          .style("text-anchor", "middle")
          .style("font-size", "10px")
          .style("font-weight", "bold")
          .style("fill", "#6b7280")  // Gray text
          .text("EXCLUDED");
      }
    });
  }
  
  // Initial draw
  if (!plotOnly) {
    fullDraw();
  }

  const onChangeNetworkData = () => fullDraw();
  const onChangeAblationCandidates = () => safeFullDraw("ablation_candidates");
  const onChangeAblationResults = () => safeFullDraw("ablation_results");
  const onChangeNodeThreshold = () => fullDraw();
  const onChangeLinkThreshold = () => fullDraw();
  const onChangeSelectedNodeIds = () => {
    if (plotOnly) {
      return;
    }
    const selectedIds = model.get("selected_node_ids") || [];
    try {
      updateNodeHighlights(selectedIds);
    } catch (e) {
      console.warn("Failed to update node highlights:", e);
    }
    if (state.allNodes.length > 0) {
      try {
        updateImportantFeatures(state.allNodes);
      } catch (e) {
        console.warn("Failed to update sidebar highlight:", e);
      }
    }
  };
  const onChangePlotMode = () => {
    if (plotOnly) {
      return;
    }
    const selectedIds = model.get("selected_node_ids") || [];
    if (selectedIds.length === 2) {
      model.send({ type: "selection", payload: selectedIds });
    }
    updatePlotModeButton();
  };

  model.on("change:network_data", onChangeNetworkData);
  model.on("change:ablation_candidates", onChangeAblationCandidates);
  model.on("change:ablation_results", onChangeAblationResults);
  model.on("change:node_threshold", onChangeNodeThreshold);
  model.on("change:link_threshold", onChangeLinkThreshold);
  model.on("change:selected_node_ids", onChangeSelectedNodeIds);
  model.on("change:plot_mode", onChangePlotMode);
  
  // Function to render plot using vega-embed
  async function renderPlot(plotData, force = false) {
    // Track plot data for re-rendering
    lastPlotData = plotData;
    
    if (!vegaEmbed || !plotData || !plotData.spec) {
      if (currentView) {
        try {
          currentView.finalize();
        } catch (e) {
          console.warn("Error finalizing previous view:", e);
        }
        currentView = null;
      }
      lastPlotSpec = null;
      vegaWrapper.innerHTML = plotData ? "" : "<p style='padding: 20px; color: #666;'>Click nodes in the network graph to see details.</p>";
      plotModeToggleBtn.style.display = "none"; // Hide button when no plot
      smoothToggleBtn.style.display = "none";
      exportBtn.style.display = "none";
      rangeToggleBtn.style.display = "none";
      rangePopover.style.display = "none";
      currentColorbarRange = null;
      return;
    }
    
    // Show/hide plot mode toggle button based on plot type
    // Only show for continuous_continuous plots (bivariate with 2 features)
    if (plotData.feature_type === "continuous_continuous" && plotData.plot_mode) {
      plotModeToggleBtn.style.display = "block";
    } else {
      plotModeToggleBtn.style.display = "none";
    }

    // Show smooth toggle for continuous scatter plots
    if (plotData.feature_type === "continuous" || plotData.feature_type === "continuous_continuous") {
      smoothToggleBtn.style.display = "block";
      updateSmoothToggleButton();
    } else {
      smoothToggleBtn.style.display = "none";
    }

    if (plotData.manual_range_supported) {
      rangeToggleBtn.style.display = "block";
      manualRangeEnabled = !!plotData.manual_range_enabled;
      const manualRange = plotData.manual_range;
      if (Array.isArray(manualRange) && manualRange.length === 2) {
        const [xr, yr] = manualRange;
        if (Array.isArray(xr) && xr.length === 2 && Array.isArray(yr) && yr.length === 2) {
          manualRangeValues = {
            xMin: Number(xr[0]),
            xMax: Number(xr[1]),
            yMin: Number(yr[0]),
            yMax: Number(yr[1]),
          };
        }
      }
      updateRangeToggleUI();
      if (!manualRangeEnabled) rangePopover.style.display = "none";
    } else {
      rangeToggleBtn.style.display = "none";
      rangePopover.style.display = "none";
    }

    exportBtn.style.display = "block";
    
    // Check if spec has actually changed (avoid unnecessary re-renders)
    const specString = JSON.stringify(plotData.spec);
    if (!force && lastPlotSpec === specString && currentView) {
      // Spec hasn't changed, skip re-render
      return;
    }
    lastPlotSpec = specString;
    
    // Clean up previous view
    if (currentView) {
      try {
        currentView.finalize();
      } catch (e) {
        console.warn("Error finalizing previous view:", e);
      }
      currentView = null;
    }
    
    // Clear container
    vegaWrapper.innerHTML = "";
    
    try {
      // Calculate container size and use fixed dimensions instead of "container"
      // This avoids vega-embed's internal ResizeObserver which can cause infinite loops
      // Button and colorbar are overlayed, so no need to reserve space
      const containerWidth = Math.max(300, vegaWrapper.clientWidth - 8); // Account for padding (4px * 2)
      const containerHeight = Math.max(250, vegaWrapper.clientHeight - 12); // Account for padding (6px * 2)
      
      // Create spec with fixed dimensions
      const spec = {
        ...plotData.spec,
        width: containerWidth,
        height: containerHeight,
        autosize: { type: "fit", contains: "padding" } // Use fit to respect padding
      };
      
      // Render with vega-embed (use canvas for better performance)
      const result = await vegaEmbed(vegaWrapper, spec, {
        actions: false,
        renderer: "canvas"
      });
      currentView = result.view;
      
      // Add color range slider for continuous color features
      // Color bar position and size settings
      if (plotData.color_feature && plotData.color_min !== undefined && plotData.color_max !== undefined) {
        const rangeSliderContainer = document.createElement("div");
        rangeSliderContainer.className = "astrolabe-color-range-slider";
        rangeSliderContainer.style.cssText = `
          position: absolute;
          right: 12px;        /* Right margin */
          bottom: 80px;       /* Bottom margin */
          width: 40px;        /* Color bar width */
          height: 55%;        /* Color bar height (%) */
          max-height: 350px;  /* Maximum height (px) */
          z-index: 50;
        `;
        
        // Initialize range values
        const dataMin = plotData.color_min;
        const dataMax = plotData.color_max;
        let rangeMin = dataMin;
        let rangeMax = dataMax;
        currentColorbarRange = { vmin: dataMin, lower: rangeMin, upper: rangeMax, vmax: dataMax };
        
        // Viridis color scale endpoints
        const leftColor = "#440154"; // Purple (viridis start)
        const rightColor = "#fde725"; // Yellow (viridis end)
        
        // Slider track with gradient (vertical)
        const sliderTrack = document.createElement("div");
        sliderTrack.style.cssText = `
          position: relative;
          width: 20px;
          height: 100%;
          margin: 0 auto;
          border-radius: 4px;
          background: linear-gradient(180deg, #fde725 0%, #3fbf73 17%, #1f9e89 34%, #277f8e 50%, #365c8d 67%, #46327e 84%, #440154 100%);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
          cursor: pointer;
          border: 1px solid rgba(255, 255, 255, 0.3);
        `;
        
        // Function to update gradient based on handle positions (vertical)
        const updateGradient = (topPercent, bottomPercent) => {
          // Build gradient: solid top color -> gradient -> solid bottom color
          // Note: top handle controls MAX (yellow), bottom handle controls MIN (purple)
          sliderTrack.style.background = `linear-gradient(180deg, 
            ${rightColor} 0%, 
            ${rightColor} ${topPercent}%, 
            #fde725 ${topPercent}%, 
            #3fbf73 ${topPercent + (bottomPercent - topPercent) * 0.17}%, 
            #1f9e89 ${topPercent + (bottomPercent - topPercent) * 0.34}%, 
            #277f8e ${topPercent + (bottomPercent - topPercent) * 0.50}%, 
            #365c8d ${topPercent + (bottomPercent - topPercent) * 0.67}%, 
            #46327e ${topPercent + (bottomPercent - topPercent) * 0.84}%, 
            #440154 ${bottomPercent}%, 
            ${leftColor} ${bottomPercent}%, 
            ${leftColor} 100%)`;
        };
        
        // Data range labels (fixed position - top and bottom of colorbar)
        const dataMaxLabel = document.createElement("div");
        dataMaxLabel.style.cssText = `
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          top: -20px;
          font-size: 10px;
          color: ${currentTheme === "dark" ? "#e0e0e0" : "#333333"};
          user-select: none;
          text-align: center;
          font-weight: 500;
          white-space: nowrap;
          text-shadow: ${currentTheme === "dark" ? "0 1px 2px rgba(0, 0, 0, 0.75)" : "0 1px 2px rgba(255, 255, 255, 0.85)"};
          z-index: 4;
        `;
        dataMaxLabel.textContent = dataMax.toLocaleString(undefined, { maximumFractionDigits: 1 });
        
        const dataMinLabel = document.createElement("div");
        dataMinLabel.style.cssText = `
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          bottom: -27px;
          font-size: 10px;
          color: ${currentTheme === "dark" ? "#e0e0e0" : "#333333"};
          user-select: none;
          text-align: center;
          font-weight: 500;
          white-space: nowrap;
          text-shadow: ${currentTheme === "dark" ? "0 1px 2px rgba(0, 0, 0, 0.75)" : "0 1px 2px rgba(255, 255, 255, 0.85)"};
          z-index: 4;
        `;
        dataMinLabel.textContent = dataMin.toLocaleString(undefined, { maximumFractionDigits: 1 });
        
        // Top handle (controls MAX - yellow end) with label inside
        // Handle center is at top edge of colorbar (0%)
        const topHandle = document.createElement("div");
        topHandle.style.cssText = `
          position: absolute;
          top: 0%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 40px;
          height: 14px;
          background: ${currentTheme === "dark" ? "#6b7280" : "#4b5563"};
          border: 2px solid ${currentTheme === "dark" ? "#9ca3af" : "#ffffff"};
          border-radius: 3px;
          cursor: ns-resize;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
          z-index: 2;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          font-weight: 600;
          color: #ffffff;
          user-select: none;
        `;
        topHandle.textContent = rangeMax.toLocaleString(undefined, { maximumFractionDigits: 1 });
        
        // Bottom handle (controls MIN - purple end) with label inside
        // Handle center is at bottom edge of colorbar (100%)
        const bottomHandle = document.createElement("div");
        bottomHandle.style.cssText = `
          position: absolute;
          top: 100%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 40px;
          height: 14px;
          background: ${currentTheme === "dark" ? "#6b7280" : "#4b5563"};
          border: 2px solid ${currentTheme === "dark" ? "#9ca3af" : "#ffffff"};
          border-radius: 3px;
          cursor: ns-resize;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
          z-index: 2;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          font-weight: 600;
          color: #ffffff;
          user-select: none;
        `;
        bottomHandle.textContent = rangeMin.toLocaleString(undefined, { maximumFractionDigits: 1 });
        
        // Update Vega color scale domain using signals
        const updateColorScale = () => {
          if (!currentView) return;
          
          try {
            // Update Vega signals that control the color scale domain
            currentView.signal("colorMin", rangeMin);
            currentView.signal("colorMax", rangeMax);
            currentView.run();
            currentColorbarRange = { vmin: dataMin, lower: rangeMin, upper: rangeMax, vmax: dataMax };
          } catch (e) {
            console.warn("Could not update color scale:", e);
          }
        };
        
        // Handle dragging
        let isDragging = false;
        let activeHandle = null;
        
        const startDrag = (handle, event) => {
          event.stopPropagation();
          event.preventDefault();
          isDragging = true;
          activeHandle = handle;
          document.body.style.cursor = 'ns-resize';
        };
        
        const onDrag = (event) => {
          if (!isDragging || !activeHandle) return;
          
          const rect = sliderTrack.getBoundingClientRect();
          const y = event.clientY - rect.top;
          const percent = Math.max(0, Math.min(1, y / rect.height));
          
          // Top handle controls MAX (yellow, top of gradient)
          if (activeHandle === topHandle) {
            const currentBottomPercent = parseFloat(bottomHandle.style.top) || 100;
            if (percent * 100 < currentBottomPercent) {
              // Value increases as we go up (inverse of percent)
              const newValue = dataMax - percent * (dataMax - dataMin);
              rangeMax = newValue;
              topHandle.style.top = `${percent * 100}%`;
              topHandle.textContent = rangeMax.toLocaleString(undefined, { maximumFractionDigits: 1 });
              updateGradient(percent * 100, currentBottomPercent);
              updateColorScale();
            }
          } 
          // Bottom handle controls MIN (purple, bottom of gradient)
          else if (activeHandle === bottomHandle) {
            const currentTopPercent = parseFloat(topHandle.style.top) || 0;
            if (percent * 100 > currentTopPercent) {
              // Value decreases as we go down (inverse of percent)
              const newValue = dataMax - percent * (dataMax - dataMin);
              rangeMin = newValue;
              bottomHandle.style.top = `${percent * 100}%`;
              bottomHandle.textContent = rangeMin.toLocaleString(undefined, { maximumFractionDigits: 1 });
              updateGradient(currentTopPercent, percent * 100);
              updateColorScale();
            }
          }
        };
        
        const endDrag = () => {
          if (isDragging) {
            isDragging = false;
            activeHandle = null;
            document.body.style.cursor = '';
          }
        };
        
        topHandle.addEventListener('mousedown', (e) => startDrag(topHandle, e));
        bottomHandle.addEventListener('mousedown', (e) => startDrag(bottomHandle, e));
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', endDrag);
        
        // Initial gradient update
        updateGradient(0, 100);
        
        // Color encoding label (vertical) - anchored to the left of the colorbar
        const colorEncodingLabel = document.createElement("div");
        colorEncodingLabel.style.cssText = `
          position: absolute;
          right: calc(100% + 6px);
          left: auto;
          top: 50%;
          transform: translateY(-50%) rotate(180deg);
          transform-origin: center;
          writing-mode: vertical-rl;
          text-orientation: mixed;
          font-size: 10px;
          font-weight: 500;
          line-height: 1;
          color: ${currentTheme === "dark" ? "#e0e0e0" : "#333333"};
          user-select: none;
          white-space: nowrap;
          text-shadow: ${currentTheme === "dark" ? "0 1px 2px rgba(0, 0, 0, 0.75)" : "0 1px 2px rgba(255, 255, 255, 0.85)"};
          z-index: 3;
          text-align: center;
          pointer-events: none;
        `;
        colorEncodingLabel.textContent = plotData.color_feature || "Color";
        
        // Assemble slider
        sliderTrack.appendChild(topHandle);
        sliderTrack.appendChild(bottomHandle);
        sliderTrack.appendChild(colorEncodingLabel);
        rangeSliderContainer.appendChild(sliderTrack);
        rangeSliderContainer.appendChild(dataMaxLabel);
        rangeSliderContainer.appendChild(dataMinLabel);
        vegaWrapper.appendChild(rangeSliderContainer);
      } else {
        currentColorbarRange = null;
      }
    } catch (error) {
      console.error("Error rendering plot:", error);
      vegaWrapper.innerHTML = `<p style='padding: 20px; color: #ef4444;'>Error rendering plot: ${error.message}</p>`;
    }
  }
  
  // Debounced plot rendering function
  function debouncedRenderPlot(plotData, force = false) {
    if (plotRenderTimeout) clearTimeout(plotRenderTimeout);
    plotRenderTimeout = setTimeout(() => {
      renderPlot(plotData, force);
    }, 100); // 100ms debounce
  }
  
  const onChangePlotData = () => {
    const plotData = model.get("plot_data") || {};
    debouncedRenderPlot(plotData);
  };
  model.on("change:plot_data", onChangePlotData);

  const onChangePlotSmooth = () => {
    updateSmoothToggleButton();
  };
  model.on("change:plot_smooth", onChangePlotSmooth);
  
  // Initial plot render
  const initialPlotData = model.get("plot_data") || {};
  if (initialPlotData.spec) {
    renderPlot(initialPlotData);
  }
  
  // Debounce function for resize (only for graph container)
  let resizeTimeout = null;
  
  const debouncedResize = () => {
    if (resizeTimeout) clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      // Only redraw graph
      fullDraw();
    }, 200); // 200ms debounce for resize
  };

  let plotResizeTimeout = null;
  const debouncedPlotResize = () => {
    if (plotResizeTimeout) clearTimeout(plotResizeTimeout);
    plotResizeTimeout = setTimeout(() => {
      if (lastPlotData && lastPlotData.spec) {
        renderPlot(lastPlotData, true);
      }
    }, 200);
  };
  
  // Resize observer for graph container only (NOT plot container to avoid infinite loops)
  const resizeObserver = new ResizeObserver(debouncedResize);
  resizeObserver.observe(container);
  window.addEventListener("resize", debouncedPlotResize);
  
  return () => {
    try {
      if (typeof model.off === "function") {
        model.off("change:network_data", onChangeNetworkData);
        model.off("change:ablation_candidates", onChangeAblationCandidates);
        model.off("change:ablation_results", onChangeAblationResults);
        model.off("change:node_threshold", onChangeNodeThreshold);
        model.off("change:link_threshold", onChangeLinkThreshold);
        model.off("change:selected_node_ids", onChangeSelectedNodeIds);
        model.off("change:plot_mode", onChangePlotMode);
        model.off("change:plot_data", onChangePlotData);
        model.off("change:plot_smooth", onChangePlotSmooth);
      }
    } catch (e) {
      console.warn("Astrolabe widget model listener cleanup:", e);
    }
    try {
      resizeObserver.unobserve(container);
    } catch {}
    if (resizeTimeout) clearTimeout(resizeTimeout);
    if (plotResizeTimeout) clearTimeout(plotResizeTimeout);
    window.removeEventListener("resize", debouncedPlotResize);
    if (plotRenderTimeout) clearTimeout(plotRenderTimeout);
    if (state.simulation) state.simulation.stop();
    if (tooltip) tooltip.remove();
    if (currentView) {
      try {
        currentView.finalize();
      } catch (e) {
        console.warn("Error finalizing view on cleanup:", e);
      }
    }
  };
}

