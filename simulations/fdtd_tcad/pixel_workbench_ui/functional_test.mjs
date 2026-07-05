import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';

const args = new Map(
  process.argv.slice(2).map((arg, index, arr) => {
    if (!arg.startsWith('--')) return [arg, true];
    const [key, inline] = arg.split('=');
    return [key, inline ?? arr[index + 1] ?? true];
  })
);

const url = args.get('--url') || 'http://127.0.0.1:8765/runs/image_sensor_pixel_studio_reference/index.html';
const out = resolve(String(args.get('--out') || '../runs/image_sensor_pixel_studio_reference/ux_functional_test_report.json'));
const screenshot = resolve(String(args.get('--screenshot') || '../runs/image_sensor_pixel_studio_reference/ux_functional_test.png'));
const chromePath = String(args.get('--chrome') || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
const port = Number(args.get('--port') || 9231);
const runSolver = Boolean(args.get('--solver'));
const runGdsSuite = Boolean(args.get('--gds-suite'));
const solverTimeoutMs = Number(args.get('--solver-timeout-ms') || 180000);
const userDir = `${tmpdir()}/pixel-workbench-functional-${Date.now()}`;
const wait = (ms) => new Promise((resolveWait) => setTimeout(resolveWait, ms));
const near = (value, expected, tolerance = 1e-6) => Math.abs(Number(value) - expected) <= tolerance;

async function fetchJson(endpoint, attempts = 80) {
  let lastError = '';
  for (let index = 0; index < attempts; index += 1) {
    try {
      const response = await fetch(endpoint);
      if (response.ok) return await response.json();
      lastError = `${response.status} ${response.statusText}`;
    } catch (error) {
      lastError = error.message;
    }
    await wait(100);
  }
  throw new Error(`CDP endpoint unavailable: ${lastError}`);
}

function makeAssert(results) {
  return (name, condition, details = {}) => {
    results.push({ name, status: condition ? 'PASS' : 'FAIL', details });
    if (!condition) throw new Error(`${name} failed: ${JSON.stringify(details)}`);
  };
}

async function run() {
  mkdirSync(dirname(out), { recursive: true });
  mkdirSync(dirname(screenshot), { recursive: true });
  const chrome = spawn(chromePath, [
    '--headless=new',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDir}`,
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--window-size=1440,900',
    'about:blank'
  ], { stdio: 'ignore' });

  const results = [];
  const assert = makeAssert(results);
  const exceptions = [];
  const states = [];

  try {
    const targets = await fetchJson(`http://127.0.0.1:${port}/json`);
    const target = targets.find((item) => item.type === 'page') || targets[0];
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    const pending = new Map();
    const events = [];
    let messageId = 0;

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        pending.get(message.id)(message);
        pending.delete(message.id);
        return;
      }
      if (message.method) {
        events.push(message);
        if (message.method === 'Runtime.exceptionThrown') {
          exceptions.push(message.params?.exceptionDetails?.text || 'exception');
        }
      }
    };

    await new Promise((resolveOpen, rejectOpen) => {
      ws.onopen = resolveOpen;
      ws.onerror = rejectOpen;
    });

    const send = (method, params = {}) => new Promise((resolveSend, rejectSend) => {
      const id = ++messageId;
      pending.set(id, (message) => {
        if (message.error) rejectSend(new Error(JSON.stringify(message.error)));
        else resolveSend(message.result);
      });
      ws.send(JSON.stringify({ id, method, params }));
    });

    const waitEvent = async (method) => {
      for (let index = 0; index < 200; index += 1) {
        const eventIndex = events.findIndex((event) => event.method === method);
        if (eventIndex >= 0) return events.splice(eventIndex, 1)[0];
        await wait(40);
      }
      throw new Error(`Timed out waiting for ${method}`);
    };

    const evaluate = async (expression) => {
      const response = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (response.exceptionDetails) throw new Error(response.exceptionDetails.text || JSON.stringify(response.exceptionDetails));
      return response.result.value;
    };

    const clickByText = async (selector, text) => evaluate(`(() => {
      const node = [...document.querySelectorAll(${JSON.stringify(selector)})].find((item) => item.textContent.includes(${JSON.stringify(text)}));
      if (!node) return false;
      node.click();
      return true;
    })()`);
    const clickSelector = async (selector, index = 0) => evaluate(`(() => {
      const node = [...document.querySelectorAll(${JSON.stringify(selector)})][${index}];
      if (!node) return false;
      node.click();
      return true;
    })()`);
    const inputValue = async (selector, value, index = 0) => evaluate(`(() => {
      const node = [...document.querySelectorAll(${JSON.stringify(selector)})][${index}];
      if (!node) return false;
      const setter = Object.getOwnPropertyDescriptor(node.constructor.prototype, 'value')?.set;
      if (setter) setter.call(node, ${JSON.stringify(String(value))});
      else node.value = ${JSON.stringify(String(value))};
      node.dispatchEvent(new Event('input', { bubbles: true }));
      node.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    const changeSelect = async (selector, value, index = 0) => inputValue(selector, value, index);
    const waitFor = async (name, expression, timeoutMs = 8000) => {
      const deadline = Date.now() + timeoutMs;
      let last = null;
      while (Date.now() < deadline) {
        last = await evaluate(expression);
        if (last) return last;
        await wait(250);
      }
      throw new Error(`${name} timed out: ${JSON.stringify(last)}`);
    };
    const snap = async (name) => {
      const state = await evaluate(`({
        h1: document.querySelector('h1')?.textContent || '',
        toast: document.querySelector('.action-toast')?.textContent || '',
        activeNav: document.querySelector('.nav-entry.active')?.textContent.trim() || '',
        activePreset: document.querySelector('.preset-card.active strong')?.textContent || '',
        log: [...document.querySelectorAll('.action-log span')].map((node) => node.textContent).slice(0, 4),
        coupled: document.querySelectorAll('.pixel-cell.coupled').length,
        responseTab: document.querySelector('.response-tabs .active')?.textContent || '',
        activeClass: document.querySelector('.class-row.active strong')?.textContent || '',
        activeOclModel: document.querySelector('.parameter-panel .select-row select')?.value || '',
        activeMode: document.querySelector('.mode-card.active strong')?.textContent || '',
        selectedCandidate: document.querySelector('tr.selected td')?.textContent || '',
        cfaThickness: [...document.querySelectorAll('.edit-row input')][0]?.value || '',
        simulationStatus: document.querySelector('.simulation-panel .status-pill')?.textContent || '',
        simulationKpis: document.querySelectorAll('.simulation-kpi').length,
        simulationImages: document.querySelectorAll('.simulation-artifacts img').length,
        simulationError: document.querySelector('.simulation-error')?.textContent || '',
        activeRequestPreview: document.querySelector('.simulation-request-preview')?.textContent || '',
        requestLinks: document.querySelectorAll('a[href*="simulation_request.json"]').length,
        solverCaseLinks: document.querySelectorAll('a[href*="solver_case.json"]').length,
        kpiSummaryLinks: document.querySelectorAll('a[href*="kpi_summary.json"]').length,
        testSuiteHeading: document.querySelector('.suite-setup h2')?.textContent || '',
        suiteCards: document.querySelectorAll('.suite-catalog button').length,
        suiteTierButtons: document.querySelectorAll('.suite-tier-row button').length,
        suiteStatus: document.querySelector('.suite-result .status-pill')?.textContent || '',
        suiteKpis: document.querySelectorAll('.suite-kpi-grid > div').length,
        suiteCharts: document.querySelectorAll('.suite-chart-card').length,
        suiteCases: document.querySelectorAll('.suite-case-table > div').length,
        suiteResultArtifactLinks: document.querySelectorAll('.suite-result-artifact-link').length,
        suiteCaseResultLinks: document.querySelectorAll('a[href*="case_result.json"]').length,
        suiteCaseInputLinks: document.querySelectorAll('a[href*="case_input.json"]').length,
        suiteCaseCommandLinks: document.querySelectorAll('a[href*="case_command.json"]').length,
        suiteReplayButtons: [...document.querySelectorAll('.suite-case-artifacts button')].filter((node) => node.textContent.includes('Replay')).length,
        suiteReplayText: document.querySelector('.suite-replay-result')?.textContent || '',
        suiteReplayManifestLinks: document.querySelectorAll('a[href*="replay_manifest.json"]').length,
        suiteReplayComparisonLinks: document.querySelectorAll('a[href*="replay_comparison.json"]').length,
        suiteSelectedCases: document.querySelectorAll('.suite-matrix-list input[type="checkbox"]:checked').length,
        readinessStatus: document.querySelector('.readiness-page .suite-result-header .status-pill')?.textContent || '',
        readinessKpis: document.querySelectorAll('.readiness-kpi-grid > div').length,
        readinessBlockerRows: document.querySelectorAll('.readiness-table .readiness-row').length,
        readinessEvidenceRows: document.querySelectorAll('.readiness-evidence-list > div').length,
        readinessArtifactLinks: document.querySelectorAll('.readiness-page .suite-result-artifact-link').length,
        readinessText: document.querySelector('.readiness-page')?.textContent || '',
        suiteCadEvidence: document.querySelectorAll('.suite-cad-evidence').length,
        suiteCadPreviews: document.querySelectorAll('.suite-cad-card img').length,
        suiteCadArtifactLinks: document.querySelectorAll('.suite-artifact-link').length,
        cadTemplateCards: document.querySelectorAll('.cad-template-catalog button').length,
        cadStarterText: document.querySelector('.cad-starter-summary')?.textContent || '',
        cadTemplateGroups: document.querySelectorAll('.cad-template-group').length,
        cadTemplateGroupHeadings: [...document.querySelectorAll('.cad-template-group h3')].map((node) => node.textContent),
        cadTemplatePreview: document.querySelectorAll('.cad-template-hero img').length,
        cadTemplateLinks: document.querySelectorAll('.cad-template-links a').length,
        cadTemplateAssumptionLinks: document.querySelectorAll('a[href*="assumption_ledger.json"]').length,
        cadTemplateVariantSourceLinks: document.querySelectorAll('a[href*="variant_source.json"]').length,
        cadTemplateMeshLinks: document.querySelectorAll('a[href*="/model.msh"]').length,
        cadTemplateTcadLinks: document.querySelectorAll('a[href*="tcad_bridge_2d"]').length,
        cadTemplateFcstdLinks: document.querySelectorAll('a[href*="/model.FCStd"]').length,
        cadTemplateFreecadValidationLinks: document.querySelectorAll('a[href*="freecad_validation_report.json"]').length,
        cadTemplateQpdGwLinks: document.querySelectorAll('a[href*="tcad_qpd_gw_3d"]').length,
        cadTemplateSummaryText: document.querySelector('.cad-template-summary')?.textContent || '',
        cadTemplatePresetMatchWarnings: document.querySelectorAll('.cad-template-summary .cad-summary-warn').length,
        cadTemplateSummaryTooltips: [...document.querySelectorAll('.cad-template-summary > div')].filter((node) => node.getAttribute('title')).length,
        cadTemplateStatus: document.querySelector('.cad-template-catalog .status-pill')?.textContent || '',
        cadTemplateActionText: [...document.querySelectorAll('.cad-workflow-panel, .cad-template-actions')].map((node) => node.textContent).join('\\n'),
        cadTemplateActionButtons: document.querySelectorAll('.cad-template-actions button').length,
        cadTemplatePrimaryButtons: document.querySelectorAll('.cad-template-actions-primary button').length,
        cadTemplateAdvancedVisible: Boolean(document.querySelector('.cad-template-actions-advanced')),
        cadTemplateAdvancedText: document.querySelector('.cad-template-actions-advanced')?.textContent || '',
        qpdComparisonRows: document.querySelectorAll('.qpd-comparison-table tbody tr').length,
        qpdComparisonText: document.querySelector('.qpd-comparison-panel')?.textContent || '',
        solverRoleRows: document.querySelectorAll('.solver-role-row').length,
        solverRoleText: document.querySelector('.solver-role-panel')?.textContent || '',
        cadSourceText: document.querySelector('.cad-source-panel')?.textContent || '',
        cadSourceRows: document.querySelectorAll('.cad-source-grid > div').length,
        cadSourceArtifactLinks: document.querySelectorAll('.cad-source-grid a').length,
        cadSourceOpenButtons: document.querySelectorAll('.cad-source-header-actions button').length,
        cadFcstdRoundtripText: document.querySelector('.fcstd-roundtrip-panel')?.textContent || '',
        cadFcstdRoundtripInputs: document.querySelectorAll('.fcstd-roundtrip-panel input').length,
        cadFcstdImportPath: document.querySelector('.fcstd-roundtrip-panel input')?.value || '',
        cadBaseTemplateText: document.querySelector('.cad-base-template-builder')?.textContent || '',
        cadBaseTemplateInputs: document.querySelectorAll('.cad-base-template-builder input').length,
        cadBaseTemplateSelects: document.querySelectorAll('.cad-base-template-builder select').length,
        cadVariantBuilderText: document.querySelector('.cad-variant-builder')?.textContent || '',
        cadVariantInputs: document.querySelectorAll('.cad-variant-builder input').length,
        cadTemplateNotes: document.querySelector('.cad-template-notes')?.textContent || '',
        suiteError: document.querySelector('.test-suite-page .simulation-error')?.textContent || ''
      })`);
      states.push({ name, state });
      return state;
    };

    await send('Page.enable');
    await send('Runtime.enable');
    await send('Page.navigate', { url });
    await waitEvent('Page.loadEventFired');
    await wait(450);

    const initial = await snap('initial');
    assert('payload loaded and Pattern Composer renders', initial.h1 === 'Pattern Composer' && initial.activePreset === 'Quad Bayer + QPD', initial);
    assert('preset cards render', await evaluate(`document.querySelectorAll('.preset-card').length === 8`));

    assert('Template nav works', await clickByText('.nav-entry', 'Template'));
    await waitFor('CAD template catalog loads', `(() => document.querySelectorAll('.cad-template-catalog button').length >= 5)()`);
    const templateState = await snap('template catalog');
    assert(
      'CAD template catalog renders',
      templateState.h1 === 'Template'
        && templateState.cadTemplateCards >= 5
        && templateState.cadTemplatePreview === 1
        && templateState.cadTemplateLinks >= 7
        && templateState.cadTemplateAssumptionLinks >= 1
        && templateState.cadTemplateSummaryText.includes('CAD mesh'),
      templateState
    );
    assert('CAD starter set readiness renders', templateState.cadStarterText.includes('9/9') && templateState.cadStarterText.includes('starter templates ready'), templateState);
    assert('CAD starter and variant groups render separately', templateState.cadTemplateGroups >= 2 && templateState.cadTemplateGroupHeadings.includes('Starter Templates') && templateState.cadTemplateGroupHeadings.includes('Registered Variants'), templateState);
    assert(
      'CAD template primary workflow actions render',
      templateState.cadTemplatePrimaryButtons >= 8
        && templateState.cadTemplateActionText.includes('Template Workflow')
        && templateState.cadTemplateActionText.includes('Open STEP in FreeCAD')
        && templateState.cadTemplateActionText.includes('Make FCStd Working Copy')
        && templateState.cadTemplateActionText.includes('Open Working Copy')
        && templateState.cadTemplateActionText.includes('Read FCStd Parameters')
        && templateState.cadTemplateActionText.includes('Create Variant From FCStd')
        && templateState.cadTemplateActionText.includes('Validate FreeCAD')
        && templateState.cadTemplateActionText.includes('Run CAD Template')
        && templateState.cadTemplateActionText.includes('Clear CAD Source')
        && templateState.cadTemplateActionText.includes('Advanced TCAD / diagnostics')
        && templateState.cadTemplateAdvancedVisible === false,
      templateState
    );
    assert('CAD advanced diagnostics can be expanded', await clickByText('.cad-advanced-toggle', 'Advanced TCAD / diagnostics'));
    await wait(100);
    const templateAdvancedState = await snap('template advanced actions');
    assert(
      'CAD template advanced diagnostic actions render only after expand',
      templateAdvancedState.cadTemplateAdvancedVisible
        && templateAdvancedState.cadTemplateAdvancedText.includes('Open BREP')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Open FCStd')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Open Assumptions')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Generate TCAD Bridge')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Run TCAD DD Smoke')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Run QPD Axis Pair')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Run QPD 3D Weighting')
        && templateAdvancedState.cadTemplateAdvancedText.includes('Run QPD 3D G*W'),
      templateAdvancedState
    );
    assert('CAD template FreeCAD validation links render', templateState.cadTemplateFcstdLinks >= 1 && templateState.cadTemplateFreecadValidationLinks >= 1, templateState);
    assert(
      'CAD source file paths render',
      templateState.cadSourceRows >= 4
        && templateState.cadSourceArtifactLinks >= 3
        && templateState.cadSourceOpenButtons === 1
        && templateState.cadSourceText.includes('CAD Source Files')
        && templateState.cadSourceText.includes('Open Source Folder')
        && templateState.cadSourceText.includes('/Users/seongcheoljeong/FDTD/runs/pixel_cad_template_library_reference')
        && templateState.cadSourceText.includes('model.step')
        && templateState.cadSourceText.includes('model.FCStd')
        && templateState.cadSourceText.includes('geometry_import.json'),
      templateState
    );
    assert('CAD template design-rule status renders', templateState.cadTemplateSummaryText.includes('Design rules') && templateState.cadTemplateSummaryText.includes('PASS'), templateState);
    assert('CAD template summary cards expose tooltips', templateState.cadTemplateSummaryTooltips >= 20, templateState);
    assert('CAD template FCStd parameter sheets render', templateState.cadTemplateSummaryText.includes('parameter sheets'), templateState);
    assert(
      'CAD template simulation fidelity is explicit',
      templateState.cadTemplateSummaryText.includes('Simulation basis3D CAD + hybrid 2D/3D')
        && templateState.cadTemplateSummaryText.includes('Full 3D DDnot available')
        && templateState.cadTemplateSummaryText.includes('FDTD geometry inputavailable'),
      templateState
    );
    assert(
      'Solver role matrix separates primary and diagnostic solvers',
      templateState.solverRoleRows >= 4
        && templateState.solverRoleText.includes('Solver Role Matrix')
        && templateState.solverRoleText.includes('Primary: FDTD Optical + 3D G*W')
        && templateState.solverRoleText.includes('Diagnostic: DEVSIM DD')
        && templateState.solverRoleText.includes('Circuit / Readout')
        && templateState.solverRoleText.includes('Current practical decisions should be FDTD/G*W-led'),
      templateState
    );
    assert(
      'CAD template pixel size and pitch policy are visible',
      templateState.cadTemplateSummaryText.includes('Pixel pitch1.400 um')
        && templateState.cadTemplateSummaryText.includes('Template span2 x 2 px · 2.800 x 2.800 um')
        && templateState.cadTemplateSummaryText.includes('OCL group pitch2x2 group · 2.800 x 2.800 um')
        && templateState.cadTemplateSummaryText.includes('Crosstalk coverage1x1 OCL groups · insufficient')
        && templateState.cadTemplateSummaryText.includes('Pitch variantconditional variant')
        && templateState.cadTemplateSummaryText.includes('Pitch scalemixed')
        && templateState.cadTemplateSummaryText.includes('Topology key2x2 pixels'),
      templateState
    );
    assert(
      'QPD template comparison renders practical variant metrics',
      templateState.qpdComparisonRows >= 7
        && templateState.qpdComparisonText.includes('QPD Template Comparison')
        && templateState.qpdComparisonText.includes('G*W phase')
        && templateState.qpdComparisonText.includes('qpd_split_pd_no_shield_2x2')
        && templateState.qpdComparisonText.includes('Full Q1-Q4 DD remains CHECK'),
      templateState
    );
    assert('CAD FCStd working-copy panel renders', templateState.cadFcstdRoundtripInputs === 1 && templateState.cadFcstdRoundtripText.includes('FCStd Working Copy') && templateState.cadFcstdRoundtripText.includes('base only'), templateState);
    assert(
      'CAD base template builder renders topology workflow',
      templateState.cadBaseTemplateInputs >= 3
        && templateState.cadBaseTemplateSelects === 1
        && templateState.cadBaseTemplateText.includes('Create Base Template')
        && templateState.cadBaseTemplateText.includes('topology changes')
        && templateState.cadBaseTemplateText.includes('Create Base Template'),
      templateState
    );
    assert('CAD quick variant builder renders', templateState.cadVariantInputs >= 9 && templateState.cadVariantBuilderText.includes('Create CAD Variant') && templateState.cadVariantBuilderText.includes('Pixel pitch and scalar geometry overrides'), templateState);
    assert('CAD template mesh role is explicit', templateState.cadTemplateNotes.includes('not calibrated DEVSIM electrical meshes'), templateState);
    const axisAwareCatalog = await evaluate(`(async () => {
      const response = await fetch('/api/cad/templates');
      if (!response.ok) return { error: response.status };
      const data = await response.json();
      const toolsResponse = await fetch('/api/cad/tools');
      const tools = toolsResponse.ok ? await toolsResponse.json() : {};
      const dryRunResponse = await fetch('/api/cad/create-base-template', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dry_run: true,
          topology_preset: 'qpd_2x2',
          id: 'functional_dry_run_qpd_2x2',
          label: 'Functional dry-run QPD 2x2',
          parameters: { pitch_um: 1.2 }
        })
      });
      const dryRun = dryRunResponse.ok ? await dryRunResponse.json() : { error: dryRunResponse.status };
      const byId = Object.fromEntries((data.templates || []).map((item) => [item.template_id, item]));
      const dualX = byId.dual_pd_x_1x1?.tcad_bridge || {};
      const dualZ = byId.dual_pd_z_1x1?.tcad_bridge || {};
      const dualZDd = dualZ.devsim_dd_smoke || {};
      const qpdAxisPair = byId.qpd_split_pd_2x2?.tcad_bridge?.axis_pair_smoke || {};
      const qpdWeighting3d = byId.qpd_split_pd_2x2?.tcad_bridge?.qpd_weighting_3d || {};
      const qpdGw3d = byId.qpd_split_pd_2x2?.tcad_bridge?.qpd_gw_3d || {};
      const qpdFidelity = byId.qpd_split_pd_2x2?.simulation_fidelity || {};
      const qpdUiDd = byId.qpd_split_pd_ui_smoke_lens_690nm?.tcad_bridge?.devsim_dd_smoke || {};
      const qpdUiCapability = byId.qpd_split_pd_ui_smoke_lens_690nm?.tcad_bridge?.electrical_capability || {};
      const qpdComparison = data.qpd_template_comparison || {};
      const qpdNoShieldRow = (qpdComparison.rows || []).find((row) => row.template_id === 'qpd_split_pd_no_shield_2x2') || {};
      const qpdDimension = byId.qpd_split_pd_2x2?.dimension_summary || {};
      const quadCrosstalkDimension = byId.quad_2x2_ocl_3x3_neighborhood?.dimension_summary || {};
      const quadPracticalCrosstalkDimension = byId.quad_2x2_ocl_5x5_crosstalk?.dimension_summary || {};
      const solverRoleMatrix = data.solver_role_matrix || {};
      const solverRoleLabels = (solverRoleMatrix.rows || []).map((row) => row.label);
      const starter = data.starter_template_set || {};
      return {
        catalogStatus: data.status,
        baseTemplateCount: data.base_template_count,
        variantCount: data.variant_count,
        starterStatus: starter.status,
        starterPassCount: starter.pass_count,
        starterTemplateCount: starter.template_count,
        dualXStatus: dualX.status,
        dualXAxis: dualX.electrical_capability?.section_axis,
        dualZStatus: dualZ.status,
        dualZAxis: dualZ.electrical_capability?.section_axis,
        dualZPhaseAxis: dualZDd.phase_result_axis,
        dualZPhaseZ: dualZDd.photo_split_phase_z_proxy,
        dualZGate: dualZDd.capability_gate,
        qpdAxisPairStatus: qpdAxisPair.status,
        qpdAxisPairFullGate: qpdAxisPair.full_q1q4_gate,
        qpdAxisPairMagnitude: qpdAxisPair.axis_phase_magnitude,
        qpdAxisPairUniformity: qpdAxisPair.axis_signal_uniformity,
        qpdWeighting3dStatus: qpdWeighting3d.status,
        qpdWeighting3dGate: qpdWeighting3d.full_q1q4_weighting_gate,
        qpdWeighting3dDdGate: qpdWeighting3d.full_q1q4_dd_gate,
        qpdWeighting3dUniformity: qpdWeighting3d.quadrant_uniformity,
        qpdGw3dStatus: qpdGw3d.status,
        qpdGw3dGate: qpdGw3d.full_q1q4_gw_gate,
        qpdGw3dDdGate: qpdGw3d.full_q1q4_dd_gate,
        qpdGw3dCaseCount: qpdGw3d.case_count,
        qpdGw3dEdgeRatio: qpdGw3d.edge_to_center_response_ratio_min ?? qpdGw3d.edge_to_center_response_ratio_max,
        qpdGw3dPhaseSlope: qpdGw3d.phase_x_slope_per_deg_max_abs,
        qpdGw3dPhaseX: qpdGw3d.phase_x_gw,
        qpdGw3dUniformity: qpdGw3d.quadrant_uniformity_gw,
        qpdFidelitySummary: qpdFidelity.summary,
        qpdFidelityOptical: qpdFidelity.optical_generation,
        qpdFidelityElectrical: qpdFidelity.electrical_dd,
        qpdFidelityFull3dDd: qpdFidelity.full_3d_drift_diffusion,
        qpdFidelityProductReady: qpdFidelity.product_accuracy_ready,
        qpdUiPhaseApplicable: qpdUiDd.phase_metric_applicable,
        qpdUiCapabilityInferred: qpdUiCapability.inferred_from_template_parameters,
        qpdComparisonStatus: qpdComparison.status,
        qpdComparisonRowCount: qpdComparison.row_count,
        qpdComparisonBaseline: qpdComparison.baseline_template_id,
        qpdNoShieldGwGate: qpdNoShieldRow.gw_gate,
        qpdNoShieldShieldMode: qpdNoShieldRow.shield_mode,
        qpdNoShieldPhaseX: qpdNoShieldRow.qpd_gw_phase_x,
        qpdPixelPitch: qpdDimension.pixel_pitch_um,
        qpdTemplateFootprintX: qpdDimension.footprint_x_um,
        qpdTemplateFootprintZ: qpdDimension.footprint_z_um,
        qpdPitchVariantPolicy: qpdDimension.pitch_variant_policy,
        quadCrosstalkTemplatePresent: Boolean(byId.quad_2x2_ocl_3x3_neighborhood),
        quadCrosstalkFootprintX: quadCrosstalkDimension.footprint_x_um,
        quadCrosstalkFootprintZ: quadCrosstalkDimension.footprint_z_um,
        quadCrosstalkOclPitchX: quadCrosstalkDimension.effective_ocl_pitch_x_um,
        quadCrosstalkOclPitchZ: quadCrosstalkDimension.effective_ocl_pitch_z_um,
        quadCrosstalkCoverage: quadCrosstalkDimension.crosstalk_kernel_label,
        quadCrosstalkCoverageStatus: quadCrosstalkDimension.crosstalk_kernel_status,
        quadPracticalCrosstalkTemplatePresent: Boolean(byId.quad_2x2_ocl_5x5_crosstalk),
        quadPracticalCrosstalkFootprintX: quadPracticalCrosstalkDimension.footprint_x_um,
        quadPracticalCrosstalkFootprintZ: quadPracticalCrosstalkDimension.footprint_z_um,
        quadPracticalCrosstalkOclPitchX: quadPracticalCrosstalkDimension.effective_ocl_pitch_x_um,
        quadPracticalCrosstalkOclPitchZ: quadPracticalCrosstalkDimension.effective_ocl_pitch_z_um,
        quadPracticalCrosstalkCoverage: quadPracticalCrosstalkDimension.crosstalk_kernel_label,
        quadPracticalCrosstalkCoverageStatus: quadPracticalCrosstalkDimension.crosstalk_kernel_status,
        solverRoleStatus: solverRoleMatrix.status,
        solverRoleRowCount: solverRoleMatrix.rows?.length || 0,
        solverRoleSummary: solverRoleMatrix.summary,
        solverRoleLabels,
        solverRolePrimaryPath: solverRoleMatrix.primary_decision_path || [],
        solverRoleDiagnosticPath: solverRoleMatrix.diagnostic_path || [],
        allowedOverrideFields: tools.allowed_override_fields || [],
        quickOverrideFields: tools.quick_override_fields || [],
        newBaseTemplateFields: tools.requires_new_base_template_fields || [],
        baseTopologyPresetIds: (tools.base_template_topology_presets || []).map((item) => item.id),
        cadToolPitchPolicy: tools.pitch_variant_policy,
        cadToolNotes: tools.notes || [],
        dryRunStatus: dryRun.status,
        dryRunTemplateId: dryRun.template_id,
        dryRunPitch: dryRun.parameters?.pitch_um,
        dryRunSplitMode: dryRun.parameters?.split_mode
      };
    })()`);
    assert(
      'CAD catalog starter set, axis-aware bridge, and QPD outputs are indexed',
      axisAwareCatalog.catalogStatus === 'PASS'
        && axisAwareCatalog.baseTemplateCount === 11
        && axisAwareCatalog.variantCount >= 6
        && axisAwareCatalog.starterStatus === 'PASS'
        && axisAwareCatalog.starterPassCount === 9
        && axisAwareCatalog.starterTemplateCount === 9
        && axisAwareCatalog.dualXStatus === 'PASS'
        && axisAwareCatalog.dualXAxis === 'x'
        && axisAwareCatalog.dualZStatus === 'PASS'
        && axisAwareCatalog.dualZAxis === 'z'
        && axisAwareCatalog.dualZPhaseAxis === 'z'
        && Number.isFinite(Number(axisAwareCatalog.dualZPhaseZ))
        && axisAwareCatalog.dualZGate === 'PASS'
        && axisAwareCatalog.qpdAxisPairStatus === 'PASS'
        && axisAwareCatalog.qpdAxisPairFullGate === 'CHECK'
        && Number.isFinite(Number(axisAwareCatalog.qpdAxisPairMagnitude))
        && Number.isFinite(Number(axisAwareCatalog.qpdAxisPairUniformity))
        && axisAwareCatalog.qpdWeighting3dStatus === 'PASS'
        && axisAwareCatalog.qpdWeighting3dGate === 'PASS'
        && axisAwareCatalog.qpdWeighting3dDdGate === 'CHECK'
        && Number.isFinite(Number(axisAwareCatalog.qpdWeighting3dUniformity))
        && axisAwareCatalog.qpdGw3dStatus === 'PASS'
        && axisAwareCatalog.qpdGw3dGate === 'PASS'
        && axisAwareCatalog.qpdGw3dDdGate === 'CHECK'
        && axisAwareCatalog.qpdGw3dCaseCount >= 4
        && Number.isFinite(Number(axisAwareCatalog.qpdGw3dEdgeRatio))
        && Number.isFinite(Number(axisAwareCatalog.qpdGw3dPhaseSlope))
        && Number.isFinite(Number(axisAwareCatalog.qpdGw3dPhaseX))
        && Number.isFinite(Number(axisAwareCatalog.qpdGw3dUniformity))
        && axisAwareCatalog.qpdFidelitySummary === '3D CAD + hybrid 2D/3D'
        && axisAwareCatalog.qpdFidelityOptical === '3D FDTD volume'
        && axisAwareCatalog.qpdFidelityElectrical === '2D DEVSIM split-response proxy'
        && axisAwareCatalog.qpdFidelityFull3dDd === false
        && axisAwareCatalog.qpdFidelityProductReady === false
        && axisAwareCatalog.qpdUiPhaseApplicable === true
        && axisAwareCatalog.qpdUiCapabilityInferred === true
        && axisAwareCatalog.qpdComparisonStatus === 'CHECK'
        && axisAwareCatalog.qpdComparisonRowCount >= 7
        && axisAwareCatalog.qpdComparisonBaseline === 'qpd_split_pd_2x2'
        && axisAwareCatalog.qpdNoShieldGwGate === 'PASS'
        && axisAwareCatalog.qpdNoShieldShieldMode === 'off'
        && Number.isFinite(Number(axisAwareCatalog.qpdNoShieldPhaseX))
        && near(axisAwareCatalog.qpdPixelPitch, 1.4)
        && near(axisAwareCatalog.qpdTemplateFootprintX, 2.8)
        && near(axisAwareCatalog.qpdTemplateFootprintZ, 2.8)
        && axisAwareCatalog.qpdPitchVariantPolicy === 'conditional_scalar_variant'
        && axisAwareCatalog.quadCrosstalkTemplatePresent === true
        && near(axisAwareCatalog.quadCrosstalkFootprintX, 8.4)
        && near(axisAwareCatalog.quadCrosstalkFootprintZ, 8.4)
        && near(axisAwareCatalog.quadCrosstalkOclPitchX, 2.8)
        && near(axisAwareCatalog.quadCrosstalkOclPitchZ, 2.8)
        && axisAwareCatalog.quadCrosstalkCoverageStatus === 'CHECK'
        && String(axisAwareCatalog.quadCrosstalkCoverage || '').includes('3x3 OCL groups')
        && axisAwareCatalog.quadPracticalCrosstalkTemplatePresent === true
        && near(axisAwareCatalog.quadPracticalCrosstalkFootprintX, 14.0)
        && near(axisAwareCatalog.quadPracticalCrosstalkFootprintZ, 14.0)
        && near(axisAwareCatalog.quadPracticalCrosstalkOclPitchX, 2.8)
        && near(axisAwareCatalog.quadPracticalCrosstalkOclPitchZ, 2.8)
        && axisAwareCatalog.quadPracticalCrosstalkCoverageStatus === 'PASS'
        && String(axisAwareCatalog.quadPracticalCrosstalkCoverage || '').includes('5x5 OCL groups')
        && axisAwareCatalog.solverRoleStatus === 'CHECK'
        && axisAwareCatalog.solverRoleRowCount === 4
        && axisAwareCatalog.solverRoleSummary.includes('FDTD/G*W-led')
        && axisAwareCatalog.solverRoleLabels.includes('FDTD Optical')
        && axisAwareCatalog.solverRoleLabels.includes('DEVSIM DD')
        && axisAwareCatalog.solverRoleLabels.includes('Circuit / Readout')
        && axisAwareCatalog.solverRolePrimaryPath.includes('FDTD Optical')
        && axisAwareCatalog.solverRolePrimaryPath.includes('3D G*W')
        && axisAwareCatalog.solverRoleDiagnosticPath.includes('DEVSIM DD')
        && axisAwareCatalog.allowedOverrideFields.includes('pitch_um')
        && !axisAwareCatalog.allowedOverrideFields.includes('nx')
        && !axisAwareCatalog.allowedOverrideFields.includes('split_mode')
        && !axisAwareCatalog.allowedOverrideFields.includes('shield_mode')
        && axisAwareCatalog.quickOverrideFields.includes('pitch_um')
        && axisAwareCatalog.newBaseTemplateFields.includes('nx')
        && axisAwareCatalog.newBaseTemplateFields.includes('split_mode')
        && axisAwareCatalog.newBaseTemplateFields.includes('shield_mode')
        && axisAwareCatalog.newBaseTemplateFields.includes('ocl_blocks')
        && axisAwareCatalog.baseTopologyPresetIds.includes('qpd_2x2')
        && axisAwareCatalog.baseTopologyPresetIds.includes('quad_2x2_ocl')
        && axisAwareCatalog.baseTopologyPresetIds.includes('quad_2x2_ocl_3x3_neighborhood')
        && axisAwareCatalog.baseTopologyPresetIds.includes('quad_2x2_ocl_5x5_crosstalk')
        && axisAwareCatalog.cadToolPitchPolicy === 'conditional_scalar_variant'
        && axisAwareCatalog.cadToolNotes.some((note) => note.includes('require a new base template'))
        && axisAwareCatalog.dryRunStatus === 'DRY_RUN'
        && axisAwareCatalog.dryRunTemplateId === 'functional_dry_run_qpd_2x2'
        && axisAwareCatalog.dryRunPitch === 1.2
        && axisAwareCatalog.dryRunSplitMode === 'quad',
      axisAwareCatalog
    );
    assert('QPD CAD template selection works', await clickByText('.cad-template-catalog button', 'QPD 2x2'));
    await wait(120);
    const qpdTemplate = await evaluate(`(async () => {
      const step = document.querySelector('a[href*="qpd_split_pd_2x2/model.step"]')?.href;
      const mesh = document.querySelector('a[href*="qpd_split_pd_2x2/model.msh"]')?.href;
      const tcadMesh = document.querySelector('a[href*="qpd_split_pd_2x2/tcad_bridge_2d/split_pixel_2d.msh"]')?.href;
      const ddSummary = document.querySelector('a[href*="qpd_split_pd_2x2/tcad_bridge_2d/devsim_smoke/summary.json"]')?.href;
      const fcstd = document.querySelector('a[href*="qpd_split_pd_2x2/model.FCStd"]')?.href;
      const freecadValidation = document.querySelector('a[href*="freecad_validation_report.json"]')?.href;
      const assumptions = document.querySelector('a[href*="qpd_split_pd_2x2/assumption_ledger.json"]')?.href;
      const geometry = document.querySelector('a[href*="qpd_split_pd_2x2/geometry_import.json"]')?.href;
      const qpdGwSummary = document.querySelector('a[href*="qpd_split_pd_2x2/tcad_qpd_gw_3d/summary.json"]')?.href;
      if (!step || !tcadMesh || !ddSummary || !fcstd || !freecadValidation || !assumptions || !geometry || !qpdGwSummary) return null;
      const response = await fetch(geometry);
      if (!response.ok) return { error: response.status };
      const data = await response.json();
      const fcstdResponse = await fetch(fcstd);
      const freecadResponse = await fetch(freecadValidation);
      const freecadData = freecadResponse.ok ? await freecadResponse.json() : {};
      const assumptionResponse = await fetch(assumptions);
      const assumptionData = assumptionResponse.ok ? await assumptionResponse.json() : {};
      const fcstdBytes = fcstdResponse.ok ? (await fcstdResponse.arrayBuffer()).byteLength : 0;
      const ddResponse = await fetch(ddSummary);
      const ddData = ddResponse.ok ? await ddResponse.json() : {};
      const qpdGwResponse = await fetch(qpdGwSummary);
      const qpdGwData = qpdGwResponse.ok ? await qpdGwResponse.json() : {};
      return {
        hasStep: Boolean(step),
        hasMesh: Boolean(mesh),
        hasTcadMesh: Boolean(tcadMesh),
        hasDdSummary: Boolean(ddSummary),
        hasFcstd: Boolean(fcstd),
        fcstdBytes,
        freecadValidationStatus: freecadData.status,
        freecadTemplateCount: freecadData.template_count,
        freecadQpdStatus: (freecadData.templates || []).find((item) => item.template_id === 'qpd_split_pd_2x2')?.status,
        hasAssumptions: Boolean(assumptions),
        blockerCount: assumptionData.measured_blockers?.length || 0,
        productAccuracyReady: assumptionData.product_accuracy_ready,
        ddNodeCount: ddData.node_count,
        ddPhase: ddData.photo_split_phase_x_proxy,
        qpdGwStatus: qpdGwData.status,
        qpdGwGate: qpdGwData.full_q1q4_gw_gate,
        qpdGwDdGate: qpdGwData.full_q1q4_dd_gate,
        qpdGwCaseCount: qpdGwData.case_count,
        qpdGwIntegrationGrid: qpdGwData.integration_grid,
        qpdGwEdgeRatio: qpdGwData.field_response_summary?.edge_to_center_response_ratio_min ?? qpdGwData.field_response_summary?.edge_to_center_response_ratio_max,
        qpdGwPhaseSlope: qpdGwData.field_response_summary?.phase_x_slope_per_deg_max_abs,
        qpdGwPhaseX: qpdGwData.cases?.[0]?.metrics?.phase_x_gw,
        qpdGwUniformity: qpdGwData.cases?.[0]?.metrics?.quadrant_uniformity_gw,
        templateId: data.cad_template?.template_id,
        cfaColors: [...new Set((data.cfa_polygons?.cells || []).map((cell) => cell.color))],
        oclCount: Object.keys(data.ocl_polygons || {}).length
      };
    })()`);
    assert(
      'QPD CAD template artifacts are reachable',
      qpdTemplate?.hasStep && qpdTemplate?.hasTcadMesh && qpdTemplate?.hasDdSummary && qpdTemplate?.hasFcstd && qpdTemplate.fcstdBytes > 0 && qpdTemplate.freecadValidationStatus === 'PASS' && qpdTemplate.freecadTemplateCount >= 5 && qpdTemplate.freecadQpdStatus === 'PASS' && qpdTemplate?.hasAssumptions && qpdTemplate.blockerCount >= 5 && qpdTemplate.productAccuracyReady === false && qpdTemplate.ddNodeCount > 0 && Number.isFinite(Number(qpdTemplate.ddPhase)) && qpdTemplate.qpdGwStatus === 'PASS' && qpdTemplate.qpdGwGate === 'PASS' && qpdTemplate.qpdGwDdGate === 'CHECK' && qpdTemplate.qpdGwCaseCount >= 4 && qpdTemplate.qpdGwIntegrationGrid === 'generation' && Number.isFinite(Number(qpdTemplate.qpdGwEdgeRatio)) && Number.isFinite(Number(qpdTemplate.qpdGwPhaseSlope)) && Number.isFinite(Number(qpdTemplate.qpdGwPhaseX)) && Number.isFinite(Number(qpdTemplate.qpdGwUniformity)) && qpdTemplate.templateId === 'qpd_split_pd_2x2' && qpdTemplate.cfaColors?.includes('green') && qpdTemplate.oclCount === 1,
      qpdTemplate || {}
    );
    assert('Bayer CAD template selection syncs active preset', await clickByText('.cad-template-catalog button', 'Bayer 1x1'));
    await wait(120);
    const bayerSynced = await snap('bayer synced');
    assert(
      'Bayer template syncs preset and shows generated TCAD smoke state',
      bayerSynced.cadTemplateSummaryText.includes('Selected presetBayer + 1x1 OCL')
        && bayerSynced.cadTemplatePresetMatchWarnings >= 2
        && bayerSynced.cadTemplateSummaryText.includes('TCAD bridgePASS 2D mesh')
        && bayerSynced.cadTemplateSummaryText.includes('TCAD scopegeneric split-PD x-section smoke mesh')
        && bayerSynced.cadTemplateSummaryText.includes('DEVSIM import326 nodes')
        && bayerSynced.cadTemplateSummaryText.includes('DD smokeconnectivity proxy')
        && bayerSynced.cadTemplateSummaryText.includes('Electrical modelproxy-pinned-split-pd'),
      bayerSynced
    );
    assert('Dual-PD Z CAD template selection works', await clickByText('.cad-template-catalog button', 'Dual-PD Z split'));
    await wait(120);
    const dualZDiagnosticText = await evaluate(`document.querySelector('.cad-template-summary')?.textContent || ''`);
    assert(
      'Dual-PD template shows axis projection but keeps QPD-only 3D diagnostics not applicable',
      dualZDiagnosticText.includes('DD smokephase-proxy z')
        && dualZDiagnosticText.includes('Axis pairCHECK')
        && dualZDiagnosticText.includes('QPD 3D Wnot applicable')
        && dualZDiagnosticText.includes('QPD 3D G*Wnot applicable'),
      { dualZDiagnosticText }
    );
    assert('Dual-PD Z template mismatch is visible', (await snap('dual z mismatch')).cadTemplatePresetMatchWarnings >= 1);
    assert('QPD CAD template can be re-selected after Dual-PD diagnostic check', await clickByText('.cad-template-catalog button', 'QPD 2x2'));
    await wait(120);
    assert('QPD template matches selected preset after re-selection', (await snap('qpd match restored')).cadTemplateSummaryText.includes('Preset matchmatches CAD template'));
    const cadAuthorityGate = await evaluate(`(async () => {
      const payload = {
        simulation_request: {
          schema: 'pixel_workbench_simulation_request_v1',
          source: 'functional_test_cad_authority_gate',
          project: { name: 'CAD authority gate test' },
          design: {
            preset_id: 'cad_authority_gate',
            preset_label: 'CAD authority gate',
            cad_template: { template_id: 'qpd_split_pd_2x2' }
          },
          condition: { wavelength_nm: 550, color_channel: 'green', cra_x_deg: 0, cra_z_deg: 0 },
          solver: {
            cad_template_id: 'qpd_split_pd_2x2',
            wavelengths_nm: '550',
            color_channel: 'green',
            cases: 'center:0:0:0:0:0:0',
            stack_overrides: {
              'geometry_um.lens_height': 9.99,
              'geometry_um.cfa_thickness': 9.99,
              'shield.mode': 'off',
              'materials.lens': { n: 1.61, k: 0, measured: false, source: 'functional-test' }
            }
          }
        }
      };
      const response = await fetch('/api/simulation/resolve-request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      const solverCase = data.solver_case || {};
      return {
        ok: response.ok,
        status: response.status,
        authority: solverCase.cad_template?.geometry_authority,
        ignored: solverCase.cad_template?.ignored_stack_override_keys || [],
        lensHeight: solverCase.stack_overrides?.['geometry_um.lens_height'],
        cfaThickness: solverCase.stack_overrides?.['geometry_um.cfa_thickness'],
        shieldMode: solverCase.stack_overrides?.['shield.mode'],
        materialLensN: solverCase.stack_overrides?.['materials.lens']?.n,
        geometryImport: solverCase.ocl_polygons
      };
    })()`);
    assert(
      'CAD authority gate ignores UI/API geometry stack overrides',
      cadAuthorityGate.ok
        && cadAuthorityGate.authority === 'cad_template'
        && cadAuthorityGate.ignored.includes('geometry_um.lens_height')
        && cadAuthorityGate.ignored.includes('geometry_um.cfa_thickness')
        && cadAuthorityGate.ignored.includes('shield.mode')
        && cadAuthorityGate.lensHeight !== 9.99
        && cadAuthorityGate.cfaThickness !== 9.99
        && cadAuthorityGate.shieldMode !== 'off'
        && cadAuthorityGate.materialLensN === 1.61
        && cadAuthorityGate.geometryImport?.includes('qpd_split_pd_2x2/geometry_import.json'),
      cadAuthorityGate
    );
    assert('QPD CAD template becomes active simulation source', (await snap('qpd cad active')).cadTemplateActionText.includes('qpd_split_pd_2x2 is active simulation geometry'));
    assert('FCStd working-copy creation button works', await clickByText('.cad-template-actions button', 'Make FCStd Working Copy'));
    const fcstdWorkingCopy = await waitFor('FCStd working copy path appears', `(() => {
      const value = document.querySelector('.fcstd-roundtrip-panel input')?.value || '';
      return value.includes('/runs/fcstd_working_copies/qpd_split_pd_2x2/') && value.endsWith('.FCStd') ? value : false;
    })()`, 12000);
    assert('FCStd working copy path is shown in UI', Boolean(fcstdWorkingCopy), { fcstdWorkingCopy });

    assert('CAD variant template selection works', await clickByText('.cad-template-catalog button', 'lens height +8%'));
    await wait(120);
    const cadVariantTemplate = await evaluate(`(async () => {
      const variantSource = document.querySelector('a[href*="qpd_split_pd_lens_high_8pct/variant_source.json"]')?.href;
      const step = document.querySelector('a[href*="qpd_split_pd_lens_high_8pct/model.step"]')?.href;
      const detailText = document.querySelector('.cad-template-detail')?.textContent || '';
      if (!variantSource || !step) return null;
      const response = await fetch(variantSource);
      const data = response.ok ? await response.json() : {};
      return {
        hasVariantSource: Boolean(variantSource),
        hasStep: Boolean(step),
        baseTemplateId: data.base_template_id,
        lensHeight: data.parameter_overrides?.lens_height_um,
        detailText
      };
    })()`);
    assert(
      'CAD variant source artifact is reachable',
      cadVariantTemplate?.hasVariantSource
        && cadVariantTemplate?.hasStep
        && cadVariantTemplate.baseTemplateId === 'qpd_split_pd_2x2'
        && cadVariantTemplate.lensHeight === 0.71
        && cadVariantTemplate.detailText.includes('of qpd_split_pd_2x2'),
      cadVariantTemplate || {}
    );
    assert('base QPD CAD template can be re-selected', await clickByText('.cad-template-catalog button', 'QPD 2x2'));
    await wait(100);

    assert('example design button works', await clickByText('.example-buttons button', 'Example A'));
    await wait(100);
    assert('example action logged', (await snap('example A')).toast.includes('Example A'));

    assert('Nona preset button works', await clickByText('.preset-card', 'Nona 3x3'));
    await wait(100);
    assert('Nona preset selected', (await snap('preset nona')).activePreset.includes('Nona'));

    assert('layer toggle works', await clickSelector('.layer-toggle input', 1));
    await wait(80);
    assert('layer action logged', (await snap('toggle OCL')).toast.includes('Layer OCL'));

    assert('highlight coupling button works', await clickByText('.primary-button', 'Highlight'));
    await wait(100);
    assert('coupled cells highlighted', (await snap('highlight')).coupled > 0);

    assert('flow detail tab button works', await clickByText('.flow-step', 'Detail Tabs'));
    await wait(120);
    assert('ML/OCL view active', (await snap('flow detail tabs')).h1 === 'ML / OCL');

    assert('OCL class button works', await clickByText('.class-row', 'OCL_3x3_Nona'));
    await wait(90);
    assert('OCL class selected', (await snap('select OCL class')).activeClass === 'OCL_3x3_Nona');

    assert('OCL model select works', await changeSelect('.parameter-panel .select-row select', 'Surface map'));
    await wait(80);
    assert('OCL model selected', (await snap('select OCL model')).activeOclModel === 'Surface map');

    assert('OCL decrement button works', await clickSelector('.parameter-row button', 0));
    await wait(80);
    assert('OCL decrement logged', (await snap('decrement OCL')).toast.includes('Radius decreased'));

    assert('OCL CFA mode select works', await changeSelect('.detail-stage select', 'Nona'));
    await wait(80);
    assert('OCL CFA mode logged', (await snap('OCL CFA mode')).toast.includes('Nona'));

    assert('CFA nav works', await clickByText('.nav-entry', 'CFA'));
    await wait(120);
    assert('CFA thickness input works', await inputValue('.edit-row input', '0.84', 0));
    assert('CFA aperture model select works', await changeSelect('.parameter-panel .select-row select', 'Inset polygon', 0));
    await wait(80);
    assert('CFA edge inset input works', await inputValue('.edit-row input', '0.120', 1));
    assert('CFA edge skew input works', await inputValue('.edit-row input', '0.025', 2));
    assert('CFA red X shift input works', await inputValue('.edit-row input', '0.020', 3));
    assert('CFA red Z shift input works', await inputValue('.edit-row input', '0.004', 4));
    assert('CFA green X shift input works', await inputValue('.edit-row input', '-0.010', 5));
    assert('CFA green Z shift input works', await inputValue('.edit-row input', '-0.006', 6));
    assert('CFA blue X shift input works', await inputValue('.edit-row input', '0.005', 7));
    assert('CFA blue Z shift input works', await inputValue('.edit-row input', '0.002', 8));
    await wait(80);
    const cfa = await snap('CFA thickness');
    assert('CFA thickness state updated', cfa.cfaThickness === '0.84' && cfa.log.some((item) => item.includes('bShiftZ')), cfa);

    assert('PDAF nav works', await clickByText('.nav-entry', 'PDAF / Shield'));
    await wait(120);
    assert('PDAF mode button works', await clickByText('.mode-card', 'Quad PDAF'));
    await wait(80);
    assert('PDAF mode selected', (await snap('PDAF mode')).activeMode === 'Quad PDAF / QPD');
    assert('PDAF layer toggle works', await clickSelector('.pdaf-stage .layer-toggle input', 3));
    await wait(80);
    assert('PDAF layer logged', (await snap('PDAF layer')).toast.includes('pdaf layer') || (await snap('PDAF layer retry')).toast.includes('PDAF layer'));

    assert('Pattern Response nav works', await clickByText('.nav-entry', 'Pattern Response'));
    await wait(120);
    assert('response tab button works', await clickByText('.response-tabs button', 'AF Signal Map'));
    await wait(80);
    assert('response tab selected', (await snap('response tab')).responseTab === 'AF Signal Map');
    assert('response wavelength select works', await changeSelect('.display-options .select-row:nth-of-type(2) select', '650 nm'));
    await wait(80);
    assert('response export button works', await clickByText('.secondary-button', 'Export Map'));
    await wait(80);
    assert('response export logged', (await snap('export map')).toast.includes('Exported response map'));

    assert('Readiness nav works', await clickByText('.nav-entry', 'Readiness'));
    await waitFor(
      'Readiness manifest loads',
      `(() => {
        const text = document.querySelector('.readiness-page')?.textContent || '';
        return text.includes('RESEARCH_READY_NOT_PRODUCT') && text.includes('Product Blockers');
      })()`,
      60000
    );
    const readiness = await snap('readiness');
    assert(
      'Readiness view renders quantitative evidence and blocker state',
      readiness.h1 === 'Readiness'
        && readiness.readinessStatus === 'RESEARCH_READY_NOT_PRODUCT'
        && readiness.readinessKpis >= 6
        && readiness.readinessBlockerRows >= 1
        && readiness.readinessEvidenceRows >= 1
        && readiness.readinessArtifactLinks >= 4
        && readiness.readinessText.includes('Product LUT')
        && readiness.readinessText.includes('Research readiness is not product LUT readiness')
        && readiness.solverRoleRows >= 4
        && readiness.solverRoleText.includes('FDTD Optical')
        && readiness.solverRoleText.includes('DEVSIM DD')
        && readiness.solverRoleText.includes('Circuit / Readout'),
      readiness
    );

    assert('Variants nav works', await clickByText('.nav-entry', 'Variants'));
    await wait(120);
    assert('optimization checkbox works', await clickSelector('.check-row input', 0));
    await wait(80);
    assert('optimization toggle logged', (await snap('optimization')).toast.includes('Optimization variable'));
    assert('candidate card works', await clickSelector('.candidate-card', 1));
    await wait(80);
    assert('candidate selected', (await snap('candidate')).selectedCandidate.length > 0);
    assert('export report button works', await clickByText('.export-grid button', 'PDF Report'));
    await wait(80);
    assert('export report logged', (await snap('export report')).toast.includes('Prepared export'));

    assert('Test Suite nav works', await clickByText('.nav-entry', 'Test Suite'));
    await wait(180);
    const suiteInitial = await snap('test suite initial');
    assert('Test Suite catalog renders', suiteInitial.h1 === 'Test Suite' && suiteInitial.suiteCards >= 5, suiteInitial);
    assert('Test Suite tiers render', suiteInitial.suiteTierButtons >= 2, suiteInitial);
    assert('Test Suite case matrix renders', await evaluate(`document.querySelectorAll('.suite-matrix-list > label').length >= 2`));
    assert('Test Suite case selection renders', await evaluate(`document.querySelectorAll('.suite-matrix-list input[type="checkbox"]').length >= 2`));
    assert('CAD variant comparison suite renders', await clickByText('.suite-catalog button', 'CAD Variant Comparison'));
    await wait(120);
    const cadVariantSuite = await evaluate(`({
      activeHeading: document.querySelector('.suite-setup h2')?.textContent || '',
      caseText: [...document.querySelectorAll('.suite-matrix-list > label')].map((node) => node.textContent).join('\\n'),
      selectedCases: document.querySelectorAll('.suite-matrix-list input[type="checkbox"]:checked').length
    })`);
    assert(
      'CAD variant comparison suite includes base and variants',
      cadVariantSuite.activeHeading === 'CAD Variant Comparison'
        && cadVariantSuite.caseText.includes('Base:')
        && cadVariantSuite.caseText.includes('lens height +8%')
        && cadVariantSuite.caseText.includes('DTI width 80 nm')
        && cadVariantSuite.selectedCases >= 3,
      cadVariantSuite
    );

    if (runGdsSuite) {
      assert('Mixed OCL suite selection works', await clickByText('.suite-catalog button', 'Mixed OCL Boundary Risk'));
      await wait(120);
      assert('Clear case selection works', await clickByText('.suite-tier-row button', 'Clear'));
      await wait(80);
      assert('all cases cleared', (await snap('gds suite cleared')).suiteSelectedCases === 0);
      assert('GDS pipeline case selection works', await evaluate(`(() => {
        const label = [...document.querySelectorAll('.suite-matrix-list > label')].find((node) => node.textContent.includes('GDS import -> OCL/CFA LUT pipeline'));
        const input = label?.querySelector('input[type="checkbox"]');
        if (!input) return false;
        input.click();
        return input.checked;
      })()`));
      await wait(80);
      assert('single GDS case selected', (await snap('gds case selected')).suiteSelectedCases === 1);
      assert('selected GDS case starts suite job', await clickSelector('.suite-tier-row .primary-button'));
      await wait(400);
      await waitFor(
        'GDS suite job completion',
        `(() => document.querySelectorAll('.suite-case-table > div').length === 1 && document.querySelectorAll('.suite-cad-evidence').length === 1 && !!document.querySelector('.suite-result .status-pill')?.textContent)()`,
        solverTimeoutMs
      );
      const gdsState = await snap('gds suite completed');
      assert('GDS CAD evidence panel renders', gdsState.suiteCadEvidence === 1 && gdsState.suiteCadPreviews === 1 && gdsState.suiteCadArtifactLinks >= 5, gdsState);
      assert('GDS suite completed without UI error', !gdsState.suiteError, gdsState);
      const gdsReport = await evaluate(`(async () => {
        const href = document.querySelector('a[href*="gds_import_report.json"]')?.href;
        if (!href) return null;
        const response = await fetch(href);
        if (!response.ok) return { error: response.status };
        const data = await response.json();
        return {
          status: data.validation_status,
          ocl: data.matched_ocl_polygon_count,
          cfa: data.matched_cfa_polygon_count,
          warnings: data.warnings?.length || 0
        };
      })()`);
      assert(
        'GDS report artifact is reachable and passes validation',
        gdsReport?.status === 'PASS' && gdsReport.ocl >= 1 && gdsReport.cfa >= 1 && gdsReport.warnings === 0,
        gdsReport || {}
      );
    }

    if (runSolver) {
      assert('solver test example reload works', await clickByText('.example-buttons button', 'Example A'));
      await wait(120);
      assert('Run FDTD Detail button starts active design solver job', await clickByText('.run-button', 'Run FDTD Detail'));
      await wait(300);
      assert('FDTD detail route active', (await snap('run fdtd')).h1 === 'FDTD Detail');
      await waitFor(
        'solver job completion',
        `(() => {
          const status = document.querySelector('.simulation-panel .status-pill')?.textContent || '';
          if (status === 'failed') return 'failed';
          return status === 'completed';
        })()`,
        solverTimeoutMs
      );
      const solverState = await snap('solver completed');
      assert('solver-backed KPI cards render', solverState.simulationStatus === 'completed' && solverState.simulationKpis >= 6, solverState);
      assert('solver artifacts render', solverState.simulationImages >= 1, solverState);
      assert('active design request artifacts render', solverState.requestLinks >= 1 && solverState.solverCaseLinks >= 1, solverState);
      assert('persistent KPI summary artifact renders', solverState.kpiSummaryLinks >= 1, solverState);
      assert('active design request includes CFA pattern and x/z shifts', solverState.activeRequestPreview.includes('CFA pattern') && solverState.activeRequestPreview.includes('x/z R+0.020/+0.004') && solverState.activeRequestPreview.includes('G-0.010/-0.006'), solverState);
      assert('active design request includes active CAD source', solverState.activeRequestPreview.includes('CAD source') && solverState.activeRequestPreview.includes('qpd_split_pd_2x2'), solverState);
      assert('active design request uses CAD template CFA/OCL geometry', solverState.activeRequestPreview.includes('CFA geometry') && solverState.activeRequestPreview.includes('CAD template cells') && solverState.activeRequestPreview.includes('OCL model') && solverState.activeRequestPreview.includes('CAD template footprint'), solverState);
      assert('active design request includes split collection mode', solverState.activeRequestPreview.includes('Collection') && solverState.activeRequestPreview.includes('split-pd'), solverState);
      const solverCase = await evaluate(`(async () => {
        const href = document.querySelector('a[href*="solver_case.json"]')?.href;
        if (!href) return null;
        const response = await fetch(href);
        if (!response.ok) return { error: response.status };
        const data = await response.json();
        return {
          mode: data.mode,
          collectionMode: data.collection_mode,
          targetLensId: data.target_lens_id,
          cadTemplateId: data.cad_template?.template_id,
          oclPolygons: data.ocl_polygons,
          cfaPolygons: data.cfa_polygons,
          hasSurfaceMap: Boolean(data.ocl_surface_map),
          hasCfaPolygons: Boolean(data.cfa_polygons),
          red: data.cfa_shifts_um?.red,
          green: data.cfa_shifts_um?.green,
          blue: data.cfa_shifts_um?.blue
        };
      })()`);
      const persistedKpi = await evaluate(`(async () => {
        const href = document.querySelector('a[href*="kpi_summary.json"]')?.href;
        if (!href) return null;
        const response = await fetch(href);
        if (!response.ok) return { error: response.status };
        const data = await response.json();
        return {
          schema: data.schema,
          status: data.status,
          templateId: data.cad_template?.template_id,
          authority: data.cad_template?.geometry_authority,
          kpiSummaryArtifact: data.artifacts?.kpi_summary
        };
      })()`);
      assert(
        'solver case includes CAD template geometry import and CFA x/z shifts',
        solverCase?.mode === 'ocl-layout'
          && solverCase.collectionMode === 'split-pd'
          && solverCase.targetLensId === 'qpd_2x2_ocl'
          && solverCase.cadTemplateId === 'qpd_split_pd_2x2'
          && solverCase.oclPolygons?.includes('qpd_split_pd_2x2/geometry_import.json')
          && solverCase.cfaPolygons?.includes('qpd_split_pd_2x2/geometry_import.json')
          && !solverCase.hasSurfaceMap
          && solverCase.hasCfaPolygons
          && solverCase.red?.x === 0.02
          && solverCase.red?.z === 0.004
          && solverCase.green?.x === -0.01
          && solverCase.green?.z === -0.006
          && solverCase.blue?.x === 0.005
          && solverCase.blue?.z === 0.002,
        solverCase || {}
      );
      assert(
        'persistent KPI summary includes CAD authority metadata',
        persistedKpi?.schema === 'pixel_workbench_solver_kpi_v1'
          && persistedKpi.templateId === 'qpd_split_pd_2x2'
          && persistedKpi.authority === 'cad_template'
          && persistedKpi.kpiSummaryArtifact?.includes('kpi_summary.json'),
        persistedKpi || {}
      );
      assert('solver completed without UI error', !solverState.simulationError, solverState);

      assert('Test Suite nav works after solver run', await clickByText('.nav-entry', 'Test Suite'));
      await wait(150);
      assert('Crosstalk suite selection works', await clickByText('.suite-catalog button', 'Crosstalk Kernel Practical'));
      await wait(100);
      assert('Run suite button starts suite job', await clickSelector('.suite-tier-row .primary-button'));
      await wait(400);
      await waitFor(
        'suite job completion',
        `(() => document.querySelectorAll('.suite-case-table > div').length >= 2 && !!document.querySelector('.suite-result .status-pill')?.textContent)()`,
        solverTimeoutMs
      );
      const suiteState = await snap('suite completed');
      assert('suite KPI cards render', suiteState.suiteKpis >= 5 && suiteState.suiteCases >= 2, suiteState);
      assert('suite charts render', suiteState.suiteCharts >= 1, suiteState);
      assert('suite persistent artifacts render', suiteState.suiteResultArtifactLinks >= 2, suiteState);
      assert('suite case result artifacts render', suiteState.suiteCaseResultLinks >= 2, suiteState);
      assert('suite case provenance artifacts render', suiteState.suiteCaseInputLinks >= 2 && suiteState.suiteCaseCommandLinks >= 1, suiteState);
      assert('suite replay buttons render', suiteState.suiteReplayButtons >= 1, suiteState);
      const suiteArtifacts = await evaluate(`(async () => {
        const resultHref = document.querySelector('a[href*="suite_result.json"]')?.href;
        const summaryHref = document.querySelector('a[href*="workbench_suite_summary.json"]')?.href;
        const caseHref = document.querySelector('a[href*="case_result.json"]')?.href;
        const caseInputHref = document.querySelector('a[href*="case_input.json"]')?.href;
        const caseCommandHref = document.querySelector('a[href*="case_command.json"]')?.href;
        if (!resultHref || !summaryHref) return null;
        const [resultResponse, summaryResponse, caseResponse, inputResponse, commandResponse] = await Promise.all([
          fetch(resultHref),
          fetch(summaryHref),
          caseHref ? fetch(caseHref) : Promise.resolve(null),
          caseInputHref ? fetch(caseInputHref) : Promise.resolve(null),
          caseCommandHref ? fetch(caseCommandHref) : Promise.resolve(null)
        ]);
        if (!resultResponse.ok || !summaryResponse.ok || (caseResponse && !caseResponse.ok) || (inputResponse && !inputResponse.ok) || (commandResponse && !commandResponse.ok)) return {
          resultStatus: resultResponse.status,
          summaryStatus: summaryResponse.status,
          caseStatus: caseResponse?.status,
          inputStatus: inputResponse?.status,
          commandStatus: commandResponse?.status
        };
        const result = await resultResponse.json();
        const summary = await summaryResponse.json();
        const caseResult = caseResponse ? await caseResponse.json() : {};
        const caseInput = inputResponse ? await inputResponse.json() : {};
        const caseCommand = commandResponse ? await commandResponse.json() : {};
        const cadCaseId = 'cad_quad_2x2_ocl_5x5_crosstalk_fdtd';
        const cadCase = (result.cases || []).find((item) => item.id === cadCaseId) || {};
        const cadCaseArtifact = (summary.case_artifacts || []).find((item) => item.case_id === cadCaseId) || {};
        const cadCommandResponse = cadCaseArtifact.case_command_url ? await fetch(cadCaseArtifact.case_command_url) : null;
        const cadCommand = cadCommandResponse?.ok ? await cadCommandResponse.json() : {};
        const cadCommandText = Array.isArray(cadCommand.command) ? cadCommand.command.join(' ') : '';
        return {
          resultSchema: result.schema,
          summarySchema: summary.schema,
          caseSchema: caseResult.schema,
          inputSchema: caseInput.schema,
          commandSchema: caseCommand.schema,
          resultSelf: result.artifacts?.suite_result,
          summarySelf: result.artifacts?.suite_summary,
          suiteId: summary.suite_id,
          caseCount: summary.case_count,
          caseArtifacts: summary.case_artifacts?.length,
          caseResultArtifacts: (summary.case_artifacts || []).filter((item) => item.case_result_url).length,
          caseInputArtifacts: (summary.case_artifacts || []).filter((item) => item.case_input_url).length,
          caseCommandArtifacts: (summary.case_artifacts || []).filter((item) => item.case_command_url).length,
          caseResultSelf: caseResult.artifacts?.case_result,
          caseInputSelf: caseResult.artifacts?.case_input,
          caseCommandSelf: caseCommand.artifacts?.case_command,
          commandLength: Array.isArray(caseCommand.command) ? caseCommand.command.length : 0,
          cadCaseRunner: cadCase.runner,
          cadCaseStatus: cadCase.status,
          cadCaseTemplateId: cadCase.kpi?.cad_template?.template_id,
          cadCaseCrosstalkCoverage: cadCase.kpi?.cad_template?.crosstalk_kernel_status,
          cadCaseCfaPolicy: cadCase.kpi?.cad_template?.cfa_geometry_policy,
          cadCaseRowCount: cadCase.kpi?.row_count,
          cadCaseOutputRowCount: cadCase.kpi?.output_row_count,
          cadCaseArtifactCommand: cadCaseArtifact.case_command_url,
          cadCommandHasCrosstalkScript: cadCommandText.includes('meep_crosstalk_kernel.py'),
          cadCommandHasLayout10: cadCommandText.includes('--layout-nx 10') && cadCommandText.includes('--layout-nz 10'),
          cadCommandHasTargetLens: cadCommandText.includes('--target-lens-id quad_4_4'),
          cadCommandUsesQuadCfaPattern: cadCommandText.includes('--cfa-pattern quad'),
          cadCommandAvoidsLargeCfaImport: !cadCommandText.includes('--cfa-polygons')
        };
      })()`);
      assert(
        'suite persistent artifacts include result and summary metadata',
        suiteArtifacts?.resultSchema === 'pixel_workbench_suite_result_v1'
          && suiteArtifacts.summarySchema === 'pixel_workbench_suite_summary_v1'
          && suiteArtifacts.caseSchema
          && suiteArtifacts.inputSchema === 'pixel_workbench_suite_case_input_v1'
          && suiteArtifacts.commandSchema === 'pixel_workbench_suite_case_command_v1'
          && suiteArtifacts.resultSelf?.includes('suite_result.json')
          && suiteArtifacts.summarySelf?.includes('workbench_suite_summary.json')
          && suiteArtifacts.suiteId === 'crosstalk_kernel_practical'
          && suiteArtifacts.caseCount >= 3
          && suiteArtifacts.caseArtifacts >= 3
          && suiteArtifacts.caseResultArtifacts >= 3
          && suiteArtifacts.caseInputArtifacts >= 3
          && suiteArtifacts.caseCommandArtifacts >= 1
          && suiteArtifacts.caseResultSelf?.includes('case_result.json')
          && suiteArtifacts.caseInputSelf?.includes('case_input.json')
          && suiteArtifacts.caseCommandSelf?.includes('case_command.json')
          && suiteArtifacts.commandLength >= 2
          && suiteArtifacts.cadCaseRunner === 'cad_template_crosstalk'
          && suiteArtifacts.cadCaseStatus === 'completed'
          && suiteArtifacts.cadCaseTemplateId === 'quad_2x2_ocl_5x5_crosstalk'
          && suiteArtifacts.cadCaseCrosstalkCoverage === 'PASS'
          && suiteArtifacts.cadCaseCfaPolicy === 'procedural_cfa_pattern_for_large_kernel'
          && suiteArtifacts.cadCaseRowCount >= 1
          && suiteArtifacts.cadCaseOutputRowCount >= 25
          && suiteArtifacts.cadCaseArtifactCommand?.includes('case_command.json')
          && suiteArtifacts.cadCommandHasCrosstalkScript
          && suiteArtifacts.cadCommandHasLayout10
          && suiteArtifacts.cadCommandHasTargetLens
          && suiteArtifacts.cadCommandUsesQuadCfaPattern
          && suiteArtifacts.cadCommandAvoidsLargeCfaImport,
        suiteArtifacts || {}
      );
      assert('suite replay button starts replay+compare', await clickByText('.suite-case-artifacts button', 'Replay + Compare'));
      await waitFor(
        'suite replay completion',
        `(() => {
          const text = document.querySelector('.suite-replay-result')?.textContent || '';
          return text.includes('Replay PASS') && text.includes('comparison PASS');
        })()`,
        solverTimeoutMs
      );
      const replayState = await snap('suite replay completed');
      assert(
        'suite replay result links render',
        replayState.suiteReplayText.includes('Replay PASS')
          && replayState.suiteReplayText.includes('comparison PASS')
          && replayState.suiteReplayManifestLinks >= 1
          && replayState.suiteReplayComparisonLinks >= 1,
        replayState
      );
      const replayArtifacts = await evaluate(`(async () => {
        const manifestHref = document.querySelector('a[href*="replay_manifest.json"]')?.href;
        const comparisonHref = document.querySelector('a[href*="replay_comparison.json"]')?.href;
        if (!manifestHref || !comparisonHref) return null;
        const [manifestResponse, comparisonResponse] = await Promise.all([fetch(manifestHref), fetch(comparisonHref)]);
        if (!manifestResponse.ok || !comparisonResponse.ok) return { manifestStatus: manifestResponse.status, comparisonStatus: comparisonResponse.status };
        const manifest = await manifestResponse.json();
        const comparison = await comparisonResponse.json();
        return {
          manifestSchema: manifest.schema,
          manifestStatus: manifest.status,
          comparisonSchema: comparison.schema,
          comparisonStatus: comparison.status,
          failureCount: comparison.failure_count
        };
      })()`);
      assert(
        'suite replay artifacts validate',
        replayArtifacts?.manifestSchema === 'pixel_workbench_case_command_replay_v1'
          && replayArtifacts.manifestStatus === 'PASS'
          && replayArtifacts.comparisonSchema === 'pixel_workbench_replay_comparison_v1'
          && replayArtifacts.comparisonStatus === 'PASS'
          && replayArtifacts.failureCount === 0,
        replayArtifacts || {}
      );
      assert('suite completed without UI error', !suiteState.suiteError, suiteState);
    } else {
      assert('Run FDTD Detail button exists', await evaluate(`!!document.querySelector('.run-button')`));
      assert('FDTD detail nav works for CAD-first request preview', await clickByText('.nav-entry', 'FDTD Detail'));
      await wait(120);
      const cadFirstState = await snap('cad-first active request');
      const starterCadSources = ['qpd_split_pd_2x2', 'nona_3x3_ocl', 'quad_2x2_ocl', 'bayer_1x1_3x3', 'pdaf_dual_x_shield_pair'];
      assert(
        'active design uses a preset-matched CAD template source',
        cadFirstState.activeRequestPreview.includes('CAD source')
          && starterCadSources.some((templateId) => cadFirstState.activeRequestPreview.includes(templateId))
          && cadFirstState.activeRequestPreview.includes('Geometry authority')
          && cadFirstState.activeRequestPreview.includes('CAD template'),
        cadFirstState
      );
    }

    const image = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    writeFileSync(screenshot, Buffer.from(image.data, 'base64'));
    ws.close();

    const failed = results.filter((item) => item.status !== 'PASS');
    const report = {
      schema: 'pixel_workbench_ux_functional_test_v1',
      url,
      solver_enabled: runSolver,
      gds_suite_enabled: runGdsSuite,
      status: failed.length ? 'FAIL' : 'PASS',
      checked_at: new Date().toISOString(),
      results,
      exceptions,
      states,
      screenshot
    };
    writeFileSync(out, JSON.stringify(report, null, 2));
    if (failed.length || exceptions.length) {
      throw new Error(`Functional test failed: ${failed.length} assertions, ${exceptions.length} runtime exceptions`);
    }
    console.log(JSON.stringify({ status: 'PASS', assertions: results.length, report: out, screenshot }, null, 2));
  } finally {
    chrome.kill('SIGTERM');
    try {
      rmSync(userDir, { recursive: true, force: true });
    } catch {
      // Best effort cleanup only.
    }
  }
}

run().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
