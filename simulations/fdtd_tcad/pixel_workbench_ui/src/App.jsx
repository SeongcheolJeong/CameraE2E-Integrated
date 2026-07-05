import { Fragment, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Aperture,
  BarChart3,
  Bookmark,
  Box,
  Camera,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Columns3,
  Cpu,
  Crosshair,
  Download,
  Eye,
  FileText,
  FolderOpen,
  Focus,
  Gauge,
  Grid2X2,
  Layers3,
  LayoutDashboard,
  LineChart,
  Microscope,
  PackageCheck,
  PanelsTopLeft,
  Play,
  Plus,
  Save,
  Settings,
  Shield,
  SlidersHorizontal,
  Target,
  TriangleAlert,
  Waves,
  Workflow,
  Zap
} from 'lucide-react';

const NAV_SECTIONS = [
  {
    label: 'DESIGN',
    items: [
      ['Template', PanelsTopLeft],
      ['Pattern Composer', Grid2X2],
      ['Stack Geometry', Box],
      ['ML / OCL', Aperture],
      ['CFA', Layers3],
      ['PDAF / Shield', Shield],
      ['DTI / Isolation', Crosshair],
      ['Materials', SlidersHorizontal]
    ]
  },
  {
    label: 'CONDITION',
    items: [
      ['Illumination / CRA', Waves],
      ['Sensor Position', Target],
      ['Readout Mode', Columns3]
    ]
  },
  {
    label: 'SIMULATE',
    items: [
      ['Fast Preview', Play],
      ['Test Suite', Gauge],
      ['FDTD Detail', Activity],
      ['Optical + Electrical', Cpu]
    ]
  },
  {
    label: 'ANALYZE',
    items: [
      ['Field Viewer', Eye],
      ['Pattern Response', BarChart3],
      ['AF Response', Focus],
      ['Readiness', PackageCheck],
      ['KPI Dashboard', LayoutDashboard]
    ]
  },
  {
    label: 'COMPARE',
    items: [
      ['Variants', Workflow],
      ['Sweep / Optimization', LineChart],
      ['Tolerance', Gauge],
      ['Report', FileText]
    ]
  }
];

const PRESETS = [
  {
    id: 'bayer_1x1',
    number: 1,
    label: 'Bayer + 1x1 OCL',
    short: 'Bayer',
    cfa: 'RGGB Bayer',
    ocl: '1x1',
    af: 'None',
    readout: 'Full resolution',
    period: 2,
    group: 1,
    complexity: 'Low'
  },
  {
    id: 'quad_2x2',
    number: 2,
    label: 'Quad Bayer + 2x2 OCL',
    short: 'Quad',
    cfa: '2x2 same-color block',
    ocl: '2x2',
    af: 'None',
    readout: '2x2 binning / remosaic',
    period: 4,
    group: 2,
    complexity: 'Medium'
  },
  {
    id: 'quad_qpd',
    number: 3,
    label: 'Quad Bayer + QPD',
    short: 'QPD',
    cfa: '2x2 same-color block',
    ocl: '2x2',
    af: 'QPD',
    readout: 'QPD + image remosaic',
    period: 4,
    group: 2,
    complexity: 'Medium'
  },
  {
    id: 'nona_3x3',
    number: 4,
    label: 'Nona 3x3 + 3x3 OCL',
    short: 'Nona',
    cfa: '3x3 same-color block',
    ocl: '3x3',
    af: 'None',
    readout: '3x3 binning / remosaic',
    period: 6,
    group: 3,
    complexity: 'High'
  },
  {
    id: 'rgbir_group',
    number: 5,
    label: 'RGB-IR + Grouped OCL',
    short: 'RGB-IR',
    cfa: 'RGB-IR',
    ocl: '2x2',
    af: 'None',
    readout: 'RGB + IR split',
    period: 4,
    group: 2,
    complexity: 'Medium'
  },
  {
    id: 'sparse_pdaf',
    number: 6,
    label: 'Sparse Half-shield PDAF',
    short: 'PDAF',
    cfa: 'Bayer green pair',
    ocl: '1x1 / mirror',
    af: 'AF-L / AF-R',
    readout: 'Sparse AF interpolation',
    period: 8,
    group: 1,
    complexity: 'Medium'
  },
  {
    id: 'optical_black',
    number: 7,
    label: 'Optical Black / Shield Row',
    short: 'OB',
    cfa: 'Shielded row',
    ocl: 'Off above OB',
    af: 'None',
    readout: 'Dark reference',
    period: 8,
    group: 1,
    complexity: 'Low'
  },
  {
    id: 'custom',
    number: 8,
    label: 'Custom Supercell',
    short: 'Custom',
    cfa: 'User-defined',
    ocl: 'Custom polygon',
    af: 'Custom',
    readout: 'Custom pipeline',
    period: 12,
    group: 2,
    complexity: 'Open'
  }
];

const DEFAULT_CAD_TEMPLATE_ID = 'qpd_split_pd_2x2';
const PRESET_CAD_TEMPLATE_MAP = {
  bayer_1x1: 'bayer_1x1_3x3',
  quad_2x2: 'quad_2x2_ocl',
  quad_qpd: 'qpd_split_pd_2x2',
  nona_3x3: 'nona_3x3_ocl',
  sparse_pdaf: 'pdaf_dual_x_shield_pair'
};

function cadTemplateIdForPreset(preset) {
  return PRESET_CAD_TEMPLATE_MAP[preset?.id] || '';
}

function presetForCadTemplate(template) {
  const templateId = template?.variant_of || template?.template_id || '';
  const presetId = Object.entries(PRESET_CAD_TEMPLATE_MAP).find(([, cadTemplateId]) => cadTemplateId === templateId)?.[0];
  return presetId ? presetById(presetId) : null;
}

const OCL_CLASSES = [
  ['OCL_1x1_Image', '1 x 1', 'Square', '#ef4444'],
  ['OCL_2x2_Quad', '2 x 2', 'Rounded', '#22d3ee'],
  ['OCL_2x2_QPD', '2 x 2', 'Rounded', '#a855f7'],
  ['OCL_3x3_Nona', '3 x 3', 'Rounded', '#f6c445'],
  ['OCL_PDAF_Left', '1 x 2', 'Rect', '#c084fc'],
  ['OCL_PDAF_Right', '1 x 2', 'Rect', '#c084fc'],
  ['OCL_Edge_CRA', '2 x 2', 'Edge', '#38bdf8'],
  ['OCL_Corner_CRA', '2 x 2', 'Corner', '#14b8a6']
];

const PDAF_MODES = [
  ['Half-shield L/R', 'Horizontal split', 'AF-L / AF-R'],
  ['Half-shield T/B', 'Vertical split', 'AF-T / AF-B'],
  ['Cross PDAF', 'Cross split', 'L/R + T/B'],
  ['Quad PDAF / QPD', '2x2 phase pixels', 'Q1-Q4'],
  ['Dual Pixel', 'Full / fractional', 'Split PD'],
  ['Optical Black', 'OB / black pixel', 'OB'],
  ['Calibration Shield', 'Shield reference', 'SH'],
  ['Custom Mask', 'User-defined', 'Custom']
];

const RESPONSE_TABS = ['QE Map', 'OCL Focus Map', 'AF Signal Map', 'Crosstalk Map', 'Binning Uniformity', 'Remosaic Risk'];

const TOPBAR_ROUTES = {
  Project: 'Template',
  Pixel: 'Pattern Composer',
  Experiment: 'Fast Preview',
  Compare: 'Variants',
  Report: 'Report'
};

const FLOW_ROUTES = {
  Template: 'Template',
  'Pattern Composer': 'Pattern Composer',
  'Detail Tabs': 'ML / OCL',
  Simulation: 'Fast Preview',
  'Pattern Response': 'Pattern Response',
  'Variant Decision': 'Variants'
};

const DEFAULT_OCL_PARAMS = {
  Radius: 1.62,
  Sag: 0.742,
  Diameter: 3.24,
  'Height (Center)': 0.86,
  'X Shift': 0.03,
  'Y Shift': 0.01,
  'Aspheric Coef. (k)': -0.12,
  'Asphere A4': 0.018,
  'Polygon Bias': 0.08,
  'Surface Edge Height': 0.2,
  'Refractive Index': 1.56
};

const OCL_SURFACE_MODELS = ['Spherical cap', 'Asphere sag', 'Polygon aperture', 'Surface map'];

const DEFAULT_CFA_PARAMS = {
  thickness: 0.8,
  apertureModel: 'Full tile',
  edgeInset: 0.12,
  edgeSkew: 0.02,
  gapBackground: 'passivation',
  rShift: 0,
  rShiftZ: 0,
  gShift: 0,
  gShiftZ: 0,
  bShift: 0,
  bShiftZ: 0,
  source: 'proxy table',
  remosaic: 'defined'
};

const DEFAULT_PDAF_PARAMS = {
  maskRatio: 0.5,
  apertureOffset: 0.04,
  pairingRule: 'Same Color',
  oclGrouping: 'Same OCL (2x2)',
  afDensity: 'Medium (12%)',
  metalHeightNm: 450,
  edgeRoundingNm: 30
};

const MIXED_OCL_LAYOUT = {
  name: 'mixed_3x3_2x2_1x1_boundary_5x3',
  nx: 5,
  nz: 3,
  descriptor: 'nona_l:0:0:3:3,quad_r:3:0:2:2,bayer_r0:3:2:1:1,bayer_r1:4:2:1:1'
};

const EXAMPLE_DESIGNS = [
  {
    id: 'example_quad_qpd',
    name: 'Example A: Quad Bayer + 2x2 OCL + QPD',
    presetId: 'quad_qpd',
    active: 'Pattern Composer',
    cfaThickness: 0.8,
    oclClass: 'OCL_2x2_QPD',
    pdafMode: 'Quad PDAF / QPD',
    responseTab: 'AF Signal Map'
  },
  {
    id: 'example_nona_cra',
    name: 'Example B: Nona 3x3 + 3x3 OCL CRA',
    presetId: 'nona_3x3',
    active: 'Pattern Response',
    cfaThickness: 0.82,
    oclClass: 'OCL_3x3_Nona',
    pdafMode: 'Optical Black',
    responseTab: 'Binning Uniformity'
  }
];

const SOLVER_TEST_EXAMPLES = [
  { id: 'ocl2x2_smoke', label: '2x2 OCL Smoke', presetIds: ['quad_2x2'], mode: 'ocl-2x2' },
  { id: 'ocl3x3_smoke', label: '3x3 OCL Smoke', presetIds: ['nona_3x3'], mode: 'ocl-3x3' },
  { id: 'split_pd_quad_smoke', label: 'QPD Split-PD Smoke', presetIds: ['quad_qpd'], mode: 'split-pd-1x1' }
];

function apiCandidates() {
  const configured = window.__PIXEL_WORKBENCH_API_BASE__;
  const candidates = [
    configured,
    '',
    'http://127.0.0.1:8766'
  ].filter((item) => item !== undefined && item !== null);
  return [...new Set(candidates.map((item) => String(item).replace(/\/$/, '')))];
}

async function fetchWorkbenchApi(path, options = {}) {
  const errors = [];
  for (const base of apiCandidates()) {
    try {
      const response = await fetch(`${base}${path}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {})
        }
      });
      if (!response.ok) {
        errors.push(`${base || 'same-origin'} ${response.status}`);
        continue;
      }
      return { base, data: await response.json() };
    } catch (error) {
      errors.push(`${base || 'same-origin'} ${error.message}`);
    }
  }
  throw new Error(errors.join(' | ') || 'Workbench backend unavailable');
}

function exampleIdForPreset(preset) {
  if (preset.id === 'nona_3x3') return 'ocl3x3_smoke';
  if (preset.id === 'quad_qpd') return 'split_pd_quad_smoke';
  if (preset.ocl === '3x3') return 'ocl3x3_smoke';
  if (preset.ocl === '2x2') return 'ocl2x2_smoke';
  return 'ocl2x2_smoke';
}

function numericValue(value, fallback = null) {
  if (value === null || value === undefined || value === '') return fallback;
  const match = String(value).match(/-?\d+(?:\.\d+)?/);
  const parsed = match ? Number(match[0]) : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function wavelengthColorChannel(wavelengthNm) {
  if (wavelengthNm <= 500) return 'blue';
  if (wavelengthNm >= 600) return 'red';
  return 'green';
}

function oclGroupForDesign(preset, oclState) {
  const label = `${preset.ocl || ''} ${oclState?.activeClass || ''}`.toLowerCase();
  if (label.includes('3x3') || label.includes('3 x 3')) return 3;
  if (label.includes('2x2') || label.includes('2 x 2') || preset.group === 2) return 2;
  return 1;
}

function solverModeForDesign(preset, oclState, pdafState) {
  if (preset.id === 'custom') {
    return { mode: 'ocl-layout', splitMode: 'quad', shieldMode: 'off' };
  }
  const pdafMode = String(pdafState?.activeMode || preset.af || '').toLowerCase();
  if (preset.id === 'quad_qpd' || pdafMode.includes('quad') || pdafMode.includes('qpd')) {
    return { mode: 'split-pd-1x1', splitMode: 'quad', shieldMode: 'off' };
  }
  if (pdafMode.includes('t/b') || pdafMode.includes('vertical')) {
    return { mode: 'split-pd-1x1', splitMode: 'dual-z', shieldMode: pdafMode.includes('shield') ? 'pdaf_pair' : 'off' };
  }
  if (preset.id === 'sparse_pdaf' || pdafMode.includes('half-shield') || pdafMode.includes('dual pixel')) {
    return { mode: 'split-pd-1x1', splitMode: 'dual-x', shieldMode: pdafMode.includes('shield') ? 'pdaf_pair' : 'off' };
  }
  const group = oclGroupForDesign(preset, oclState);
  if (group >= 3) return { mode: 'ocl-3x3', splitMode: 'quad', shieldMode: 'off' };
  if (group >= 2) return { mode: 'ocl-2x2', splitMode: 'quad', shieldMode: 'off' };
  return { mode: 'split-pd-1x1', splitMode: 'dual-x', shieldMode: 'off' };
}

function cfaPatternForPreset(preset) {
  if (preset.id === 'nona_3x3') return 'nona';
  if (preset.id === 'custom') return 'nona';
  if (preset.id === 'quad_2x2' || preset.id === 'quad_qpd') return 'quad';
  if (preset.id === 'bayer_1x1' || preset.id === 'sparse_pdaf') return 'bayer';
  return 'uniform';
}

function activeOclLayoutForModel(group, surfaceModel) {
  const safeGroup = Math.max(1, Math.min(3, Number(group) || 1));
  const id = 'active_ocl';
  return {
    mode: 'ocl-layout',
    layout_nx: safeGroup,
    layout_nz: safeGroup,
    ocl_layout: `${id}:0:0:${safeGroup}:${safeGroup}`,
    ocl_layout_name: `${surfaceModel.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_${safeGroup}x${safeGroup}_active_ocl`,
    lensId: id,
    group: safeGroup
  };
}

function oclAdvancedSolverFields({ selectedPreset, oclState, model, baseMode }) {
  const surfaceModel = oclState?.surfaceModel || 'Spherical cap';
  if (surfaceModel === 'Spherical cap') return { solverFields: {}, designFields: { surface_model: surfaceModel } };
  const group = oclGroupForDesign(selectedPreset, oclState);
  const pitch = numericValue(model.geometry?.pitch, 1.4);
  const edgeGap = numericValue(model.geometry?.lens_edge_gap, 0.04);
  const lensHeight = numericValue(oclState?.params?.['Height (Center)'], model.geometry?.lens_height || 0.657);
  const half = Math.max(0.5 * group * pitch - edgeGap, 0.05);
  const layout = activeOclLayoutForModel(group, surfaceModel);
  const solverFields = {
    mode: layout.mode,
    layout_nx: layout.layout_nx,
    layout_nz: layout.layout_nz,
    ocl_layout: layout.ocl_layout,
    ocl_layout_name: layout.ocl_layout_name
  };
  const designFields = { surface_model: surfaceModel, active_lens_id: layout.lensId };

  if (surfaceModel === 'Asphere sag') {
    solverFields.ocl_sag = {
      [layout.lensId]: {
        type: 'asphere',
        conic_k: numericValue(oclState?.params?.['Aspheric Coef. (k)'], -0.12),
        a4: numericValue(oclState?.params?.['Asphere A4'], 0.018),
        normalize_edge: true
      }
    };
    return { solverFields, designFields };
  }

  if (surfaceModel === 'Polygon aperture') {
    const bias = Math.max(0, Math.min(numericValue(oclState?.params?.['Polygon Bias'], 0.08), half * 0.45));
    solverFields.ocl_polygons = {
      [layout.lensId]: [
        [-half, -half],
        [half - bias, -half + 0.5 * bias],
        [half, half - bias],
        [-half + bias, half]
      ]
    };
    designFields.polygon_bias_um = bias;
    return { solverFields, designFields };
  }

  if (surfaceModel === 'Surface map') {
    const edgeHeight = Math.max(0, Math.min(numericValue(oclState?.params?.['Surface Edge Height'], 0.2), lensHeight));
    solverFields.ocl_surface_map = {
      [layout.lensId]: {
        source: 'UI inline surface map',
        x_um: [-half, 0, half],
        z_um: [-half, 0, half],
        height_um: [
          [0, edgeHeight, 0],
          [edgeHeight * 0.9, lensHeight, edgeHeight * 1.1],
          [0, edgeHeight, 0]
        ]
      }
    };
    designFields.surface_edge_height_um = edgeHeight;
    return { solverFields, designFields };
  }

  return { solverFields: {}, designFields: { surface_model: surfaceModel, base_mode: baseMode } };
}

function cfaPolygonSolverFields(cfaState, model) {
  const apertureModel = cfaState?.apertureModel || 'Full tile';
  if (apertureModel === 'Full tile') {
    return { solverFields: {}, designFields: { aperture_model: apertureModel } };
  }
  const pitch = numericValue(model.geometry?.pitch, 1.4);
  const inset = Math.max(0, Math.min(numericValue(cfaState?.edgeInset, 0.12), pitch * 0.35));
  const skew = Math.max(0, Math.min(numericValue(cfaState?.edgeSkew, 0.02), pitch * 0.12));
  const half = Math.max(0.5 * pitch - inset, 0.05);
  const background = cfaState?.gapBackground || 'passivation';
  const base = [
    [-half, -half],
    [half, -half],
    [half, half],
    [-half, half]
  ];
  const skewed = {
    red: [
      [-half, -half],
      [half - skew, -half],
      [half, half - skew],
      [-half + skew, half]
    ],
    green: base,
    blue: [
      [-half + skew, -half],
      [half, -half + skew],
      [half - skew, half],
      [-half, half - skew]
    ]
  };
  return {
    solverFields: {
      cfa_polygons: {
        background,
        red: skewed.red,
        green: skewed.green,
        blue: skewed.blue
      }
    },
    designFields: {
      aperture_model: apertureModel,
      edge_inset_um: inset,
      edge_skew_um: skew,
      gap_background: background
    }
  };
}

function buildSimulationRequest({ model, selectedPreset, oclState, cfaState, pdafState, responseState, projectName, cadTemplate }) {
  const wavelengthNm = numericValue(responseState?.wavelength, model.edgeCase?.wavelength_nm || 550);
  const craDeg = numericValue(responseState?.cra, model.edgeCase?.cra_x_deg || 20);
  const lensShiftX = oclState?.craCompensation ? numericValue(oclState?.params?.['X Shift'], 0) : 0;
  const lensShiftZ = oclState?.craCompensation ? numericValue(oclState?.params?.['Y Shift'], 0) : 0;
  const solver = solverModeForDesign(selectedPreset, oclState, pdafState);
  const lensIndex = numericValue(oclState?.params?.['Refractive Index'], null);
  const lensHeight = numericValue(oclState?.params?.['Height (Center)'], model.geometry?.lens_height || 0.657);
  const cfaThickness = numericValue(cfaState?.thickness, model.geometry?.cfa_thickness || 0.8);
  const cfaPattern = cfaPatternForPreset(selectedPreset);
  const cfaShifts = {
    red: { x: numericValue(cfaState?.rShift, 0), z: numericValue(cfaState?.rShiftZ, 0) },
    green: { x: numericValue(cfaState?.gShift, 0), z: numericValue(cfaState?.gShiftZ, 0) },
    blue: { x: numericValue(cfaState?.bShift, 0), z: numericValue(cfaState?.bShiftZ, 0) }
  };
  const pitch = numericValue(model.geometry?.pitch, 1.4);
  const safeCra = Number.isFinite(craDeg) ? craDeg : 20;
  const cases = [
    'center:0:0:0:0:0:0',
    `field${String(safeCra).replace('.', 'p')}x:${safeCra}:0:1:0:${lensShiftX || 0}:${lensShiftZ || 0}`
  ].join(',');
  const layoutFields = solver.mode === 'ocl-layout'
    ? {
        layout_nx: MIXED_OCL_LAYOUT.nx,
        layout_nz: MIXED_OCL_LAYOUT.nz,
        ocl_layout: MIXED_OCL_LAYOUT.descriptor,
        ocl_layout_name: MIXED_OCL_LAYOUT.name
      }
    : {};
  const advancedOcl = oclAdvancedSolverFields({ selectedPreset, oclState, model, baseMode: solver.mode });
  const cfaGeometry = cfaPolygonSolverFields(cfaState, model);
  const advancedSplitCollection = advancedOcl.solverFields?.mode === 'ocl-layout' && solver.mode === 'split-pd-1x1'
    ? {
        collection_mode: 'split-pd',
        target_lens_id: advancedOcl.designFields?.active_lens_id || 'active_ocl'
      }
    : {};
  const cadDefaults = cadTemplate?.solver_defaults || {};
  const cadSolverDefaults = cadDefaults.solver || {};
  const cadStackDefaults = cadDefaults.stack_overrides || {};
  const stackOverrides = cadTemplate ? {} : {
    ...cadStackDefaults,
    'geometry_um.pitch': pitch,
    'geometry_um.lens_height': lensHeight,
    'geometry_um.cfa_thickness': cfaThickness
  };
  if (lensIndex) {
    stackOverrides['materials.lens'] = {
      n: lensIndex,
      k: 0,
      measured: false,
      source: 'UI active design override',
      usage: 'on-chip microlens'
    };
  }
  const warnings = [];
  if (wavelengthNm > 700) {
    warnings.push('Current proxy CFA n,k tables only cover visible wavelengths up to 700 nm; 940 nm should use a measured/imported IR stack before running.');
  }
  const solverPayload = {
    mode: solver.mode,
    split_mode: solver.splitMode,
    shield_mode: solver.shieldMode,
    ...(cadTemplate ? {} : layoutFields),
    ...(cadTemplate ? {} : advancedOcl.solverFields),
    ...advancedSplitCollection,
    ...(cadTemplate ? {} : cfaGeometry.solverFields),
    ...(cadTemplate ? cadSolverDefaults : {}),
    cad_template_id: cadTemplate?.template_id,
    wavelengths_nm: String(wavelengthNm),
    color_channel: cadSolverDefaults.color_channel || wavelengthColorChannel(wavelengthNm),
    cfa_pattern: cadTemplate ? (cadSolverDefaults.cfa_pattern || cfaPattern) : cfaPattern,
    cfa_shifts_um: cfaShifts,
    cases,
    resolution: 18,
    after_source_time: 2,
    pml_um: 0.45,
    stack_overrides: stackOverrides
  };
  return {
    schema: 'pixel_workbench_simulation_request_v1',
    source: cadTemplate ? 'ui_active_design_with_cad_template' : 'ui_active_design',
    project: { name: projectName || model.projectName || 'Pixel Workbench' },
    design: {
      preset_id: selectedPreset.id,
      preset_label: selectedPreset.label,
      cad_template: cadTemplate ? {
        template_id: cadTemplate.template_id,
        label: cadTemplate.label,
        source_truth_level: cadTemplate.source_truth_level,
        geometry_import: cadDefaults.geometry_import,
        parameters: cadDefaults.parameters,
        solver_ready: cadTemplate.solver_ready === true,
        geometry_authority: 'cad_template',
        geometry_override_policy: 'Use CAD variants or FCStd round-trip for geometry edits; UI stack overrides only apply to non-CAD active designs.'
      } : null,
      cfa: selectedPreset.cfa,
      cfa_geometry: cfaGeometry.designFields,
      ocl: selectedPreset.ocl,
      ocl_group: oclGroupForDesign(selectedPreset, oclState),
      ocl_class: oclState?.activeClass,
      pdaf_mode: pdafState?.activeMode,
      readout: selectedPreset.readout,
      cfa_parameters: cfaState,
      ocl_parameters: oclState?.params,
      ocl_surface: advancedOcl.designFields,
      collection_geometry: advancedSplitCollection.collection_mode ? advancedSplitCollection : { collection_mode: 'auto' },
      cra_compensation: Boolean(oclState?.craCompensation)
    },
    condition: {
      wavelength_nm: wavelengthNm,
      color_channel: wavelengthColorChannel(wavelengthNm),
      cra_x_deg: safeCra,
      cra_z_deg: 0,
      field_x_norm: safeCra === 0 ? 0 : 1,
      field_z_norm: 0,
      polarization: responseState?.polarization || 'TM',
      analysis_plane: responseState?.plane || 'OCL Exit Plane'
    },
    solver: solverPayload,
    gates: {
      product_lut_ready_expected: false,
      geometry_authority: cadTemplate ? 'cad_template' : 'ui_controls',
      reason: cadTemplate
        ? 'CAD template footprints reduce hidden geometry assumptions, but current stack/material/device values are still not measured product calibration.'
        : 'UI active-design request uses the current proxy stack unless measured stack/material files are imported.'
    },
    warnings
  };
}

function artifactUrl(path, apiBase) {
  if (!path) return '';
  if (/^https?:\/\//.test(path)) return path;
  if (path.startsWith('/') && apiBase && /^https?:\/\//.test(apiBase)) return `${apiBase}${path}`;
  return path;
}

function parentPath(path) {
  const value = String(path || '').replace(/\/+$/, '');
  const index = value.lastIndexOf('/');
  return index > 0 ? value.slice(0, index) : '';
}

function number(value, digits = 2, fallback = '-') {
  if (value === null || value === undefined || value === '') return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : fallback;
}

function percent(value, digits = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(digits)}%` : '-';
}

function compactValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object') {
    return Object.entries(value).map(([key, item]) => `${key}: ${item}`).join(', ');
  }
  return String(value);
}

function diagnosticText(available, value, applies = true) {
  if (available) return value;
  return applies ? 'not run' : 'not applicable';
}

function diagnosticTitle(available, label, applies = true) {
  if (available) return `${label} artifact is present for this template.`;
  return applies
    ? `${label} can apply to this template, but no generated artifact is indexed yet.`
    : `${label} does not apply to this template geometry.`;
}

function CadSummaryItem({ label, value, title, warning = false }) {
  return (
    <div className={warning ? 'cad-summary-warn' : ''} title={title || ''}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function factorSummary(factors = {}) {
  return Object.entries(factors).map(([key, value]) => `${key}: ${compactValue(value)}`).join(' · ');
}

const CAD_QUICK_VARIANT_FIELDS = [
  ['pitch_um', 'Pixel pitch', 'um'],
  ['lens_height_um', 'Lens height', 'um'],
  ['lens_edge_gap_um', 'Lens edge gap', 'um'],
  ['cfa_thickness_um', 'CFA thickness', 'um'],
  ['passivation_thickness_um', 'Passivation', 'um'],
  ['dti_width_um', 'DTI width', 'um'],
  ['dti_depth_um', 'DTI depth', 'um'],
  ['pd_margin_um', 'PD margin', 'um'],
  ['pd_depth_max_um', 'PD max depth', 'um']
];

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function presetById(id) {
  return PRESETS.find((preset) => preset.id === id) || PRESETS[0];
}

function selectedCellForPreset(preset, row = 5, col = 5) {
  return { row, col, color: colorForCell(preset, row, col), role: roleForCell(preset, row, col, 12) };
}

function formatSigned(value, digits = 3) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return '-';
  return `${parsed >= 0 ? '+' : ''}${parsed.toFixed(digits)}`;
}

function nextId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

function bayerColor(row, col) {
  if (row % 2 === 0) return col % 2 === 0 ? 'R' : 'G';
  return col % 2 === 0 ? 'G' : 'B';
}

function blockColor(row, col, group) {
  const br = Math.floor(row / group);
  const bc = Math.floor(col / group);
  return bayerColor(br, bc);
}

function colorForCell(preset, row, col) {
  if (preset.id === 'optical_black' && row >= 10) return 'OB';
  if (preset.id === 'rgbir_group') {
    const key = `${row % 2}${col % 2}`;
    return { '00': 'R', '01': 'G', '10': 'B', '11': 'IR' }[key];
  }
  if (preset.id === 'nona_3x3') return blockColor(row, col, 3);
  if (preset.id === 'quad_2x2' || preset.id === 'quad_qpd') return blockColor(row, col, 2);
  if (preset.id === 'custom') return (row + col) % 5 === 0 ? 'SH' : bayerColor(row, col);
  return bayerColor(row, col);
}

function roleForCell(preset, row, col, size) {
  if (preset.id === 'optical_black' && row >= size - 2) return 'OB';
  if (preset.id === 'sparse_pdaf') {
    if (row === 2 && col === 3) return 'AF-L';
    if (row === 2 && col === 4) return 'AF-R';
    if (row === 7 && col === 2) return 'AF-L';
    if (row === 7 && col === 3) return 'AF-R';
    if (row >= size - 1) return 'OB';
  }
  if (preset.id === 'quad_qpd') {
    const start = 5;
    const roles = [
      ['Q1', 'Q2'],
      ['Q3', 'Q4']
    ];
    if (row >= start && row < start + 2 && col >= start && col < start + 2) {
      return roles[row - start][col - start];
    }
  }
  return 'Img';
}

function colorClass(label) {
  return `color-${String(label || 'img').toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
}

function oclGroupsForPreset(preset, size = 12) {
  if (preset.id === 'bayer_1x1' || preset.id === 'optical_black') return [];
  if (preset.id === 'nona_3x3') {
    return [
      { id: 'OCL_Nona_R', x: 0, y: 0, w: 3, h: 3, kind: '3x3' },
      { id: 'OCL_Nona_G1', x: 3, y: 0, w: 3, h: 3, kind: '3x3' },
      { id: 'OCL_Nona_G2', x: 0, y: 3, w: 3, h: 3, kind: '3x3' },
      { id: 'OCL_Nona_B', x: 3, y: 3, w: 3, h: 3, kind: '3x3' },
      { id: 'OCL_Nona_edge', x: 6, y: 6, w: 3, h: 3, kind: '3x3' },
      { id: 'OCL_Nona_corner', x: 9, y: 6, w: 3, h: 3, kind: '3x3' }
    ];
  }
  if (preset.id === 'sparse_pdaf') {
    return [
      { id: 'OCL_AF_pair_1', x: 3, y: 2, w: 2, h: 1, kind: 'AF pair' },
      { id: 'OCL_AF_pair_2', x: 2, y: 7, w: 2, h: 1, kind: 'AF pair' }
    ];
  }
  const groups = [];
  for (let y = 0; y < size; y += 4) {
    for (let x = 0; x < size; x += 4) {
      groups.push({ id: `OCL_${x}_${y}`, x, y, w: 2, h: 2, kind: '2x2' });
    }
  }
  if (preset.id === 'quad_qpd') groups.push({ id: 'OCL_QPD_center', x: 5, y: 5, w: 2, h: 2, kind: 'QPD' });
  return groups;
}

function binningGroupsForPreset(preset) {
  if (preset.id === 'nona_3x3') {
    return [
      { id: 'BIN_Nona_1', x: 0, y: 0, w: 6, h: 3 },
      { id: 'BIN_Nona_2', x: 0, y: 3, w: 6, h: 3 },
      { id: 'BIN_Nona_3', x: 6, y: 6, w: 6, h: 3 }
    ];
  }
  if (preset.group === 2 || preset.id === 'quad_qpd') {
    return [
      { id: 'BIN_2x2_1', x: 0, y: 0, w: 2, h: 2 },
      { id: 'BIN_2x2_2', x: 4, y: 0, w: 2, h: 2 },
      { id: 'BIN_2x2_3', x: 8, y: 0, w: 2, h: 2 }
    ];
  }
  return [];
}

function useWorkbenchModel(payload) {
  const stack = payload?.stack || {};
  const profile = payload?.profile || {};
  const geometry = stack.geometry_um || {};
  const pixel = profile.geometry || {};
  const cases = payload?.gw_coupling?.cases || [];
  const centerCase = cases.find((item) => item.case === 'center') || cases[0] || {};
  const edgeCase = cases.find((item) => item.case === 'edge20x') || cases[1] || centerCase;
  const generated = Number(edgeCase.generated_current_a_per_cm || centerCase.generated_current_a_per_cm || 0);
  const native = Number(edgeCase.native_total_abs_delta_a_per_cm || centerCase.native_total_abs_delta_a_per_cm || 0);
  const qeProxy = generated ? clamp(native / generated, 0, 1.2) : 0.72;
  const splitError = Math.abs(Number(edgeCase.gw_devsim_laplace_split_phase_error || 0.042));
  const rows = payload?.variant_comparison_rows || [];
  const variants = (rows.length ? rows.filter((row) => row.case === 'edge20x') : []).slice(0, 6).map((row, index) => {
    const rel = Number(row.total_photo_delta_rel_change);
    const response = Number.isFinite(rel) ? 0.72 * (1 + rel) : 0.72 + index * 0.01;
    const crosstalk = clamp(Math.abs(Number(row.split_phase_delta || row.gw_devsim_laplace_split_phase_error || 0.04)) * 8, 0.03, 0.12);
    const cra = clamp(1 - Math.abs(Number(row.gw_devsim_laplace_total_reference_scaled_rel_error || 0.12)), 0.4, 1);
    const score = clamp(0.38 * response + 0.3 * cra + 0.32 * (1 - crosstalk));
    return {
      id: row.variant_id || `variant_${index + 1}`,
      label: row.variant_label || row.variant_id || `Variant ${index + 1}`,
      ocl: index % 3 === 0 ? '1x1' : index % 3 === 1 ? '2x2' : '3x3',
      pdaf: index % 2 === 0 ? 'None' : 'PDAF Sparse',
      cfa: index % 3 === 2 ? '3x3' : index % 3 === 1 ? '2x2' : '1x1',
      qe: response,
      crosstalk,
      cra,
      remosaic: clamp(0.28 + index * 0.05, 0, 1),
      score
    };
  });
  if (!variants.length) {
    variants.push(
      { id: 'baseline', label: 'Baseline Bayer 1x1 OCL', ocl: '1x1', pdaf: 'None', cfa: '1x1', qe: 0.716, crosstalk: 0.063, cra: 0.72, remosaic: 0.42, score: 0.68 },
      { id: 'quad', label: 'Quad Bayer + 2x2 OCL', ocl: '2x2', pdaf: 'None', cfa: '2x2', qe: 0.751, crosstalk: 0.051, cra: 0.83, remosaic: 0.33, score: 0.77 },
      { id: 'qpd', label: 'Quad Bayer + QPD', ocl: '2x2', pdaf: 'QPD', cfa: '2x2', qe: 0.738, crosstalk: 0.054, cra: 0.84, remosaic: 0.35, score: 0.81 },
      { id: 'nona', label: 'Nona 3x3 + 3x3 OCL', ocl: '3x3', pdaf: 'None', cfa: '3x3', qe: 0.746, crosstalk: 0.047, cra: 0.86, remosaic: 0.28, score: 0.83 }
    );
  }
  const best = variants.reduce((winner, item) => (item.score > winner.score ? item : winner), variants[0]);
  return {
    payload,
    stack,
    profile,
    geometry,
    pixel,
    centerCase,
    edgeCase,
    variants,
    best,
    metrics: {
      qeProxy,
      splitError,
      crosstalk: 0.047,
      oclUniformity: 0.94,
      pdafBalance: clamp(1 - splitError, 0.8, 1),
      afCoverage: 0.873,
      remosaicRisk: 0.72
    },
    projectName: payload?.project?.project?.name || 'Image Sensor Pixel Workbench'
  };
}

function StatusPill({ state = 'neutral', children }) {
  return <span className={`status-pill ${state}`}>{children}</span>;
}

function IconButton({ label, onClick, active = false, children }) {
  return (
    <button type="button" className={`icon-button ${active ? 'active' : ''}`} aria-label={label} title={label} onClick={onClick}>
      {children}
    </button>
  );
}

function SectionNav({ active, onSelect, onNewPattern }) {
  return (
    <aside className="side-nav">
      <div className="side-brand">
        <Grid2X2 size={18} />
        <span>Image Sensor Pixel Workbench</span>
      </div>
      <div className="nav-groups">
        {NAV_SECTIONS.map((section) => (
          <section className="nav-group" key={section.label}>
            <div className="nav-heading">{section.label}</div>
            {section.items.map(([label, Icon]) => (
              <button
                type="button"
                key={label}
                className={`nav-entry ${active === label ? 'active' : ''}`}
                onClick={() => onSelect(label)}
              >
                <Icon size={15} />
                <span>{label}</span>
              </button>
            ))}
          </section>
        ))}
      </div>
      <button type="button" className="new-pattern-button" onClick={onNewPattern}>
        <Plus size={15} />
        New Pattern
      </button>
    </aside>
  );
}

function AppTopbar({ model, activeTopTab, projectName, bookmarked, onTopTab, onProjectChange, onAction }) {
  return (
    <header className="app-topbar">
      <div className="topbar-tabs">
        {['Project', 'Pixel', 'Experiment', 'Compare', 'Report'].map((item) => (
          <button type="button" key={item} className={activeTopTab === item ? 'active' : ''} onClick={() => onTopTab(item)}>
            {item}
          </button>
        ))}
      </div>
      <div className="project-picker">
        <span>Project</span>
        <select value={projectName} aria-label="Project" onChange={(event) => onProjectChange(event.target.value)}>
          <option>Automotive_HDR_Pixel_v2</option>
          <option>{model.projectName}</option>
        </select>
      </div>
      <div className="topbar-actions">
        <IconButton label="Save" onClick={() => onAction('save')}><Save size={17} /></IconButton>
        <IconButton label="Bookmark" active={bookmarked} onClick={() => onAction('bookmark')}><Bookmark size={17} /></IconButton>
        <IconButton label="Help" onClick={() => onAction('help')}><CircleHelp size={17} /></IconButton>
        <IconButton label="Settings" onClick={() => onAction('settings')}><Settings size={17} /></IconButton>
      </div>
    </header>
  );
}

function PageHeader({ active, selectedPreset, model }) {
  const subtitle =
    active === 'Pattern Composer'
      ? 'CFA + OCL grouping + PDAF/Shield + Readout topology'
      : active === 'Test Suite'
        ? 'Practical pattern, OCL, material, crosstalk, and PDAF validation matrix'
      : active === 'Pattern Response'
        ? 'Field, top-view response, crosstalk, AF balance, and remosaic risk'
        : active === 'ML / OCL'
          ? 'Group-class OCL editing'
          : active === 'PDAF / Shield'
            ? 'Phase pixel mask and pair balance'
            : active === 'Variants'
              ? 'Robust pattern decision'
              : 'Pattern-aware pixel design';
  return (
    <div className="page-header">
      <div>
        <h1>{active}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="header-badges">
        <StatusPill state="ok">framework ready</StatusPill>
        <StatusPill state="accent">{selectedPreset.short}</StatusPill>
        <StatusPill state="warn">lambda {number(model.edgeCase.wavelength_nm || 550, 0)} nm</StatusPill>
      </div>
    </div>
  );
}

function PresetMini({ preset }) {
  const cells = Array.from({ length: 4 }, (_, index) => {
    const row = Math.floor(index / 2);
    const col = index % 2;
    const label = preset.id === 'rgbir_group' && index === 3 ? 'IR' : blockColor(row, col, 1);
    return <span className={`mini-cell ${colorClass(label)}`} key={index}>{label}</span>;
  });
  return (
    <div className="preset-mini">
      <div className="mini-cfa">{cells}</div>
      <div className={`mini-ocl group-${preset.group}`}>{preset.ocl}</div>
    </div>
  );
}

function PresetCards({ selectedPreset, onSelect }) {
  return (
    <div className="preset-strip">
      {PRESETS.map((preset) => (
        <button
          type="button"
          className={`preset-card ${selectedPreset.id === preset.id ? 'active' : ''}`}
          key={preset.id}
          onClick={() => onSelect(preset)}
        >
          <span className="preset-number">{preset.number}</span>
          <strong>{preset.label}</strong>
          <PresetMini preset={preset} />
          <em>{preset.cfa}</em>
        </button>
      ))}
    </div>
  );
}

function LayerToggles({ layers, onToggle }) {
  const entries = [
    ['cfa', 'CFA'],
    ['ocl', 'OCL'],
    ['pdaf', 'PDAF'],
    ['shield', 'Shield'],
    ['binning', 'Binning'],
    ['dti', 'DTI'],
    ['readout', 'Readout']
  ];
  return (
    <div className="layer-toggle-row">
      {entries.map(([key, label]) => (
        <label key={key} className="layer-toggle">
          <input type="checkbox" checked={layers[key]} onChange={() => onToggle(key)} />
          <span>{label}</span>
        </label>
      ))}
    </div>
  );
}

function OverlayBox({ item, size, tone, label }) {
  return (
    <div
      className={`overlay-box ${tone}`}
      style={{
        left: `${(item.x / size) * 100}%`,
        top: `${(item.y / size) * 100}%`,
        width: `${(item.w / size) * 100}%`,
        height: `${(item.h / size) * 100}%`
      }}
    >
      {label || item.kind}
    </div>
  );
}

function SupercellGrid({ preset, layers, selectedCell, onCellSelect, compact = false, highlightCouplings = false }) {
  const size = 12;
  const oclGroups = oclGroupsForPreset(preset, size);
  const binningGroups = binningGroupsForPreset(preset, size);
  const cells = Array.from({ length: size * size }, (_, index) => {
    const row = Math.floor(index / size);
    const col = index % size;
    const color = colorForCell(preset, row, col);
    const role = roleForCell(preset, row, col, size);
    const selected = selectedCell.row === row && selectedCell.col === col;
    return (
      <button
        type="button"
        className={`pixel-cell ${colorClass(color)} ${role !== 'Img' ? 'has-role' : ''} ${selected ? 'selected' : ''} ${highlightCouplings && (selected || role === selectedCell.role || color === selectedCell.color) ? 'coupled' : ''}`}
        key={`${row}-${col}`}
        onClick={() => onCellSelect({ row, col, color, role })}
      >
        {layers.cfa ? <span>{color}</span> : null}
        {layers.pdaf && role !== 'Img' ? <b>{role}</b> : null}
        {layers.readout && !compact ? <i /> : null}
      </button>
    );
  });
  return (
    <div className={`supercell-grid-wrap ${compact ? 'compact' : ''}`}>
      <div className="grid-axis x-axis">{Array.from({ length: size }, (_, i) => <span key={i}>{i + 1}</span>)}</div>
      <div className="grid-axis y-axis">{Array.from({ length: size }, (_, i) => <span key={i}>{i + 1}</span>)}</div>
      <div className="supercell-grid" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}>
        {cells}
        {layers.ocl && oclGroups.map((item) => <OverlayBox key={item.id} item={item} size={size} tone={item.kind === '3x3' ? 'three' : item.kind === 'QPD' ? 'pdaf' : 'two'} />)}
        {layers.binning && binningGroups.map((item) => <OverlayBox key={item.id} item={item} size={size} tone="binning" label="" />)}
        {layers.dti ? <div className="dti-overlay" /> : null}
      </div>
    </div>
  );
}

function PatternTree({ preset }) {
  const rows = [
    ['Supercell', '12 x 12 repeat'],
    ['CFA', preset.cfa],
    ['OCL Topology', `${preset.ocl} shared`],
    ['PDAF', preset.af],
    ['Readout', preset.readout],
    ['LCM Cell', preset.id === 'nona_3x3' ? '6 x 6 minimum' : preset.group === 2 ? '4 x 4 minimum' : '2 x 2 minimum']
  ];
  return (
    <aside className="pattern-tree panel-surface">
      <h2>Pattern Tree</h2>
      {rows.map(([label, value]) => (
        <div className="tree-row" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
      <div className="tree-footer">
        <StatusPill state={preset.id === 'custom' ? 'warn' : 'ok'}>{preset.id === 'custom' ? 'custom mode' : 'rule checked'}</StatusPill>
      </div>
    </aside>
  );
}

function CouplingInspector({ selectedCell, preset, highlightCouplings, onAction }) {
  const role = selectedCell.role || 'Img';
  const isAf = role.startsWith('AF') || role.startsWith('Q');
  const rows = [
    ['Color Filter', selectedCell.color || 'G', 'ok'],
    ['OCL Group', preset.ocl, preset.id === 'custom' ? 'warn' : 'ok'],
    ['PDAF Pair', isAf ? role : preset.af, isAf || preset.af !== 'None' ? 'ok' : 'neutral'],
    ['Shield Mask', isAf ? 'Mirrored half' : 'None', isAf ? 'ok' : 'neutral'],
    ['Binning Group', preset.group > 1 ? `${preset.group}x${preset.group}` : '1x1', 'ok'],
    ['Remosaic Group', preset.readout.includes('remosaic') ? 'Defined' : 'Image path', 'ok'],
    ['Validation', preset.id === 'custom' ? 'Needs rule pass' : 'OK', preset.id === 'custom' ? 'warn' : 'ok']
  ];
  return (
    <aside className="inspector panel-surface">
      <header>
        <h2>Coupling Inspector</h2>
        <StatusPill state="accent">Cell ({selectedCell.col + 1}, {selectedCell.row + 1})</StatusPill>
      </header>
      {rows.map(([label, value, state]) => (
        <div className="inspector-row" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
          <StatusDot state={state} />
        </div>
      ))}
      <button type="button" className={`primary-button ${highlightCouplings ? 'active' : ''}`} onClick={() => onAction('toggle-highlight-couplings')}>
        <Eye size={15} />
        {highlightCouplings ? 'Hide Couplings' : 'Highlight Couplings'}
      </button>
    </aside>
  );
}

function StatusDot({ state }) {
  return <span className={`status-dot ${state}`} />;
}

function ValidationStrip({ preset }) {
  const items = [
    ['AF pair same CFA color', preset.af === 'None' ? 'not used' : '2 / 2 pairs OK', preset.id === 'custom' ? 'warn' : 'ok'],
    ['AF pair shares OCL group', preset.af === 'None' ? 'not used' : 'same OCL', preset.id === 'sparse_pdaf' ? 'warn' : 'ok'],
    ['QPD roles complete', preset.af === 'QPD' ? 'Q1-Q4 assigned' : 'not used', 'ok'],
    ['3x3 OCL aligned to CFA', preset.ocl === '3x3' ? '1 / 1 group OK' : 'not used', preset.id === 'nona_3x3' ? 'ok' : 'neutral'],
    ['Remosaic complexity', preset.complexity, preset.complexity === 'High' ? 'warn' : 'ok'],
    ['Shield pattern valid', preset.id === 'custom' ? 'review' : 'no optical path conflict', preset.id === 'custom' ? 'warn' : 'ok']
  ];
  return (
    <div className="validation-strip">
      {items.map(([label, value, state]) => (
        <div className={`validation-card ${state}`} key={label}>
          {state === 'ok' ? <CheckCircle2 size={18} /> : state === 'warn' ? <TriangleAlert size={18} /> : <CheckCircle2 size={18} />}
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function FlowRail({ active, onSelect }) {
  const steps = [
    ['Template', PanelsTopLeft],
    ['Pattern Composer', Grid2X2],
    ['Detail Tabs', Layers3],
    ['Simulation', Activity],
    ['Pattern Response', BarChart3],
    ['Variant Decision', Target]
  ];
  return (
    <div className="flow-rail">
      {steps.map(([label, Icon], index) => (
        <button type="button" className={`flow-step ${FLOW_ROUTES[label] === active ? 'active' : ''}`} key={label} onClick={() => onSelect(FLOW_ROUTES[label])}>
          <Icon size={26} />
          <span>{label}</span>
          {index < steps.length - 1 ? <ChevronRight size={18} className="flow-arrow" /> : null}
        </button>
      ))}
    </div>
  );
}

function PatternComposerView({
  active,
  selectedPreset,
  setSelectedPreset,
  setSelectedCadTemplateId,
  selectedCell,
  setSelectedCell,
  composerLayers,
  setComposerLayers,
  composerSettings,
  setComposerSettings,
  highlightCouplings,
  onSelectView,
  onAction
}) {
  const onToggle = (key) => {
    setComposerLayers((current) => ({ ...current, [key]: !current[key] }));
    onAction(`Layer ${key.toUpperCase()} toggled`);
  };
  return (
    <div className="pattern-composer">
      <PresetCards selectedPreset={selectedPreset} onSelect={(preset) => {
        setSelectedPreset(preset);
        setSelectedCell(selectedCellForPreset(preset));
        const cadTemplateId = cadTemplateIdForPreset(preset);
        setSelectedCadTemplateId(cadTemplateId);
        onAction(cadTemplateId
          ? `Preset loaded: ${preset.label} · CAD source ${cadTemplateId}`
          : `Preset loaded: ${preset.label} · no exact CAD template`);
      }} />
      <section className="composer-main">
        <PatternTree preset={selectedPreset} />
        <main className="layout-viewer panel-surface">
          <header className="viewer-toolbar">
            <div>
              <h2>Supercell Layout Viewer</h2>
              <span>CFA + OCL + PDAF + Shield + Binning</span>
            </div>
            <div className="toolbar-controls">
              <select value={composerSettings.size} aria-label="Supercell Size" onChange={(event) => {
                setComposerSettings((current) => ({ ...current, size: event.target.value }));
                onAction(`Supercell size set to ${event.target.value}`);
              }}>
                <option>12x12</option>
                <option>8x8</option>
                <option>6x6</option>
              </select>
              <select value={composerSettings.view} aria-label="View" onChange={(event) => {
                setComposerSettings((current) => ({ ...current, view: event.target.value }));
                onAction(`Composer view set to ${event.target.value}`);
              }}>
                <option value="top">Top</option>
                <option value="slice">Slice</option>
              </select>
            </div>
          </header>
          <LayerToggles layers={composerLayers} onToggle={onToggle} />
          <SupercellGrid preset={selectedPreset} layers={composerLayers} selectedCell={selectedCell} highlightCouplings={highlightCouplings} onCellSelect={(cell) => {
            setSelectedCell(cell);
            onAction(`Selected cell (${cell.col + 1}, ${cell.row + 1}) ${cell.color}/${cell.role}`);
          }} />
          <div className="legend-row">
            <span><i className="legend red" />R</span>
            <span><i className="legend green" />G</span>
            <span><i className="legend blue" />B</span>
            <span><i className="legend cyan-outline" />2x2 OCL</span>
            <span><i className="legend yellow-outline" />3x3 OCL / Binning</span>
            <span><i className="legend purple-outline" />PDAF Pair</span>
            <span><i className="legend gray" />Shield / OB</span>
          </div>
        </main>
        <CouplingInspector selectedCell={selectedCell} preset={selectedPreset} highlightCouplings={highlightCouplings} onAction={onAction} />
      </section>
      <ValidationStrip preset={selectedPreset} />
      <FlowRail active={active} onSelect={onSelectView} />
    </div>
  );
}

function CrossSectionSvg({ variant = 'ocl' }) {
  const cra = variant === 'pdaf' ? 42 : 30;
  return (
    <svg className="cross-section-svg" viewBox="0 0 760 310" role="img" aria-label="Linked pixel stack cross-section">
      <defs>
        <linearGradient id="lens-ui" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#d7efff" />
          <stop offset="100%" stopColor="#245c99" />
        </linearGradient>
        <linearGradient id="field-ui" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#0ea5e9" />
          <stop offset="50%" stopColor="#84cc16" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
      </defs>
      <rect width="760" height="310" rx="8" fill="#061018" />
      <rect x="38" y="98" width="684" height="34" fill="#12304a" />
      <rect x="38" y="132" width="684" height="22" fill="#be3e5e" opacity="0.9" />
      <rect x="38" y="154" width="684" height="18" fill="#c7bfa9" />
      <rect x="38" y="172" width="684" height="98" fill="#111a24" />
      <path d="M112 98 C160 22 280 22 326 98 L326 132 L112 132 Z" fill="url(#lens-ui)" stroke="#38bdf8" />
      <path d="M326 98 C390 14 568 14 634 98 L634 132 L326 132 Z" fill="url(#lens-ui)" stroke="#f6c445" opacity="0.92" />
      <rect x="190" y="172" width="26" height="94" fill="#475569" />
      <rect x="544" y="172" width="26" height="94" fill="#475569" />
      <path d="M300 230 C330 198 432 198 462 230 L462 266 L300 266 Z" fill="#c9a52b" stroke="#f6d34f" />
      <rect x="328" y="270" width="108" height="14" rx="3" fill="#0ea5e9" />
      {variant === 'pdaf' ? (
        <>
          <rect x="300" y="158" width="76" height="12" fill="#737373" />
          <rect x="386" y="158" width="76" height="12" fill="#0f172a" stroke="#a855f7" />
          <path d="M381 158 V268" stroke="#d8b4fe" strokeDasharray="4 5" />
          <text x="337" y="292" fill="#c084fc" fontSize="12" textAnchor="middle">AF-L</text>
          <text x="424" y="292" fill="#c084fc" fontSize="12" textAnchor="middle">AF-R</text>
        </>
      ) : null}
      <path d={`M615 32 L382 218 L655 58`} stroke="#bdefff" strokeWidth="2" strokeDasharray="7 7" fill="none" opacity="0.84" />
      <path d="M382 50 C354 120 349 186 382 218 C410 184 418 119 430 50" fill="url(#field-ui)" opacity="0.72" />
      <circle cx="382" cy="218" r="5" fill="#f6c445" />
      <text x="665" y="52" fill="#dff1f8" fontSize="13">{cra}deg</text>
      <g fill="#a8c4d1" fontSize="12">
        <text x="54" y="118">Microlens / OCL</text>
        <text x="54" y="148">CFA</text>
        <text x="54" y="168">Passivation</text>
        <text x="54" y="226">DTI</text>
        <text x="381" y="252" fill="#fff3c4" fontWeight="800" textAnchor="middle">Photodiode</text>
      </g>
    </svg>
  );
}

function MetricSpark({ color = '#22d3ee', values = [0.4, 0.62, 0.55, 0.74, 0.68, 0.8, 0.72] }) {
  const points = values.map((value, index) => {
    const x = 10 + index * 28;
    const y = 70 - value * 52;
    return [x, y];
  });
  return (
    <svg className="spark" viewBox="0 0 190 76">
      <path d="M10 64 H180 M10 16 H180" stroke="#233745" />
      <path d={points.map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ')} fill="none" stroke={color} strokeWidth="2.5" />
      {points.map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="2.6" fill={color} />)}
    </svg>
  );
}

function MLOclView({ selectedPreset, model, oclState, setOclState, onAction }) {
  const activeClass = oclState.activeClass;
  const surfaceModel = oclState.surfaceModel || 'Spherical cap';
  const classRows = [
    ['Radius', oclState.params.Radius, 'um', 0.02],
    ['Sag', oclState.params.Sag, 'um', 0.01],
    ['Diameter', oclState.params.Diameter, 'um', 0.02],
    ['Height (Center)', oclState.params['Height (Center)'], 'um', 0.01],
    ['X Shift', oclState.params['X Shift'], 'um', 0.005],
    ['Y Shift', oclState.params['Y Shift'], 'um', 0.005],
    ['Aspheric Coef. (k)', oclState.params['Aspheric Coef. (k)'], '', 0.01],
    ['Refractive Index', oclState.params['Refractive Index'], '@ 550 nm', 0.005]
  ];
  const advancedRows = surfaceModel === 'Asphere sag'
    ? [['Asphere A4', oclState.params['Asphere A4'], '', 0.002]]
    : surfaceModel === 'Polygon aperture'
      ? [['Polygon Bias', oclState.params['Polygon Bias'], 'um', 0.005]]
      : surfaceModel === 'Surface map'
        ? [['Surface Edge Height', oclState.params['Surface Edge Height'], 'um', 0.01]]
        : [];
  const updateParam = (label, delta) => {
    setOclState((current) => ({
      ...current,
      params: { ...current.params, [label]: Number((numericValue(current.params[label], 0) + delta).toFixed(4)) }
    }));
    onAction(`${activeClass} ${label} ${delta > 0 ? 'increased' : 'decreased'}`);
  };
  const renderParameterRow = ([label, value, unit, step]) => (
    <label className="parameter-row" key={label}>
      <span>{label}</span>
      <button type="button" onClick={() => updateParam(label, -step)}>-</button>
      <input value={label.includes('Shift') ? formatSigned(value, 3) : number(value, label === 'Refractive Index' ? 3 : 3)} onChange={(event) => {
        const parsed = Number(event.target.value);
        if (Number.isFinite(parsed)) {
          setOclState((current) => ({ ...current, params: { ...current.params, [label]: parsed } }));
        }
      }} />
      <button type="button" onClick={() => updateParam(label, step)}>+</button>
      <em>{unit}</em>
    </label>
  );
  return (
    <div className="detail-layout three-col">
      <aside className="panel-surface class-library">
        <h2>OCL Class Library</h2>
        {OCL_CLASSES.map(([name, size, shape, color]) => (
          <button type="button" className={`class-row ${activeClass === name ? 'active' : ''}`} key={name} onClick={() => {
            setOclState((current) => ({ ...current, activeClass: name }));
            onAction(`OCL class selected: ${name}`);
          }}>
            <span className="class-thumb" style={{ borderColor: color }} />
            <strong>{name}</strong>
            <em>{size} · {shape}</em>
          </button>
        ))}
      </aside>
      <main className="panel-surface detail-stage">
        <header className="viewer-toolbar">
          <div>
            <h2>Top View - OCL Class Map</h2>
            <span>{selectedPreset.cfa} / {selectedPreset.ocl}</span>
          </div>
          <select value={oclState.cfaMode} aria-label="CFA mode" onChange={(event) => {
            setOclState((current) => ({ ...current, cfaMode: event.target.value }));
            onAction(`OCL map CFA mode: ${event.target.value}`);
          }}><option>RGGB</option><option>Quad</option><option>Nona</option></select>
        </header>
        <SupercellGrid
          preset={selectedPreset}
          layers={{ cfa: true, ocl: true, pdaf: true, shield: false, binning: false, dti: false, readout: false }}
          selectedCell={oclState.selectedCell}
          onCellSelect={(cell) => {
            setOclState((current) => ({ ...current, selectedCell: cell }));
            onAction(`OCL map selected cell (${cell.col + 1}, ${cell.row + 1})`);
          }}
          compact
        />
        <h2 className="section-title">Linked Cross-Section Viewer</h2>
        <CrossSectionSvg />
      </main>
      <aside className="panel-surface parameter-panel">
        <header className="selected-class">
          <span>Selected Class</span>
          <strong>{activeClass}</strong>
          <StatusPill state="ok">Active</StatusPill>
        </header>
        <label className="select-row">
          <span>OCL Model</span>
          <select value={surfaceModel} onChange={(event) => {
            setOclState((current) => ({ ...current, surfaceModel: event.target.value }));
            onAction(`OCL model selected: ${event.target.value}`);
          }}>
            {OCL_SURFACE_MODELS.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        {classRows.map(renderParameterRow)}
        {advancedRows.map(renderParameterRow)}
        <div className="toggle-block">
          <span>CRA Compensation</span>
          <input type="checkbox" checked={oclState.craCompensation} onChange={() => {
            setOclState((current) => ({ ...current, craCompensation: !current.craCompensation }));
            onAction('CRA compensation toggled');
          }} />
        </div>
        <div className="small-metric-grid">
          <MiniMetric title="Group Uniformity" value="1.82%" tone="cyan" />
          <MiniMetric title="CRA Response @ 35deg" value="72.4%" tone="purple" />
          <MiniMetric title="Intra-group Crosstalk" value="3.45%" tone="yellow" />
          <MiniMetric title="Binning Uniformity" value="2.11%" tone="green" />
        </div>
        <div className="warning-callout">
          <TriangleAlert size={18} />
          <span>3x3 OCL requires aligned 3x3 CFA block or Custom Mode.</span>
        </div>
      </aside>
    </div>
  );
}

function MiniMetric({ title, value, tone }) {
  return (
    <div className={`mini-metric ${tone}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <MetricSpark color={tone === 'purple' ? '#c084fc' : tone === 'yellow' ? '#f6c445' : tone === 'green' ? '#76d16a' : '#22d3ee'} />
    </div>
  );
}

function CfaView({ selectedPreset, cfaState, setCfaState, onAction }) {
  const updateCfa = (key, value) => {
    setCfaState((current) => ({ ...current, [key]: value }));
    onAction(`CFA ${key} set to ${value}`);
  };
  return (
    <div className="detail-layout two-col-right">
      <main className="panel-surface cfa-main">
        <header className="viewer-toolbar">
          <div>
            <h2>CFA / Binning Color</h2>
            <span>{selectedPreset.cfa}</span>
          </div>
          <StatusPill state="accent">{selectedPreset.short}</StatusPill>
        </header>
        <div className="cfa-grid-large">
          {Array.from({ length: 64 }, (_, index) => {
            const row = Math.floor(index / 8);
            const col = index % 8;
            const color = colorForCell(selectedPreset, row, col);
            return <span className={colorClass(color)} key={index}>{color}</span>;
          })}
        </div>
        <div className="filter-stack">
          {[
            ['R', 'Red n,k', '#ef4444'],
            ['G', 'Green n,k', '#22c55e'],
            ['B', 'Blue n,k', '#3b82f6'],
            ['IR', 'IR / Clear', '#94a3b8']
          ].map(([label, name, color]) => (
            <div className="filter-row" key={label}>
              <i style={{ background: color }} />
              <span>{name}</span>
              <strong>{label === 'IR' ? 'optional' : 'proxy table'}</strong>
            </div>
          ))}
        </div>
      </main>
      <aside className="panel-surface parameter-panel">
        <h2>CFA Detail</h2>
        <div className="plain-row"><span>Pattern source</span><strong>Pattern Composer</strong></div>
        <div className="plain-row"><span>CFA group</span><strong>{selectedPreset.cfa}</strong></div>
        <label className="edit-row"><span>Thickness</span><input type="number" step="0.01" value={cfaState.thickness} onChange={(event) => updateCfa('thickness', Number(event.target.value))} /><em>um</em></label>
        <label className="select-row"><span>Aperture model</span><select value={cfaState.apertureModel} onChange={(event) => updateCfa('apertureModel', event.target.value)}><option>Full tile</option><option>Inset polygon</option></select></label>
        {cfaState.apertureModel === 'Inset polygon' ? (
          <>
            <label className="edit-row"><span>Edge inset</span><input type="number" step="0.005" value={cfaState.edgeInset} onChange={(event) => updateCfa('edgeInset', Number(event.target.value))} /><em>um</em></label>
            <label className="edit-row"><span>Edge skew</span><input type="number" step="0.005" value={cfaState.edgeSkew} onChange={(event) => updateCfa('edgeSkew', Number(event.target.value))} /><em>um</em></label>
            <label className="select-row"><span>Gap fill</span><select value={cfaState.gapBackground} onChange={(event) => updateCfa('gapBackground', event.target.value)}><option>passivation</option><option>nearest</option><option>air</option></select></label>
          </>
        ) : null}
        <label className="edit-row"><span>R shift X</span><input type="number" step="0.005" value={cfaState.rShift} onChange={(event) => updateCfa('rShift', Number(event.target.value))} /><em>um</em></label>
        <label className="edit-row"><span>R shift Z</span><input type="number" step="0.005" value={cfaState.rShiftZ} onChange={(event) => updateCfa('rShiftZ', Number(event.target.value))} /><em>um</em></label>
        <label className="edit-row"><span>G shift X</span><input type="number" step="0.005" value={cfaState.gShift} onChange={(event) => updateCfa('gShift', Number(event.target.value))} /><em>um</em></label>
        <label className="edit-row"><span>G shift Z</span><input type="number" step="0.005" value={cfaState.gShiftZ} onChange={(event) => updateCfa('gShiftZ', Number(event.target.value))} /><em>um</em></label>
        <label className="edit-row"><span>B shift X</span><input type="number" step="0.005" value={cfaState.bShift} onChange={(event) => updateCfa('bShift', Number(event.target.value))} /><em>um</em></label>
        <label className="edit-row"><span>B shift Z</span><input type="number" step="0.005" value={cfaState.bShiftZ} onChange={(event) => updateCfa('bShiftZ', Number(event.target.value))} /><em>um</em></label>
        <label className="select-row"><span>n,k source</span><select value={cfaState.source} onChange={(event) => updateCfa('source', event.target.value)}><option>proxy table</option><option>public literature</option><option>measured import placeholder</option></select></label>
        <label className="select-row"><span>Remosaic</span><select value={cfaState.remosaic} onChange={(event) => updateCfa('remosaic', event.target.value)}><option>defined</option><option>not required</option><option>custom pipeline required</option></select></label>
        <div className="compatibility-list">
          <h3>Compatibility</h3>
          <StatusPill state="ok">same-color group rule</StatusPill>
          <StatusPill state={selectedPreset.id === 'custom' ? 'warn' : 'ok'}>CFA / OCL alignment</StatusPill>
          <StatusPill state="warn">measured spectrum missing</StatusPill>
        </div>
      </aside>
    </div>
  );
}

function PdafShieldView({ selectedPreset, pdafState, setPdafState, onAction }) {
  const activeMode = pdafState.activeMode;
  const setActiveMode = (name) => {
    setPdafState((current) => ({ ...current, activeMode: name }));
    onAction(`PDAF mode selected: ${name}`);
  };
  const updateParam = (key, value) => {
    setPdafState((current) => ({ ...current, params: { ...current.params, [key]: value } }));
    onAction(`PDAF ${key} set to ${value}`);
  };
  const toggleLayer = (key) => {
    setPdafState((current) => ({ ...current, layers: { ...current.layers, [key]: !current.layers[key] } }));
    onAction(`PDAF layer ${key} toggled`);
  };
  return (
    <div className="pdaf-layout">
      <aside className="panel-surface mode-library">
        <h2>PDAF / Shield Mode</h2>
        {PDAF_MODES.map(([name, desc, role]) => (
          <button type="button" className={`mode-card ${activeMode === name ? 'active' : ''}`} key={name} onClick={() => setActiveMode(name)}>
            <span className="mode-glyph">{role}</span>
            <strong>{name}</strong>
            <em>{desc}</em>
          </button>
        ))}
      </aside>
      <main className="panel-surface pdaf-stage">
        <header className="viewer-toolbar">
          <div>
            <h2>Supercell Top View</h2>
            <span>CFA + OCL + PDAF groups</span>
          </div>
          <LayerToggles
            layers={pdafState.layers}
            onToggle={toggleLayer}
          />
        </header>
        <SupercellGrid
          preset={selectedPreset.id === 'quad_qpd' ? selectedPreset : PRESETS.find((preset) => preset.id === 'sparse_pdaf')}
          layers={pdafState.layers}
          selectedCell={pdafState.selectedCell}
          onCellSelect={(cell) => {
            setPdafState((current) => ({ ...current, selectedCell: cell }));
            onAction(`PDAF selected cell (${cell.col + 1}, ${cell.row + 1})`);
          }}
          compact
        />
        <div className="pdaf-cross-sections">
          <div>
            <h2>A. Half-shield L/R Pair</h2>
            <CrossSectionSvg variant="pdaf" />
          </div>
          <div>
            <h2>B. QPD 2x2 Group</h2>
            <CrossSectionSvg />
          </div>
        </div>
      </main>
      <aside className="panel-surface parameter-panel">
        <h2>PDAF / Shield Parameters</h2>
        <div className="plain-row"><span>Shield Mask Type</span><strong>{activeMode}</strong></div>
        <label className="edit-row"><span>Mask Ratio</span><input type="range" min="0.1" max="0.9" step="0.01" value={pdafState.params.maskRatio} onChange={(event) => updateParam('maskRatio', Number(event.target.value))} /><em>{number(pdafState.params.maskRatio, 2)}</em></label>
        <label className="edit-row"><span>Aperture Offset</span><input type="number" step="0.005" value={pdafState.params.apertureOffset} onChange={(event) => updateParam('apertureOffset', Number(event.target.value))} /><em>um</em></label>
        <label className="select-row"><span>Pairing Rule</span><select value={pdafState.params.pairingRule} onChange={(event) => updateParam('pairingRule', event.target.value)}><option>Same Color</option><option>Same Row</option><option>Custom Pair</option></select></label>
        <label className="select-row"><span>OCL Grouping</span><select value={pdafState.params.oclGrouping} onChange={(event) => updateParam('oclGrouping', event.target.value)}><option>Same OCL (2x2)</option><option>Mirror OCL</option><option>Independent OCL</option></select></label>
        <label className="select-row"><span>AF Density</span><select value={pdafState.params.afDensity} onChange={(event) => updateParam('afDensity', event.target.value)}><option>Low (3%)</option><option>Medium (12%)</option><option>High (25%)</option></select></label>
        <div className="chart-pair">
          <MiniMetric title="L/R Angular Response" value="4.2%" tone="purple" />
          <MiniMetric title="AF Confidence" value="0.87" tone="cyan" />
        </div>
        <div className="validation-list">
          <h3>PDAF / Shield Validation</h3>
          {['Mirrored shield masks', 'Same CFA color in pair', 'Same OCL group', 'Pair baseline valid'].map((item) => (
            <div className="validation-line ok" key={item}><CheckCircle2 size={15} />{item}</div>
          ))}
          <div className="validation-line warn"><TriangleAlert size={15} />AF-L/R QE mismatch: 4.2%</div>
        </div>
      </aside>
    </div>
  );
}

function HeatMap({ preset }) {
  return (
    <div className="response-map">
      {Array.from({ length: 48 }, (_, index) => {
        const row = Math.floor(index / 12);
        const col = index % 12;
        const color = colorForCell(preset, row, col);
        const qe = 69 + ((row * 7 + col * 3) % 16) + (color === 'R' ? 3 : color === 'G' ? 6 : 0);
        return <div className={`response-cell ${colorClass(color)}`} key={index}><strong>{color}</strong><span>{qe.toFixed(1)}</span></div>;
      })}
    </div>
  );
}

function PatternMetrics({ model }) {
  const rows = [
    ['OCL Group Uniformity (3x3)', model.metrics.oclUniformity, '0.94'],
    ['Intra-group Crosstalk', model.metrics.crosstalk, '3.1%'],
    ['Inter-group Crosstalk', 0.064, '6.4%'],
    ['PDAF Pair Balance', model.metrics.pdafBalance, '95.8%'],
    ['AF Coverage', model.metrics.afCoverage, '87.3%'],
    ['Binning Uniformity', 0.96, '0.96'],
    ['Remosaic Risk', model.metrics.remosaicRisk, '0.72'],
    ['Image Penalty', 0.68, '0.68 dB']
  ];
  return (
    <div className="pattern-metrics panel-surface">
      <h2>Pattern Metrics</h2>
      {rows.map(([label, value, text]) => (
        <div className="metric-bar-row" key={label}>
          <span>{label}</span>
          <strong>{text}</strong>
          <i><b style={{ width: `${clamp(Number(value), 0, 1) * 100}%` }} /></i>
        </div>
      ))}
    </div>
  );
}

function PatternResponseView({ selectedPreset, model, responseState, setResponseState, onAction }) {
  const tab = responseState.tab;
  const updateOption = (key, value) => {
    setResponseState((current) => ({ ...current, [key]: value }));
    onAction(`Response ${key} set to ${value}`);
  };
  return (
    <div className="response-layout">
      <main className="panel-surface response-main">
        <div className="response-tabs">
          {RESPONSE_TABS.map((item) => (
            <button type="button" key={item} className={tab === item ? 'active' : ''} onClick={() => {
              setResponseState((current) => ({ ...current, tab: item }));
              onAction(`Response tab selected: ${item}`);
            }}>
              {item}
            </button>
          ))}
        </div>
        <HeatMap preset={selectedPreset} />
        <div className="map-caption">
          <span>Supercell: 12 (H) x 4 (V)</span>
          <span>Metric: {tab} @ 550 nm, 30deg CRA</span>
        </div>
      </main>
      <PatternMetrics model={model} />
      <aside className="panel-surface display-options">
        <h2>Display Options</h2>
        <label className="select-row"><span>Metric</span><select value={tab} onChange={(event) => updateOption('tab', event.target.value)}>{RESPONSE_TABS.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label className="select-row"><span>Wavelength</span><select value={responseState.wavelength} onChange={(event) => updateOption('wavelength', event.target.value)}><option>450 nm</option><option>550 nm</option><option>650 nm</option><option>940 nm</option></select></label>
        <label className="select-row"><span>CRA</span><select value={responseState.cra} onChange={(event) => updateOption('cra', event.target.value)}><option>0deg</option><option>20deg</option><option>30deg</option><option>45deg</option><option>60deg</option></select></label>
        <label className="select-row"><span>Polarization</span><select value={responseState.polarization} onChange={(event) => updateOption('polarization', event.target.value)}><option>TE</option><option>TM</option><option>Unpol</option></select></label>
        <label className="select-row"><span>Plane</span><select value={responseState.plane} onChange={(event) => updateOption('plane', event.target.value)}><option>OCL Exit Plane</option><option>CFA Mid Plane</option><option>PD Top</option></select></label>
        <button type="button" className="secondary-button" onClick={() => onAction(`Exported response map: ${tab} ${responseState.wavelength} ${responseState.cra}`)}><Download size={15} />Export Map</button>
      </aside>
      <section className="panel-surface field-response">
        <h2>FDTD Field Viewer</h2>
        <CrossSectionSvg />
      </section>
      <section className="panel-surface chart-grid-response">
        <MiniMetric title="3x3 OCL Center vs Edge" value="69.4%" tone="cyan" />
        <MiniMetric title="PDAF QPD Balance" value="4.2%" tone="purple" />
        <MiniMetric title="Crosstalk Matrix" value="3.7%" tone="yellow" />
        <MiniMetric title="CRA Response" value="74.3%" tone="green" />
      </section>
      <aside className="panel-surface explain-panel">
        <h2>Explain Why</h2>
        {[
          ['3x3 corner subpixel QE lower', 'Corner subpixel shows 8-10% lower QE due to CRA vignetting.'],
          ['Focus spot shift at high CRA', 'At 35deg CRA, focus spot shifts toward DTI edge.'],
          ['PDAF L/R mismatch 4.2%', 'Shield and OCL asymmetry causes AF-L/R imbalance.'],
          ['Remosaic risk high near boundary', 'Mixed-OCL boundary can create zipper artifacts.']
        ].map(([title, body], index) => (
          <div className="explain-row" key={title}>
            {index % 2 ? <TriangleAlert size={20} /> : <Target size={20} />}
            <strong>{title}</strong>
            <span>{body}</span>
          </div>
        ))}
      </aside>
    </div>
  );
}

function Radar({ variant }) {
  const axes = [
    ['QE', variant.qe],
    ['CRA', variant.cra],
    ['AF Balance', variant.pdaf === 'None' ? 0.62 : 0.82],
    ['OCL Uniformity', 1 - variant.crosstalk],
    ['Remosaic Risk', 1 - variant.remosaic],
    ['Process Window', 0.78]
  ];
  const cx = 125;
  const cy = 118;
  const radius = 88;
  const points = axes.map(([, value], index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / axes.length;
    return [cx + Math.cos(angle) * radius * clamp(value), cy + Math.sin(angle) * radius * clamp(value)];
  });
  return (
    <svg className="radar" viewBox="0 0 250 236">
      {[0.33, 0.66, 1].map((scale) => <polygon key={scale} points={axes.map((_, index) => {
        const angle = -Math.PI / 2 + (index * Math.PI * 2) / axes.length;
        return `${cx + Math.cos(angle) * radius * scale},${cy + Math.sin(angle) * radius * scale}`;
      }).join(' ')} fill="none" stroke="#29495d" />)}
      {axes.map(([label], index) => {
        const angle = -Math.PI / 2 + (index * Math.PI * 2) / axes.length;
        return <text key={label} x={cx + Math.cos(angle) * (radius + 18)} y={cy + Math.sin(angle) * (radius + 18)} fill="#9db7c8" fontSize="10" textAnchor="middle">{label}</text>;
      })}
      <polygon points={points.map((point) => point.join(',')).join(' ')} fill="rgba(34,211,238,.22)" stroke="#22d3ee" strokeWidth="2" />
    </svg>
  );
}

function CompareView({ model, compareState, setCompareState, onAction }) {
  const active = model.variants.find((variant) => variant.id === compareState.activeId) || model.best;
  const toggleVariable = (name) => {
    setCompareState((current) => ({
      ...current,
      variables: { ...current.variables, [name]: !current.variables[name] }
    }));
    onAction(`Optimization variable toggled: ${name}`);
  };
  return (
    <div className="compare-page">
      <aside className="panel-surface optimization-setup">
        <h2>Optimization Setup</h2>
        {['ML Radius', 'OCL Shift', 'CFA Shift', 'DTI Depth', 'Shield Aperture', 'AF Density'].map((item) => (
          <label className="check-row" key={item}><input type="checkbox" checked={compareState.variables[item]} onChange={() => toggleVariable(item)} />{item}<span>{compareState.variables[item] ? 'range set' : 'held fixed'}</span></label>
        ))}
        <h2>Objective Weights</h2>
        {['QE', 'Crosstalk', 'CRA Score', 'AF Balance', 'Binning Uniformity', 'Remosaic Risk'].map((item, index) => (
          <div className="weight-row" key={item}><span>{item}</span><i><b style={{ width: `${80 - index * 8}%` }} /></i></div>
        ))}
        <div className="goal-card"><Target size={26} />Maximize robust perception score</div>
      </aside>
      <main className="panel-surface variant-decision">
          <div className="candidate-row">
          {model.variants.slice(0, 6).map((variant, index) => (
            <button type="button" className={`candidate-card ${variant.id === compareState.activeId ? 'active' : ''}`} key={variant.id} onClick={() => {
              setCompareState((current) => ({ ...current, activeId: variant.id }));
              onAction(`Variant selected: ${variant.label}`);
            }}>
              <span>{String.fromCharCode(65 + index)}</span>
              <strong>{variant.label}</strong>
              <em>{variant.cfa} CFA / {variant.ocl} OCL</em>
            </button>
          ))}
        </div>
        <div className="variant-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Pattern</th>
                <th>OCL Type</th>
                <th>PDAF Type</th>
                <th>CFA Period</th>
                <th>QE (%)</th>
                <th>XT (%)</th>
                <th>CRA Score</th>
                <th>Remosaic</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {model.variants.map((variant) => (
                <tr key={variant.id} className={variant.id === compareState.activeId ? 'selected' : ''}>
                  <td>{variant.label}</td>
                  <td>{variant.ocl}</td>
                  <td>{variant.pdaf}</td>
                  <td>{variant.cfa}</td>
                  <td>{number(variant.qe * 100, 1)}</td>
                  <td>{number(variant.crosstalk * 100, 1)}</td>
                  <td>{number(variant.cra, 2)}</td>
                  <td>{number(variant.remosaic, 2)}</td>
                  <td>{number(variant.score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="sensitivity-row">
          <MiniMetric title="ML Shift vs QE" value="76.3%" tone="cyan" />
          <MiniMetric title="DTI Depth vs Crosstalk" value="4.3%" tone="yellow" />
          <MiniMetric title="Shield Ratio vs PDAF" value="6.2%" tone="purple" />
          <MiniMetric title="3x3 OCL Shift" value="0.082" tone="green" />
        </div>
      </main>
      <aside className="panel-surface compare-aside">
        <h2>Pattern Comparison</h2>
        <Radar variant={active} />
        <div className="takeaway">
          <PackageCheck size={22} />
          <strong>Key Takeaway</strong>
          <span>{active.label} is the current robust trade-off across CRA, crosstalk, AF balance, and remosaic risk.</span>
        </div>
        <h2>Reports & Export</h2>
        <div className="export-grid">
          {['PDF Report', 'Pattern Package', 'FDTD Setup', 'Measurement Correlation', 'Design Rule Summary'].map((item) => (
            <button type="button" key={item} onClick={() => onAction(`Prepared export: ${item}`)}><FileText size={20} />{item}</button>
          ))}
        </div>
      </aside>
    </div>
  );
}

function SimulationRunPanel({ selectedPreset, simulation, onRunSimulation }) {
  const examples = simulation.examples.length ? simulation.examples : SOLVER_TEST_EXAMPLES;
  const suggested = simulation.suggestedExampleId || exampleIdForPreset(selectedPreset);
  const job = simulation.currentJob;
  const kpi = job?.kpi;
  const activeRequest = simulation.activeRequest;
  const status = job?.status || simulation.backendStatus;
  const isRunning = status === 'queued' || status === 'running' || simulation.starting;
  const responseImage = artifactUrl(kpi?.artifacts?.response_maps, simulation.apiBase);
  const focalImage = artifactUrl(kpi?.artifacts?.focal_maps, simulation.apiBase);
  const outputUrl = artifactUrl(job?.output_url, simulation.apiBase);
  const logUrl = artifactUrl(job?.log_url, simulation.apiBase);
  const requestUrl = artifactUrl(job?.request_url, simulation.apiBase);
  const solverCaseUrl = artifactUrl(job?.solver_case_url, simulation.apiBase);
  const kpiUrl = artifactUrl(job?.kpi_url || kpi?.artifacts?.kpi_summary, simulation.apiBase);
  const statusState =
    status === 'completed' ? 'ok' : status === 'failed' || status === 'offline' ? 'warn' : status === 'ready' ? 'ok' : 'accent';
  const numericalGate = kpi?.numerical_gate;
  const kpiRows = [
    ['KPI Status', kpi?.status, ''],
    ['Grid Gate', numericalGate?.passed ? 'PASS' : numericalGate?.available ? 'CHECK' : '-', ''],
    ['Rows', kpi?.row_count, ''],
    ['Center Response', kpi?.center_total_response, ''],
    ['Edge / Center', kpi?.edge_to_center_response, 'x'],
    ['CRA Delta', kpi?.edge_delta_pct, '%'],
    ['Region Imbalance', kpi?.region_imbalance_proxy, ''],
    ['Split Phase X', kpi?.split_phase_x_max, ''],
    ['Split Phase Z', kpi?.split_phase_z_max, ''],
    ['Phase Amp Max', kpi?.split_phase_amplitude_max, ''],
    ['Collection', kpi?.collection_modes?.join(', ') || '-', ''],
    ['CAD Template', kpi?.cad_template?.template_id || '-', ''],
    ['CFA Polygons', kpi?.cfa_polygon_count_max, ''],
    ['Geometry Import', kpi?.imported_geometry_sources?.join(', ') || '-', ''],
    ['Si Pixels', numericalGate?.min_si_internal_wavelength_pixels, 'px'],
    ['Feature Pixels', numericalGate?.min_critical_feature_pixels, 'px'],
    ['Negative Flux', kpi?.negative_signed_flux_count, 'rows']
  ];
  return (
    <section className={`simulation-panel panel-surface ${status || 'idle'}`}>
      <header className="simulation-panel-header">
        <div>
          <h2>Solver-backed Simulation Test</h2>
          <span>Active Design/Condition request, actual Meep run, KPI parser, and artifact viewer</span>
        </div>
        <StatusPill state={statusState}>{status || 'idle'}</StatusPill>
      </header>
      {activeRequest ? (
        <div className="simulation-request-preview">
          <div>
            <span>Active design request</span>
            <strong>{activeRequest.design?.preset_label}</strong>
          </div>
          <div><span>Mode</span><strong>{activeRequest.solver?.mode}</strong></div>
          <div><span>λ / channel</span><strong>{activeRequest.condition?.wavelength_nm} nm / {activeRequest.condition?.color_channel}</strong></div>
          <div><span>CRA X</span><strong>{activeRequest.condition?.cra_x_deg} deg</strong></div>
          <div><span>CFA pattern</span><strong>{activeRequest.solver?.cfa_pattern} · x/z R{formatSigned(activeRequest.solver?.cfa_shifts_um?.red?.x || 0, 3)}/{formatSigned(activeRequest.solver?.cfa_shifts_um?.red?.z || 0, 3)} G{formatSigned(activeRequest.solver?.cfa_shifts_um?.green?.x || 0, 3)}/{formatSigned(activeRequest.solver?.cfa_shifts_um?.green?.z || 0, 3)} B{formatSigned(activeRequest.solver?.cfa_shifts_um?.blue?.x || 0, 3)}/{formatSigned(activeRequest.solver?.cfa_shifts_um?.blue?.z || 0, 3)} um</strong></div>
          {activeRequest.design?.cad_template ? (
            <div><span>CAD source</span><strong>{activeRequest.design.cad_template.template_id}</strong></div>
          ) : null}
          <div><span>Geometry authority</span><strong>{activeRequest.design?.cad_template ? 'CAD template' : 'UI controls'}</strong></div>
          <div><span>CFA geometry</span><strong>{activeRequest.design?.cad_template ? 'CAD template cells' : (activeRequest.design?.cfa_geometry?.aperture_model || 'Full tile')}</strong></div>
          <div><span>OCL model</span><strong>{activeRequest.design?.cad_template ? 'CAD template footprint' : (activeRequest.design?.ocl_surface?.surface_model || 'Spherical cap')}</strong></div>
          <div><span>Collection</span><strong>{activeRequest.solver?.collection_mode || 'auto'}</strong></div>
          <button type="button" className="primary-button" disabled={isRunning || activeRequest.warnings?.length} onClick={() => onRunSimulation()}>
            <Play size={14} />
            Run Active Design
          </button>
        </div>
      ) : null}
      {activeRequest?.warnings?.length ? (
        <div className="simulation-check">
          <TriangleAlert size={16} />
          {activeRequest.warnings[0]}
        </div>
      ) : null}
      <div className="simulation-example-row">
        {examples.map((example) => (
          <button
            type="button"
            key={example.id}
            className={suggested === example.id ? 'suggested' : ''}
            disabled={isRunning}
            onClick={() => onRunSimulation(example.id)}
          >
            <Play size={14} />
            <strong>{example.label}</strong>
            <span>{example.mode || example.description || example.id}</span>
          </button>
        ))}
      </div>
      <div className="simulation-status-grid">
        <div><span>Backend</span><strong>{simulation.backendStatus}</strong></div>
        <div><span>Recommended</span><strong>{suggested}</strong></div>
        <div><span>Job</span><strong>{job?.id || '-'}</strong></div>
        <div><span>Output</span><strong>{outputUrl ? <a href={outputUrl}>open run</a> : '-'}</strong></div>
        <div><span>Log</span><strong>{logUrl ? <a href={logUrl}>job.log</a> : '-'}</strong></div>
        <div><span>Request</span><strong>{requestUrl ? <a href={requestUrl}>simulation_request.json</a> : '-'}</strong></div>
        <div><span>Solver Case</span><strong>{solverCaseUrl ? <a href={solverCaseUrl}>solver_case.json</a> : '-'}</strong></div>
        <div><span>KPI</span><strong>{kpiUrl ? <a href={kpiUrl}>kpi_summary.json</a> : '-'}</strong></div>
      </div>
      {simulation.error ? <div className="simulation-error"><TriangleAlert size={16} />{simulation.error}</div> : null}
      {kpi?.status === 'CHECK' ? (
        <div className="simulation-check">
          <TriangleAlert size={16} />
          KPI generated, but numerical gate needs review: {numericalGate?.notes?.[0] || 'resolution/convergence gate did not pass.'}
        </div>
      ) : null}
      {kpi ? (
        <div className="simulation-kpi-wrap">
          <div className="simulation-kpi-grid">
            {kpiRows.map(([label, value, unit]) => (
              <div className="simulation-kpi" key={label}>
                <span>{label}</span>
                <strong>{typeof value === 'number' ? number(value, label.includes('Response') ? 5 : 3) : value ?? '-'}</strong>
                {unit ? <em>{unit}</em> : null}
              </div>
            ))}
          </div>
          <div className="simulation-artifacts">
            {responseImage ? <img src={responseImage} alt="Meep response map" /> : <div className="artifact-placeholder">response_maps.png</div>}
            {focalImage ? <img src={focalImage} alt="Meep focal map" /> : <div className="artifact-placeholder">focal_maps.png</div>}
          </div>
        </div>
      ) : (
        <div className="simulation-empty">
          <Cpu size={19} />
          <span>Run a solver test to populate KPI values from camera_lut_summary.csv and camera_lut.json.</span>
        </div>
      )}
      {job?.log_tail?.length ? (
        <pre className="simulation-log">{job.log_tail.slice(-10).join('\n')}</pre>
      ) : null}
    </section>
  );
}

function SuiteLineChart({ title, points = [] }) {
  const clean = points.filter((point) => Number.isFinite(Number(point.cra_x_deg)) && Number.isFinite(Number(point.total_response)));
  const xs = clean.map((point) => Number(point.cra_x_deg));
  const ys = clean.map((point) => Number(point.total_response));
  const minX = xs.length ? Math.min(...xs) : 0;
  const maxX = xs.length ? Math.max(...xs) : 1;
  const minY = ys.length ? Math.min(...ys) : 0;
  const maxY = ys.length ? Math.max(...ys) : 1;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const path = clean.map((point, index) => {
    const x = 18 + ((Number(point.cra_x_deg) - minX) / spanX) * 174;
    const y = 106 - ((Number(point.total_response) - minY) / spanY) * 86;
    return `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
  return (
    <div className="suite-chart-card">
      <h3>{title}</h3>
      <svg viewBox="0 0 210 124">
        <path d="M18 106 H196 M18 20 V106" stroke="#24445a" fill="none" />
        {path ? <path d={path} stroke="#22d3ee" strokeWidth="2.5" fill="none" /> : null}
        {clean.map((point) => {
          const x = 18 + ((Number(point.cra_x_deg) - minX) / spanX) * 174;
          const y = 106 - ((Number(point.total_response) - minY) / spanY) * 86;
          return <circle key={`${point.case}-${point.cra_x_deg}`} cx={x} cy={y} r="3" fill="#f6c445" />;
        })}
      </svg>
      <span>{clean.length ? `${clean.length} CRA anchors` : 'no curve data'}</span>
    </div>
  );
}

function SuiteMatrixChart({ title, cells = [] }) {
  const clean = cells.filter((cell) => Number.isFinite(Number(cell.ix)) && Number.isFinite(Number(cell.iz)));
  const responses = clean.map((cell) => Number(cell.response)).filter(Number.isFinite);
  const maxResponse = responses.length ? Math.max(...responses) : 1;
  return (
    <div className="suite-chart-card">
      <h3>{title}</h3>
      <div className="suite-matrix">
        {clean.slice(0, 9).map((cell) => {
          const level = maxResponse ? clamp(Number(cell.response) / maxResponse, 0, 1) : 0;
          return (
            <span key={cell.region_id} style={{ opacity: 0.35 + level * 0.65 }}>
              {number(Number(cell.response), 4)}
            </span>
          );
        })}
      </div>
      <span>{clean.length} regions</span>
    </div>
  );
}

function SuiteTornado({ rows = [] }) {
  const clean = rows.filter((row) => Number.isFinite(Number(row.delta_pct_vs_nominal)));
  const maxAbs = clean.length ? Math.max(...clean.map((row) => Math.abs(Number(row.delta_pct_vs_nominal)))) || 1 : 1;
  return (
    <div className="suite-chart-card">
      <h3>Material Sensitivity Tornado</h3>
      <div className="suite-tornado">
        {clean.length ? clean.map((row) => {
          const value = Number(row.delta_pct_vs_nominal);
          return (
            <div key={row.case_id}>
              <span>{row.label}</span>
              <i><b className={value >= 0 ? 'pos' : 'neg'} style={{ width: `${Math.abs(value) / maxAbs * 100}%` }} /></i>
              <strong>{formatSigned(value, 2)}%</strong>
            </div>
          );
        }) : <em>Run material stack sensitivity to populate this chart.</em>}
      </div>
    </div>
  );
}

function ArtifactLink({ href, label, icon: Icon = FileText }) {
  if (!href) return null;
  return (
    <a href={href} className="suite-artifact-link">
      <Icon size={13} />
      {label}
    </a>
  );
}

function SuiteCadEvidence({ cases = [], apiBase }) {
  const evidenceCases = cases.filter((item) => item.kpi?.gds_import_pipeline || item.kpi?.gmsh_mesh_bridge);
  if (!evidenceCases.length) return null;
  return (
    <section className="suite-cad-evidence">
      <header>
        <div>
          <h3>CAD Import & Mesh Evidence</h3>
          <span>GDS layer-map validation, imported footprint preview, and Gmsh TCAD bridge artifacts</span>
        </div>
      </header>
      <div className="suite-cad-list">
        {evidenceCases.map((item) => {
          const kpi = item.kpi || {};
          const artifacts = kpi.artifacts || {};
          const validation = kpi.gds_import_validation || {};
          const bridge = kpi.gmsh_bridge || {};
          const previewUrl = artifactUrl(artifacts.gds_import_preview, apiBase);
          const bbox = validation.bbox_um || {};
          const bridgeStatus = bridge.status || (kpi.gmsh_mesh_bridge ? 'PASS' : '-');
          const nativeMesh = kpi.gmsh_native_mask_polygon_mesh === true || bridge.native_mask_polygon_mesh === true;
          return (
            <article className="suite-cad-card" key={item.id}>
              <div className="suite-cad-meta">
                <div className="suite-cad-title">
                  <strong>{item.label}</strong>
                  <StatusPill state={validation.status === 'PASS' ? 'ok' : validation.status === 'FAIL' ? 'warn' : 'accent'}>
                    GDS {validation.status || '-'}
                  </StatusPill>
                  <StatusPill state={bridgeStatus === 'PASS' ? 'ok' : bridgeStatus === 'FAIL' ? 'warn' : 'accent'}>
                    Gmsh {bridgeStatus}
                  </StatusPill>
                </div>
                <div className="suite-cad-grid">
                  <div><span>Polygons</span><strong>{validation.polygon_count ?? '-'}</strong></div>
                  <div><span>OCL / CFA</span><strong>{validation.matched_ocl_polygon_count ?? '-'} / {validation.matched_cfa_polygon_count ?? '-'}</strong></div>
                  <div><span>BBox</span><strong>{bbox.width ? `${number(bbox.width, 3)} x ${number(bbox.height, 3)} um` : '-'}</strong></div>
                  <div><span>Mesh type</span><strong>{nativeMesh ? 'native polygon' : 'bbox proxy'}</strong></div>
                </div>
                {!nativeMesh ? (
                  <div className="suite-cad-warning">
                    <TriangleAlert size={14} />
                    Gmsh mesh is a TCAD bridge mesh, not a polygon-preserving CAD mesh.
                  </div>
                ) : null}
                {validation.warnings?.length ? (
                  <div className="suite-cad-warning">
                    <TriangleAlert size={14} />
                    {validation.warnings[0]}
                  </div>
                ) : null}
                <div className="suite-artifact-links">
                  <ArtifactLink href={artifactUrl(artifacts.gds_import_report, apiBase)} label="GDS report" />
                  <ArtifactLink href={previewUrl} label="Preview SVG" icon={Eye} />
                  <ArtifactLink href={artifactUrl(artifacts.converted_geometry_json, apiBase)} label="Geometry JSON" />
                  <ArtifactLink href={artifactUrl(artifacts.gmsh_bridge_report, apiBase)} label="Gmsh bridge" />
                  <ArtifactLink href={artifactUrl(artifacts.gmsh_mesh_metadata, apiBase)} label="Mesh metadata" />
                  <ArtifactLink href={artifactUrl(artifacts.gmsh_mesh_2d, apiBase)} label="2D mesh" icon={Download} />
                </div>
              </div>
              {previewUrl ? <img src={previewUrl} alt={`${item.label} GDS import preview`} /> : <div className="artifact-placeholder">gds_import_preview.svg</div>}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SuiteTemplateEvidence({ cases = [], apiBase }) {
  const evidenceCases = cases.filter((item) => item.kpi?.cad_template);
  if (!evidenceCases.length) return null;
  return (
    <section className="suite-cad-evidence">
      <header>
        <div>
          <h3>CAD Template Source Evidence</h3>
          <span>FreeCAD-openable template source, FDTD footprint JSON, and solver provenance</span>
        </div>
      </header>
      <div className="suite-cad-list">
        {evidenceCases.map((item) => {
          const template = item.kpi?.cad_template || {};
          const importedSources = item.kpi?.imported_geometry_sources || [];
          const tcad = template.tcad_bridge || {};
          const freecadValidation = template.freecad_validation || {};
          const designRuleValidation = template.design_rule_validation || {};
          const ddSmoke = tcad.devsim_dd_smoke || {};
          const ddGate = ddSmoke.capability_gate || tcad.capability_gate || 'PASS';
          const ddAxis = ddSmoke.phase_result_axis || tcad.electrical_capability?.represented_split_axis || 'x';
          const ddPhase = ddAxis === 'z' ? (ddSmoke.photo_split_phase_z_proxy ?? ddSmoke.photo_split_phase_x_proxy) : ddSmoke.photo_split_phase_x_proxy;
          const coupledDd = template.coupled_tcad_dd_smoke || {};
          const coupledSummary = coupledDd.summary || {};
          return (
            <article className="suite-cad-card" key={item.id}>
              <div className="suite-cad-meta">
                <div className="suite-cad-title">
                  <strong>{template.template_id || item.label}</strong>
                  <StatusPill state={item.kpi?.status === 'PASS' ? 'ok' : 'warn'}>{item.kpi?.status || item.status}</StatusPill>
                </div>
                <div className="suite-cad-grid">
                  <div><span>Template</span><strong>{template.label || '-'}</strong></div>
                  <div><span>Truth level</span><strong>{template.source_truth_level || '-'}</strong></div>
                  <div><span>FreeCAD check</span><strong>{freecadValidation.available ? `${freecadValidation.status || 'CHECK'} · ${freecadValidation.step_solid_count || '-'} solids` : '-'}</strong></div>
                  <div><span>Design rules</span><strong>{designRuleValidation.available ? `${designRuleValidation.status || 'CHECK'} · ${designRuleValidation.fail_count || 0} fails` : '-'}</strong></div>
                  <div><span>Collection</span><strong>{item.kpi?.collection_modes?.join(', ') || '-'}</strong></div>
                  <div><span>Target lens</span><strong>{item.kpi?.target_lens_ids?.join(', ') || '-'}</strong></div>
                  <div><span>TCAD bridge</span><strong>{tcad.available ? tcad.status || 'available' : 'not generated'}</strong></div>
                  <div><span>TCAD scope</span><strong>{tcad.electrical_capability?.phase_result_scope || '-'}</strong></div>
                  <div><span>DD split phase</span><strong>{ddSmoke.available ? `${ddGate === 'CHECK' ? 'CHECK · ' : ''}phase ${ddAxis} ${number(ddPhase, 5)}` : '-'}</strong></div>
                  <div><span>Coupled DD</span><strong>{coupledDd.available ? number(coupledSummary.photo_split_phase_x_proxy, 5) : coupledDd.status || '-'}</strong></div>
                  <div><span>Coupled mode</span><strong>{coupledSummary.coupling_mode || '-'}</strong></div>
                  <div><span>Map scale</span><strong>{coupledSummary.generation_map_scale ? number(coupledSummary.generation_map_scale, 4) : '-'}</strong></div>
                </div>
                <div className="suite-artifact-links">
                  <ArtifactLink href={artifactUrl(String(template.geometry_import || '').replace(/^@/, '/'), apiBase)} label="Geometry JSON" />
                  <ArtifactLink href={artifactUrl(String(template.parameters || '').replace(/^@/, '/'), apiBase)} label="Parameters" />
                  <ArtifactLink href={artifactUrl(template.artifacts?.fcstd, apiBase)} label="FCStd" />
                  <ArtifactLink href={artifactUrl(template.artifacts?.freecad_validation_report, apiBase)} label="FreeCAD check" />
                  <ArtifactLink href={artifactUrl(template.artifacts?.tcad_bridge_report, apiBase)} label="TCAD bridge" />
                  <ArtifactLink href={artifactUrl(template.artifacts?.devsim_dd_summary, apiBase)} label="DD smoke" />
                  <ArtifactLink href={artifactUrl(coupledDd.summary_url, apiBase)} label="Coupled DD" />
                </div>
                {importedSources.length ? (
                  <div className="suite-cad-warning">
                    <CheckCircle2 size={14} />
                    Solver imported {importedSources.join(', ')}
                  </div>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SuiteCadVariantDeltaTable({ rows = [] }) {
  if (!rows.length) return null;
  return (
    <section className="suite-delta-table">
      <header>
        <div>
          <h3>CAD Variant Delta</h3>
          <span>Base template 대비 FDTD response와 scaled coupled-DD smoke 변화</span>
        </div>
      </header>
      <div className="suite-delta-grid">
        <span>Variant</span>
        <span>Base</span>
        <span>Override</span>
        <span>FDTD response</span>
        <span>FDTD delta</span>
        <span>DD phase</span>
        <span>DD delta</span>
        {rows.map((row) => (
          <Fragment key={row.case_id || row.template_id}>
            <strong>{row.label || row.template_id}</strong>
            <span>{row.base_label || row.variant_of || '-'}</span>
            <span>{compactValue(row.overrides)}</span>
            <span>{number(row.variant_center_total_response, 4)}</span>
            <b className={Number(row.delta_pct_vs_base) >= 0 ? 'pos' : 'neg'}>{number(row.delta_pct_vs_base, 1)}%</b>
            <span>{number(row.variant_dd_split_phase_x_proxy, 5)}</span>
            <b className={Number(row.dd_split_phase_delta_abs) >= 0 ? 'pos' : 'neg'}>{number(row.dd_split_phase_delta_abs, 5)}</b>
          </Fragment>
        ))}
      </div>
    </section>
  );
}

function SuiteResultView({ job, apiBase, onReplayCase, replayingCaseId, replayResult, replayJob }) {
  const result = job?.suite_result;
  if (!result) {
    return (
      <section className="suite-result-empty panel-surface">
        <Cpu size={20} />
        <span>Run a test suite to generate KPI cards, charts, and gate evidence.</span>
      </section>
    );
  }
  const suiteResultUrl = artifactUrl(job?.suite_result_url || result.artifacts?.suite_result, apiBase);
  const suiteSummaryUrl = artifactUrl(job?.suite_summary_url || result.artifacts?.suite_summary, apiBase);
  const cameraArtifactLinks = [
    ['camera_system_suite_export.json', result.artifacts?.camera_system_export],
    ['field_response.csv', result.artifacts?.camera_system_field_response_csv],
    ['pdaf_response.csv', result.artifacts?.camera_system_pdaf_response_csv],
    ['crosstalk_summary.csv', result.artifacts?.camera_system_crosstalk_summary_csv],
    ['crosstalk_cells.csv', result.artifacts?.camera_system_crosstalk_cells_csv],
    ['gate_report.csv', result.artifacts?.camera_system_gate_report_csv],
    ['consumer_validation.json', result.artifacts?.camera_system_validation],
    ['consumer_validation.md', result.artifacts?.camera_system_validation_md],
    ['field_query.csv', result.artifacts?.camera_system_field_query_csv],
    ['crosstalk_index.csv', result.artifacts?.camera_system_crosstalk_index_csv],
    ['gate_summary.csv', result.artifacts?.camera_system_gate_summary_csv]
  ].map(([label, href]) => [label, artifactUrl(href, apiBase)]).filter(([, href]) => href);
  const summaryRows = [
    ['Suite Status', result.status],
    ['Cases', result.case_count],
    ['Gate Failures', result.kpi_summary?.gate_failure_count],
    ['Edge min/max', `${number(result.kpi_summary?.edge_to_center_min, 3)} / ${number(result.kpi_summary?.edge_to_center_max, 3)}`],
    ['Phase amp max', result.kpi_summary?.split_phase_amplitude_max],
    ['Split cases', result.kpi_summary?.split_collection_case_count],
    ['CFA polygon cases', result.kpi_summary?.cfa_polygon_case_count],
    ['Imported geometry cases', result.kpi_summary?.imported_geometry_case_count],
    ['GDS import cases', result.kpi_summary?.gds_import_case_count],
    ['Gmsh bridge cases', result.kpi_summary?.gmsh_bridge_case_count],
    ['CAD variant deltas', result.kpi_summary?.cad_variant_delta_count],
    ['Max variant delta', result.kpi_summary?.cad_variant_max_abs_delta_pct === null || result.kpi_summary?.cad_variant_max_abs_delta_pct === undefined ? '-' : `${number(result.kpi_summary?.cad_variant_max_abs_delta_pct, 1)}%`],
    ['DD evidence rows', result.kpi_summary?.cad_variant_dd_available_count],
    ['Max DD phase delta', result.kpi_summary?.cad_variant_dd_max_abs_split_delta],
    ['Product LUT', result.product_lut_ready ? 'READY' : 'NOT READY']
  ];
  const firstCra = result.cases?.flatMap((item) => item.charts?.cra_response_curve?.points || [])[0]
    ? result.cases.flatMap((item) => item.charts?.cra_response_curve ? [{ title: item.label, points: item.charts.cra_response_curve.points }] : [])
    : [];
  const firstMatrix = result.cases?.find((item) => item.charts?.subpixel_response_matrix)?.charts?.subpixel_response_matrix;
  const crosstalkImage = result.cases?.map((item) => item.charts?.crosstalk_kernel_heatmap?.image_url).find(Boolean);
  const tornadoRows = result.charts?.material_sensitivity_tornado?.rows || [];
  const cadVariantDeltaRows = result.charts?.cad_variant_deltas?.rows || [];
  const replayDisplay = replayResult || replayJob?.replay_result || null;
  const replayStatus = replayDisplay?.status || (replayJob?.status === 'running' || replayJob?.status === 'queued' ? replayJob.status : replayJob?.status || '');
  const replayComparisonStatus = replayDisplay?.replay?.replay_comparison_status || '';
  const replayOutputUrl = replayDisplay?.output_url || replayJob?.output_url;
  const replayManifestUrl = replayDisplay?.replay_manifest_url || replayJob?.replay_manifest_url;
  const replayComparisonUrl = replayDisplay?.replay_comparison_url || replayJob?.replay_comparison_url;
  return (
    <section className="suite-result panel-surface">
      <header className="suite-result-header">
        <div>
          <h2>{result.suite_label}</h2>
          <span>{result.tier} tier · {result.gates?.measured_accuracy}</span>
        </div>
        <StatusPill state={result.status === 'PASS' ? 'ok' : 'warn'}>{result.status}</StatusPill>
      </header>
      <div className="suite-artifact-row">
        {suiteResultUrl ? <a className="suite-result-artifact-link" href={suiteResultUrl}>suite_result.json</a> : null}
        {suiteSummaryUrl ? <a className="suite-result-artifact-link" href={suiteSummaryUrl}>workbench_suite_summary.json</a> : null}
        {cameraArtifactLinks.map(([label, href]) => (
          <a key={label} className="suite-result-artifact-link" href={href}>{label}</a>
        ))}
      </div>
      {replayDisplay || replayJob ? (
        <div className={`suite-replay-result ${replayStatus === 'PASS' ? 'pass' : 'fail'}`}>
          <strong>Replay {replayStatus || '-'}</strong>
          {replayOutputUrl ? <a href={artifactUrl(replayOutputUrl, apiBase)}>output</a> : null}
          {replayManifestUrl ? <a href={artifactUrl(replayManifestUrl, apiBase)}>replay_manifest.json</a> : null}
          {replayComparisonUrl ? <a href={artifactUrl(replayComparisonUrl, apiBase)}>replay_comparison.json</a> : null}
          <span>{replayComparisonStatus ? `comparison ${replayComparisonStatus}` : replayJob?.error || ''}</span>
        </div>
      ) : null}
      <div className="suite-kpi-grid">
        {summaryRows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value ?? '-'}</strong>
          </div>
        ))}
      </div>
      {result.gates?.gate_failures?.length ? (
        <div className="suite-gate-note">
          <TriangleAlert size={16} />
          {result.gates.gate_failures.length} case(s) need review. Smoke results are not product LUT evidence.
        </div>
      ) : null}
      <div className="suite-chart-grid">
        {firstCra.slice(0, 3).map((chart) => <SuiteLineChart key={chart.title} title={chart.title} points={chart.points} />)}
        {firstMatrix ? <SuiteMatrixChart title="Subpixel Response Matrix" cells={firstMatrix.cells} /> : null}
        {tornadoRows.length ? <SuiteTornado rows={tornadoRows} /> : null}
        {crosstalkImage ? (
          <div className="suite-chart-card">
            <h3>Crosstalk Kernel</h3>
            <img src={artifactUrl(crosstalkImage, apiBase)} alt="Crosstalk kernel" />
          </div>
        ) : null}
      </div>
      <SuiteCadVariantDeltaTable rows={cadVariantDeltaRows} />
      <SuiteCadEvidence cases={result.cases || []} apiBase={apiBase} />
      <SuiteTemplateEvidence cases={result.cases || []} apiBase={apiBase} />
      <div className="suite-case-table">
        {result.cases?.map((item) => (
          <div key={item.id}>
            <strong>{item.label}</strong>
            <StatusPill state={item.kpi?.status === 'PASS' ? 'ok' : 'warn'}>{item.kpi?.status || item.status}</StatusPill>
            <span>{factorSummary(item.design_factors || {})}</span>
            <span className="suite-case-artifacts">
              {item.artifacts?.case_result ? <a href={artifactUrl(item.artifacts.case_result, apiBase)}>case_result.json</a> : null}
              {item.artifacts?.case_input ? <a href={artifactUrl(item.artifacts.case_input, apiBase)}>case_input.json</a> : null}
              {item.artifacts?.case_command ? <a href={artifactUrl(item.artifacts.case_command, apiBase)}>case_command.json</a> : null}
              {item.artifacts?.solver_case ? <a href={artifactUrl(item.artifacts.solver_case, apiBase)}>solver_case.json</a> : null}
              {item.artifacts?.case_command ? (
                <button type="button" onClick={() => onReplayCase(item)} disabled={replayingCaseId === item.id}>
                  {replayingCaseId === item.id ? 'Replaying' : 'Replay + Compare'}
                </button>
              ) : null}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function TestSuiteView({ simulation, suiteState, setSuiteState, onRunSuite, onReplayCase }) {
  const suites = simulation.testSuites || [];
  const selectedSuite = suites.find((suite) => suite.id === suiteState.selectedSuiteId) || suites[0];
  const tier = suiteState.tier || selectedSuite?.recommended_tier || 'smoke';
  const job = simulation.currentJob?.kind === 'suite' ? simulation.currentJob : null;
  const isRunning = ['queued', 'running'].includes(job?.status) || simulation.startingSuite;
  const suiteCases = selectedSuite?.cases || [];
  const allCaseIds = suiteCases.map((testCase) => testCase.id);
  const selectedCaseIds = suiteState.selectedCaseIds === undefined ? allCaseIds : suiteState.selectedCaseIds;
  const selectedCaseIdSet = new Set(selectedCaseIds);
  const runLabel = isRunning ? 'Suite Running' : `Run ${selectedCaseIds.length || 'No'} Case${selectedCaseIds.length === 1 ? '' : 's'}`;
  const setAllCases = () => setSuiteState((current) => ({ ...current, selectedCaseIds: allCaseIds }));
  const clearCases = () => setSuiteState((current) => ({ ...current, selectedCaseIds: [] }));
  const toggleCase = (caseId) => {
    setSuiteState((current) => {
      const currentIds = current.selectedCaseIds === undefined ? allCaseIds : current.selectedCaseIds;
      const next = currentIds.includes(caseId)
        ? currentIds.filter((item) => item !== caseId)
        : [...currentIds, caseId];
      return { ...current, selectedCaseIds: next };
    });
  };
  return (
    <div className="test-suite-page">
      <aside className="panel-surface suite-catalog">
        <h2>Practical Test Suite</h2>
        {suites.map((suite) => (
          <button
            type="button"
            key={suite.id}
            className={selectedSuite?.id === suite.id ? 'active' : ''}
            onClick={() => setSuiteState((current) => ({
              ...current,
              selectedSuiteId: suite.id,
              tier: suite.recommended_tier || 'smoke',
              selectedCaseIds: (suite.cases || []).map((testCase) => testCase.id)
            }))}
          >
            <strong>{suite.label}</strong>
            <span>{suite.category} · {suite.runtime_hint}</span>
          </button>
        ))}
      </aside>
      <main className="test-suite-main">
        <section className="panel-surface suite-setup">
          <header className="suite-result-header">
            <div>
              <h2>{selectedSuite?.label || 'Test Suite'}</h2>
              <span>{selectedSuite?.decision_goal || 'Backend test catalog is loading.'}</span>
            </div>
            <StatusPill state={simulation.backendStatus === 'ready' ? 'ok' : 'warn'}>{simulation.backendStatus}</StatusPill>
          </header>
          <div className="suite-tier-row">
            {(selectedSuite?.tiers || ['smoke']).map((item) => (
              <button type="button" key={item} className={tier === item ? 'active' : ''} onClick={() => setSuiteState((current) => ({ ...current, tier: item }))}>{item}</button>
            ))}
            <span>{selectedCaseIds.length} / {allCaseIds.length} cases selected</span>
            <button type="button" onClick={setAllCases} disabled={isRunning || selectedCaseIds.length === allCaseIds.length}>All</button>
            <button type="button" onClick={clearCases} disabled={isRunning || !selectedCaseIds.length}>Clear</button>
            <button type="button" className="primary-button" disabled={!selectedSuite || isRunning || !selectedCaseIds.length} onClick={() => onRunSuite(selectedSuite.id, tier, selectedCaseIds)}>
              <Play size={15} />
              {runLabel}
            </button>
          </div>
          <div className="suite-matrix-list">
            {suiteCases.map((testCase) => (
              <label className={selectedCaseIdSet.has(testCase.id) ? 'selected' : ''} key={testCase.id}>
                <input
                  type="checkbox"
                  checked={selectedCaseIdSet.has(testCase.id)}
                  disabled={isRunning}
                  onChange={() => toggleCase(testCase.id)}
                />
                <strong>{testCase.label}</strong>
                <span>{testCase.runner} · {(testCase.tiers || []).join(', ')}</span>
                <em>{factorSummary(testCase.design_factors || {})}</em>
              </label>
            ))}
          </div>
          {simulation.error ? <div className="simulation-error"><TriangleAlert size={16} />{simulation.error}</div> : null}
          {job?.progress ? <div className="suite-progress">Progress: {job.progress.completed} / {job.progress.total} {job.progress.current_case ? `· ${job.progress.current_case}` : ''}</div> : null}
        </section>
        <SuiteResultView
          job={job}
          apiBase={simulation.apiBase}
          onReplayCase={onReplayCase}
          replayingCaseId={simulation.replayingCaseId}
          replayResult={simulation.replayResult}
          replayJob={simulation.replayJob}
        />
      </main>
    </div>
  );
}

function readinessPillState(value) {
  if (value === true || value === 'PASS' || value === 'PRODUCT_READY') return 'ok';
  if (value === false || value === 'FAIL' || value === 'RESEARCH_READY_NOT_PRODUCT') return 'warn';
  return 'accent';
}

function SolverRoleMatrixPanel({ matrix = {}, compact = false }) {
  const rows = matrix.rows || [];
  if (!rows.length) return null;
  return (
    <section className={`solver-role-panel ${compact ? 'compact' : ''}`}>
      <header>
        <div>
          <h3>Solver Role Matrix</h3>
          <span>{matrix.summary || 'Solver scope and decision role are not indexed.'}</span>
        </div>
        <StatusPill state={matrix.status === 'PASS' ? 'ok' : 'warn'}>{matrix.status || 'CHECK'}</StatusPill>
      </header>
      <div className="solver-role-paths">
        <span>Primary: {(matrix.primary_decision_path || []).join(' + ') || '-'}</span>
        <span>Diagnostic: {(matrix.diagnostic_path || []).join(' + ') || '-'}</span>
        <span>Missing: {(matrix.missing_accuracy_track || []).join(' + ') || '-'}</span>
      </div>
      <div className="solver-role-grid">
        {rows.map((row) => (
          <article className="solver-role-row" key={row.id}>
            <div>
              <strong>{row.label}</strong>
              <span>{row.role} · {row.availability} · {row.evidence_count ?? 0} evidence</span>
            </div>
            <p>{row.current_use}</p>
            <em>Not for: {row.not_for}</em>
            <b>{row.product_gate || 'CHECK'}</b>
          </article>
        ))}
      </div>
    </section>
  );
}

function ReadinessView({ model, simulation }) {
  const payload = model.payload || {};
  const solverRoleMatrix = simulation?.cadTemplateCatalog?.solver_role_matrix || {};
  const staticEvidence = (payload.results || []).find((item) => String(item.path || '').includes('camera_system_quantitative_evidence.json'));
  const [state, setState] = useState({ loading: true, apiBase: '', data: null, error: '' });
  useEffect(() => {
    let cancelled = false;
    setState((current) => ({ ...current, loading: true, error: '' }));
    fetchWorkbenchApi('/api/simulation/quantitative-evidence', {
      method: 'POST',
      body: JSON.stringify({
        config: payload.project_config || 'configs/image_sensor_pixel_studio_reference.json',
        output_dir: 'runs/camera_system_quantitative_evidence_reference'
      })
    })
      .then(({ base, data }) => {
        if (!cancelled) setState({ loading: false, apiBase: base, data, error: '' });
      })
      .catch((error) => {
        if (!cancelled) setState({ loading: false, apiBase: '', data: null, error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, [payload.project_config]);

  const data = state.data || {};
  const blockers = data.blockers || [];
  const evidenceRows = data.evidence || [];
  const artifacts = data.artifacts || {};
  const artifactLinks = [
    ['quantitative_evidence.json', artifacts.manifest_json],
    ['evidence.csv', artifacts.evidence_csv],
    ['blockers.csv', artifacts.blockers_csv],
    ['report.md', artifacts.report_md],
    ['static payload entry', staticEvidence?.path]
  ].map(([label, href]) => [label, artifactUrl(href, state.apiBase)]).filter(([, href]) => href);
  const cards = [
    ['Framework', data.framework_ready],
    ['Research LUT', data.research_lut_ready],
    ['Quant Evidence', data.quantitative_evidence_pass],
    ['Accuracy', data.accuracy_ready],
    ['Product LUT', data.product_lut_ready],
    ['Blockers', data.blocker_count ?? blockers.length]
  ];
  return (
    <div className="readiness-page">
      <section className="panel-surface readiness-main">
        <header className="suite-result-header">
          <div>
            <h2>Quantitative Readiness</h2>
            <span>Convergence, LUT validation, and product-accuracy blocker index</span>
          </div>
          <StatusPill state={readinessPillState(data.status)}>{state.loading ? 'loading' : data.status || 'CHECK'}</StatusPill>
        </header>
        <div className="suite-artifact-row">
          {artifactLinks.map(([label, href]) => (
            <a key={label} className="suite-result-artifact-link" href={href}>{label}</a>
          ))}
        </div>
        {state.error ? (
          <div className="simulation-error"><TriangleAlert size={16} />{state.error}</div>
        ) : null}
        <div className="suite-kpi-grid readiness-kpi-grid">
          {cards.map(([label, value]) => (
            <div key={label}>
              <span>{label}</span>
              <strong>{value === true ? 'PASS' : value === false ? 'NO' : value ?? '-'}</strong>
            </div>
          ))}
        </div>
        <div className="readiness-note">
          <TriangleAlert size={16} />
          <span>Research readiness is not product LUT readiness. Product LUT remains blocked until measured stack/material/device calibration and full quantitative convergence pass.</span>
        </div>
        <SolverRoleMatrixPanel matrix={solverRoleMatrix} />
        <section className="readiness-table">
          <header>
            <h3>Product Blockers</h3>
            <span>{blockers.length} rows</span>
          </header>
          {(blockers.length ? blockers.slice(0, 12) : [{ id: 'none', category: 'status', status: 'PASS', details: 'No blockers reported by the current evidence manifest.' }]).map((item, index) => (
            <div key={`${item.category}-${item.id}-${index}`} className="readiness-row">
              <strong>{item.id}</strong>
              <StatusPill state={item.status === 'PASS' ? 'ok' : 'warn'}>{item.status}</StatusPill>
              <span>{item.category}</span>
              <em>{item.details}</em>
            </div>
          ))}
        </section>
      </section>
      <aside className="panel-surface readiness-side">
        <header>
          <h2>Evidence Rows</h2>
          <span>{evidenceRows.length || data.evidence_count || 0} checks</span>
        </header>
        <div className="readiness-evidence-list">
          {evidenceRows.slice(0, 12).map((item, index) => (
            <div key={`${item.category}-${item.id}-${index}`}>
              <strong>{item.label || item.id}</strong>
              <StatusPill state={item.status === 'PASS' ? 'ok' : item.status === 'FAIL' ? 'warn' : 'accent'}>{item.status}</StatusPill>
              <span>{item.summary || item.category}</span>
            </div>
          ))}
          {!evidenceRows.length ? (
            <div>
              <strong>Evidence API</strong>
              <StatusPill state={state.error ? 'warn' : 'accent'}>{state.error ? 'offline' : 'loading'}</StatusPill>
              <span>{staticEvidence?.path || 'No static evidence row found in payload.'}</span>
            </div>
          ) : null}
        </div>
      </aside>
    </div>
  );
}

function CadTemplateView({ selectedPreset, setSelectedPreset, setSelectedCell, simulation, cadTemplateId, setCadTemplateId, onAction, onRunSimulation, onOpenCadArtifact, onOpenWorkspacePath, onOpenFcstdWorkingCopy, onCreateFcstdWorkingCopy, onCreateCadVariant, onCreateBaseTemplate, onExtractFcstdParameters, onCreateVariantFromFcstd, onValidateFreecad, onGenerateTcadBridge, onRunTcadDdSmoke, onRunTcadAxisPairSmoke, onRunTcadQpdWeighting3d, onRunQpdGw3d }) {
  const catalog = simulation.cadTemplateCatalog || {};
  const templates = catalog.templates || [];
  const qpdComparison = catalog.qpd_template_comparison || {};
  const qpdComparisonRows = qpdComparison.rows || [];
  const solverRoleMatrix = catalog.solver_role_matrix || {};
  const starterSet = catalog.starter_template_set || {};
  const starterRows = starterSet.templates || [];
  const starterIds = new Set(starterRows.map((row) => row.template_id));
  const starterTemplates = templates.filter((template) => template.starter_template || starterIds.has(template.template_id));
  const variantTemplates = templates.filter((template) => template.variant_of);
  const otherBaseTemplates = templates.filter((template) => !template.variant_of && !starterIds.has(template.template_id));
  const templateGroups = [
    ['Starter Templates', starterTemplates, 'common design starting points'],
    ['Other Base Templates', otherBaseTemplates, 'additional controlled sources'],
    ['Registered Variants', variantTemplates, 'recorded design changes']
  ].filter(([, groupTemplates]) => groupTemplates.length);
  const [browsedTemplateId, setBrowsedTemplateId] = useState(cadTemplateId || templates[0]?.template_id || '');
  const [variantOverrides, setVariantOverrides] = useState({});
  const [variantLabel, setVariantLabel] = useState('');
  const [variantId, setVariantId] = useState('');
  const [fcstdImportPath, setFcstdImportPath] = useState('');
  const [advancedTemplateActionsOpen, setAdvancedTemplateActionsOpen] = useState(false);
  const baseTopologyPresets = simulation.cadTools?.base_template_topology_presets || [];
  const [baseTemplateTopology, setBaseTemplateTopology] = useState('qpd_2x2');
  const [baseTemplateLabel, setBaseTemplateLabel] = useState('');
  const [baseTemplateId, setBaseTemplateId] = useState('');
  const [baseTemplatePitch, setBaseTemplatePitch] = useState('1.400');
  const activeTemplateId = cadTemplateId || browsedTemplateId;
  const activeTemplate = templates.find((item) => item.template_id === activeTemplateId) || templates[0];
  const selectedBaseTopology = baseTopologyPresets.find((item) => item.id === baseTemplateTopology) || baseTopologyPresets[0] || {};
  const artifacts = activeTemplate?.artifacts || {};
  const assumptionLedger = activeTemplate?.assumption_ledger || {};
  const simulationFidelity = activeTemplate?.simulation_fidelity || {};
  const dimensionSummary = activeTemplate?.dimension_summary || {};
  const pitchScaledFields = Array.isArray(dimensionSummary.pitch_scaled_fields) ? dimensionSummary.pitch_scaled_fields : [];
  const pitchAbsoluteFields = Array.isArray(dimensionSummary.pitch_absolute_fields) ? dimensionSummary.pitch_absolute_fields : [];
  const pitchScaleTitle = [
    dimensionSummary.notes?.find?.((note) => String(note).includes('not a uniform scale operation')) || 'Changing pitch is not a uniform scale operation.',
    pitchScaledFields.length ? `Pitch-linked: ${pitchScaledFields.join(', ')}.` : '',
    pitchAbsoluteFields.length ? `Absolute unless overridden: ${pitchAbsoluteFields.join(', ')}.` : ''
  ].filter(Boolean).join(' ');
  const tcadBridge = activeTemplate?.tcad_bridge || {};
  const freecadValidation = activeTemplate?.freecad_validation || {};
  const designRuleValidation = activeTemplate?.design_rule_validation || {};
  const devsimSmoke = tcadBridge.devsim_import_smoke || {};
  const devsimDdSmoke = tcadBridge.devsim_dd_smoke || {};
  const axisPairSmoke = tcadBridge.axis_pair_smoke || {};
  const qpdWeighting3d = tcadBridge.qpd_weighting_3d || {};
  const qpdGw3d = tcadBridge.qpd_gw_3d || {};
  const tcadCapability = tcadBridge.electrical_capability || {};
  const tcadCapabilityGate = tcadBridge.capability_gate || devsimDdSmoke.capability_gate || 'PASS';
  const ddSolverGate = devsimDdSmoke.solver_gate || 'PASS';
  const ddSmokeGate = tcadCapabilityGate === 'FAIL' || ddSolverGate === 'FAIL' ? 'FAIL' : (tcadCapabilityGate === 'CHECK' || ddSolverGate === 'CHECK' ? 'CHECK' : 'PASS');
  const ddPhaseAxis = devsimDdSmoke.phase_result_axis || tcadCapability.represented_split_axis || 'x';
  const ddPhaseValue = ddPhaseAxis === 'z'
    ? (devsimDdSmoke.photo_split_phase_z_proxy ?? devsimDdSmoke.photo_split_phase_x_proxy)
    : devsimDdSmoke.photo_split_phase_x_proxy;
  const splitPhaseMetricApplicable = devsimDdSmoke.phase_metric_applicable !== false && tcadCapability.requested_split_axis !== 'none';
  const templateText = `${activeTemplate?.template_id || ''} ${activeTemplate?.label || ''}`;
  const isQpdTemplate = /\bqpd\b|qpd_/i.test(templateText);
  const hasElectricalBridge = Boolean(tcadBridge.available);
  const devsimImportValue = diagnosticText(devsimSmoke.available, `${devsimSmoke.node_count || '-'} nodes`, hasElectricalBridge);
  const ddSmokeMetric = splitPhaseMetricApplicable
    ? `phase-proxy ${ddPhaseAxis} ${number(ddPhaseValue, 4)}`
    : `connectivity proxy · ${devsimDdSmoke.node_count || '-'} nodes`;
  const ddSmokeValue = diagnosticText(devsimDdSmoke.available, `${ddSmokeGate === 'PASS' ? '' : `${ddSmokeGate} · `}${ddSmokeMetric}`, hasElectricalBridge);
  const axisPairValue = diagnosticText(axisPairSmoke.available, `${axisPairSmoke.status || 'CHECK'} · x ${number(axisPairSmoke.phase_x_proxy, 4)} z ${number(axisPairSmoke.phase_z_proxy, 4)}`, isQpdTemplate);
  const qpdWeightingValue = diagnosticText(qpdWeighting3d.available, `${qpdWeighting3d.full_q1q4_weighting_gate || 'CHECK'} · uniform ${number(qpdWeighting3d.quadrant_uniformity, 4)}`, isQpdTemplate);
  const qpdGwGate = qpdGw3d.generation_volume_gate === 'CHECK' || qpdGw3d.full_q1q4_gw_gate === 'CHECK' ? 'CHECK' : (qpdGw3d.full_q1q4_gw_gate || 'CHECK');
  const qpdGwValue = diagnosticText(qpdGw3d.available, `${qpdGwGate} · ${qpdGw3d.case_count || 0} cases · phase-proxy x ${number(qpdGw3d.phase_x_gw, 4)} z ${number(qpdGw3d.phase_z_gw, 4)}`, isQpdTemplate);
  const qpdGwCurveValue = diagnosticText(qpdGw3d.available, `E/C ${number(qpdGw3d.edge_to_center_response_ratio_min ?? qpdGw3d.edge_to_center_response_ratio_max, 4)} · phase-proxy slope ${number(qpdGw3d.phase_x_slope_per_deg_max_abs, 5)}/deg`, isQpdTemplate);
  const electricalBridgeApplies = (activeTemplate?.counts?.photodiode || 0) > 0 || hasElectricalBridge;
  const devsimImportDisplay = diagnosticText(devsimSmoke.available, devsimImportValue, electricalBridgeApplies);
  const ddSmokeDisplay = diagnosticText(devsimDdSmoke.available, ddSmokeValue, electricalBridgeApplies);
  const electricalModel = devsimDdSmoke.electrical_model || '-';
  const electricalScope = tcadCapability.phase_result_scope || '-';
  const electricalModelIsProxy = /proxy|generic|projection|mapped/i.test(`${electricalModel} ${electricalScope}`);
  const full3dDdReady = simulationFidelity.full_3d_drift_diffusion === true;
  const pixelPitchUm = dimensionSummary.pixel_pitch_um ?? activeTemplate?.parameters?.pitch_um;
  const footprintXUm = dimensionSummary.footprint_x_um ?? (dimensionSummary.array_nx && pixelPitchUm ? dimensionSummary.array_nx * pixelPitchUm : null);
  const footprintZUm = dimensionSummary.footprint_z_um ?? (dimensionSummary.array_nz && pixelPitchUm ? dimensionSummary.array_nz * pixelPitchUm : null);
  const pitchVariantPolicy = dimensionSummary.pitch_variant_policy === 'conditional_scalar_variant' ? 'conditional variant' : 'new base template';
  const opticalGenerationPath = simulationFidelity.optical_generation || (artifacts.fdtd_generation_volume_3d?.exists ? '3D FDTD volume' : 'not generated');
  const electricalDdPath = simulationFidelity.electrical_dd || (devsimDdSmoke.available ? '2D DEVSIM cross-section proxy' : 'not run');
  const freecadBbox = freecadValidation.bbox_um || {};
  const fcstdHasSheets = freecadValidation.fcstd?.contains_parameter_sheet && freecadValidation.fcstd?.contains_validation_sheet;
  const previewUrl = artifactUrl(artifacts.footprint_preview?.url, simulation.apiBase);
  const validationStatus = catalog.validation?.status || catalog.status || 'CHECK';
  const freecad = simulation.cadTools?.freecad || {};
  const freecadLabel = freecad.installed ? `FreeCAD ${freecad.bundle_version || ''}`.trim() : 'FreeCAD not found';
  const activeFcstdWorkingCopy = simulation.fcstdWorkingCopy?.template_id === activeTemplate?.template_id ? simulation.fcstdWorkingCopy : null;
  const activeFcstdWorkingPath = activeFcstdWorkingCopy?.fcstd_path || activeFcstdWorkingCopy?.artifact?.path || '';
  const activeFcstdExtract = simulation.fcstdParameterExtract?.template_id === activeTemplate?.template_id ? simulation.fcstdParameterExtract : null;
  const fcstdOverrideCount = Object.keys(activeFcstdExtract?.overrides || {}).length;
  const fcstdBlockedCount = (activeFcstdExtract?.unsupported_changes || []).length;
  const cadSourceFolder = parentPath(artifacts.step?.path || artifacts.fcstd?.path || artifacts.geometry_import?.path);
  const presetMatchedTemplateId = cadTemplateIdForPreset(selectedPreset);
  const templateMatchesPreset = !presetMatchedTemplateId || presetMatchedTemplateId === activeTemplate?.template_id || presetMatchedTemplateId === activeTemplate?.variant_of;
  const cadSourceRows = [
    ['Source folder', cadSourceFolder, ''],
    ['STEP opened by FreeCAD', artifacts.step?.path, artifacts.step?.url],
    ['FreeCAD native FCStd', artifacts.fcstd?.path, artifacts.fcstd?.url],
    ['BREP solid', artifacts.brep?.path, artifacts.brep?.url],
    ['Geometry import JSON', artifacts.geometry_import?.path, artifacts.geometry_import?.url]
  ].filter(([, path]) => path);

  useEffect(() => {
    if (!browsedTemplateId && templates[0]?.template_id) setBrowsedTemplateId(templates[0].template_id);
  }, [browsedTemplateId, templates]);

  useEffect(() => {
    setVariantOverrides({});
    setVariantLabel('');
    setVariantId('');
  }, [activeTemplate?.template_id]);

  useEffect(() => {
    setFcstdImportPath(activeFcstdWorkingPath || '');
  }, [activeTemplate?.template_id, activeFcstdWorkingPath]);

  useEffect(() => {
    if (pixelPitchUm) setBaseTemplatePitch(number(pixelPitchUm, 3));
  }, [activeTemplate?.template_id, pixelPitchUm]);

  if (!templates.length) {
    return (
      <div className="cad-template-page">
        <section className="panel-surface suite-result-empty">
          <TriangleAlert size={18} />
          <span>CAD template library is not loaded. Generate it with .tcad-env/bin/python pixel_cad_template_library.py.</span>
        </section>
      </div>
    );
  }

  const artifactLinks = [
    ['STEP', artifacts.step?.url, Box],
    ['BREP', artifacts.brep?.url, Box],
    ['FCStd', artifacts.fcstd?.url, Box],
    ['3D Mesh', artifacts.mesh?.url, Download],
    ['Geometry JSON', artifacts.geometry_import?.url, FileText],
    ['Parameters', artifacts.parameters?.url, Settings],
    ['Variant Source', artifacts.variant_source?.url, FileText],
    ['Assumptions', artifacts.assumption_ledger?.url, TriangleAlert],
    ['Preview SVG', artifacts.footprint_preview?.url, Eye],
    ['TCAD Bridge', artifacts.tcad_bridge_report?.url, FileText],
    ['Axis Pair Summary', artifacts.tcad_axis_pair_summary?.url, FileText],
    ['Axis Pair Plot', artifacts.tcad_axis_pair_plot?.url, Eye],
    ['QPD 3D Weighting', artifacts.tcad_qpd_weighting_3d_summary?.url, FileText],
    ['QPD 3D Plot', artifacts.tcad_qpd_weighting_3d_plot?.url, Eye],
    ['QPD 3D G*W', artifacts.tcad_qpd_gw_3d_summary?.url, FileText],
    ['QPD G*W Plot', artifacts.tcad_qpd_gw_3d_plot?.url, Eye],
    ['QPD G*W CSV', artifacts.tcad_qpd_gw_3d_csv?.url, FileText],
    ['FDTD 3D Volume', artifacts.fdtd_generation_volume_3d?.url, Download],
    ['FDTD 2D Map', artifacts.fdtd_generation_map_2d?.url, Download],
    ['FDTD Smoke KPI', artifacts.fdtd_smoke_kpi?.url, FileText],
    ['2D TCAD Mesh', artifacts.tcad_mesh_2d?.url, Download],
    ['DEVSIM Smoke', artifacts.devsim_import_summary?.url, CheckCircle2],
    ['DD Smoke', artifacts.devsim_dd_summary?.url, CheckCircle2],
    ['DD Currents', artifacts.devsim_split_currents?.url, FileText],
    ['DD Plot', artifacts.devsim_split_currents_plot?.url, Eye],
    ['Validation', catalog.validation_report?.url, CheckCircle2],
    ['FreeCAD Validation', catalog.freecad_validation_report?.url || artifacts.freecad_validation_report?.url, CheckCircle2],
    ['Manifest', catalog.manifest?.url, FileText]
  ];
  const quickVariantOverrides = Object.fromEntries(
    Object.entries(variantOverrides).filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== '')
  );
  const quickVariantCount = Object.keys(quickVariantOverrides).length;
  const basePitchNumber = Number(baseTemplatePitch);
  const generatedBaseTemplateId = (baseTemplateId || `${baseTemplateTopology}_${Number.isFinite(basePitchNumber) ? `${basePitchNumber.toFixed(3).replace('.', 'p')}um` : 'custom'}`).toLowerCase();
  const generatedBaseTemplateLabel = baseTemplateLabel || `${selectedBaseTopology.label || 'CAD base template'} ${Number.isFinite(basePitchNumber) ? `${basePitchNumber.toFixed(3)} um` : ''}`.trim();

  return (
    <div className="cad-template-page">
      <aside className="panel-surface cad-template-catalog">
        <header>
          <h2>CAD Templates</h2>
          <StatusPill state={validationStatus === 'PASS' ? 'ok' : 'warn'}>{validationStatus}</StatusPill>
        </header>
        <span>{catalog.base_template_count ?? starterTemplates.length} base · {catalog.variant_count ?? variantTemplates.length} variants · {catalog.template_count || templates.length} total</span>
        <div className="cad-starter-summary">
          <div>
            <strong>{starterSet.pass_count ?? 0}/{starterSet.template_count ?? starterTemplates.length}</strong>
            <span>starter templates ready</span>
          </div>
          <StatusPill state={starterSet.status === 'PASS' ? 'ok' : 'warn'}>{starterSet.status || 'CHECK'}</StatusPill>
        </div>
        {templateGroups.map(([groupLabel, groupTemplates, groupHint]) => (
          <section className="cad-template-group" key={groupLabel}>
            <h3>{groupLabel}</h3>
            <span>{groupHint}</span>
            {groupTemplates.map((template) => (
              <button
                type="button"
                key={template.template_id}
                className={activeTemplate?.template_id === template.template_id ? 'active' : ''}
                onClick={() => {
                  const matchingPreset = presetForCadTemplate(template);
                  setBrowsedTemplateId(template.template_id);
                  setCadTemplateId(template.template_id);
                  if (matchingPreset && matchingPreset.id !== selectedPreset.id) {
                    setSelectedPreset(matchingPreset);
                    setSelectedCell(selectedCellForPreset(matchingPreset));
                    onAction(`CAD template selected for simulation: ${template.label} · preset synced to ${matchingPreset.label}`);
                  } else {
                    onAction(`CAD template selected for simulation: ${template.label}`);
                  }
                }}
              >
                <strong>{template.label}</strong>
                <span>{template.template_id}</span>
                <em>{template.readiness?.status === 'PASS' ? 'starter-ready geometry source' : (template.solver_ready ? 'solver-ready geometry source' : template.source_truth_level)}</em>
              </button>
            ))}
          </section>
        ))}
      </aside>
      <main className="panel-surface cad-template-detail">
        <header className="suite-result-header">
          <div>
            <h2>{activeTemplate?.label || 'CAD Template'}</h2>
            <span>{catalog.generated_with} · {catalog.accuracy_status}</span>
          </div>
          <StatusPill state="warn">not measured</StatusPill>
        </header>
        <div className="cad-template-hero">
          {previewUrl ? <img src={previewUrl} alt={`${activeTemplate.label} footprint preview`} /> : <div className="artifact-placeholder">footprint_preview.svg</div>}
          <div className="cad-template-summary">
            <CadSummaryItem label="Selected preset" value={selectedPreset.label} title="Currently selected design intent in the workbench." />
            <CadSummaryItem label="Pixel pitch" value={`${number(pixelPitchUm, 3)} um`} title="Single-pixel pitch from template_parameters.json. This is the primary pixel-size value for the CAD template." />
            <CadSummaryItem label="Template span" value={`${dimensionSummary.array_nx || '-'} x ${dimensionSummary.array_nz || '-'} px · ${number(footprintXUm, 3)} x ${number(footprintZUm, 3)} um`} title="Template footprint equals array_nx * pixel pitch by array_nz * pixel pitch." />
            <CadSummaryItem label="OCL group pitch" value={dimensionSummary.effective_ocl_group_pitch_label || '-'} title={dimensionSummary.effective_binning_note || 'OCL/binning group pitch equals OCL span times single-pixel pitch.'} />
            <CadSummaryItem label="Crosstalk coverage" value={dimensionSummary.crosstalk_kernel_label || '-'} warning={dimensionSummary.crosstalk_kernel_status !== 'PASS'} title={dimensionSummary.crosstalk_kernel_note || 'Shows whether the template has enough neighboring OCL groups for crosstalk kernel analysis.'} />
            <CadSummaryItem label="Pitch variant" value={pitchVariantPolicy} warning={dimensionSummary.pitch_variant_policy === 'conditional_scalar_variant'} title={dimensionSummary.pitch_variant_rule || 'Changing pixel pitch requires regenerated CAD, mesh, FDTD, and TCAD artifacts before comparison.'} />
            <CadSummaryItem label="Pitch scale" value={dimensionSummary.pitch_scaling_label || 'mixed scale'} warning title={pitchScaleTitle} />
            <CadSummaryItem label="Topology key" value={dimensionSummary.topology_signature || '-'} title="Changing nx/nz, OCL blocks, CFA pattern, split mode, or shield topology should be a new base template." />
            <CadSummaryItem label="Source role" value="parametric CAD source" title="This geometry is generated from controlled parametric CAD, not measured product CAD." />
            <CadSummaryItem label="Simulation basis" value={simulationFidelity.summary || '3D CAD + hybrid 2D/3D'} warning title="3D CAD geometry is used, but the current electrical path is hybrid rather than a native full 3D drift-diffusion solve." />
            <CadSummaryItem label="Optical path" value={opticalGenerationPath} warning={!artifacts.fdtd_generation_volume_3d?.exists && !qpdGw3d.available} title="Shows whether a 3D FDTD generation volume exists for this CAD template." />
            <CadSummaryItem label="Electrical DD" value={electricalDdPath} warning title="Current drift-diffusion smoke uses 2D DEVSIM cross-sections/proxy collection, not full 3D DD." />
            <CadSummaryItem label="Full 3D DD" value={full3dDdReady ? 'available' : 'not available'} warning={!full3dDdReady} title={simulationFidelity.not_full_3d_reason || 'Full 3D drift-diffusion is not available for the current template.'} />
            <CadSummaryItem label="Preset match" value={templateMatchesPreset ? 'matches CAD template' : `mismatch · expected ${presetMatchedTemplateId}`} warning={!templateMatchesPreset} title="Whether the selected CAD template matches the current design preset." />
            <CadSummaryItem label="Variant" value={activeTemplate.variant_of ? `of ${activeTemplate.variant_of}` : 'base template'} title="Shows whether this template is a base geometry or a registered variant." />
            <CadSummaryItem label="FreeCAD" value={activeTemplate.freecad_openable ? 'STEP/BREP openable' : '-'} title="Whether the CAD artifact can be opened for visual inspection in FreeCAD." />
            <CadSummaryItem label="FreeCAD check" value={freecadValidation.available ? `${freecadValidation.status || 'CHECK'} · ${freecadValidation.step_solid_count || '-'} solids` : 'not run'} title="FreeCAD validation of exported solids and parameter sheets." />
            <CadSummaryItem label="Design rules" value={designRuleValidation.available ? `${designRuleValidation.status || 'CHECK'} · ${designRuleValidation.fail_count || 0} fails` : 'not indexed'} title="Basic geometry rule validation for the generated template." />
            <CadSummaryItem label="CAD bbox" value={freecadBbox.xlen ? `${number(freecadBbox.xlen, 2)} x ${number(freecadBbox.ylen, 2)} x ${number(freecadBbox.zlen, 2)} um` : '-'} title="Bounding-box size of the CAD solid in micrometers." />
            <CadSummaryItem label="FCStd" value={artifacts.fcstd?.exists ? (fcstdHasSheets ? 'generated · parameter sheets' : 'generated') : 'not generated'} title="FreeCAD native file availability and embedded parameter-sheet status." />
            <CadSummaryItem label="FCStd copy" value={activeFcstdWorkingPath ? 'working copy ready' : 'create before edit'} title="Working copies are the editable files; base FCStd files are controlled references." />
            <CadSummaryItem label="Overrides" value={Object.keys(activeTemplate.parameter_overrides || {}).length || '-'} title="Number of scalar CAD parameter overrides applied to this template." />
            <CadSummaryItem label="Assumptions" value={assumptionLedger.available ? `${assumptionLedger.assumption_count} tracked / ${assumptionLedger.measured_blocker_count} blockers` : 'not indexed'} title="Tracked assumptions and measured-data blockers for product accuracy." />
            <CadSummaryItem label="CAD mesh" value={artifacts.mesh?.exists ? '3D Gmsh available' : 'not generated'} title="3D Gmsh review mesh availability; this is not a calibrated product TCAD mesh." />
            <CadSummaryItem label="Mesh validation" value={`${catalog.validation?.mesh_pass_count ?? 0} / ${catalog.template_count || templates.length} pass`} title="How many catalog templates passed mesh validation." />
            <CadSummaryItem label="TCAD bridge" value={tcadBridge.available ? `${tcadBridge.status || 'CHECK'} 2D mesh` : 'not generated'} title="Availability of the parameter-derived 2D electrical bridge mesh." />
            <CadSummaryItem label="TCAD scope" value={electricalScope} warning={electricalModelIsProxy && hasElectricalBridge} title="Electrical split/phase capability represented by the current template." />
            <CadSummaryItem label="DEVSIM import" value={devsimImportDisplay} title={diagnosticTitle(devsimSmoke.available, 'DEVSIM import smoke', electricalBridgeApplies)} />
            <CadSummaryItem label="DD smoke" value={ddSmokeDisplay} warning={devsimDdSmoke.available && (ddSmokeGate !== 'PASS' || !splitPhaseMetricApplicable)} title={devsimDdSmoke.solver_gate_reason || devsimDdSmoke.phase_metric_reason || 'Phase-proxy is normalized split photocurrent imbalance, (right-left)/(right+left), not optical phase in radians.'} />
            <CadSummaryItem label="Electrical model" value={electricalModel} warning={electricalModelIsProxy && devsimDdSmoke.available} title="DEVSIM electrical model used by the smoke run; proxy models prove connectivity, not product accuracy." />
            <CadSummaryItem label="Axis pair" value={axisPairValue} title={diagnosticTitle(axisPairSmoke.available, 'QPD x/z axis-pair diagnostic', isQpdTemplate)} />
            <CadSummaryItem label="QPD 3D W" value={qpdWeightingValue} title={diagnosticTitle(qpdWeighting3d.available, 'QPD 3D weighting-potential diagnostic', isQpdTemplate)} />
            <CadSummaryItem label="QPD 3D G*W" value={qpdGwValue} warning={qpdGw3d.available && qpdGwGate !== 'PASS'} title={qpdGw3d.generation_volume_reason || diagnosticTitle(qpdGw3d.available, 'QPD 3D optical-generation times weighting diagnostic', isQpdTemplate)} />
            <CadSummaryItem label="G*W field curve" value={qpdGwCurveValue} title={diagnosticTitle(qpdGw3d.available, 'QPD 3D G*W field curve', isQpdTemplate)} />
            <CadSummaryItem label="OCL / CFA / PD" value={`${activeTemplate.counts?.ocl ?? '-'} / ${activeTemplate.counts?.cfa ?? '-'} / ${activeTemplate.counts?.photodiode ?? '-'}`} title="Counts of microlens/OCL, color-filter, and photodiode objects in the CAD template." />
            <CadSummaryItem label="DTI / Shield" value={`${activeTemplate.counts?.dti ?? '-'} / ${activeTemplate.counts?.shield ?? '-'}`} title="Counts of isolation and PDAF shield objects in the CAD template." />
            <CadSummaryItem label="FDTD geometry input" value={artifacts.geometry_import?.exists ? 'available' : 'missing'} title="Whether CAD-derived geometry_import.json exists for FDTD runs; this is solver input availability, not proof that FDTD has already run." />
          </div>
        </div>
        <div className="cad-template-links">
          {artifactLinks.map(([label, path, Icon]) => (
            path ? (
              <a href={artifactUrl(path, simulation.apiBase)} key={label}>
                <Icon size={15} />
                {label}
              </a>
            ) : null
          ))}
        </div>
        <SolverRoleMatrixPanel matrix={solverRoleMatrix} compact />
        {qpdComparisonRows.length ? (
          <section className="qpd-comparison-panel">
            <header>
              <div>
                <h3>QPD Template Comparison</h3>
                <span>3D FDTD generation × 3D weighting surrogate; compare lens, DTI, shield, and field coverage before choosing a QPD variant.</span>
              </div>
              <StatusPill state={qpdComparison.status === 'PASS' ? 'ok' : 'warn'}>{qpdComparison.status || 'CHECK'}</StatusPill>
            </header>
            <div className="variant-table-wrap qpd-comparison-table">
              <table>
                <thead>
                  <tr>
                    <th>Template</th>
                    <th>Lens H</th>
                    <th>DTI W</th>
                    <th>Shield</th>
                    <th>DD</th>
                    <th>G*W phase</th>
                    <th>Uniform</th>
                    <th>Qsum</th>
                    <th>Field</th>
                    <th>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {qpdComparisonRows.map((row) => {
                    const isActive = row.template_id === activeTemplate?.template_id;
                    const phase = `x ${number(row.qpd_gw_phase_x, 4)} / z ${number(row.qpd_gw_phase_z, 4)}`;
                    const qsumDelta = row.qsum_delta_pct_from_base === null || row.qsum_delta_pct_from_base === undefined
                      ? ''
                      : ` (${number(row.qsum_delta_pct_from_base, 1)}%)`;
                    const fieldText = row.field_curve_gate === 'PASS'
                      ? `PASS · ${row.case_count} cases`
                      : `CHECK · ${row.case_count || 0} case`;
                    return (
                      <tr key={row.template_id} className={isActive ? 'active' : ''} title={row.label || row.template_id}>
                        <td>
                          <strong>{row.label || row.template_id}</strong>
                          <span>{row.template_id}</span>
                        </td>
                        <td>{number(row.lens_height_um, 3)} um <span>{number(row.lens_height_delta_pct_from_base, 1, '0.0')}%</span></td>
                        <td>{number(row.dti_width_um === null || row.dti_width_um === undefined ? null : row.dti_width_um * 1000, 0)} nm <span>{number(row.dti_width_delta_nm_from_base, 0, '0')} nm</span></td>
                        <td>{row.shield_mode || '-'}</td>
                        <td>{row.dd_phase_metric_applicable ? `phase x ${number(row.dd_phase_proxy_x, 4)}` : 'connectivity'}</td>
                        <td>{phase}</td>
                        <td>{number(row.quadrant_uniformity_gw, 4)} <span>{number(row.uniformity_delta_from_base, 4, '0.0000')}</span></td>
                        <td>{percent(row.generation_weighted_qsum_fraction, 2)}<span>{qsumDelta}</span></td>
                        <td><StatusPill state={row.field_curve_gate === 'PASS' ? 'ok' : 'warn'}>{fieldText}</StatusPill></td>
                        <td>{(row.decision_flags || []).join(', ') || 'ready'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p>Baseline: {qpdComparison.baseline_template_id || 'not set'} · Full Q1-Q4 DD remains CHECK until native calibrated 3D drift-diffusion is available.</p>
          </section>
        ) : null}
        <section className="cad-source-panel">
          <header>
            <div>
              <h3>CAD Source Files</h3>
              <span>Open STEP in FreeCAD launches the STEP file below; FCStd working copies should be edited instead of base files.</span>
            </div>
            <div className="cad-source-header-actions">
              <button type="button" disabled={!cadSourceFolder || simulation.openingWorkspacePath} onClick={() => onOpenWorkspacePath(cadSourceFolder, 'CAD source folder')}>
                <FolderOpen size={15} />
                {simulation.openingWorkspacePath ? 'Opening Folder' : 'Open Source Folder'}
              </button>
              <StatusPill state={artifacts.step?.exists ? 'ok' : 'warn'}>{artifacts.step?.exists ? 'STEP ready' : 'missing STEP'}</StatusPill>
            </div>
          </header>
          <div className="cad-source-grid">
            {cadSourceRows.map(([label, path, url]) => (
              <div key={label}>
                <span>{label}</span>
                <strong title={path}>{path}</strong>
                {url ? <a href={artifactUrl(url, simulation.apiBase)}>artifact</a> : null}
              </div>
            ))}
          </div>
        </section>
        <section className="cad-workflow-panel">
          <header>
            <div>
              <h3>Template Workflow</h3>
              <span>Select geometry, edit a copy, register a variant, then run solver cases from that CAD source.</span>
            </div>
            <StatusPill state={cadTemplateId ? 'ok' : 'warn'}>{cadTemplateId ? 'CAD source active' : 'select source'}</StatusPill>
          </header>
          <div className="cad-template-actions cad-template-actions-primary">
            <button type="button" disabled={!artifacts.step?.exists || simulation.openingCadArtifact} onClick={() => onOpenCadArtifact(activeTemplate.template_id, 'step', true)}>
              <Box size={15} />
              Open STEP in FreeCAD
            </button>
            <button type="button" disabled={!artifacts.fcstd?.exists || simulation.creatingFcstdWorkingCopy} onClick={() => onCreateFcstdWorkingCopy(activeTemplate.template_id)}>
              <Save size={15} />
              {simulation.creatingFcstdWorkingCopy ? 'Making FCStd Copy' : 'Make FCStd Working Copy'}
            </button>
            <button type="button" disabled={!fcstdImportPath || simulation.openingFcstdWorkingCopy} onClick={() => onOpenFcstdWorkingCopy(fcstdImportPath)}>
              <Box size={15} />
              {simulation.openingFcstdWorkingCopy ? 'Opening Working Copy' : 'Open Working Copy'}
            </button>
            <button type="button" disabled={!artifacts.fcstd?.exists || simulation.extractingFcstdParameters} onClick={() => onExtractFcstdParameters(activeTemplate.template_id, fcstdImportPath)}>
              <FileText size={15} />
              {simulation.extractingFcstdParameters ? 'Reading FCStd' : 'Read FCStd Parameters'}
            </button>
            <button type="button" disabled={!artifacts.fcstd?.exists || simulation.creatingFcstdVariant} onClick={() => onCreateVariantFromFcstd(activeTemplate.template_id, fcstdImportPath)}>
              <Plus size={15} />
              {simulation.creatingFcstdVariant ? 'Creating FCStd Variant' : 'Create Variant From FCStd'}
            </button>
            <button type="button" disabled={!activeTemplate?.template_id || simulation.validatingFreecad} onClick={() => onValidateFreecad(activeTemplate.template_id)}>
              <CheckCircle2 size={15} />
              {simulation.validatingFreecad ? 'Validating FreeCAD' : 'Validate FreeCAD'}
            </button>
            <button type="button" className="primary-button" disabled={!activeTemplate?.solver_ready || simulation.starting} onClick={() => onRunSimulation()}>
              <Play size={15} />
              Run CAD Template
            </button>
            <button type="button" disabled={!cadTemplateId} onClick={() => {
              setCadTemplateId('');
              onAction('CAD template source cleared from active simulation');
            }}>
              Clear CAD Source
            </button>
            <span>{freecadLabel} · {cadTemplateId ? `${cadTemplateId} is active simulation geometry` : 'Select a template to use CAD geometry in solver runs'}</span>
          </div>
          <button
            type="button"
            className={`cad-advanced-toggle ${advancedTemplateActionsOpen ? 'open' : ''}`}
            aria-expanded={advancedTemplateActionsOpen}
            onClick={() => setAdvancedTemplateActionsOpen((current) => !current)}
          >
            <ChevronRight size={15} />
            Advanced TCAD / diagnostics
          </button>
          {advancedTemplateActionsOpen ? (
            <div className="cad-template-actions cad-template-actions-advanced">
              <button type="button" disabled={!artifacts.brep?.exists || simulation.openingCadArtifact} onClick={() => onOpenCadArtifact(activeTemplate.template_id, 'brep', true)}>
                <Box size={15} />
                Open BREP
              </button>
              <button type="button" disabled={!artifacts.fcstd?.exists || simulation.openingCadArtifact} onClick={() => onOpenCadArtifact(activeTemplate.template_id, 'fcstd', true)}>
                <Box size={15} />
                Open FCStd
              </button>
              <button type="button" disabled={!artifacts.assumption_ledger?.exists || simulation.openingCadArtifact} onClick={() => onOpenCadArtifact(activeTemplate.template_id, 'assumption_ledger', false)}>
                <TriangleAlert size={15} />
                Open Assumptions
              </button>
              <button type="button" disabled={!activeTemplate?.template_id || simulation.generatingTcadBridge} onClick={() => onGenerateTcadBridge(activeTemplate.template_id)}>
                <Workflow size={15} />
                {simulation.generatingTcadBridge ? 'Generating TCAD' : 'Generate TCAD Bridge'}
              </button>
              <button type="button" disabled={!activeTemplate?.template_id || simulation.runningTcadDdSmoke} onClick={() => onRunTcadDdSmoke(activeTemplate.template_id)}>
                <Cpu size={15} />
                {simulation.runningTcadDdSmoke ? 'Running DD Smoke' : 'Run TCAD DD Smoke'}
              </button>
              <button type="button" disabled={!activeTemplate?.template_id || simulation.runningTcadAxisPairSmoke} onClick={() => onRunTcadAxisPairSmoke(activeTemplate.template_id)}>
                <Gauge size={15} />
                {simulation.runningTcadAxisPairSmoke ? 'Running Axis Pair' : 'Run QPD Axis Pair'}
              </button>
              <button type="button" disabled={!activeTemplate?.template_id || simulation.runningTcadQpdWeighting3d} onClick={() => onRunTcadQpdWeighting3d(activeTemplate.template_id)}>
                <Grid2X2 size={15} />
                {simulation.runningTcadQpdWeighting3d ? 'Running 3D Weighting' : 'Run QPD 3D Weighting'}
              </button>
              <button type="button" disabled={!activeTemplate?.template_id || simulation.runningQpdGw3d} onClick={() => onRunQpdGw3d(activeTemplate.template_id)}>
                <Zap size={15} />
                {simulation.runningQpdGw3d ? 'Running 3D G*W' : 'Run QPD 3D G*W'}
              </button>
            </div>
          ) : null}
        </section>
        <section className="fcstd-roundtrip-panel">
          <header>
            <div>
              <h3>FCStd Working Copy</h3>
              <span>Use a copied FreeCAD file for edits; keep base templates as controlled references.</span>
            </div>
            <StatusPill state={activeFcstdWorkingPath ? 'ok' : 'warn'}>{activeFcstdWorkingPath ? 'copy ready' : 'base only'}</StatusPill>
          </header>
          <label>
            <span>FCStd import path</span>
            <input
              value={fcstdImportPath}
              placeholder={artifacts.fcstd?.path || 'Create a working copy or paste a workspace FCStd path'}
              onChange={(event) => setFcstdImportPath(event.target.value)}
            />
          </label>
          <div className="fcstd-roundtrip-status">
            {activeFcstdWorkingCopy?.fcstd_url ? <a href={artifactUrl(activeFcstdWorkingCopy.fcstd_url, simulation.apiBase)}>FCStd copy artifact</a> : <span>No working copy has been created for this template yet.</span>}
            {activeFcstdExtract ? <span>{fcstdOverrideCount} scalar override{fcstdOverrideCount === 1 ? '' : 's'} · {fcstdBlockedCount} blocked topology change{fcstdBlockedCount === 1 ? '' : 's'}</span> : <span>Read parameters after saving the working copy in FreeCAD.</span>}
          </div>
        </section>
        <section className="cad-base-template-builder">
          <header>
            <div>
              <h3>Create Base Template</h3>
              <span>Use this for topology changes: pixel count, OCL grouping, CFA pattern, split mode, or shield mode.</span>
            </div>
            <StatusPill state="accent">{baseTopologyPresets.length || 0} presets</StatusPill>
          </header>
          <div className="cad-base-template-grid">
            <label>
              <span>Topology preset</span>
              <select value={baseTemplateTopology} onChange={(event) => setBaseTemplateTopology(event.target.value)}>
                {baseTopologyPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>{preset.label || preset.id}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Pixel pitch</span>
              <input value={baseTemplatePitch} onChange={(event) => setBaseTemplatePitch(event.target.value)} />
            </label>
            <label>
              <span>Template id</span>
              <input value={baseTemplateId} placeholder={generatedBaseTemplateId} onChange={(event) => setBaseTemplateId(event.target.value)} />
            </label>
            <label>
              <span>Label</span>
              <input value={baseTemplateLabel} placeholder={generatedBaseTemplateLabel} onChange={(event) => setBaseTemplateLabel(event.target.value)} />
            </label>
          </div>
          <div className="cad-base-template-rule">
            <TriangleAlert size={15} />
            <span>New base templates regenerate CAD source files and reset solver artifacts; run FDTD/G*W/TCAD diagnostics again before comparing to existing templates.</span>
          </div>
          <div className="cad-template-actions compact">
            <button
              type="button"
              className="primary-button"
              disabled={!baseTopologyPresets.length || !Number.isFinite(basePitchNumber) || simulation.creatingCadBaseTemplate}
              onClick={() => onCreateBaseTemplate({
                topology_preset: baseTemplateTopology,
                id: generatedBaseTemplateId,
                label: generatedBaseTemplateLabel,
                parameters: { pitch_um: basePitchNumber }
              })}
            >
              <Plus size={15} />
              {simulation.creatingCadBaseTemplate ? 'Creating Base' : 'Create Base Template'}
            </button>
            <span>Topology fields are not variant overrides; generated base templates become selectable CAD sources.</span>
          </div>
        </section>
        <section className="cad-variant-builder">
          <header>
            <div>
              <h3>Create CAD Variant</h3>
              <span>Pixel pitch and scalar geometry overrides only; topology changes need a new base template.</span>
            </div>
            <StatusPill state={quickVariantCount ? 'accent' : 'warn'}>{quickVariantCount || 'no'} override{quickVariantCount === 1 ? '' : 's'}</StatusPill>
          </header>
          <div className="cad-variant-meta">
            <label>
              <span>Variant label</span>
              <input value={variantLabel} placeholder={`${activeTemplate?.label || 'Template'} variant`} onChange={(event) => setVariantLabel(event.target.value)} />
            </label>
            <label>
              <span>Variant id</span>
              <input value={variantId} placeholder="auto-generated if empty" onChange={(event) => setVariantId(event.target.value)} />
            </label>
          </div>
          <div className="cad-variant-grid">
            {CAD_QUICK_VARIANT_FIELDS.map(([field, label, unit]) => (
              <label key={field}>
                <span>{label}</span>
                <em>current {compactValue(activeTemplate?.parameters?.[field])}{unit ? ` ${unit}` : ''}</em>
                <input
                  value={variantOverrides[field] || ''}
                  placeholder="leave unchanged"
                  onChange={(event) => setVariantOverrides((current) => ({ ...current, [field]: event.target.value }))}
                />
              </label>
            ))}
          </div>
          <div className="cad-template-actions compact">
            <button
              type="button"
              className="primary-button"
              disabled={!activeTemplate?.template_id || !quickVariantCount || simulation.creatingCadVariant}
              onClick={() => onCreateCadVariant({
                base_template: activeTemplate.template_id,
                id: variantId,
                label: variantLabel,
                overrides: quickVariantOverrides
              })}
            >
              <Plus size={15} />
              {simulation.creatingCadVariant ? 'Creating Variant' : 'Create Variant'}
            </button>
            <button type="button" disabled={!quickVariantCount || simulation.creatingCadVariant} onClick={() => setVariantOverrides({})}>Clear Overrides</button>
            <span>Generated variants are registered in the CAD catalog and become solver-selectable geometry.</span>
          </div>
        </section>
        <div className="suite-gate-note">
          <TriangleAlert size={16} />
          These templates reduce hidden assumptions. 3D mesh artifacts are CAD review meshes; TCAD bridge meshes are parameter-derived 2D electrical meshes, not calibrated full product meshes. Product accuracy still requires measured geometry, n,k, implants, traps, and convergence pass.
        </div>
        <div className="cad-template-notes">
          {catalog.mesh_role ? <span>{catalog.mesh_role}</span> : null}
          {(activeTemplate.notes || []).map((note) => <span key={note}>{note}</span>)}
          {assumptionLedger.available ? <span>Assumption ledger is linked so parametric choices and measured-data blockers are explicit.</span> : null}
          {(catalog.validation?.notes || []).slice(0, 2).map((note) => <span key={note}>{note}</span>)}
        </div>
      </main>
    </div>
  );
}

function GenericView({ active, selectedPreset, model, onAction, simulation, onRunSimulation }) {
  const controls = {
    Template: [['Project', model.projectName], ['Pixel class', '1.4um BSI split-PD'], ['Pattern', selectedPreset.label], ['Solver chain', 'Meep + DEVSIM']],
    'Stack Geometry': [['Pixel pitch', `${number(model.geometry.pitch || 1.4, 3)} um`], ['Lens height', `${number(model.geometry.lens_height || 0.657, 3)} um`], ['CFA thickness', `${number(model.geometry.cfa_thickness || 0.8, 3)} um`], ['DTI depth', `${number(model.pixel.dti_depth_um || 3, 3)} um`]],
    'DTI / Isolation': [['DTI depth', `${number(model.pixel.dti_depth_um || 3, 3)} um`], ['DTI width', `${number(model.pixel.dti_width_um || 0.06, 3)} um`], ['Split gap', `${number(model.pixel.split_gap_um || 0.08, 3)} um`], ['Isolation rule', 'linked to OCL boundary']],
    Materials: [['Silicon n,k', 'public table'], ['Lens material', 'polymer proxy'], ['CFA material', 'RGB proxy'], ['Measured n,k', 'not loaded']],
    'Illumination / CRA': [['Wavelength', `${number(model.edgeCase.wavelength_nm || 550, 0)} nm`], ['CRA X', `${number(model.edgeCase.cra_x_deg || 20, 0)} deg`], ['CRA Z', `${number(model.edgeCase.cra_z_deg || 0, 0)} deg`], ['Polarization', 'TM / unpolarized preview']],
    'Sensor Position': [['Field X', number(model.edgeCase.field_x_norm || 1, 2)], ['Field Y', '0.00'], ['Array region', 'center + edge'], ['CRA guard', 'enabled']],
    'Readout Mode': [['Readout', selectedPreset.readout], ['Binning', selectedPreset.group > 1 ? `${selectedPreset.group}x${selectedPreset.group}` : '1x1'], ['Remosaic', selectedPreset.readout.includes('remosaic') ? 'defined' : 'not required'], ['ADC', 'column proxy']],
    'Fast Preview': [['Optical mode', 'pattern preview'], ['Electrical mode', 'G*W proxy'], ['Runtime', 'seconds'], ['Use case', 'screening']],
    'FDTD Detail': [['Resolution', '90 px/um reference'], ['PML', '0.45-0.60 um'], ['Convergence', 'latest pass'], ['Export', 'FDTD setup package']],
    'Optical + Electrical': [['Optical input', 'FDTD G(y) + lateral model'], ['Electrical', 'native DEVSIM proxy'], ['Coupling', 'G*W'], ['Accuracy', 'needs measured calibration']],
    'Field Viewer': [['Plane', 'OCL exit'], ['Metric', '|E|2 / absorption'], ['CRA', '30 deg'], ['Overlay', 'stack + rays']],
    'AF Response': [['AF type', selectedPreset.af], ['Phase slope', '0.84 a.u.'], ['Pair mismatch', '4.2%'], ['Usable CRA', '0-60 deg']],
    'KPI Dashboard': [['QE proxy', percent(model.metrics.qeProxy)], ['Crosstalk', '4.7%'], ['OCL uniformity', '0.94'], ['Remosaic risk', '0.72']],
    'Sweep / Optimization': [['Variables', '6 enabled'], ['Constraints', 'same-color / same-OCL'], ['Objective', 'robust perception'], ['Best variant', model.best.label]],
    Tolerance: [['ML shift', '+/-20 nm'], ['CFA shift', '+/-20 nm'], ['DTI depth', '+/-10%'], ['AF density', '+/-15%']],
    Report: [['PDF', 'KPI summary'], ['Pattern package', 'cells + layout'], ['FDTD setup', 'geometry + mesh'], ['Design rules', 'guardrail summary']]
  }[active] || [];
  const showSimulationPanel = ['Fast Preview', 'FDTD Detail', 'Optical + Electrical', 'KPI Dashboard'].includes(active);
  return (
    <div className={`generic-page ${showSimulationPanel ? 'with-simulation' : ''}`}>
      {showSimulationPanel ? (
        <SimulationRunPanel selectedPreset={selectedPreset} simulation={simulation} onRunSimulation={onRunSimulation} />
      ) : null}
      <section className="panel-surface generic-primary">
        <header className="viewer-toolbar">
          <div>
            <h2>{active}</h2>
            <span>{selectedPreset.label}</span>
          </div>
          <StatusPill state="accent">{selectedPreset.ocl} OCL</StatusPill>
        </header>
        <div className="generic-content-grid">
          <div className="generic-preview">
            <SupercellGrid
              preset={selectedPreset}
              layers={{ cfa: true, ocl: true, pdaf: true, shield: true, binning: true, dti: false, readout: false }}
              selectedCell={{ row: 5, col: 5 }}
              onCellSelect={(cell) => onAction(`${active} preview selected cell (${cell.col + 1}, ${cell.row + 1})`)}
              compact
            />
          </div>
          <CrossSectionSvg variant={active.includes('PDAF') ? 'pdaf' : 'ocl'} />
        </div>
      </section>
      <aside className="panel-surface parameter-panel">
        <h2>Parameters</h2>
        {controls.map(([label, value]) => (
          <div className="plain-row" key={label}><span>{label}</span><strong>{value}</strong></div>
        ))}
        <div className="validation-list">
          <h3>Rule State</h3>
          <div className="validation-line ok"><CheckCircle2 size={15} />Pattern dependency linked</div>
          <div className="validation-line ok"><CheckCircle2 size={15} />Minimum simulation cell calculated</div>
          <div className="validation-line warn"><TriangleAlert size={15} />Measured calibration not loaded</div>
        </div>
      </aside>
    </div>
  );
}

function ActiveView({ active, selectedPreset, setSelectedPreset, model, ui, setters, actions, simulation, onRunSimulation, onOpenCadArtifact, onOpenWorkspacePath, onOpenFcstdWorkingCopy, onCreateFcstdWorkingCopy, onCreateCadVariant, onCreateBaseTemplate, onExtractFcstdParameters, onCreateVariantFromFcstd, onValidateFreecad, onGenerateTcadBridge, onRunTcadDdSmoke, onRunTcadAxisPairSmoke, onRunTcadQpdWeighting3d, onRunQpdGw3d, suiteState, setSuiteState, onRunSuite, onReplayCase }) {
  if (active === 'Template') {
    return (
      <CadTemplateView
        selectedPreset={selectedPreset}
        setSelectedPreset={setSelectedPreset}
        setSelectedCell={setters.setSelectedCell}
        simulation={simulation}
        cadTemplateId={ui.selectedCadTemplateId}
        setCadTemplateId={setters.setSelectedCadTemplateId}
        onAction={actions.handleAction}
        onRunSimulation={onRunSimulation}
        onOpenCadArtifact={onOpenCadArtifact}
        onOpenWorkspacePath={onOpenWorkspacePath}
        onOpenFcstdWorkingCopy={onOpenFcstdWorkingCopy}
        onCreateFcstdWorkingCopy={onCreateFcstdWorkingCopy}
        onCreateCadVariant={onCreateCadVariant}
        onCreateBaseTemplate={onCreateBaseTemplate}
        onExtractFcstdParameters={onExtractFcstdParameters}
        onCreateVariantFromFcstd={onCreateVariantFromFcstd}
        onValidateFreecad={onValidateFreecad}
        onGenerateTcadBridge={onGenerateTcadBridge}
        onRunTcadDdSmoke={onRunTcadDdSmoke}
        onRunTcadAxisPairSmoke={onRunTcadAxisPairSmoke}
        onRunTcadQpdWeighting3d={onRunTcadQpdWeighting3d}
        onRunQpdGw3d={onRunQpdGw3d}
      />
    );
  }
  if (active === 'Pattern Composer') {
    return (
      <PatternComposerView
        active={active}
        selectedPreset={selectedPreset}
        setSelectedPreset={setSelectedPreset}
        setSelectedCadTemplateId={setters.setSelectedCadTemplateId}
        selectedCell={ui.selectedCell}
        setSelectedCell={setters.setSelectedCell}
        composerLayers={ui.composerLayers}
        setComposerLayers={setters.setComposerLayers}
        composerSettings={ui.composerSettings}
        setComposerSettings={setters.setComposerSettings}
        highlightCouplings={ui.highlightCouplings}
        onSelectView={actions.setActive}
        onAction={actions.handleAction}
      />
    );
  }
  if (active === 'ML / OCL') return <MLOclView selectedPreset={selectedPreset} model={model} oclState={ui.oclState} setOclState={setters.setOclState} onAction={actions.handleAction} />;
  if (active === 'CFA') return <CfaView selectedPreset={selectedPreset} cfaState={ui.cfaState} setCfaState={setters.setCfaState} onAction={actions.handleAction} />;
  if (active === 'PDAF / Shield') return <PdafShieldView selectedPreset={selectedPreset} pdafState={ui.pdafState} setPdafState={setters.setPdafState} onAction={actions.handleAction} />;
  if (active === 'Test Suite') return <TestSuiteView simulation={simulation} suiteState={suiteState} setSuiteState={setSuiteState} onRunSuite={onRunSuite} onReplayCase={onReplayCase} />;
  if (active === 'Pattern Response') return <PatternResponseView selectedPreset={selectedPreset} model={model} responseState={ui.responseState} setResponseState={setters.setResponseState} onAction={actions.handleAction} />;
  if (active === 'Readiness') return <ReadinessView model={model} simulation={simulation} />;
  if (active === 'Variants') return <CompareView model={model} compareState={ui.compareState} setCompareState={setters.setCompareState} onAction={actions.handleAction} />;
  return <GenericView active={active} selectedPreset={selectedPreset} model={model} onAction={actions.handleAction} simulation={simulation} onRunSimulation={onRunSimulation} />;
}

function ActionToast({ toast }) {
  if (!toast) return null;
  return <div className="action-toast" role="status">{toast}</div>;
}

function ActionDrawer({ actionLog, onExample }) {
  return (
    <aside className="action-drawer">
      <div className="example-buttons">
        {EXAMPLE_DESIGNS.map((example) => (
          <button type="button" key={example.id} onClick={() => onExample(example)}>{example.name}</button>
        ))}
      </div>
      <div className="action-log">
        {actionLog.slice(0, 4).map((item) => (
          <span key={item.id}>{item.text}</span>
        ))}
      </div>
    </aside>
  );
}

export function App({ payload }) {
  const model = useWorkbenchModel(payload || {});
  const [active, setActive] = useState('Pattern Composer');
  const [selectedPreset, setSelectedPreset] = useState(() => PRESETS.find((preset) => preset.id === 'quad_qpd') || PRESETS[0]);
  const [selectedCadTemplateId, setSelectedCadTemplateId] = useState(DEFAULT_CAD_TEMPLATE_ID);
  const [activeTopTab, setActiveTopTab] = useState('Pixel');
  const [projectName, setProjectName] = useState('Automotive_HDR_Pixel_v2');
  const [bookmarked, setBookmarked] = useState(false);
  const [toast, setToast] = useState('');
  const [actionLog, setActionLog] = useState([]);
  const [selectedCell, setSelectedCell] = useState(() => selectedCellForPreset(presetById('quad_qpd')));
  const [composerLayers, setComposerLayers] = useState({ cfa: true, ocl: true, pdaf: true, shield: true, binning: true, dti: true, readout: true });
  const [composerSettings, setComposerSettings] = useState({ size: '12x12', view: 'top' });
  const [highlightCouplings, setHighlightCouplings] = useState(false);
  const [oclState, setOclState] = useState({
    activeClass: 'OCL_2x2_Quad',
    cfaMode: 'RGGB',
    selectedCell: selectedCellForPreset(presetById('quad_qpd'), 2, 2),
    surfaceModel: 'Spherical cap',
    params: DEFAULT_OCL_PARAMS,
    craCompensation: true
  });
  const [cfaState, setCfaState] = useState(DEFAULT_CFA_PARAMS);
  const [pdafState, setPdafState] = useState({
    activeMode: 'Half-shield L/R',
    selectedCell: selectedCellForPreset(presetById('sparse_pdaf'), 2, 3),
    layers: { cfa: true, ocl: true, pdaf: true, shield: true, binning: false, dti: false, readout: false },
    params: DEFAULT_PDAF_PARAMS
  });
  const [responseState, setResponseState] = useState({
    tab: 'QE Map',
    wavelength: '550 nm',
    cra: '30deg',
    polarization: 'TM',
    plane: 'OCL Exit Plane'
  });
  const [compareState, setCompareState] = useState({
    activeId: model.best.id,
    variables: {
      'ML Radius': true,
      'OCL Shift': true,
      'CFA Shift': true,
      'DTI Depth': true,
      'Shield Aperture': true,
      'AF Density': true
    }
  });
  const [simulationState, setSimulationState] = useState({
    backendStatus: 'checking',
    apiBase: '',
    examples: [],
    testSuites: [],
    cadTemplateCatalog: null,
    cadTemplates: [],
    cadTools: null,
    currentJob: null,
    starting: false,
    startingSuite: false,
    openingCadArtifact: false,
    openingWorkspacePath: false,
    openingFcstdWorkingCopy: false,
    creatingCadVariant: false,
    creatingCadBaseTemplate: false,
    validatingFreecad: false,
    creatingFcstdWorkingCopy: false,
    extractingFcstdParameters: false,
    creatingFcstdVariant: false,
    generatingTcadBridge: false,
    runningTcadDdSmoke: false,
    runningTcadAxisPairSmoke: false,
    runningTcadQpdWeighting3d: false,
    runningQpdGw3d: false,
    replayingCaseId: '',
    replayJob: null,
    replayResult: null,
    fcstdWorkingCopy: null,
    error: ''
  });
  const [suiteState, setSuiteState] = useState({ selectedSuiteId: 'pattern_baseline', tier: 'smoke' });
  const accuracyReady = Boolean(payload?.accuracy?.accuracy_ready);
  const headerModel = useMemo(() => model, [model]);
  const recordAction = (text) => {
    if (text === 'toggle-highlight-couplings') {
      setHighlightCouplings((current) => !current);
      text = highlightCouplings ? 'Coupling highlight disabled' : 'Coupling highlight enabled';
    }
    const item = { id: nextId('action'), text };
    setActionLog((current) => [item, ...current].slice(0, 12));
    setToast(text);
    window.clearTimeout(window.__pixelWorkbenchToastTimer);
    window.__pixelWorkbenchToastTimer = window.setTimeout(() => setToast(''), 1800);
  };
  const refreshSimulationJob = async (jobId) => {
    const { base, data } = await fetchWorkbenchApi(`/api/simulation/jobs/${jobId}`);
    setSimulationState((current) => ({
      ...current,
      apiBase: base,
      backendStatus: 'ready',
      currentJob: data,
      starting: false,
      error: data.error || ''
    }));
    return data;
  };
  const refreshReplayJob = async (jobId) => {
    const { base, data } = await fetchWorkbenchApi(`/api/simulation/jobs/${jobId}`);
    const done = ['completed', 'failed'].includes(data.status);
    setSimulationState((current) => ({
      ...current,
      apiBase: base,
      backendStatus: 'ready',
      replayJob: data,
      replayingCaseId: done ? '' : current.replayingCaseId,
      replayResult: done ? (data.replay_result || current.replayResult) : current.replayResult,
      error: data.error || ''
    }));
    return data;
  };
  useEffect(() => {
    let cancelled = false;
    const loadBackend = async () => {
      try {
        const { base, data } = await fetchWorkbenchApi('/api/health');
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          apiBase: base,
          backendStatus: data.meep_python_exists ? 'ready' : 'offline',
          error: data.meep_python_exists ? '' : 'Meep Python environment was not found.'
        }));
        const examples = await fetchWorkbenchApi('/api/simulation/examples');
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          apiBase: examples.base,
          examples: examples.data.examples || current.examples
        }));
        const catalog = await fetchWorkbenchApi('/api/simulation/test-suite');
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          apiBase: catalog.base,
          testSuites: catalog.data.suites || current.testSuites
        }));
        const cadTemplates = await fetchWorkbenchApi('/api/cad/templates');
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          apiBase: cadTemplates.base,
          cadTemplateCatalog: cadTemplates.data,
          cadTemplates: cadTemplates.data.templates || current.cadTemplates
        }));
        const cadTools = await fetchWorkbenchApi('/api/cad/tools');
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          apiBase: cadTools.base,
          cadTools: cadTools.data
        }));
      } catch (error) {
        if (cancelled) return;
        setSimulationState((current) => ({
          ...current,
          backendStatus: 'offline',
          error: error.message
        }));
      }
    };
    loadBackend();
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    const job = simulationState.currentJob;
    if (!job?.id || !['queued', 'running'].includes(job.status)) return undefined;
    const timer = window.setInterval(() => {
      refreshSimulationJob(job.id).catch((error) => {
        setSimulationState((current) => ({ ...current, backendStatus: 'offline', error: error.message, starting: false, startingSuite: false }));
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [simulationState.currentJob?.id, simulationState.currentJob?.status]);
  useEffect(() => {
    const job = simulationState.replayJob;
    if (!job?.id || !['queued', 'running'].includes(job.status)) return undefined;
    const timer = window.setInterval(() => {
      refreshReplayJob(job.id).catch((error) => {
        setSimulationState((current) => ({ ...current, backendStatus: 'ready', error: error.message, replayingCaseId: '' }));
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [simulationState.replayJob?.id, simulationState.replayJob?.status]);
  const selectedCadTemplate = useMemo(
    () => simulationState.cadTemplates.find((template) => template.template_id === selectedCadTemplateId) || null,
    [simulationState.cadTemplates, selectedCadTemplateId]
  );
  const activeSimulationRequest = useMemo(
    () => buildSimulationRequest({
      model,
      selectedPreset,
      oclState,
      cfaState,
      pdafState,
      responseState,
      projectName,
      cadTemplate: selectedCadTemplate
    }),
    [model, selectedPreset, oclState, cfaState, pdafState, responseState, projectName, selectedCadTemplate]
  );
  const startSimulation = async (exampleId = null) => {
    setActive('FDTD Detail');
    setActiveTopTab('Experiment');
    setSimulationState((current) => ({ ...current, starting: true, error: '' }));
    const payloadBody = exampleId
      ? { example_id: exampleId }
      : { simulation_request: activeSimulationRequest };
    try {
      const { base, data } = await fetchWorkbenchApi('/api/simulation/run', {
        method: 'POST',
        body: JSON.stringify(payloadBody)
      });
      setSimulationState((current) => ({
        ...current,
          apiBase: base,
          backendStatus: 'ready',
          currentJob: data,
          starting: false,
          startingSuite: false,
          error: ''
        }));
      recordAction(`Simulation job started: ${data.example?.label || activeSimulationRequest.design?.preset_label || exampleId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'offline',
        starting: false,
        startingSuite: false,
        error: error.message
      }));
      recordAction(`Simulation backend unavailable: ${error.message}`);
    }
  };
  const startSuite = async (suiteId, tier = 'smoke', caseIds = []) => {
    setActive('Test Suite');
    setActiveTopTab('Experiment');
    setSuiteState((current) => ({ ...current, selectedSuiteId: suiteId, tier, selectedCaseIds: caseIds }));
    setSimulationState((current) => ({ ...current, startingSuite: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/simulation/run-suite', {
        method: 'POST',
        body: JSON.stringify({ suite_id: suiteId, tier, case_ids: caseIds })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        currentJob: data,
        starting: false,
        startingSuite: false,
        error: ''
      }));
      recordAction(`Test suite started: ${data.suite?.label || suiteId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'offline',
        starting: false,
        startingSuite: false,
        error: error.message
      }));
      recordAction(`Test suite backend unavailable: ${error.message}`);
    }
  };
  const replaySuiteCase = async (caseResult) => {
    const caseCommand = caseResult?.artifacts?.case_command;
    if (!caseCommand) return;
    setSimulationState((current) => ({ ...current, replayingCaseId: caseResult.id, replayJob: null, replayResult: null, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/simulation/replay-case-job', {
        method: 'POST',
        body: JSON.stringify({
          case_command: caseCommand,
          compare_source: true,
          timeout_sec: 300
        })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        replayJob: data,
        replayingCaseId: caseResult.id,
        replayResult: data.replay_result || null,
        error: ''
      }));
      recordAction(`Replay job started: ${caseResult.label}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        replayingCaseId: '',
        replayJob: null,
        error: error.message
      }));
      recordAction(`Replay failed: ${error.message}`);
    }
  };
  const openCadArtifact = async (templateId, artifact, preferFreecad = true) => {
    setSimulationState((current) => ({ ...current, openingCadArtifact: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/open', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId, artifact, prefer_freecad: preferFreecad })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTools: data.freecad ? { schema: 'pixel_workbench_cad_tools_v1', freecad: data.freecad } : current.cadTools,
        openingCadArtifact: false,
        error: ''
      }));
      recordAction(`Opened ${artifact} for ${templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        openingCadArtifact: false,
        error: error.message
      }));
      recordAction(`CAD open failed: ${error.message}`);
    }
  };
  const createFcstdWorkingCopy = async (templateId) => {
    setSimulationState((current) => ({ ...current, creatingFcstdWorkingCopy: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/create-fcstd-working-copy', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        fcstdWorkingCopy: data,
        creatingFcstdWorkingCopy: false,
        error: ''
      }));
      recordAction(`FCStd working copy created: ${data.fcstd_path || templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        creatingFcstdWorkingCopy: false,
        error: error.message
      }));
      recordAction(`FCStd working copy failed: ${error.message}`);
    }
  };
  const openFcstdWorkingCopy = async (fcstdPath) => {
    setSimulationState((current) => ({ ...current, openingFcstdWorkingCopy: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/open-file', {
        method: 'POST',
        body: JSON.stringify({ path: fcstdPath, prefer_freecad: true })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTools: data.freecad ? { schema: 'pixel_workbench_cad_tools_v1', freecad: data.freecad } : current.cadTools,
        openingFcstdWorkingCopy: false,
        error: ''
      }));
      recordAction(`Opened FCStd working copy: ${fcstdPath}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        openingFcstdWorkingCopy: false,
        error: error.message
      }));
      recordAction(`FCStd working copy open failed: ${error.message}`);
    }
  };
  const openWorkspacePath = async (path, label = 'workspace path') => {
    setSimulationState((current) => ({ ...current, openingWorkspacePath: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/open-file', {
        method: 'POST',
        body: JSON.stringify({ path, prefer_freecad: false })
      });
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        openingWorkspacePath: false,
        error: ''
      }));
      recordAction(`Opened ${label}: ${data.path || path}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        openingWorkspacePath: false,
        error: error.message
      }));
      recordAction(`${label} open failed: ${error.message}`);
    }
  };
  const createCadVariant = async ({ base_template, id, label, overrides }) => {
    setSimulationState((current) => ({ ...current, creatingCadVariant: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/create-variant', {
        method: 'POST',
        body: JSON.stringify({ base_template, id, label, overrides })
      });
      const templateId = data.variant?.template_id;
      if (templateId) setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        creatingCadVariant: false,
        error: ''
      }));
      recordAction(`CAD variant created: ${templateId || base_template}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        creatingCadVariant: false,
        error: error.message
      }));
      recordAction(`CAD variant failed: ${error.message}`);
    }
  };
  const createBaseTemplate = async ({ topology_preset, id, label, parameters }) => {
    setSimulationState((current) => ({ ...current, creatingCadBaseTemplate: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/create-base-template', {
        method: 'POST',
        body: JSON.stringify({ topology_preset, id, label, parameters })
      });
      const templateId = data.template_id || data.record?.template_id;
      if (templateId) setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        creatingCadBaseTemplate: false,
        error: ''
      }));
      recordAction(`CAD base template created: ${templateId || id}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        creatingCadBaseTemplate: false,
        error: error.message
      }));
      recordAction(`CAD base template failed: ${error.message}`);
    }
  };
  const extractFcstdParameters = async (templateId, fcstdPath = '') => {
    setSimulationState((current) => ({ ...current, extractingFcstdParameters: true, error: '' }));
    try {
      const payload = { template_id: templateId };
      if (fcstdPath) payload.fcstd_path = fcstdPath;
      const { base, data } = await fetchWorkbenchApi('/api/cad/extract-fcstd-parameters', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      const overrideCount = Object.keys(data.overrides || {}).length;
      const blockedCount = (data.unsupported_changes || []).length;
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        fcstdParameterExtract: data,
        extractingFcstdParameters: false,
        error: ''
      }));
      recordAction(`FCStd parameters read: ${overrideCount} scalar override${overrideCount === 1 ? '' : 's'}${blockedCount ? `, ${blockedCount} blocked change${blockedCount === 1 ? '' : 's'}` : ''}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        extractingFcstdParameters: false,
        error: error.message
      }));
      recordAction(`FCStd parameter read failed: ${error.message}`);
    }
  };
  const createVariantFromFcstd = async (templateId, fcstdPath = '') => {
    setSimulationState((current) => ({ ...current, creatingFcstdVariant: true, error: '' }));
    try {
      const payload = { template_id: templateId };
      if (fcstdPath) payload.fcstd_path = fcstdPath;
      const { base, data } = await fetchWorkbenchApi('/api/cad/create-variant-from-fcstd', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      const templateIdOut = data.variant?.template_id;
      if (templateIdOut) setSelectedCadTemplateId(templateIdOut);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        fcstdParameterExtract: data.fcstd_import || current.fcstdParameterExtract,
        creatingFcstdVariant: false,
        error: ''
      }));
      recordAction(`FCStd CAD variant created: ${templateIdOut || templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        creatingFcstdVariant: false,
        error: error.message
      }));
      recordAction(`FCStd variant failed: ${error.message}`);
    }
  };
  const validateFreecad = async (templateId) => {
    setSimulationState((current) => ({ ...current, validatingFreecad: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/validate-freecad', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId, write_fcstd: true })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        validatingFreecad: false,
        error: ''
      }));
      recordAction(`FreeCAD validation ${data.status || 'completed'}: ${templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        validatingFreecad: false,
        error: error.message
      }));
      recordAction(`FreeCAD validation failed: ${error.message}`);
    }
  };
  const generateTcadBridge = async (templateId) => {
    setSimulationState((current) => ({ ...current, generatingTcadBridge: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/generate-tcad-bridge', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        generatingTcadBridge: false,
        error: ''
      }));
      recordAction(`TCAD bridge generated: ${templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        generatingTcadBridge: false,
        error: error.message
      }));
      recordAction(`TCAD bridge failed: ${error.message}`);
    }
  };
  const runTcadDdSmoke = async (templateId) => {
    setSimulationState((current) => ({ ...current, runningTcadDdSmoke: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/run-tcad-dd-smoke', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        runningTcadDdSmoke: false,
        error: ''
      }));
      recordAction(`TCAD DD smoke completed: ${templateId}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        runningTcadDdSmoke: false,
        error: error.message
      }));
      recordAction(`TCAD DD smoke failed: ${error.message}`);
    }
  };
  const runTcadAxisPairSmoke = async (templateId) => {
    setSimulationState((current) => ({ ...current, runningTcadAxisPairSmoke: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/run-tcad-axis-pair-smoke', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        runningTcadAxisPairSmoke: false,
        error: ''
      }));
      const summary = data.summary || {};
      recordAction(`TCAD axis-pair smoke ${data.status || 'completed'}: ${templateId} x ${number(summary.phase_x_proxy, 4)} z ${number(summary.phase_z_proxy, 4)}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        runningTcadAxisPairSmoke: false,
        error: error.message
      }));
      recordAction(`TCAD axis-pair smoke failed: ${error.message}`);
    }
  };
  const runTcadQpdWeighting3d = async (templateId) => {
    setSimulationState((current) => ({ ...current, runningTcadQpdWeighting3d: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/run-tcad-qpd-weighting-3d', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        runningTcadQpdWeighting3d: false,
        error: ''
      }));
      const metrics = data.summary?.metrics || {};
      recordAction(`QPD 3D weighting ${data.status || 'completed'}: ${templateId} uniform ${number(metrics.quadrant_uniformity, 4)}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        runningTcadQpdWeighting3d: false,
        error: error.message
      }));
      recordAction(`QPD 3D weighting failed: ${error.message}`);
    }
  };
  const runQpdGw3d = async (templateId) => {
    setSimulationState((current) => ({ ...current, runningQpdGw3d: true, error: '' }));
    try {
      const { base, data } = await fetchWorkbenchApi('/api/cad/run-qpd-gw-3d', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId, integration_grid: 'generation' })
      });
      setSelectedCadTemplateId(templateId);
      setSimulationState((current) => ({
        ...current,
        apiBase: base,
        backendStatus: 'ready',
        cadTemplateCatalog: data.catalog || current.cadTemplateCatalog,
        cadTemplates: data.catalog?.templates || current.cadTemplates,
        runningQpdGw3d: false,
        error: ''
      }));
      const firstCase = data.summary?.cases?.[0] || {};
      const metrics = firstCase.metrics || {};
      recordAction(`QPD 3D G*W ${data.status || 'completed'}: ${templateId} phase x ${number(metrics.phase_x_gw, 4)} z ${number(metrics.phase_z_gw, 4)}`);
    } catch (error) {
      setSimulationState((current) => ({
        ...current,
        backendStatus: 'ready',
        runningQpdGw3d: false,
        error: error.message
      }));
      recordAction(`QPD 3D G*W failed: ${error.message}`);
    }
  };
  const selectActive = (view) => {
    setActive(view);
    const top = Object.entries(TOPBAR_ROUTES).find(([, route]) => route === view)?.[0];
    if (top) setActiveTopTab(top);
  };
  const handleTopTab = (tab) => {
    setActiveTopTab(tab);
    setActive(TOPBAR_ROUTES[tab] || 'Pattern Composer');
    recordAction(`Top workspace switched to ${tab}`);
  };
  const handleTopAction = (type) => {
    if (type === 'bookmark') {
      setBookmarked((current) => !current);
      recordAction(bookmarked ? 'Bookmark removed' : 'Bookmark added');
      return;
    }
    if (type === 'save') recordAction(`Saved variant state: ${selectedPreset.label}`);
    if (type === 'help') recordAction('Help opened: Pattern topology guide');
    if (type === 'settings') recordAction('Settings opened: solver/profile preferences');
  };
  const createNewPattern = () => {
    const custom = presetById('custom');
    setSelectedPreset(custom);
    setSelectedCell(selectedCellForPreset(custom));
    setSelectedCadTemplateId('');
    setActive('Pattern Composer');
    setActiveTopTab('Pixel');
    recordAction('New custom supercell draft created · no exact CAD template');
  };
  const applyExample = (example) => {
    const preset = presetById(example.presetId);
    setSelectedPreset(preset);
    setSelectedCell(selectedCellForPreset(preset));
    const cadTemplateId = cadTemplateIdForPreset(preset);
    setSelectedCadTemplateId(cadTemplateId);
    setActive(example.active);
    setActiveTopTab(example.active === 'Variants' ? 'Compare' : example.active === 'Report' ? 'Report' : 'Pixel');
    setCfaState((current) => ({ ...current, thickness: example.cfaThickness }));
    setOclState((current) => ({ ...current, activeClass: example.oclClass }));
    setPdafState((current) => ({ ...current, activeMode: example.pdafMode }));
    setResponseState((current) => ({ ...current, tab: example.responseTab }));
    recordAction(cadTemplateId
      ? `Loaded design test example: ${example.name} · CAD source ${cadTemplateId}`
      : `Loaded design test example: ${example.name} · no exact CAD template`);
  };
  const runFDTDSetup = () => {
    startSimulation();
  };
  const ui = { selectedCell, composerLayers, composerSettings, highlightCouplings, oclState, cfaState, pdafState, responseState, compareState, selectedCadTemplateId };
  const setters = { setSelectedCell, setComposerLayers, setComposerSettings, setOclState, setCfaState, setPdafState, setResponseState, setCompareState, setSelectedCadTemplateId };
  const actions = { setActive: selectActive, handleAction: recordAction };
  const simulationIsRunning = ['queued', 'running'].includes(simulationState.currentJob?.status) || simulationState.starting || simulationState.startingSuite;
  const simulation = {
    ...simulationState,
    examples: simulationState.examples.length ? simulationState.examples : SOLVER_TEST_EXAMPLES,
    testSuites: simulationState.testSuites,
    suggestedExampleId: exampleIdForPreset(selectedPreset),
    activeRequest: activeSimulationRequest
  };
  if (!payload) {
    return (
      <div className="missing-payload">
        <Microscope size={32} />
        <h1>Pixel Workbench payload is missing</h1>
        <p>Regenerate the studio with image_sensor_pixel_studio.py.</p>
      </div>
    );
  }
  return (
    <div className="app-shell">
      <SectionNav active={active} onSelect={selectActive} onNewPattern={createNewPattern} />
      <main className="main-shell">
        <AppTopbar
          model={headerModel}
          activeTopTab={activeTopTab}
          projectName={projectName}
          bookmarked={bookmarked}
          onTopTab={handleTopTab}
          onProjectChange={(value) => {
            setProjectName(value);
            recordAction(`Project selected: ${value}`);
          }}
          onAction={handleTopAction}
        />
        <PageHeader active={active} selectedPreset={selectedPreset} model={model} />
        <ActionDrawer actionLog={actionLog} onExample={applyExample} />
        <ActiveView
          active={active}
          selectedPreset={selectedPreset}
          setSelectedPreset={setSelectedPreset}
          model={model}
          ui={ui}
          setters={setters}
          actions={actions}
          simulation={simulation}
          onRunSimulation={startSimulation}
          onOpenCadArtifact={openCadArtifact}
          onOpenWorkspacePath={openWorkspacePath}
          onOpenFcstdWorkingCopy={openFcstdWorkingCopy}
          onCreateFcstdWorkingCopy={createFcstdWorkingCopy}
          onCreateCadVariant={createCadVariant}
          onCreateBaseTemplate={createBaseTemplate}
          onExtractFcstdParameters={extractFcstdParameters}
          onCreateVariantFromFcstd={createVariantFromFcstd}
          onValidateFreecad={validateFreecad}
          onGenerateTcadBridge={generateTcadBridge}
          onRunTcadDdSmoke={runTcadDdSmoke}
          onRunTcadAxisPairSmoke={runTcadAxisPairSmoke}
          onRunTcadQpdWeighting3d={runTcadQpdWeighting3d}
          onRunQpdGw3d={runQpdGw3d}
          suiteState={suiteState}
          setSuiteState={setSuiteState}
          onRunSuite={startSuite}
          onReplayCase={replaySuiteCase}
        />
      </main>
      <ActionToast toast={toast} />
      <footer className="status-bar">
        <StatusPill state="ok">Ready</StatusPill>
        <span>Active Variant: {selectedPreset.label}</span>
        <span>Pixel Pitch: {number(model.geometry.pitch || 1.4, 3)} um</span>
        <span>OCL: {selectedPreset.ocl}</span>
        <span>CFA: {selectedPreset.cfa}</span>
        <span>CRA Range: 0-60 deg</span>
        <StatusPill state={accuracyReady ? 'ok' : 'warn'}>{accuracyReady ? 'accuracy ready' : 'research stack'}</StatusPill>
        <button type="button" className="run-button" disabled={simulationIsRunning} onClick={runFDTDSetup}>
          <Play size={15} />{simulationIsRunning ? 'Solver Running' : 'Run FDTD Detail'}
        </button>
      </footer>
    </div>
  );
}
