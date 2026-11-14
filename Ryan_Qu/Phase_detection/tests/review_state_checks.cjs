/* Run with: node tests/review_state_checks.cjs. No browser, dataset or npm packages. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const template = fs.readFileSync(
  path.join(__dirname, "../stage1_python_worker/obb/orientation_review.html"),
  "utf8",
);
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1];
// Exercise the actual pure state functions, before browser event registration.
const prefix = script.slice(
  0,
  script.indexOf('$("export-json").addEventListener'),
);
assert(prefix.length > 1000);
const data = {
  fingerprint: "unit-test-only",
  pilot_id: "unit-test-only",
  model: "pretrained-fixture",
  crop_source: "ground-truth",
  manifest_sha256: "fixture",
  predictions_sha256: "fixture",
  selection_sha256: "fixture",
  samples: [0, 1, 2, 3].map((i) => ({
    example_id: "fixture_" + i,
    predicted_orientation: i === 1 ? 180 : 0,
    confidence: 0.9,
    status: "success",
    error: "",
    source_image_path: "fixture.png",
    crop_source: "ground-truth",
    original_crop_path: "original.png",
    corrected_crop_path: "corrected.png",
  })),
};
const context = vm.createContext({
  document: {
    getElementById: (id) => ({
      textContent: id === "pilot-data" ? JSON.stringify(data) : "{}",
    }),
  },
  localStorage: { getItem: () => null },
  console,
});
vm.runInContext(prefix, context);
const decisions = ["0", "180", "uncertain", "bad_crop"];
const payload = {
  schema_version: 1,
  fingerprint: data.fingerprint,
  reviewer: "fixture reviewer",
  reviews: data.samples.map((sample, i) => ({
    ...sample,
    review_decision: decisions[i],
    crop_issue: i === 0,
    notes: i === 0 ? 'A note, with "quotes"\nand a newline' : "",
    reviewed_at_utc: "2026-10-27T00:00:00.000Z",
  })),
};
context.payload = payload;
context.imported = vm.runInContext("validateImport(payload)", context);
vm.runInContext("state=imported", context);
const exported = JSON.parse(
  vm.runInContext("JSON.stringify(exportPayload())", context),
);
assert.equal(exported.counts.reviewed, 4);
assert.deepEqual(
  exported.reviews.map((row) => row.review_decision),
  decisions,
);
assert.equal(exported.reviews[0].notes, payload.reviews[0].notes);
assert.equal(exported.reviews[0].model_decision_correct, true);
assert.equal(exported.reviews[0].eligible_for_orientation_analysis, false);
assert.equal(exported.reviews[1].eligible_for_orientation_analysis, true);
assert.equal(exported.reviews[2].manual_orientation_deg, null);
assert.equal(exported.reviews[3].model_decision_correct, null);
const before = vm.runInContext("JSON.stringify(state.items)", context);
context.invalid = JSON.parse(JSON.stringify(payload));
context.invalid.fingerprint = "other experiment";
assert.throws(
  () => vm.runInContext("validateImport(invalid)", context),
  /does not match/,
);
context.invalid = JSON.parse(JSON.stringify(payload));
context.invalid.reviews[0].confidence = 0.01;
assert.throws(
  () => vm.runInContext("validateImport(invalid)", context),
  /predictions differ/,
);
assert.equal(vm.runInContext("JSON.stringify(state.items)", context), before);
vm.runInContext("state.items.fixture_0.decision='180'", context);
assert.equal(
  JSON.parse(vm.runInContext("JSON.stringify(exportPayload())", context))
    .reviews[0].model_decision_correct,
  false,
);
context.snapshot = JSON.parse(
  vm.runInContext("JSON.stringify(state)", context),
);
vm.runInContext(
  "state={fingerprint:DATA.fingerprint,reviewer:'',items:{},cursor:0,updated_at:null};restore(snapshot)",
  context,
);
assert.equal(
  JSON.parse(vm.runInContext("JSON.stringify(exportPayload())", context)).counts
    .reviewed,
  4,
);
assert.equal(
  vm.runInContext("csvCell('hello,\"quoted\"')", context),
  '"hello,""quoted"""',
);
assert.equal(
  vm.runInContext('csvCell("=not_a_formula")', context),
  '"\'=not_a_formula"',
);
console.log(
  "PASS: review imports/exports, null labels, notes, CSV escaping, snapshot restoration and mismatched-file rejection",
);
