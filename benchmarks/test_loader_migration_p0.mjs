/** Verify that old positional ComfyUI Loader values survive P0 flag insertion. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../custom_nodes/minimax_h3_tp2/web/loader_migration.js', import.meta.url), 'utf8');
const runtime = { app: { registerExtension() {} } };
runInNewContext(
    source.replace(/^import \{ app \} from .*;\s*/m, '').replaceAll('export ', '') +
        '\nthis.restoreWidgets = restoreWidgets; this.v4Widgets = v4Widgets; this.v5Widgets = v5Widgets; this.v6Widgets = v6Widgets;',
    runtime,
);
const { restoreWidgets, v4Widgets, v5Widgets, v6Widgets } = runtime;
assert.equal(v4Widgets.length, 35);
assert.equal(v4Widgets.at(-1), 'debug');
assert.ok(v4Widgets.includes('sol_tau'));
assert.ok(!v4Widgets.includes('dit_p0_2_lifetime'));
assert.ok(!v4Widgets.includes('dit_p0_3_lifetime'));

const schema = {
    debug: ['BOOLEAN', { default: false }],
    dit_p0_2_lifetime: ['BOOLEAN', { default: true }],
    dit_p0_3_lifetime: ['BOOLEAN', { default: true }],
    dit_p0_b_inplace: ['BOOLEAN', { default: true }],
    dit_p0_a1_turing: ['BOOLEAN', { default: true }],
    dit_p0_c_local_rotation: ['BOOLEAN', { default: true }],
};
const node = {
    widgets: Object.keys(schema).map(name => ({ name, value: null })),
    properties: {},
};
const savedValues = v4Widgets.map(name => name === 'debug' ? false : 'saved_' + name);
restoreWidgets(node, {
    properties: { h3_configuration_version: 4 },
    widgets_values: savedValues,
}, schema);
assert.equal(node.widgets[0].value, false);
assert.equal(node.widgets[1].value, true);
assert.equal(node.widgets[2].value, true);
assert.equal(node.widgets[3].value, true);
assert.equal(node.widgets[4].value, true);
assert.equal(node.widgets[5].value, true);
assert.equal(node.properties.h3_configuration_version, 7);

// v5 positional flags must survive insertion of the two v6 controls.
assert.equal(v5Widgets.at(-1), 'debug');
const oldV5 = { widgets: Object.keys(schema).map(name => ({ name, value: null })), properties: {} };
const oldV5Values = v5Widgets.map(name => name === 'dit_p0_2_lifetime' ? false :
    name === 'dit_p0_3_lifetime' ? true : name === 'debug' ? false : 'old_' + name);
restoreWidgets(oldV5, { properties: { h3_configuration_version: 5 },
    widgets_values: oldV5Values }, schema);
assert.equal(oldV5.widgets[0].value, false);
assert.equal(oldV5.widgets[1].value, false);
assert.equal(oldV5.widgets[2].value, true);
assert.equal(oldV5.widgets[3].value, true);
assert.equal(oldV5.widgets[4].value, true);
assert.equal(oldV5.widgets[5].value, true);
assert.equal(oldV5.properties.h3_configuration_version, 7);

// Preserve v6 positional P0-A1/P0-C and introduce P0-B before debug.
assert.equal(v6Widgets.at(-1), 'debug');
const oldV6 = { widgets: Object.keys(schema).map(name => ({ name, value: null })), properties: {} };
const oldV6Values = v6Widgets.map(name => name === 'dit_p0_3_lifetime' ? false :
    name === 'dit_p0_c_local_rotation' ? false : name === 'debug' || name.startsWith('dit_p0_') ? true : 'old_' + name);
restoreWidgets(oldV6, { properties: { h3_configuration_version: 6 },
    widgets_values: oldV6Values }, schema);
assert.equal(oldV6.widgets[0].value, true);
assert.equal(oldV6.widgets[1].value, true);
assert.equal(oldV6.widgets[2].value, false);
assert.equal(oldV6.widgets[3].value, true);
assert.equal(oldV6.widgets[4].value, true);
assert.equal(oldV6.widgets[5].value, false);
assert.equal(oldV6.properties.h3_configuration_version, 7);

// A disabled v7 switch must never be re-enabled on reload.
const modern = { widgets: [{ name: 'dit_p0_2_lifetime', value: false }], properties: {} };
restoreWidgets(modern, {
    properties: { h3_configuration_version: 7 },
    widgets_values: [false],
}, schema);
assert.equal(modern.widgets[0].value, false);

// An explicit named v4 opt-out takes precedence over the new default.
const v4Explicit = {
    widgets: [{ name: 'dit_p0_2_lifetime', value: true }],
    properties: {},
};
restoreWidgets(v4Explicit, {
    properties: { h3_configuration_version: 4 },
    widgets_values: savedValues,
    widgets_values_named: { dit_p0_2_lifetime: false },
}, schema);
assert.equal(v4Explicit.widgets[0].value, false);

console.log('H3_LOADER_P0_MIGRATION_PASS');
