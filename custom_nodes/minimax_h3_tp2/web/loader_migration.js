import { app } from "../../scripts/app.js";

// The original ordered widgets, including later appended fields. Old workflows
// may contain only a prefix. Link inputs such as lora_stack are not widgets.
export const legacyWidgets = [
    "text_encoder", "diffusion_model", "gpu_weight_window_gib", "ram_cache_per_rank_gib",
    "idle_residency", "tp_quantization", "idle_weight_gib", "lora_row_partial_reduce",
    "te_attention", "dit_attention", "row_linear_backend", "swiglu_padding", "selective_b",
    "scalar_score_workspace_mib", "adaptive_lookahead", "fc2_quantizer", "lora_mode",
    "tensor_flow", "residual_exponent", "vision_residual_exponent", "video_residual_exponent",
    "video_decode_reserve_gib", "prefetch_lifecycle", "memory_pressure_aware", "dit_reserve_gib",
    "command_timeout_seconds", "rank_checkpoint_dir", "checkpoint_read_chunk_mib",
    "lora_selective_qkv_b", "column_linear_backend",
];

export function restoreWidgets(node, info, schema) {
    if ((info.properties?.h3_configuration_version ?? 1) >= 2) return;
    const saved = Object.fromEntries(legacyWidgets.slice(0, info.widgets_values?.length ?? 0)
        .map((name, i) => [name, info.widgets_values[i]]));
    Object.assign(saved, info.widgets_values_named ?? {});
    for (const widget of node.widgets ?? []) {
        const spec = schema[widget.name];
        if (!spec) continue;
        const fallback = spec[1]?.default ?? (Array.isArray(spec[0]) ? spec[0][0] : undefined);
        // LiteGraph has already restored widgets positionally before onConfigure.
        // Restore every surviving widget by name, including absent tail defaults.
        let value = Object.hasOwn(saved, widget.name) ? saved[widget.name] : fallback;
        if (widget.name === "column_linear_backend" && value === "tilelang_exact") {
            value = "cuda_int8_tilelang";
        }
        widget.value = value;
    }
    node.properties ??= {};
    node.properties.h3_configuration_version = 2;
}

app.registerExtension({
    name: "MiniMaxH3.TP2.ConfigurationMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3TP2Loader") return;
        const schema = { ...nodeData.input.required, ...nodeData.input.optional };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.properties ??= {};
            this.properties.h3_configuration_version = 2;
            return result;
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = configure?.call(this, info);
            restoreWidgets(this, info, schema);
            return result;
        };
    },
});
