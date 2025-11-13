import React from "react";
import { Box, Paper, RadioGroup, FormControlLabel, Radio, Typography } from "@mui/material";

// layers: array of LayerResult for current frame
// value: currently selected source_file (string)
// onChange: (source_file_string) => void
export default function ActiveLayerPicker({ layers = [], value, onChange }) {
    if (!Array.isArray(layers) || layers.length <= 1) return null;

    const items = layers
        .map((L, idx) => {
            const src = L?.source_file || null;
            const label = buildLabel(L, idx);
            if (!src) return null;
            return { value: src, label };
        })
        .filter(Boolean);

    if (items.length <= 1) return null;

    return (
        <Paper elevation={3} sx={{ position: "fixed", top: 12, right: 12, zIndex: 1000, p: 1.5, maxWidth: 360 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                Capa activa para herramientas
            </Typography>
            <RadioGroup
                value={value ?? items[0].value}
                onChange={(e) => onChange?.(e.target.value)}
                name="active-layer-picker"
            >
                {items.map((it) => (
                    <FormControlLabel
                        key={it.value}
                        value={it.value}
                        control={<Radio size="small" />}
                        label={<Typography variant="body2" noWrap title={it.label}>{it.label}</Typography>}
                    />
                ))}
            </RadioGroup>
        </Paper>
    );
}

function buildLabel(L) {
    const field = L?.field || "Layer";
    const src = L?.source_file || "";
    const base = basename(src);
    // Try to make it compact: FIELD — filename
    return `${field} — ${base}`;
}

function basename(path) {
    if (!path || typeof path !== "string") return "";
    const norm = path.replace(/\\/g, "/");
    const parts = norm.split("/");
    return parts[parts.length - 1] || path;
}
