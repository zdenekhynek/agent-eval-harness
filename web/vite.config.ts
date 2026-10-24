import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `base: "./"` keeps every URL relative, so the same build works at a domain
// root (harness serve), under a GitHub Pages project path, or from a folder.
// In dev, data/ is proxied to `harness serve` on :8000.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { proxy: { "/data": "http://127.0.0.1:8000" } },
});
