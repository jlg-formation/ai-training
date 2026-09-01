import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import { createRequire } from "node:module";
import path from "node:path";

// onnxruntime-web charge ses fichiers WebAssembly (.wasm / .mjs) à l'exécution.
// On les copie depuis node_modules vers la racine servie par Vite, pour que le
// runtime les trouve en local (aucune dépendance à un CDN, marche hors-ligne).
//
// Les node_modules sont factorisés à la racine du dépôt (workspaces Bun) : le
// paquet n'est donc plus dans front/node_modules. On résout son dossier dist via
// Node (à partir d'un fichier .wasm exporté par le paquet), pour le trouver où
// qu'il soit installé.
const require = createRequire(import.meta.url);
const distOnnx = path
  .dirname(require.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"))
  .replace(/\\/g, "/");

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: `${distOnnx}/*.{wasm,mjs}`,
          dest: ".",
        },
      ],
    }),
  ],
});
