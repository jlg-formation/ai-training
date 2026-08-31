import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// onnxruntime-web charge ses fichiers WebAssembly (.wasm / .mjs) à l'exécution.
// On les copie depuis node_modules vers la racine servie par Vite, pour que le
// runtime les trouve en local (aucune dépendance à un CDN, marche hors-ligne).
export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
          dest: ".",
        },
      ],
    }),
  ],
});
