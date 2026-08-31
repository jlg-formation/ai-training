// ---------------------------------------------------------------------------
// TP 7 - Front : générer un chiffre avec le GAN, dans le navigateur.
//
// Le GÉNÉRATEUR a été entraîné en Python (main.py) puis exporté en ONNX. Ici, on
// le charge avec onnxruntime-web. À chaque clic, on tire un vecteur de bruit z
// AU HASARD (un point de l'espace d'entrée du générateur) et on affiche l'image
// 28x28 que le modèle produit. Aucune image de MNIST n'est copiée : tout est
// fabriqué à partir du bruit.
// ---------------------------------------------------------------------------

import * as ort from "onnxruntime-web";
import "./style.css";

// Les fichiers WebAssembly d'onnxruntime sont copiés à la racine servie par Vite
// (voir vite.config.ts). On indique au runtime de les y chercher.
ort.env.wasm.wasmPaths = "/";

// DOIT correspondre à DIM_BRUIT de main.py (la taille du vecteur d'entrée z).
const DIM_BRUIT = 64;
const TAILLE = 28; // le générateur sort une image 28x28

// ---- Éléments de la page ----
const canvas = document.getElementById("sortie") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const boutonNouvelle = document.getElementById("nouvelle") as HTMLButtonElement;
const elStatut = document.getElementById("statut") as HTMLDivElement;

// ---------------------------------------------------------------------------
// 1) Tirer un bruit z ~ Normale(0, 1), comme torch.randn en Python
// ---------------------------------------------------------------------------
// La méthode de Box-Muller transforme deux tirages uniformes dans [0, 1) en un
// tirage suivant une loi normale centrée réduite (la même loi que torch.randn).
function tirerBruitNormal(taille: number): Float32Array {
  const z = new Float32Array(taille);
  for (let i = 0; i < taille; i++) {
    const u1 = Math.random() || 1e-12; // évite log(0)
    const u2 = Math.random();
    z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return z;
}

// ---------------------------------------------------------------------------
// 2) Afficher une image 28x28 (pixels dans [-1, 1]) dans le canvas
// ---------------------------------------------------------------------------
// On inverse les couleurs pour un rendu clair : fond BLANC, chiffre NOIR.
function afficherImage(pixels: Float32Array) {
  // Petit canvas 28x28 qu'on agrandira ensuite (sans lissage) vers 280x280.
  const petit = document.createElement("canvas");
  petit.width = TAILLE;
  petit.height = TAILLE;
  const pctx = petit.getContext("2d")!;
  const imageData = pctx.createImageData(TAILLE, TAILLE);

  for (let i = 0; i < TAILLE * TAILLE; i++) {
    // [-1, 1] -> [0, 1] : 1 = trait du chiffre, 0 = fond.
    const v01 = (pixels[i] + 1) / 2;
    // Inversion : trait -> noir (0), fond -> blanc (255).
    const gris = Math.round(255 * (1 - v01));
    imageData.data[i * 4 + 0] = gris;
    imageData.data[i * 4 + 1] = gris;
    imageData.data[i * 4 + 2] = gris;
    imageData.data[i * 4 + 3] = 255;
  }
  pctx.putImageData(imageData, 0, 0);

  // Agrandissement au carré, en gardant les pixels nets (pas de flou).
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(petit, 0, 0, canvas.width, canvas.height);
}

// ---------------------------------------------------------------------------
// 3) L'inférence ONNX : bruit z -> image
// ---------------------------------------------------------------------------
let session: ort.InferenceSession | null = null;

async function genererNouvelleImage() {
  if (!session) return;
  const bruit = tirerBruitNormal(DIM_BRUIT);
  const tenseur = new ort.Tensor("float32", bruit, [1, DIM_BRUIT]);
  const sorties = await session.run({ bruit: tenseur });
  const image = sorties.image.data as Float32Array;
  afficherImage(image);
}

async function chargerModele() {
  session = await ort.InferenceSession.create("/generateur.onnx");
  // Préchauffage : la première inférence WebAssembly compile le modèle et est
  // lente. On la fait une fois "à vide" pour que le premier clic soit instantané.
  const vide = new ort.Tensor("float32", new Float32Array(DIM_BRUIT), [1, DIM_BRUIT]);
  await session.run({ bruit: vide });

  elStatut.textContent = "Modèle prêt. Clique pour générer !";
  boutonNouvelle.disabled = false;
  genererNouvelleImage(); // une première image dès le chargement
}

boutonNouvelle.addEventListener("click", () => {
  genererNouvelleImage().catch((err) => {
    console.error(err);
    elStatut.textContent = "Erreur pendant la génération.";
  });
});

// ---------------------------------------------------------------------------
// Démarrage
// ---------------------------------------------------------------------------
chargerModele().catch((err) => {
  console.error(err);
  elStatut.textContent = "Erreur : impossible de charger le modèle ONNX.";
});
