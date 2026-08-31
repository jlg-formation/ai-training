// ---------------------------------------------------------------------------
// TP 6 - Front : reconnaître un chiffre dessiné à la souris, dans le navigateur.
//
// Le modèle a été entraîné en Python (main.py) puis exporté en ONNX. Ici, on le
// charge avec onnxruntime-web et on lui envoie le chiffre dessiné, APRÈS l'avoir
// prétraité EXACTEMENT comme les images MNIST d'entraînement (sinon le modèle,
// qui n'a vu que des images "façon MNIST", se trompe).
// ---------------------------------------------------------------------------

import * as ort from "onnxruntime-web";
import "./style.css";

// Les fichiers WebAssembly d'onnxruntime sont copiés à la racine servie par Vite
// (voir vite.config.ts). On indique au runtime de les y chercher.
ort.env.wasm.wasmPaths = "/";

// Normalisation MNIST : DOIT être identique à celle de main.py (Normalize).
const MNIST_MOYENNE = 0.1307;
const MNIST_ECART_TYPE = 0.3081;

// Le chiffre est mis à l'échelle dans une boîte 20x20, puis centré dans 28x28
// (par son centre de masse) : c'est la recette d'origine du jeu MNIST.
const TAILLE = 28;
const TAILLE_BOITE = 20;

// ---- Éléments de la page ----
const canvas = document.getElementById("dessin") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
const boutonEffacer = document.getElementById("effacer") as HTMLButtonElement;
const elPrediction = document.getElementById("prediction") as HTMLDivElement;
const elStatut = document.getElementById("statut") as HTMLDivElement;
const elBarres = document.getElementById("barres") as HTMLDivElement;

// ---------------------------------------------------------------------------
// 1) Le dessin à la souris
// ---------------------------------------------------------------------------
function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";        // chiffre BLANC sur fond NOIR, comme MNIST
  ctx.lineWidth = 18;               // trait épais : proche des chiffres MNIST
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

let enTrainDeDessiner = false;

function positionSouris(e: PointerEvent): [number, number] {
  const rect = canvas.getBoundingClientRect();
  // Le canvas peut être affiché à une taille différente de sa résolution réelle.
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  return [x, y];
}

canvas.addEventListener("pointerdown", (e) => {
  enTrainDeDessiner = true;
  const [x, y] = positionSouris(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("pointermove", (e) => {
  if (!enTrainDeDessiner) return;
  const [x, y] = positionSouris(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});

function terminerTrait() {
  if (!enTrainDeDessiner) return;
  enTrainDeDessiner = false;
  reconnaitre();                    // on lance la reconnaissance à chaque trait fini
}

canvas.addEventListener("pointerup", terminerTrait);
canvas.addEventListener("pointerleave", terminerTrait);

boutonEffacer.addEventListener("click", () => {
  initCanvas();
  elPrediction.textContent = "?";
  dessinerBarres(new Array(10).fill(0), -1);
});

// ---------------------------------------------------------------------------
// 2) Le prétraitement "façon MNIST"
// ---------------------------------------------------------------------------
// Rend un Float32Array [1, 1, 28, 28] prêt pour le modèle, ou null si le canvas
// est vide.
function preparerEntree(): Float32Array | null {
  const largeur = canvas.width;
  const hauteur = canvas.height;
  const pixels = ctx.getImageData(0, 0, largeur, hauteur).data;

  // (a) Intensité en niveaux de gris dans [0, 1] (le canal rouge suffit : le
  //     trait est blanc, le fond noir). On cherche au passage la boîte
  //     englobante du tracé (les bords du chiffre).
  const gris = new Float32Array(largeur * hauteur);
  let minX = largeur, minY = hauteur, maxX = -1, maxY = -1;
  for (let y = 0; y < hauteur; y++) {
    for (let x = 0; x < largeur; x++) {
      const v = pixels[(y * largeur + x) * 4] / 255; // canal R normalisé
      gris[y * largeur + x] = v;
      if (v > 0.05) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return null;         // rien n'a été dessiné

  // (b) Recadrage sur la boîte englobante, puis mise à l'échelle pour tenir dans
  //     une boîte 20x20 en conservant les proportions.
  const larg = maxX - minX + 1;
  const haut = maxY - minY + 1;
  const echelle = TAILLE_BOITE / Math.max(larg, haut);
  const largRedim = Math.max(1, Math.round(larg * echelle));
  const hautRedim = Math.max(1, Math.round(haut * echelle));

  // On dessine le chiffre recadré et redimensionné dans un petit canvas.
  const tmp = document.createElement("canvas");
  tmp.width = largRedim;
  tmp.height = hautRedim;
  const tctx = tmp.getContext("2d", { willReadFrequently: true })!;
  tctx.drawImage(
    canvas,
    minX, minY, larg, haut,        // source : la boîte englobante
    0, 0, largRedim, hautRedim,    // destination : la boîte 20x20
  );
  const petit = tctx.getImageData(0, 0, largRedim, hautRedim).data;

  // (c) Centre de masse du chiffre redimensionné (pondéré par l'intensité).
  let somme = 0, sommeX = 0, sommeY = 0;
  for (let y = 0; y < hautRedim; y++) {
    for (let x = 0; x < largRedim; x++) {
      const v = petit[(y * largRedim + x) * 4] / 255;
      somme += v;
      sommeX += v * x;
      sommeY += v * y;
    }
  }
  const cmX = sommeX / somme;
  const cmY = sommeY / somme;

  // (d) On place le chiffre dans l'image finale 28x28 de sorte que son centre de
  //     masse tombe au centre (13.5, 13.5) : c'est le centrage de MNIST.
  const decalageX = Math.round(TAILLE / 2 - cmX);
  const decalageY = Math.round(TAILLE / 2 - cmY);

  const image = new Float32Array(TAILLE * TAILLE); // fond noir (0) par défaut
  for (let y = 0; y < hautRedim; y++) {
    for (let x = 0; x < largRedim; x++) {
      const dx = x + decalageX;
      const dy = y + decalageY;
      if (dx < 0 || dx >= TAILLE || dy < 0 || dy >= TAILLE) continue;
      image[dy * TAILLE + dx] = petit[(y * largRedim + x) * 4] / 255;
    }
  }

  // (e) Normalisation identique à l'entraînement : (x - moyenne) / ecart_type.
  const entree = new Float32Array(TAILLE * TAILLE);
  for (let i = 0; i < entree.length; i++) {
    entree[i] = (image[i] - MNIST_MOYENNE) / MNIST_ECART_TYPE;
  }
  return entree;
}

// ---------------------------------------------------------------------------
// 3) L'inférence ONNX
// ---------------------------------------------------------------------------
let session: ort.InferenceSession | null = null;

async function chargerModele() {
  session = await ort.InferenceSession.create("/modele.onnx");
  // Préchauffage : la toute première inférence WebAssembly est lente (compilation
  // du modèle). On en lance une "à vide" pour que le premier chiffre dessiné soit
  // reconnu instantanément.
  const vide = new ort.Tensor("float32", new Float32Array(TAILLE * TAILLE), [1, 1, TAILLE, TAILLE]);
  await session.run({ input: vide });
  elStatut.textContent = "Modèle prêt. À toi de dessiner !";
}

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const somme = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / somme);
}

async function reconnaitre() {
  if (!session) return;
  const entree = preparerEntree();
  if (!entree) return;

  const tenseur = new ort.Tensor("float32", entree, [1, 1, TAILLE, TAILLE]);
  const sorties = await session.run({ input: tenseur });
  const logits = sorties.logits.data as Float32Array;

  const probas = softmax(logits);
  let meilleur = 0;
  for (let i = 1; i < probas.length; i++) {
    if (probas[i] > probas[meilleur]) meilleur = i;
  }

  elPrediction.textContent = String(meilleur);
  dessinerBarres(probas, meilleur);
}

// ---------------------------------------------------------------------------
// 4) Affichage des probabilités (une barre par chiffre)
// ---------------------------------------------------------------------------
function dessinerBarres(probas: number[], gagnant: number) {
  elBarres.innerHTML = "";
  for (let chiffre = 0; chiffre < 10; chiffre++) {
    const p = probas[chiffre] ?? 0;

    const ligne = document.createElement("div");
    ligne.className = "barre-ligne" + (chiffre === gagnant ? " gagnant" : "");

    const lab = document.createElement("span");
    lab.className = "barre-chiffre";
    lab.textContent = String(chiffre);

    const piste = document.createElement("div");
    piste.className = "barre-piste";
    const remplissage = document.createElement("div");
    remplissage.className = "barre-remplissage";
    remplissage.style.width = `${(p * 100).toFixed(1)}%`;
    piste.appendChild(remplissage);

    const val = document.createElement("span");
    val.className = "barre-valeur";
    val.textContent = `${(p * 100).toFixed(1)}%`;

    ligne.append(lab, piste, val);
    elBarres.appendChild(ligne);
  }
}

// ---------------------------------------------------------------------------
// Démarrage
// ---------------------------------------------------------------------------
initCanvas();
dessinerBarres(new Array(10).fill(0), -1);
chargerModele().catch((err) => {
  console.error(err);
  elStatut.textContent = "Erreur : impossible de charger le modèle ONNX.";
});
