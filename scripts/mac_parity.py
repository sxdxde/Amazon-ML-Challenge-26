"""Parity: MLX encoder (CLS + L2, fp16) vs sentence-transformers BAAI/bge-m3 on CPU (fp32).
Pass = mean cosine >= 0.999. Also shows what mean pooling would have scored (the trap)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache"))
import numpy as np
import mlx.core as mx
from er_mlx_encoder import load_encoder, encode

# (mlx repo, mlx rev, sentence-transformers repo, st rev, pooling, prefix)
MODELS = [
    ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3",
     "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "mean", "query: "),  # notebook default
    ("mlx-community/bge-m3-mlx-fp16", "a37eddded9a6a1273a87fb8b0da0d1cdbd98aeec",
     "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "cls", ""),  # validated; too slow on M1 for 24M records
]
TEXTS = [
    "Dacy Cox Kogyo, LLC", "38 Kelsey Road, Sanford, NY", "PIEDMONT DENTAL GROUP", "508 BRIDAL PATH PIEDMONT OK 73078",
    "Bombay Management Private Limited", "बॉम्बे मैनेजमेंट प्राइवेट लिमिटेड", "Surya Impex Pnvie Limited",
    "Plot No 13, Telangana, Hyderabad, Hig Colony Vanasthalipuram", "Near SBI ATM, MG Road, Bengaluru, Karnataka 560001",
    "Chennai City Corporation Energy Ltd", "DOOR NO 108 NEW NO 100/2B, T2, THIRD FLOOR, CHENNAI",
    "Médiation Visages Fêtes EURL", "Dunkerque, 3 Rue de la Bienfaisance, Hauts-de-France",
    "Établissements Faire", "7 Avenue de la Reousse Dune l'Herbe, Lège-Cap-Ferret, Nouvelle-Aquitaine",
    "Pharmacie Dame", "26 Rue Mercière, Bordeaux", "Boulangerie Pâtisserie du Château SARL",
    "41 Place de la Cinquième République, Pessac", "Société Générale d'Électricité SAS", "12 bd Haussmann 75009 Paris",
    "Café-Brasserie « Le Français »", "en face de la mairie, 2 impasse des Lilas", "Hôtel Les Tilleuls SASU",
    "Eye Health Inc", "EYE EYE INC HEALTH", "Lopez End Inc.", "The-Carr Trust", "tsílvercom", "Silver Twin, Inc",
    "9521 Riverside Parkway, Unit SUITE E, Tulsa, OK", "EHW Industries", "618 Willow Court, Bellingham, WA",
    "Jhaveri Advisors", "707-C, 7Th Floor, Jaina Tower Ii Distt. Centre, Janakpuri, West Delhi",
    "सनराइज मीडिया प्राइवेट लिमिटेड", "QOA It Limited", "Ram Technology Pvt Ltd", "Aditiya Promoters Corp",
    "Flat No. F 203, Amish Park E & F Chs Ltd, Near Don School, Mira Road East, Thane", "A&B Holdings LLP",
    "H.No.4200, Daroga Street, Jogiwara, Nai Sarak, New Delhi", "x", "12345", "",
    "Ünïcödé Tëst Çafé", "AMAZON.COM SERVICES LLC", "Tata Consultancy Services Limited", "Rue du Faubourg Saint-Honoré",
    "S/O Zamil Akhtar, Line Bazar, Kishanganj, Bihar",
]
print(f"parity: {len(TEXTS)} strings (incl. French accents, Devanagari, empty)")

from sentence_transformers import SentenceTransformer
ok = True
for REPO, REV, BASE, BASE_REV, POOL, PREFIX in MODELS:
    texts = [PREFIX + t for t in TEXTS]
    model, tok, cfg = load_encoder(REPO, REV)
    a = encode(model, tok, texts, batch_size=16, pooling=POOL).astype(np.float32)
    wrong = "mean" if POOL == "cls" else "cls"
    aw = encode(model, tok, texts, batch_size=16, pooling=wrong).astype(np.float32)
    del model; mx.clear_cache()
    st = SentenceTransformer(BASE, revision=BASE_REV, device="cpu")
    b = st.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    cos, cosw = (a * b).sum(1), (aw * b).sum(1)
    print(f"[{REPO}] ST pooling: {st[1]}")
    print(f"  MLX({POOL}+L2, fp16) vs ST(cpu fp32): mean cos {cos.mean():.5f}  min {cos.min():.5f}")
    print(f"  MLX({wrong}) vs ST: mean cos {cosw.mean():.5f}   <- wrong pooling, for reference")
    print("  PARITY", "PASS" if cos.mean() >= 0.999 else "FAIL")
    ok &= cos.mean() >= 0.999
sys.exit(0 if ok else 1)
