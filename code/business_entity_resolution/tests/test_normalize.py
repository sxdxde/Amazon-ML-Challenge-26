"""Regression tests for normalize.py. Run directly or via pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize import fold, normalize_address, normalize_name, translit_brahmic  # noqa: E402


def test_transliteration_scripts():
    assert translit_brahmic("राज फूड्स प्राइवेट लिमिटेड") == "raj fuds praivet limited"      # Devanagari
    assert translit_brahmic("జై లాజిస్టిక్స్ లిమిటెడ్") == "jai lajistiks limited"           # Telugu
    assert "hospittalitti" in translit_brahmic("ഹോസ്പിറ്റാലിറ്റി")                        # Malayalam റ്റ
    assert translit_brahmic("ஆல்ஃபா").startswith("alf")                                     # Tamil ஃப


def test_legal_extraction_anywhere():
    n = normalize_name("LLC Southern Currencyshares", "US")
    assert n["name_core"] == "southern currencyshares" and n["name_legal"] == "llc"
    n = normalize_name("ग्रीन लॉजिस्टिक्स प्रा. लि.", "India")
    assert n["name_legal"] == "ltd pvt" and n["name_core"] == "grin lojistiks"
    n = normalize_name("Nantes Sport SARL", "France")
    assert n["name_legal"] == "sarl"


def test_country_word_and_accents():
    n = normalize_name("Entreprendre (Frànce) Club SAS", "France")
    assert n["name_core"] == "entreprendre club" and n["name_legal"] == "sas"
    assert fold("Trois & Frèrês Développement") == "trois and freres developpement"


def test_address_components():
    a = normalize_address("N° 14 R. DE BOULMGNE, TOURCOING")
    assert a["addr_stnum"] == "14" and a["addr_stword"] == "boulmgne" and "rue" in a["addr_norm"]
    a = normalize_address("Flat 12, 400 001, Mumbai")
    assert a["addr_postal"] == "400001"
    a = normalize_address("761 Hathaway Saint, Rocky Mt, North Carolina")
    b = normalize_address("4761 Hathaway Street, Rocky Mount, NC")
    assert a["addr_stword"] == b["addr_stword"] == "hathaway"
    assert "nc" in a["addr_norm"].split() and "nc" in b["addr_norm"].split()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL NORMALIZE TESTS PASSED")
