import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Load model and feature list
with open("apartment_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load enriched BFS municipality data (includes distance_to_zurich)
df_bfs = pd.read_csv("bfs_municipality_data_enriched.csv", sep=",", encoding="utf-8")

# Municipality name -> BFS number mapping
locations = {
    "Zürich": 261, "Kloten": 62, "Uster": 198, "Illnau-Effretikon": 296,
    "Feuerthalen": 27, "Pfäffikon": 177, "Ottenbach": 11, "Dübendorf": 191,
    "Richterswil": 138, "Maur": 195, "Embrach": 56, "Bülach": 53,
    "Winterthur": 230, "Oetwil am See": 157, "Russikon": 178, "Obfelden": 10,
    "Wald (ZH)": 120, "Niederweningen": 91, "Dällikon": 84, "Buchs (ZH)": 83,
    "Rüti (ZH)": 118, "Hittnau": 173, "Bassersdorf": 52, "Glattfelden": 58,
    "Opfikon": 66, "Hinwil": 117, "Regensberg": 95, "Langnau am Albis": 136,
    "Dietikon": 243, "Erlenbach (ZH)": 151, "Kappel am Albis": 6, "Stäfa": 158,
    "Zell (ZH)": 231, "Turbenthal": 228, "Oberglatt": 92, "Winkel": 72,
    "Volketswil": 199, "Kilchberg (ZH)": 135, "Wetzikon (ZH)": 121,
    "Zumikon": 160, "Weisslingen": 180, "Elsau": 219, "Hettlingen": 221,
    "Rüschlikon": 139, "Stallikon": 13, "Dielsdorf": 86, "Wallisellen": 69,
    "Dietlikon": 54, "Meilen": 156, "Wangen-Brüttisellen": 200, "Flaach": 28,
    "Regensdorf": 96, "Niederhasli": 90, "Bauma": 297, "Aesch (ZH)": 241,
    "Schlieren": 247, "Dürnten": 113, "Unterengstringen": 249,
    "Gossau (ZH)": 115, "Oberengstringen": 245, "Schleinikon": 98,
    "Aeugst am Albis": 1, "Rheinau": 38, "Höri": 60, "Rickenbach (ZH)": 225,
    "Rafz": 67, "Adliswil": 131, "Zollikon": 161, "Urdorf": 250,
    "Hombrechtikon": 153, "Birmensdorf (ZH)": 242, "Fehraltorf": 172,
    "Weiach": 102, "Männedorf": 155, "Küsnacht (ZH)": 154,
    "Hausen am Albis": 4, "Hochfelden": 59, "Fällanden": 193,
    "Greifensee": 194, "Mönchaltorf": 196, "Dägerlen": 214,
    "Thalheim an der Thur": 39, "Uetikon am See": 159, "Seuzach": 227,
    "Uitikon": 248, "Affoltern am Albis": 2, "Geroldswil": 244,
    "Niederglatt": 89, "Thalwil": 141, "Rorbas": 68, "Pfungen": 224,
    "Weiningen (ZH)": 251, "Bubikon": 112, "Neftenbach": 223,
    "Mettmenstetten": 9, "Otelfingen": 94, "Flurlingen": 29, "Stadel": 100,
    "Grüningen": 116, "Henggart": 31, "Dachsen": 25, "Bonstetten": 3,
    "Bachenbülach": 51, "Horgen": 295,
}


def predict_apartment(rooms, area, town):
    """Predict monthly apartment rent in CHF."""
    bfs_number = locations[town]
    bfs_row = df_bfs[df_bfs["bfs_number"] == bfs_number]

    if len(bfs_row) != 1:
        return -1

    bfs_row = bfs_row.iloc[0]

    # Build feature vector matching training feature order
    area_per_room = area / rooms if rooms > 0 else area
    features = {
        "rooms": rooms,
        "area": area,
        "pop": bfs_row["pop"],
        "pop_dens": bfs_row["pop_dens"],
        "frg_pct": bfs_row["frg_pct"],
        "emp": bfs_row["emp"],
        "tax_income": bfs_row["tax_income"],
        "distance_to_zurich": bfs_row["distance_to_zurich"],
        "area_per_room": area_per_room,
    }

    X = np.array([[features[f] for f in feature_names]])

    # Model was trained on log1p(price), so we invert with expm1
    prediction_log = model.predict(X)[0]
    prediction = np.expm1(prediction_log)

    return int(np.round(prediction, 0))


# Gradio interface
iface = gr.Interface(
    fn=predict_apartment,
    inputs=[
        gr.Number(label="Rooms", value=3.5),
        gr.Number(label="Area (m²)", value=80),
        gr.Dropdown(
            choices=sorted(locations.keys()),
            label="Town",
            value="Zürich",
        ),
    ],
    outputs=gr.Number(label="Predicted Monthly Rent (CHF)"),
    title="Zurich Apartment Price Predictor",
    description="Predict monthly rental prices for apartments in the Canton of Zurich. "
    "The model uses a Random Forest trained on apartment listings, enriched with "
    "municipality data and distance to Zurich city center.",
    examples=[
        [4.5, 120, "Dietlikon"],
        [3.5, 60, "Winterthur"],
        [3, 100, "Zürich"],
        [2.5, 50, "Adliswil"],
        [5, 150, "Küsnacht (ZH)"],
    ],
)

if __name__ == "__main__":
    iface.launch()
