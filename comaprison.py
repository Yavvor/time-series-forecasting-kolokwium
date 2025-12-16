import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from poisson_forecaster import KolokwiumForecaster as Forecaster1
from poisson_forecaster2 import KolokwiumForecaster2 as Forecaster2



def calculate_metrics(y_true, y_pred):
    """Pomocnicza funkcja do liczenia metryk."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_comparison(data_path):


    # 1. Wczytanie danych
    raw_forecaster = Forecaster1(data_path)
    full_df = raw_forecaster.load()

    # Podział na zbiór uczący i testowy (ostatnie 30 dni) oraz inne parametry do zabawy
    test_days = 30

    ht1 = 0.4
    ht2 = 0.7
    noise1 = False
    noise2 = False

    ns1 = 0.3
    ns2 = 0.3


    train_df = full_df.iloc[:-test_days].copy()
    test_df = full_df.iloc[-test_days:].copy()

    print(f"Zbiór uczący: {len(train_df)} dni")
    print(f"Zbiór testowy: {len(test_df)} dni")

    # =========================================================================
    # FAZA 1: WALIDACJA NA ZBIORZE TESTOWYM (Ostatnie 30 dni)
    # =========================================================================


    # --- Model 1 (Trening) ---
    f1_val = Forecaster1(data_path)
    f1_val.df = train_df  # Nadpisujemy df zbiorem treningowym
    f1_val.decompose(trend_method="linear", yearly_method="fourier", holiday_threshold=ht1)
    # Prognoza na 30 dni (okres testowy)
    pred1_val = f1_val.forecast_from_decomposer(periods=test_days, calibrate=True, add_noise=noise1, noise_scale=ns1)

    # --- Model 2 (Trening) ---
    f2_val = Forecaster2(data_path)
    f2_val.df = train_df  # Nadpisujemy df zbiorem treningowym
    f2_val.decompose(trend_method="linear", yearly_method="fourier", holiday_threshold=ht2)
    # Prognoza na 30 dni (okres testowy)
    pred2_val = f2_val.forecast_from_decomposer(periods=test_days, calibrate=True, add_noise=noise2, noise_scale=ns2)

    # Obliczenie metryk
    metrics1 = calculate_metrics(test_df['count'], pred1_val['expected_count'])
    metrics2 = calculate_metrics(test_df['count'], pred2_val['expected_count'])

    # Wyświetlenie tabeli wyników
    results_df = pd.DataFrame([metrics1, metrics2], index=["Model 1 (Oryginał)", "Model 2 (Poprawiony)"])
    print("\nWYNIKI NA ZBIORZE TESTOWYM:")
    print("=" * 60)
    print(results_df)
    print("=" * 60)

    # =========================================================================
    # WYKRES 1: PORÓWNANIE NA ZBIORZE TESTOWYM
    # =========================================================================
    plt.figure(figsize=(12, 6))

    # Prawdziwe dane (Test)
    plt.plot(test_df.index, test_df['count'], color='black', linewidth=2, label='Dane Rzeczywiste (Test)')

    # Końcówka treningu (dla kontekstu)
    plt.plot(train_df.index[-30:], train_df['count'].iloc[-30:], color='gray', alpha=0.5, label='Koniec Treningu')

    # Prognozy walidacyjne
    plt.plot(pred1_val.index, pred1_val['expected_count'], color='blue', marker='o', markersize=4, linestyle='--',
             label=f'Model 1 (MAE: {metrics1["MAE"]:.2f})')
    plt.plot(pred2_val.index, pred2_val['expected_count'], color='red', marker='o', markersize=4, linestyle='--',
             label=f'Model 2 (MAE: {metrics2["MAE"]:.2f})')

    plt.title("Rzeczywistość vs Prognozy (ostatnie 30 dni)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # FAZA 2: PROGNOZA NA PRZYSZŁOŚĆ (Rok do przodu)
    # =========================================================================


    # Aby prognozować przyszłość, uczymy modele na CAŁYM dostępnym zbiorze
    # (włącznie z tymi ostatnimi 30 dniami testowymi)
    # --- Model 1 (Full) ---
    f1_full = Forecaster1(data_path)
    f1_full.load()  # Ładuje całość
    f1_full.decompose(trend_method="linear", yearly_method="fourier", holiday_threshold=ht1)
    future1 = f1_full.forecast_from_decomposer(periods=365, calibrate=True, add_noise=noise1, noise_scale=ns1)

    # --- Model 2 (Full) ---
    f2_full = Forecaster2(data_path)
    f2_full.load()  # Ładuje całość
    f2_full.decompose(trend_method="linear", yearly_method="fourier", holiday_threshold=ht2)
    # Włączamy szum dla Modelu 2 na wykresie przyszłości, żeby pokazać zmienność w święta
    future2 = f2_full.forecast_from_decomposer(periods=365, calibrate=True, add_noise=noise2, noise_scale=ns2)

    # =========================================================================
    # WYKRES 2: PORÓWNANIE PROGNOZY W PRZÓD
    # =========================================================================
    plt.figure(figsize=(14, 7))

    # Ostatnie 2 miesiące historii dla kontekstu
    recent_history = full_df.iloc[-60:]
    plt.plot(recent_history.index, recent_history['count'], color='black', alpha=0.4,
             label='Historia (ostatnie 60 dni)')

    # Model 1
    plt.plot(future1.index, future1['expected_count'], color='blue', linewidth=0.7,
             label='Model 1: Prognoza (z "Spikes")')

    # Model 2 (Linia + Chmura punktów symulacji)
    plt.plot(future2.index, future2['expected_count'], color='red', linewidth=0.7, label='Model 2: Prognoza (Średnia)')
    plt.scatter(future2.index, future2['simulated_count'], color='purple', s=3, alpha=0.65,
                label='Model 2: Symulacja (Z losowością)')

    plt.axvline(full_df.index.max(), color='black', linestyle='--', label='Start Prognozy')

    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ścieżka do pliku
    DATA_PATH = r"kolokwium-dane-0912.csv"

    if os.path.exists(DATA_PATH):
        run_comparison(DATA_PATH)
    else:
        print(f"Nie znaleziono pliku: {DATA_PATH}")