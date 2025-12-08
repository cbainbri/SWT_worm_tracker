#!/usr/bin/env python3
"""Test Richards Analysis with Synthetic Data"""
import numpy as np
import pandas as pd

def richards_curve(t, y_initial, y_final, B, M, nu):
    A = y_final - y_initial
    denominator = (1 + np.exp(-B * (t - M)))**(1/nu)
    return y_initial + A / denominator

def generate_synthetic_animal(animal_id, y_initial=0.6, y_final=0.3, B=0.1, M=5.0, nu=1.0, noise_level=0.02):
    assay_num, track_num = animal_id
    duration, fps = 240, 10
    encounter_time = 120
    time = np.linspace(0, duration, int(duration * fps))
    time_rel = time - encounter_time
    
    speed_true = richards_curve(time_rel, y_initial, y_final, B, M, nu)
    speed_obs = np.maximum(speed_true + np.random.normal(0, noise_level, len(time)), 0)
    
    x, y = np.zeros(len(time)), np.zeros(len(time))
    for i in range(1, len(time)):
        angle = np.random.uniform(0, 2 * np.pi)
        dt = time[i] - time[i-1]
        distance = speed_obs[i] * dt
        x[i] = x[i-1] + distance * np.cos(angle)
        y[i] = y[i-1] + distance * np.sin(angle)
    
    food_encounter = np.where(np.abs(time - encounter_time) < 0.05, 'food', '')
    
    df = pd.DataFrame({
        'source_file': f'synthetic_{assay_num}_{track_num}.csv',
        'assay_num': assay_num, 'track_num': track_num,
        'pc_number': 'PC1', 'sex': 'm', 'strain_genotype': 'synthetic',
        'treatment': 'test', 'time': time, 'x': x, 'y': y,
        'centroid_on_food': 0, 'nose_on_food': 0, 'food_encounter': food_encounter
    })
    return df, {'y_initial': y_initial, 'y_final': y_final, 'A': y_final - y_initial, 'B': B, 'M': M, 'nu': nu}

print("Generating synthetic dataset...")
animals, true_params = [], {}
for i in range(5):
    df, params = generate_synthetic_animal((1, i), y_initial=0.6, y_final=0.3, B=0.1, M=5, nu=1.0)
    df['strain_genotype'] = 'wt'
    animals.append(df)
    true_params[(1, i)] = params

for i in range(5):
    df, params = generate_synthetic_animal((2, i), y_initial=0.5, y_final=0.5, B=0.1, M=0, nu=1.0)
    df['strain_genotype'] = 'mutant'
    animals.append(df)
    true_params[(2, i)] = params

composite = pd.concat(animals, ignore_index=True)
composite.to_csv('synthetic_composite.csv', index=False)
print("Created: synthetic_composite.csv")
print("Run: python richards_speed_analysis.py synthetic_composite.csv --output synthetic_results/")
