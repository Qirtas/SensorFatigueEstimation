# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:25:09 2024

@author: Marco Sica
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from Segmentation.TrendAnalysis import validate_specific_data
from Segmentation.Segmentation import load_borg_scale, process_and_segment_data, add_borg_column_to_processed_data
from Features.jerk_features import extract_all_jerk_features, print_missing_values, interpolate_missing_shoulder_features, fill_missing_values
from Features.movement_frequency_features import extract_all_movement_frequency_features
from Features.mpsd_features import extract_all_mpsd_features
from Features.rt_variability_feature_extraction import extract_all_rt_variability_features
from model_trainer import train_and_evaluate, train_baseline_model, train_all_test_individual, prepare_and_train_models
from FeatureImportance.featureImpVisualisation import load_feature_importance_scores, normalize_scores, aggregate_and_select_features, create_heatmap
from Features.gyr_3axis_feature_extraction import extract_all_gyr_3axis_features
from Features.zero_crossing_feature_extraction import extract_all_emg_zc_features
from Features.emg_mav_features import extract_all_mav_features
from Features.acc_3axis_feature_extraction import extract_all_acc_3axis_features
from Features.stft_features_extraction import extract_all_stft_features
from Features.emg_statistical_features import extract_all_emg_statistical_features
from Features.wavelength_emg_feature_extraction import extract_all_wavelength_emg_features
from Features.borg_mapping import merge_features, map_borg_to_features, interpolate_borg_values, compute_repetition_times_for_all_subjects, add_borg_to_features, map_borg_to_repetitions
from Segmentation.filter_data import analyze_frequency
from Features.rom_features import extract_all_rom_features
from classify_fatigue import train_and_evaluate_classification_models
from Features.integratedEMG_feature_extraction import extract_all_integrated_emg_features


if __name__ == "__main__":

    # ****************************
    #       35 Internal
    # # ****************************


    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------

    '''
    # Parameters for Segmentation
    exercise = 35
    select_internal_external = 'i'
    borg_scale_path = 'borg_scale.npy'  # Path to Borg scale file
    output_dir = '35Internal/processed_data_35_i'  # Output directory for saving segmented data
    numer_of_subject = 34  # Total number of subjects

    # Call the segmentation function
    process_and_segment_data(
        exercise=exercise,
        select_internal_external=select_internal_external,
        borg_scale_path=borg_scale_path,
        output_dir=output_dir,
        numer_of_subject=numer_of_subject
    )
    '''


    ############################################################


    '''
    # Perform Frequency Analysis
    base_directory = "processed_data_35_i"
    body_part = "forearm"
    data_type = "acc"

    # List of subjects to analyze
    #subjects = [12]
    # subjects = [10] #Noise around higher freq
    subjects = [8] # Increase in the high freq at later reps

    # Perform frequency analysis for each subject
    for subject in subjects:
        print(f"Analyzing frequency for Subject {subject}...")
        analyze_frequency(subject, body_part, data_type, base_directory)
    '''

    ############################################################

    '''
    # Basic Statistical Features Extraction
    # Base directory containing the segmented data
    base_directory = "processed_data_35_i"
    output_directory = "Features"

    # Configuration for feature extraction
    movement = "Internal"
    body_part = "Torso"  # (Shoulder, Pelvis, Palm, Forearm, Torso)
    data_type = "acc"  # Data type (e.g., acc, gyr, mag)
    relevant_axis = "Y"

    # Call the feature extraction function
    feature_extraction_for_all_subjects(
        base_directory=base_directory,
        movement=movement,
        body_part=body_part,
        data_type=data_type,
        relevant_axis=relevant_axis,
        output_directory=output_directory
    )
    '''

    ############################################################

    '''
    # Base directory containing the segmented data
    base_directory = "35Internal/processed_data_35_i"
    output_directory = "Features"

    # Configuration for feature extraction
    subject = 1
    movement = "Internal"
    body_part = "Forearm"  # (Shoulder, Pelvis, Palm, Forearm, Torso)
    data_type = "gyr"  # Data type (e.g., acc, gyr, mag)

    # Plot the segmented data
    validate_specific_data(base_directory, subject, body_part, data_type)
    '''

    ############################################################

    '''
    # Features Visualisation
    # Base directory containing feature data
    base_directory = "Features"

    # Configure the parameters
    movement = "Internal"  # Movement type (e.g., "Internal", "External")
    body_part = "Forearm"  # Body part sensor (e.g., "Shoulder", "Forearm")

    # Call the visualization function
    visualize_features(base_directory, movement, body_part)
    '''

    ############################################################

    '''
    # Features Trend Analysis
    base_directory = "Features/Internal"

    # Parameters for the analysis
    subject = 4
    body_part = "Forearm"  # Shoulder, Pelvis, Palm, Forearm, Torso
    data_type = "gyr"  # gyr, acc

    # Perform trend analysis
    trend_analysis_for_subject(base_directory, subject, body_part, data_type)
    '''

############################################################

    '''
    # Merging all features for all subjects into a single file for global analysis
    # Parameters
    base_feature_dir = "Features"
    body_parts = ["Shoulder", "Pelvis", "Torso", "Forearm", "Palm"]
    movement = "Internal"
    sensors = ["acc", "gyr"]
    output_file = f"merged_features_{movement}.csv"

    # Run the merging function
    merge_features(base_feature_dir, body_parts, movement, sensors, output_file)
    '''

    ############################################################

    '''
    # Performing analysis on merged features file (GLOBAL)
    compute_sensor_correlations('merged_features_Internal.csv', 'gyr')
    compute_body_part_sensor_correlations('merged_features_Internal.csv', 'acc')
    '''

    ############################################################

    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------

    '''
    i35_base_directory = "35Internal/processed_data_35_i"
    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        i35_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="35Internal/Features/Extracted/jerk_features_IMU.csv",
        metadata_file=None  # Set to None if not using
    )

    extract_all_rom_features(
        i35_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="35Internal/Features/Extracted/rom_features_IMU.csv",
        metadata_file=None
    )


    extract_all_movement_frequency_features(
        i35_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="35Internal/Features/Extracted/movement_frequency_features_IMU.csv"
    )


    extract_all_mpsd_features(
        i35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35Internal/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        i35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35Internal/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        i35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        i35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35Internal/Features/Extracted/gyr_3axis_features_IMU.csv"
    )
    '''


    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------

    '''
    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "35Internal/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "35Internal/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        i35_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="35Internal/Features/Extracted/integratedEMG_features_EMG.csv",
        metadata_file=None  # or "metadata.csv" if you have it
    )

    extract_all_emg_statistical_features(
        i35_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="35Internal/Features/Extracted/statistical_features_EMG.csv",
        metadata_file=None  # Or "SubjectMetadata.csv" if you have it
    )

    extract_all_mav_features(
        i35_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="35Internal/Features/Extracted/mav_features_EMG.csv",
        metadata_file=None  # or "SubjectDemographics.csv" if you have it
    )

    extract_all_stft_features(i35_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)

    extract_all_wavelength_emg_features(i35_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)

    extract_all_emg_zc_features(i35_base_directory, emg_muscle_folders, 1000,  output_csv="35Internal/Features/Extracted/zeroCrossing_features_EMG.csv")
    '''

    # -----------------------
    # 4. Handling Missing Values
    # -----------------------
    '''
    feature_files = [
        "35Internal/Features/Extracted/jerk_features_IMU.csv",
        "35Internal/Features/Extracted/rom_features_IMU.csv",
        "35Internal/Features/Extracted/movement_frequency_features_IMU.csv",
        "35Internal/Features/Extracted/mpsd_features_IMU.csv",
        "35Internal/Features/Extracted/integratedEMG_features_EMG.csv",
        "35Internal/Features/Extracted/statistical_features_EMG.csv",
        "35Internal/Features/Extracted/mav_features_EMG.csv",
        "35Internal/Features/Extracted/stft_features_EMG.csv",
        "35Internal/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "35Internal/Features/Extracted/gyr_3axis_features_IMU.csv",
        "35Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        "35Internal/Features/Extracted/rt_variability_features_IMU.csv",
        "35Internal/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="35Internal/Features/Extracted/Filled",
        fill_method='both'
    )
    '''

    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------
    '''
    feature_files = [
        "35Internal/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "35Internal/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "35Internal/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "35Internal/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "35Internal/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "35Internal/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "35Internal/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
        ]

    merge_features(feature_files, "35Internal/Features/Extracted/allmerged_features.csv")

    print_missing_values("35Internal/Features/Extracted/allmerged_features.csv")
    '''



    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "35Internal/processed_data_35_i/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "35Internal/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "35Internal/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "35Internal/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)

    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='35i')

    # Adding BORG values to Features file
    features_file = "35Internal/Features/Extracted/allmerged_features.csv"
    repetition_file = "35Internal/Repitition Times/repetition_times_with_borg.csv"
    output_file = "35Internal/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    ############################################################

    # -----------------------
    # 7. Training Regression Models
    # -----------------------

    # Internal35_merged_features_file = "35Internal/Features/Extracted/allmerged_features_with_borg.csv"
    #
    #
    # # Regression Models
    # train_and_evaluate(Internal35_merged_features_file, '35Internal')

    # Classification Models
    # train_and_evaluate_classification_models(merged_features_file, 'Results CSVs/all_features_results.csv', 'Results CSVs/features_imp.csv')





# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #       45 Internal
    # # ****************************

    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------

    '''
    # Parameters for Segmentation
    exercise = 45
    select_internal_external = 'i'
    borg_scale_path = 'borg_scale.npy'
    output_dir = '45Internal/processed_data_45_i'
    numer_of_subject = 34

    # Call the segmentation function
    process_and_segment_data(
        exercise=exercise,
        select_internal_external=select_internal_external,
        borg_scale_path=borg_scale_path,
        output_dir=output_dir,
        numer_of_subject=numer_of_subject
    )
    '''


    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------
    '''
    i45_base_directory = "45Internal/processed_data_45_i"
    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        i45_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="45Internal/Features/Extracted/jerk_features_IMU.csv",
        metadata_file=None  # Set to None if not using
    )

    extract_all_rom_features(
        i45_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="45Internal/Features/Extracted/rom_features_IMU.csv",
        metadata_file=None
    )


    extract_all_movement_frequency_features(
        i45_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="45Internal/Features/Extracted/movement_frequency_features_IMU.csv"
    )


    extract_all_mpsd_features(
        i45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45Internal/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        i45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45Internal/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        i45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        i45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45Internal/Features/Extracted/gyr_3axis_features_IMU.csv"
    )
    '''

    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------
    '''
    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "45Internal/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "45Internal/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        i45_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="45Internal/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        i45_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="45Internal/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        i45_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="45Internal/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(i45_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(i45_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(i45_base_directory, emg_muscle_folders, 1000,  output_csv="45Internal/Features/Extracted/zeroCrossing_features_EMG.csv")
    '''

    # -----------------------
    # 4. Handling Missing Values
    # -----------------------
    '''
    feature_files = [
        "45Internal/Features/Extracted/jerk_features_IMU.csv",
        "45Internal/Features/Extracted/rom_features_IMU.csv",
        "45Internal/Features/Extracted/movement_frequency_features_IMU.csv",
        "45Internal/Features/Extracted/mpsd_features_IMU.csv",
        "45Internal/Features/Extracted/integratedEMG_features_EMG.csv",
        "45Internal/Features/Extracted/statistical_features_EMG.csv",
        "45Internal/Features/Extracted/mav_features_EMG.csv",
        "45Internal/Features/Extracted/stft_features_EMG.csv",
        "45Internal/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "45Internal/Features/Extracted/gyr_3axis_features_IMU.csv",
        "45Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        "45Internal/Features/Extracted/rt_variability_features_IMU.csv",
        "45Internal/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="45Internal/Features/Extracted/Filled",
        fill_method='both'
    )
    '''

    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------
    '''
    feature_files = [
        "45Internal/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "45Internal/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "45Internal/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "45Internal/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "45Internal/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "45Internal/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "45Internal/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
        ]

    merge_features(feature_files, "45Internal/Features/Extracted/allmerged_features.csv")

    print_missing_values("45Internal/Features/Extracted/allmerged_features.csv")
    '''

    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "45Internal/processed_data_45_i/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "45Internal/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "45Internal/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "45Internal/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)


    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='45i')

    # Adding BORG values to Features file
    features_file = "45Internal/Features/Extracted/allmerged_features.csv"
    repetition_file = "45Internal/Repitition Times/repetition_times_with_borg.csv"
    output_file = "45Internal/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    # -----------------------
    # 7. Training Regression Models
    # -----------------------

    # Internal45_merged_features_file = "45Internal/Features/Extracted/allmerged_features_with_borg.csv"
    # #
    # # Regression Models
    # train_and_evaluate(Internal45_merged_features_file, '45Internal')








# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #       55 Internal
    # # ****************************

    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------


    # # Parameters for Segmentation
    # exercise = 55
    # select_internal_external = 'i'
    # borg_scale_path = 'borg_scale.npy'
    # output_dir = '55Internal/processed_data_55_i'
    # numer_of_subject = 34
    #
    # # Call the segmentation function
    # process_and_segment_data(
    #     exercise=exercise,
    #     select_internal_external=select_internal_external,
    #     borg_scale_path=borg_scale_path,
    #     output_dir=output_dir,
    #     numer_of_subject=numer_of_subject
    # )



    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------

    '''
    i55_base_directory = "55Internal/processed_data_55_i"
    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        i55_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="55Internal/Features/Extracted/jerk_features_IMU.csv"
    )

    extract_all_rom_features(
        i55_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="55Internal/Features/Extracted/rom_features_IMU.csv"
    )


    extract_all_movement_frequency_features(
        i55_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="55Internal/Features/Extracted/movement_frequency_features_IMU.csv"
    )


    extract_all_mpsd_features(
        i55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55Internal/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        i55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55Internal/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        i55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        i55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55Internal/Features/Extracted/gyr_3axis_features_IMU.csv"
    )


    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------

    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "55Internal/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "55Internal/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        i55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55Internal/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        i55_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="55Internal/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        i55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55Internal/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(i55_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(i55_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(i55_base_directory, emg_muscle_folders, 1000,  output_csv="55Internal/Features/Extracted/zeroCrossing_features_EMG.csv")
    '''

    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------
    '''
    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "55Internal/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "55Internal/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        i55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55Internal/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        i55_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="55Internal/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        i55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55Internal/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(i55_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(i55_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(i55_base_directory, emg_muscle_folders, 1000,  output_csv="55Internal/Features/Extracted/zeroCrossing_features_EMG.csv")


    # -----------------------
    # 4. Handling Missing Values
    # -----------------------

    feature_files = [
        "55Internal/Features/Extracted/jerk_features_IMU.csv",
        "55Internal/Features/Extracted/rom_features_IMU.csv",
        "55Internal/Features/Extracted/movement_frequency_features_IMU.csv",
        "55Internal/Features/Extracted/mpsd_features_IMU.csv",
        "55Internal/Features/Extracted/integratedEMG_features_EMG.csv",
        "55Internal/Features/Extracted/statistical_features_EMG.csv",
        "55Internal/Features/Extracted/mav_features_EMG.csv",
        "55Internal/Features/Extracted/stft_features_EMG.csv",
        "55Internal/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "55Internal/Features/Extracted/gyr_3axis_features_IMU.csv",
        "55Internal/Features/Extracted/acc_3axis_features_IMU.csv",
        "55Internal/Features/Extracted/rt_variability_features_IMU.csv",
        "55Internal/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="55Internal/Features/Extracted/Filled",
        fill_method='both'
    )
    '''

    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------
    '''
    feature_files = [
        "55Internal/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "55Internal/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "55Internal/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "55Internal/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "55Internal/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "55Internal/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "55Internal/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
        ]

    merge_features(feature_files, "55Internal/Features/Extracted/allmerged_features.csv")

    print_missing_values("55Internal/Features/Extracted/allmerged_features.csv")
    '''

    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "55Internal/processed_data_55_i/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "55Internal/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "55Internal/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "55Internal/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)

    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='55i')

    # Adding BORG values to Features file
    features_file = "55Internal/Features/Extracted/allmerged_features.csv"
    repetition_file = "55Internal/Repitition Times/repetition_times_with_borg.csv"
    output_file = "55Internal/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    # -----------------------
    # 7. Training Regression Models
    # -----------------------


    # Internal55_merged_features_file = "55Internal/Features/Extracted/allmerged_features_with_borg.csv"
    #
    # # Regression Models
    # train_and_evaluate(Internal55_merged_features_file, '55Internal')














# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #       35 External
    # # ****************************

    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------

    '''
    # Parameters for Segmentation
    exercise = 35
    select_internal_external = 'e'
    borg_scale_path = 'borg_scale.npy'
    output_dir = '35External/processed_data_35_e'
    numer_of_subject = 34

    # Call the segmentation function
    process_and_segment_data(
        exercise=exercise,
        select_internal_external=select_internal_external,
        borg_scale_path=borg_scale_path,
        output_dir=output_dir,
        numer_of_subject=numer_of_subject
    )
    '''



    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------
    '''
    e35_base_directory = "35External/processed_data_35_e"

    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        e35_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="35External/Features/Extracted/jerk_features_IMU.csv"
    )

    extract_all_rom_features(
        e35_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="35External/Features/Extracted/rom_features_IMU.csv"
    )


    extract_all_movement_frequency_features(
        e35_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="35External/Features/Extracted/movement_frequency_features_IMU.csv"
    )


    extract_all_mpsd_features(
        e35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35External/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        e35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35External/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        e35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35External/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        e35_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="35External/Features/Extracted/gyr_3axis_features_IMU.csv"
    )


    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------

    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "35External/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "35External/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        e35_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="35External/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        e35_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="35External/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        e35_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="35External/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(e35_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(e35_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(e35_base_directory, emg_muscle_folders, 1000,  output_csv="35External/Features/Extracted/zeroCrossing_features_EMG.csv")



    # -----------------------
    # 4. Handling Missing Values
    # -----------------------
    
    
    feature_files = [
        "35External/Features/Extracted/jerk_features_IMU.csv",
        "35External/Features/Extracted/rom_features_IMU.csv",
        "35External/Features/Extracted/movement_frequency_features_IMU.csv",
        "35External/Features/Extracted/mpsd_features_IMU.csv",
        "35External/Features/Extracted/integratedEMG_features_EMG.csv",
        "35External/Features/Extracted/statistical_features_EMG.csv",
        "35External/Features/Extracted/mav_features_EMG.csv",
        "35External/Features/Extracted/stft_features_EMG.csv",
        "35External/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "35External/Features/Extracted/gyr_3axis_features_IMU.csv",
        "35External/Features/Extracted/acc_3axis_features_IMU.csv",
        "35External/Features/Extracted/rt_variability_features_IMU.csv",
        "35External/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="35External/Features/Extracted/Filled",
        fill_method='both'
    )


    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------

    feature_files = [
        "35External/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "35External/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "35External/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "35External/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "35External/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "35External/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "35External/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
        ]

    merge_features(feature_files, "35External/Features/Extracted/allmerged_features.csv")

    print_missing_values("35External/Features/Extracted/allmerged_features.csv")
    '''


    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "35External/processed_data_35_e/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "35External/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "35External/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "35External/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)

    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='35e')

    # Adding BORG values to Features file
    features_file = "35External/Features/Extracted/allmerged_features.csv"
    repetition_file = "35External/Repitition Times/repetition_times_with_borg.csv"
    output_file = "35External/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    # -----------------------
    # 7. Training Regression Models
    # -----------------------


    # External35_merged_features_file = "35External/Features/Extracted/allmerged_features_with_borg.csv"
    #
    # # Regression Models
    # train_and_evaluate(External35_merged_features_file, '35External')











# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #       45 External
    # # ****************************


    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------

    '''
    # Parameters for Segmentation
    exercise = 45
    select_internal_external = 'e'
    borg_scale_path = 'borg_scale.npy'
    output_dir = '45External/processed_data_45_e'
    numer_of_subject = 34

    # Call the segmentation function
    process_and_segment_data(
        exercise=exercise,
        select_internal_external=select_internal_external,
        borg_scale_path=borg_scale_path,
        output_dir=output_dir,
        numer_of_subject=numer_of_subject
    )


    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------

    e45_base_directory = "45External/processed_data_45_e"

    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        e45_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="45External/Features/Extracted/jerk_features_IMU.csv"
    )

    extract_all_rom_features(
        e45_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="45External/Features/Extracted/rom_features_IMU.csv"
    )


    extract_all_movement_frequency_features(
        e45_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="45External/Features/Extracted/movement_frequency_features_IMU.csv"
    )


    extract_all_mpsd_features(
        e45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45External/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        e45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45External/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        e45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45External/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        e45_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="45External/Features/Extracted/gyr_3axis_features_IMU.csv"
    )


    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------

    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi", "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "45External/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "45External/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        e45_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="45External/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        e45_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="45External/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        e45_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="45External/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(e45_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(e45_base_directory, emg_muscle_folders, sampling_rate=1000, output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(e45_base_directory, emg_muscle_folders, 1000,  output_csv="45External/Features/Extracted/zeroCrossing_features_EMG.csv")

    # -----------------------
    # 4. Handling Missing Values
    # -----------------------

    feature_files = [
        "45External/Features/Extracted/jerk_features_IMU.csv",
        "45External/Features/Extracted/rom_features_IMU.csv",
        "45External/Features/Extracted/movement_frequency_features_IMU.csv",
        "45External/Features/Extracted/mpsd_features_IMU.csv",
        "45External/Features/Extracted/integratedEMG_features_EMG.csv",
        "45External/Features/Extracted/statistical_features_EMG.csv",
        "45External/Features/Extracted/mav_features_EMG.csv",
        "45External/Features/Extracted/stft_features_EMG.csv",
        "45External/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "45External/Features/Extracted/gyr_3axis_features_IMU.csv",
        "45External/Features/Extracted/acc_3axis_features_IMU.csv",
        "45External/Features/Extracted/rt_variability_features_IMU.csv",
        "45External/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="45External/Features/Extracted/Filled",
        fill_method='both'
    )

    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------

    feature_files = [
        "45External/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "45External/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "45External/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "45External/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "45External/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "45External/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "45External/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
    ]

    merge_features(feature_files, "45External/Features/Extracted/allmerged_features.csv")

    print_missing_values("45External/Features/Extracted/allmerged_features.csv")
    '''

    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "45External/processed_data_45_e/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "45External/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "45External/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "45External/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)

    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='45e')

    # Adding BORG values to Features file
    features_file = "45External/Features/Extracted/allmerged_features.csv"
    repetition_file = "45External/Repitition Times/repetition_times_with_borg.csv"
    output_file = "45External/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    # -----------------------
    # 7. Training Regression Models
    # -----------------------

    # External45_merged_features_file = "45External/Features/Extracted/allmerged_features_with_borg.csv"
    #
    # # Regression Models
    # train_and_evaluate(External45_merged_features_file, '45External')










# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #       55 External
    # # ****************************


    # ---------------------------------------------
    # 1. Segmentation of raw data based on repititions
    # ---------------------------------------------

    '''
    # Parameters for Segmentation
    exercise = 55
    select_internal_external = 'e'
    borg_scale_path = 'borg_scale.npy'
    output_dir = '55External/processed_data_55_e'
    numer_of_subject = 34

    # Call the segmentation function
    process_and_segment_data(
        exercise=exercise,
        select_internal_external=select_internal_external,
        borg_scale_path=borg_scale_path,
        output_dir=output_dir,
        numer_of_subject=numer_of_subject
    )

    # -----------------------
    # 2. Extracting IMU Features
    # -----------------------

    e55_base_directory = "55External/processed_data_55_e"

    imu_body_parts = ['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm']
    imu_sampling_rate = 100

    extract_all_jerk_features(
        e55_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="55External/Features/Extracted/jerk_features_IMU.csv"
    )

    extract_all_rom_features(
        e55_base_directory,
        body_parts=['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'],
        sampling_rate=100,
        output_csv="55External/Features/Extracted/rom_features_IMU.csv"
    )

    extract_all_movement_frequency_features(
        e55_base_directory,
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="55External/Features/Extracted/movement_frequency_features_IMU.csv"
    )

    extract_all_mpsd_features(
        e55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55External/Features/Extracted/mpsd_features_IMU.csv"
    )

    extract_all_rt_variability_features(
        e55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55External/Features/Extracted/rt_variability_features_IMU.csv"
    )

    extract_all_acc_3axis_features(
        e55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55External/Features/Extracted/acc_3axis_features_IMU.csv",
        metadata_file=None  # Or specify a metadata file if available
    )

    extract_all_gyr_3axis_features(
        e55_base_directory,
        imu_body_parts,
        imu_sampling_rate,
        output_csv="55External/Features/Extracted/gyr_3axis_features_IMU.csv"
    )

    # -----------------------
    # 3. Extracting EMG Features
    # -----------------------

    emg_muscle_folders = ["emg_deltoideus_anterior", "emg_infraspinatus", "emg_latissimus_dorsi",
                          "emg_pectoralis_major", "emg_deltoideus_posterior", "emg_trapezius_ascendens"]
    emg_stft_output_csv = "55External/Features/Extracted/stft_features_EMG.csv"
    emg_wavelength_output_csv = "55External/Features/Extracted/wavelengthEMG_features_EMG.csv"

    extract_all_integrated_emg_features(
        e55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55External/Features/Extracted/integratedEMG_features_EMG.csv"
    )

    extract_all_emg_statistical_features(
        e55_base_directory,
        emg_muscle_folders,
        fs=1000,
        output_csv="55External/Features/Extracted/statistical_features_EMG.csv"
    )

    extract_all_mav_features(
        e55_base_directory,
        emg_muscle_folders,
        sampling_rate=1000,
        output_csv="55External/Features/Extracted/mav_features_EMG.csv"
    )

    extract_all_stft_features(e55_base_directory, emg_muscle_folders, sampling_rate=1000,
                              output_csv=emg_stft_output_csv)
    extract_all_wavelength_emg_features(e55_base_directory, emg_muscle_folders, sampling_rate=1000,
                                        output_csv=emg_wavelength_output_csv)
    extract_all_emg_zc_features(e55_base_directory, emg_muscle_folders, 1000,
                                output_csv="55External/Features/Extracted/zeroCrossing_features_EMG.csv")



    # -----------------------
    # 4. Handling Missing Values
    # -----------------------

    feature_files = [
        "55External/Features/Extracted/jerk_features_IMU.csv",
        "55External/Features/Extracted/rom_features_IMU.csv",
        "55External/Features/Extracted/movement_frequency_features_IMU.csv",
        "55External/Features/Extracted/mpsd_features_IMU.csv",
        "55External/Features/Extracted/integratedEMG_features_EMG.csv",
        "55External/Features/Extracted/statistical_features_EMG.csv",
        "55External/Features/Extracted/mav_features_EMG.csv",
        "55External/Features/Extracted/stft_features_EMG.csv",
        "55External/Features/Extracted/wavelengthEMG_features_EMG.csv",
        "55External/Features/Extracted/gyr_3axis_features_IMU.csv",
        "55External/Features/Extracted/acc_3axis_features_IMU.csv",
        "55External/Features/Extracted/rt_variability_features_IMU.csv",
        "55External/Features/Extracted/zeroCrossing_features_EMG.csv"
    ]

    filled_dfs = fill_missing_values(
        feature_files,
        output_path="55External/Features/Extracted/Filled",
        fill_method='both'
    )

    # -----------------------
    # 5. Merging Features into a single file
    # -----------------------

    feature_files = [
        "55External/Features/Extracted/Filled/mav_features_EMG_filled.csv",
        "55External/Features/Extracted/Filled/statistical_features_EMG_filled.csv",
        "55External/Features/Extracted/Filled/integratedEMG_features_EMG_filled.csv",
        "55External/Features/Extracted/Filled/mpsd_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/movement_frequency_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/rom_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/jerk_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/stft_features_EMG_filled.csv",
        "55External/Features/Extracted/Filled/wavelengthEMG_features_EMG_filled.csv",
        "55External/Features/Extracted/Filled/gyr_3axis_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/acc_3axis_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/rt_variability_features_IMU_filled.csv",
        "55External/Features/Extracted/Filled/zeroCrossing_features_EMG_filled.csv"
    ]

    merge_features(feature_files, "55External/Features/Extracted/allmerged_features.csv")

    print_missing_values("55External/Features/Extracted/allmerged_features.csv")
    '''



    # --------------------------------------------
    # 6. Adding BORG Column to Merged Features File
    # --------------------------------------------

    '''
    segmented_folder_path = "55External/processed_data_55_e/Upperarm/acc"  # Path to the folder containing all subject files
    timestamps_output_csv = "55External/OutputCSVFiles/repetition_times_all_subjects.csv"  # Path to save the output CSV

    repetition_times = compute_repetition_times_for_all_subjects(segmented_folder_path)
    repetition_times.to_csv(timestamps_output_csv, index=False)
    print(f"Repetition times saved to {timestamps_output_csv}.")

    repetition_file = "55External/OutputCSVFiles/repetition_times_all_subjects.csv"
    borg_file = "Borg data/borg_data.csv"
    output_file = "55External/Repitition Times/repetition_times_with_borg.csv"

    # Mid point time based
    # map_borg_to_repetitions(repetition_file, borg_file, output_file)

    interpolate_borg_values(repetition_file, borg_file, output_file, target_task='55e')

    # Adding BORG values to Features file
    features_file = "55External/Features/Extracted/allmerged_features.csv"
    repetition_file = "55External/Repitition Times/repetition_times_with_borg.csv"
    output_file = "55External/Features/Extracted/allmerged_features_with_borg.csv"
    add_borg_to_features(features_file, repetition_file, output_file)
    '''

    # -----------------------
    # 7. Training Regression Models
    # -----------------------

    # External55_merged_features_file = "55External/Features/Extracted/allmerged_features_with_borg.csv"
    #
    # # Regression Models
    # train_and_evaluate(External55_merged_features_file, '55External')










# ----------------------------------------------------------------------------------------------------------------

    # ****************************
    #    35 Internal + 35 External
    # # ****************************

    # # Read both CSV files
    # internal_df = pd.read_csv('35Internal+35External/35InternalFeatures.csv')
    # external_df = pd.read_csv('35Internal+35External/35ExternalFeatures.csv')
    #
    # combined_df = pd.concat([internal_df, external_df], axis=0, ignore_index=True)
    # combined_df.to_csv('35Internal+35External/35Combined_Internal_External_Features.csv', index=False)
    #
    # Internal35External35_merged_features_file = "35Internal+35External/35Combined_Internal_External_Features.csv"
    # train_and_evaluate(Internal35External35_merged_features_file, 'Internal35External35')










    # ****************************
    #    45 Internal + 45 External
    # # ****************************

    # Read both CSV files
    # internal_df = pd.read_csv('45Internal+45External/45InternalFeatures.csv')
    # external_df = pd.read_csv('45Internal+45External/45ExternalFeatures.csv')
    #
    # combined_df = pd.concat([internal_df, external_df], axis=0, ignore_index=True)
    # combined_df.to_csv('45Internal+45External/45Combined_Internal_External_Features.csv', index=False)
    #
    # Internal35External35_merged_features_file = "45Internal+45External/45Combined_Internal_External_Features.csv"
    # train_and_evaluate(Internal35External35_merged_features_file, 'Internal45External45')













    # ****************************
    #    35 Internal + 35 External + 45 Internal + 45 External
    # # ****************************

    # Read both CSV files
    # internal35_df = pd.read_csv('35I+35E+45I+45E/35InternalFeatures.csv')
    # external35_df = pd.read_csv('35I+35E+45I+45E/35ExternalFeatures.csv')
    #
    # internal45_df = pd.read_csv('35I+35E+45I+45E/45InternalFeatures.csv')
    # external45_df = pd.read_csv('35I+35E+45I+45E/45ExternalFeatures.csv')
    #
    # all_columns_match = (internal35_df.columns.tolist() == external35_df.columns.tolist() ==
    #                      internal45_df.columns.tolist() == external45_df.columns.tolist())
    # print("All columns match:", all_columns_match)
    #
    # # Concatenate all dataframes vertically
    # combined_df = pd.concat([internal35_df, external35_df, internal45_df, external45_df],
    #                         axis=0, ignore_index=True)
    #
    #
    # combined_df.to_csv('35I+35E+45I+45E/CombinedFeatures.csv', index=False)
    # CombinedFeatures = "35I+35E+45I+45E/CombinedFeatures.csv"
    # train_and_evaluate(CombinedFeatures, '35i35e45i45e')






    # ****************************
    #   All Movements
    # # ****************************

    '''
    # Read all CSV files
    internal35_df = pd.read_csv('AllMovements/35InternalFeatures.csv')
    external35_df = pd.read_csv('AllMovements/35ExternalFeatures.csv')

    internal45_df = pd.read_csv('AllMovements/45InternalFeatures.csv')
    external45_df = pd.read_csv('AllMovements/45ExternalFeatures.csv')

    internal55_df = pd.read_csv('AllMovements/55InternalFeatures.csv')
    external55_df = pd.read_csv('AllMovements/55ExternalFeatures.csv')

    all_columns_match = (internal35_df.columns.tolist() == external35_df.columns.tolist() ==
                         internal45_df.columns.tolist() == external45_df.columns.tolist() ==
                         internal55_df.columns.tolist() == external55_df.columns.tolist()
                         )
    print("All columns match:", all_columns_match)

    # Concatenate all dataframes vertically
    combined_df = pd.concat([internal35_df, external35_df, internal45_df, external45_df, internal55_df, external55_df],
                            axis=0, ignore_index=True)


    combined_df.to_csv('AllMovements/CombinedFeatures.csv', index=False)
    CombinedFeatures = "AllMovements/CombinedFeatures.csv"
    train_and_evaluate(CombinedFeatures, 'AllMovements')
    '''






# -----------------------------------------------------------------------------------------------------------------------------

    # ****************************
    #   Baseline Model
    # # **************************

    # CombinedFeatures = "35Internal/Features/Extracted/allmerged_features_with_borg.csv"
    # baseline_metrics = train_baseline_model(CombinedFeatures, '35Internal')






# -----------------------------------------------------------------------------------------------------------------------------

    # ****************************
    #   Feature Importance
    # # *********************


    folder_path = "FeatureImportance"
    feature_df = load_feature_importance_scores(folder_path)

    top_features_df = aggregate_and_select_features(feature_df, top_n=15)

    normalized_df = normalize_scores(top_features_df)

    desired_order = ['35Internal', '35External', '45Internal', '45External', '35+45', 'all', 'Aggregated_Importance']
    normalized_df = normalized_df.reindex(columns=desired_order)

    create_heatmap(normalized_df, title="")







 # ****************************
    #  Plots
 # *********************




'''
# Set style for research paper quality plots
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

# Data preparation
conditions = ['35I', '35E', '45I', '45E', '55I', '55E', '35I-45E', 'All']

# MAE values for each model
mae_values = {
    'XGBoost': [2.1425, 1.6445, 2.0298, 1.3211, 1.8687, 1.2627, 2.0615, 2.0845],
    'SVR': [2.2370, 1.6426, 1.9950, 1.4243, 1.9270, 1.3152, 2.1051, 2.1403],
    'RF': [2.1654, 1.8389, 2.1672, 1.4457, 1.9754, 1.3321, 2.1637, 2.1208]
}

# Set width of bars and positions of the bars
bar_width = 0.25
r1 = np.arange(len(conditions))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create figure and axis with larger size
plt.figure(figsize=(12, 6))

# Create bars
plt.bar(r1, mae_values['XGBoost'], width=bar_width, label='XGBoost', color='#1f77b4', alpha=0.8)
plt.bar(r2, mae_values['SVR'], width=bar_width, label='SVR', color='#ff7f0e', alpha=0.8)
plt.bar(r3, mae_values['RF'], width=bar_width, label='RF', color='#2ca02c', alpha=0.8)

# Customize the plot
plt.xlabel('Conditions', fontsize=15)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=15)
plt.title('')
plt.xticks([r + bar_width for r in range(len(conditions))], conditions, rotation=45, fontsize=15)
plt.yticks(fontsize=15)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('grouped_bar_chart_mae.png', dpi=300, bbox_inches='tight')
plt.close()



# Created/Modified files during execution:
print("Created files:")
print("grouped_bar_chart_mae.png")
print("grouped_bar_chart_rmse.png")
'''















        # *********************************************
        #   Leave One Condition Out
        # # *********************************************



# Read all CSV files
# internal35_df = pd.read_csv('AllMovements/35InternalFeatures.csv')
# external35_df = pd.read_csv('AllMovements/35ExternalFeatures.csv')
# internal45_df = pd.read_csv('AllMovements/45InternalFeatures.csv')
# external45_df = pd.read_csv('AllMovements/45ExternalFeatures.csv')
# internal55_df = pd.read_csv('AllMovements/55InternalFeatures.csv')
# external55_df = pd.read_csv('AllMovements/55ExternalFeatures.csv')
#
# # Add condition column to each dataframe
# internal35_df['condition'] = '35Internal'
# external35_df['condition'] = '35External'
# internal45_df['condition'] = '45Internal'
# external45_df['condition'] = '45External'
# internal55_df['condition'] = '55Internal'
# external55_df['condition'] = '55External'
#
# # Verify columns match
# all_columns_match = (internal35_df.columns.tolist() == external35_df.columns.tolist() ==
#                      internal45_df.columns.tolist() == external45_df.columns.tolist() ==
#                      internal55_df.columns.tolist() == external55_df.columns.tolist()
#                      )
# print("All columns match:", all_columns_match)
#
# # Concatenate all dataframes vertically
# combined_df = pd.concat([internal35_df, external35_df, internal45_df, external45_df, internal55_df, external55_df],
#                         axis=0, ignore_index=True)
#
# # Save combined dataset
# combined_df.to_csv('AllMovements/CombinedFeaturesWithConditionColumn.csv', index=False)
# CombinedFeatures = "AllMovements/CombinedFeaturesWithConditionColumn.csv"
#
# # Print distribution of conditions
# print("\nDistribution of conditions:")
# print(combined_df['condition'].value_counts())


# train_all_test_individual("AllMovements/CombinedFeaturesWithConditionColumn.csv")












# *********************************************
#   Seperate Internal and External
# # *********************************************



# file_path = 'AllMovements/CombinedFeaturesWithConditionColumn.csv'
# internal_metrics, external_metrics = prepare_and_train_models(file_path)




