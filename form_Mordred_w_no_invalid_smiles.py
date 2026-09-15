#!/usr/bin/env python3
"""
Generate Mordred descriptors from RU_SMILES column and save to Excel files.

This script:
1. Reads RU_SMILES.xlsx file
2. Generates Mordred descriptors for each SMILES in RU_SMILES column
3. Skips invalid SMILES that cannot be processed (IMPROVED)
4. Fills NaN values with 0
5. Saves the data with descriptors to data_w_Md_from_ru_smiles.xlsx
6. Saves descriptor information to mordred_descriptor_info.xlsx
7. Saves invalid SMILES log to invalid_smiles_log.xlsx (NEW)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import sys
import warnings

# Suppress RDKit warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def generate_mordred_descriptors(smiles_list, pids):
    """
    Generate Mordred descriptors for a list of SMILES strings.
    Invalid SMILES are skipped and logged.

    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    pids : list
        List of PIDs corresponding to SMILES

    Returns:
    --------
    descriptor_df : pandas.DataFrame
        DataFrame with Mordred descriptors (only valid molecules)
    descriptor_info : pandas.DataFrame
        DataFrame with descriptor information
    invalid_smiles : list
        List of tuples (PID, SMILES, error_message) for invalid molecules
    """

    print("Initializing Mordred calculator...")
    # Create calculator with all 2D descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    print(f"Total number of descriptors: {len(calc.descriptors)}")

    # Get descriptor names
    descriptor_names = [str(d) for d in calc.descriptors]

    # Initialize results dictionary
    results = {f'Md_{i}': [] for i in range(len(descriptor_names))}
    valid_pids = []
    invalid_smiles = []

    print(f"\nProcessing {len(smiles_list)} SMILES strings...")

    for idx, (pid, smiles) in enumerate(zip(pids, smiles_list)):
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(smiles_list)} molecules...")

        # Convert SMILES to molecule
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                error_msg = "Invalid SMILES structure"
                print(f"⚠ Skipping PID {pid}: {error_msg}")
                invalid_smiles.append((pid, smiles, error_msg))
                continue
            
            # Try to calculate descriptors
            desc_values = calc(mol)
            
            # Check if descriptor calculation was successful
            desc_list = []
            has_error = False
            
            for i, value in enumerate(desc_values):
                try:
                    # Convert to float, handle None and error values
                    if value is None or isinstance(value, str):
                        desc_list.append(np.nan)
                    else:
                        desc_list.append(float(value))
                except (ValueError, TypeError) as e:
                    # If conversion fails, mark as having error
                    has_error = True
                    break
            
            # Only add if no errors occurred
            if not has_error:
                for i, val in enumerate(desc_list):
                    results[f'Md_{i}'].append(val)
                valid_pids.append(pid)
            else:
                error_msg = "Descriptor calculation failed"
                print(f"⚠ Skipping PID {pid}: {error_msg}")
                invalid_smiles.append((pid, smiles, error_msg))
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"⚠ Skipping PID {pid}: {error_msg}")
            invalid_smiles.append((pid, smiles, error_msg))
            continue

    print(f"\n✓ Successfully processed: {len(valid_pids)} molecules")
    print(f"✗ Skipped invalid SMILES: {len(invalid_smiles)} molecules")

    # Create DataFrame with descriptors (only valid molecules)
    descriptor_df = pd.DataFrame(results)
    descriptor_df.insert(0, 'PID', valid_pids)

    # Fill NaN values with 0 for all descriptor columns
    descriptor_cols = [col for col in descriptor_df.columns if col != 'PID']
    nan_count = descriptor_df[descriptor_cols].isna().sum().sum()
    if nan_count > 0:
        descriptor_df[descriptor_cols] = descriptor_df[descriptor_cols].fillna(0)
        print(f"✓ Filled {nan_count} NaN values with 0 in descriptor columns")

    # Create descriptor information DataFrame
    descriptor_info = pd.DataFrame({
        'Descriptor_Code': [f'Md_{i}' for i in range(len(descriptor_names))],
        'Descriptor_Name': descriptor_names,
        'Description': [str(d.__class__.__doc__).strip() if d.__class__.__doc__ else 'N/A'
                        for d in calc.descriptors]
    })

    return descriptor_df, descriptor_info, invalid_smiles


def main():
    """Main function to process RU_SMILES.xlsx and generate Mordred descriptors."""

    # File paths
    input_file = 'bo_raw_03.xlsx'
    output_file = 'bo_raw_04.xlsx'
    info_file = 'mordred_descriptor_info.xlsx'
    invalid_log_file = 'invalid_smiles_log.xlsx'

    print("=" * 70)
    print("Mordred Descriptor Generator (Skip Invalid SMILES)")
    print("=" * 70)

    # Read the Excel file
    print(f"\nReading input file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✓ Successfully loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"✗ Error: File '{input_file}' not found!")
        print("Please make sure the file is in the current directory.")
        return
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return

    # Check required columns
    if 'PID' not in df.columns or 'RU_SMILES' not in df.columns:
        print("✗ Error: Required columns 'PID' or 'RU_SMILES' not found!")
        print(f"Available columns: {list(df.columns)}")
        return

    # Extract PID and RU_SMILES
    pids = df['PID'].tolist()
    smiles_list = df['RU_SMILES'].tolist()

    # Generate Mordred descriptors
    print("\n" + "=" * 70)
    print("Generating Mordred Descriptors")
    print("=" * 70)

    descriptor_df, descriptor_info, invalid_smiles = generate_mordred_descriptors(
        smiles_list, pids
    )

    # Merge with original data (only keep valid molecules)
    print("\nMerging descriptors with original data...")
    result_df = pd.merge(df, descriptor_df, on='PID', how='inner')  # Changed to 'inner' join
    print(f"✓ Final dataset contains {len(result_df)} valid molecules")

    # Save results
    print(f"\nSaving results to {output_file}...")
    result_df.to_excel(output_file, index=False)
    print(f"✓ Main data saved successfully!")

    print(f"\nSaving descriptor information to {info_file}...")
    descriptor_info.to_excel(info_file, index=False)
    print(f"✓ Descriptor information saved successfully!")

    # Save invalid SMILES log
    if invalid_smiles:
        print(f"\nSaving invalid SMILES log to {invalid_log_file}...")
        invalid_df = pd.DataFrame(invalid_smiles, columns=['PID', 'SMILES', 'Error_Message'])
        invalid_df.to_excel(invalid_log_file, index=False)
        print(f"✓ Invalid SMILES log saved successfully!")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total molecules in input: {len(pids)}")
    print(f"✓ Valid molecules processed: {len(result_df)}")
    print(f"✗ Invalid SMILES skipped: {len(invalid_smiles)}")
    print(f"Total Mordred descriptors generated: {len(descriptor_info)}")
    print(f"\nOutput files:")
    print(f"  1. {output_file} - Data with Mordred descriptors ({len(result_df)} molecules)")
    print(f"  2. {info_file} - Descriptor information")
    if invalid_smiles:
        print(f"  3. {invalid_log_file} - Invalid SMILES log ({len(invalid_smiles)} entries)")

    if invalid_smiles:
        print(f"\n⚠ Invalid SMILES details (first 10):")
        for pid, smiles, error in invalid_smiles[:10]:
            print(f"  PID: {pid}")
            print(f"    SMILES: {smiles}")
            print(f"    Error: {error}")
        if len(invalid_smiles) > 10:
            print(f"  ... and {len(invalid_smiles) - 10} more (see {invalid_log_file})")

    print("\n" + "=" * 70)
    print("Processing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
