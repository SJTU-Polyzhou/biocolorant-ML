#!/usr/bin/env python3
"""
Generate Mordred descriptors for my_test_dyes.xlsx file and save to Excel files.

This script:
1. Reads my_test_dyes.xlsx file
2. Generates Mordred descriptors for each SMILES in SMILES column
3. Fills NaN values with 0
4. Saves the data with descriptors to my_test_dyes_with_Md.xlsx
5. Saves descriptor information to mordred_descriptor_info.xlsx
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import sys
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# 35个指定的描述符
selected_bits = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]


def generate_mordred_descriptors(smiles_list, compound_names):
    """
    Generate Mordred descriptors for a list of SMILES strings.

    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    compound_names : list
        List of compound names corresponding to SMILES

    Returns:
    --------
    descriptor_df : pandas.DataFrame
        DataFrame with Mordred descriptors
    descriptor_info : pandas.DataFrame
        DataFrame with descriptor information
    invalid_compounds : list
        List of tuples (compound_name, smiles) for invalid SMILES
    """

    print("Initializing Mordred calculator...")
    # Create calculator with all 2D descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    print(f"Total number of descriptors: {len(calc.descriptors)}")

    # Get descriptor names
    descriptor_names = [str(d) for d in calc.descriptors]

    # Initialize results dictionary
    results = {f'Md_{i}': [] for i in range(len(descriptor_names))}
    valid_names = []
    invalid_compounds = []

    print(f"\nProcessing {len(smiles_list)} compounds...")

    for idx, (name, smiles) in enumerate(zip(compound_names, smiles_list)):
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(smiles_list)} compounds...")

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            print(f"Warning: Invalid SMILES for compound '{name}': {smiles}")
            invalid_compounds.append((name, smiles))
            # Add NaN values for this compound
            for i in range(len(descriptor_names)):
                results[f'Md_{i}'].append(np.nan)
            valid_names.append(name)
        else:
            # Calculate descriptors
            try:
                desc_values = calc(mol)

                # Store descriptor values
                for i, value in enumerate(desc_values):
                    try:
                        # Convert to float, handle None and error values
                        if value is None or isinstance(value, str):
                            results[f'Md_{i}'].append(np.nan)
                        else:
                            results[f'Md_{i}'].append(float(value))
                    except (ValueError, TypeError):
                        results[f'Md_{i}'].append(np.nan)

                valid_names.append(name)
            except Exception as e:
                print(f"Error calculating descriptors for compound '{name}': {e}")
                invalid_compounds.append((name, smiles))
                for i in range(len(descriptor_names)):
                    results[f'Md_{i}'].append(np.nan)
                valid_names.append(name)

    print(f"\nCompleted processing. Invalid compounds: {len(invalid_compounds)}")

    # Create DataFrame with descriptors
    descriptor_df = pd.DataFrame(results)
    descriptor_df.insert(0, 'Name', valid_names)

    # Fill NaN values with 0 for all descriptor columns
    descriptor_cols = [col for col in descriptor_df.columns if col != 'Name']
    nan_count = descriptor_df[descriptor_cols].isna().sum().sum()
    descriptor_df[descriptor_cols] = descriptor_df[descriptor_cols].fillna(0)
    print(f"✓ Filled {nan_count} NaN values with 0 in descriptor columns")

    # Create descriptor information DataFrame
    descriptor_info = pd.DataFrame({
        'Descriptor_Code': [f'Md_{i}' for i in range(len(descriptor_names))],
        'Descriptor_Name': descriptor_names,
        'Description': [str(d.__class__.__doc__).strip()[:200] + '...' if d.__class__.__doc__ else 'N/A'
                        for d in calc.descriptors]
    })

    return descriptor_df, descriptor_info, invalid_compounds


def extract_model_descriptors(descriptor_df, selected_bits):
    """
    Extract only the descriptors used in the MLR model.

    Parameters:
    -----------
    descriptor_df : pandas.DataFrame
        DataFrame with all Mordred descriptors
    selected_bits : list
        List of descriptor indices used in the model

    Returns:
    --------
    filtered_df : pandas.DataFrame
        DataFrame with only model descriptors
    """
    # Get the descriptor columns needed for the model
    model_descriptors = [f'Md_{bit}' for bit in selected_bits]

    # Check which descriptors are available
    available_descriptors = []
    missing_descriptors = []

    for desc in model_descriptors:
        if desc in descriptor_df.columns:
            available_descriptors.append(desc)
        else:
            missing_descriptors.append(desc)

    print(f"Model requires {len(model_descriptors)} descriptors")
    print(f"Found {len(available_descriptors)} descriptors in data")

    if missing_descriptors:
        print(f"Missing {len(missing_descriptors)} descriptors:")
        for desc in missing_descriptors[:10]:
            print(f"  - {desc}")
        if len(missing_descriptors) > 10:
            print(f"  ... and {len(missing_descriptors) - 10} more")

    # Create filtered DataFrame
    filtered_df = pd.DataFrame()
    filtered_df['Name'] = descriptor_df['Name']

    for desc in model_descriptors:
        if desc in descriptor_df.columns:
            filtered_df[desc] = descriptor_df[desc]
        else:
            # If descriptor is missing, fill with 0
            print(f"Warning: Adding missing column {desc} with zeros")
            filtered_df[desc] = 0.0

    return filtered_df, available_descriptors, missing_descriptors


def main():
    """Main function to process my_test_dyes.xlsx and generate Mordred descriptors."""

    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'my_test_dyes.xlsx')
    output_dir = os.path.join(base_dir, 'descriptor_results')
    output_file = os.path.join(output_dir, 'my_test_dyes_with_Md.xlsx')
    model_output_file = os.path.join(output_dir, 'my_test_dyes_model_descriptors.xlsx')
    info_file = os.path.join(output_dir, 'mordred_descriptor_info.xlsx')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Mordred Descriptor Generator for my_test_dyes.xlsx")
    print("=" * 70)

    # Read the Excel file
    print(f"\nReading input file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"Successfully loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Please make sure the file is in the current directory.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Check required columns
    if 'SMILES' not in df.columns:
        print("Error: Required column 'SMILES' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Get compound names - use Name column if available, otherwise create from index
    if 'Name' in df.columns:
        compound_names = df['Name'].tolist()
        print(f"Using 'Name' column for compound names")
    elif 'Compound' in df.columns:
        compound_names = df['Compound'].tolist()
        print(f"Using 'Compound' column for compound names")
    else:
        compound_names = [f'Compound_{i + 1}' for i in range(len(df))]
        print(f"Created compound names from index")
        df['Name'] = compound_names

    # Extract SMILES
    smiles_list = df['SMILES'].tolist()

    # Generate Mordred descriptors
    print("\n" + "=" * 70)
    print("Generating Mordred Descriptors")
    print("=" * 70)

    descriptor_df, descriptor_info, invalid_compounds = generate_mordred_descriptors(
        smiles_list, compound_names
    )

    # Merge with original data
    print("\nMerging descriptors with original data...")

    # Keep all original columns
    result_df = df.copy()

    # Add descriptor columns
    for col in descriptor_df.columns:
        if col != 'Name':  # Already have Name from original df
            result_df[col] = descriptor_df[col]

    # Save full results (all descriptors)
    print(f"\nSaving full descriptor results to {output_file}...")
    result_df.to_excel(output_file, index=False)
    print(f"✓ Full data with {descriptor_df.shape[1] - 1} descriptors saved!")

    # Extract only the descriptors needed for MLR model
    print("\n" + "=" * 70)
    print("Extracting MLR Model Descriptors")
    print("=" * 70)

    model_descriptor_df, available, missing = extract_model_descriptors(descriptor_df, selected_bits)

    # Merge model descriptors with original data
    model_result_df = df.copy()
    for col in model_descriptor_df.columns:
        if col != 'Name':
            model_result_df[col] = model_descriptor_df[col]

    # Save model descriptor results
    print(f"\nSaving model descriptor results to {model_output_file}...")
    model_result_df.to_excel(model_output_file, index=False)
    print(f"✓ Model data with {len(selected_bits)} descriptors saved!")

    # Save descriptor information
    print(f"\nSaving descriptor information to {info_file}...")
    descriptor_info.to_excel(info_file, index=False)
    print(f"✓ Descriptor information saved successfully!")

    # Create a summary file
    summary_file = os.path.join(output_dir, 'processing_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MORDRED DESCRIPTOR PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Input File: {input_file}\n")
        f.write(f"Total Compounds Processed: {len(df)}\n")
        f.write(f"Invalid SMILES: {len(invalid_compounds)}\n")
        f.write(f"Total Mordred Descriptors Generated: {descriptor_df.shape[1] - 1}\n")
        f.write(f"MLR Model Descriptors Required: {len(selected_bits)}\n")
        f.write(f"MLR Model Descriptors Found: {len(available)}\n")
        f.write(f"MLR Model Descriptors Missing: {len(missing)}\n\n")

        f.write("Output Files:\n")
        f.write(f"  1. {output_file} - Full data with all descriptors\n")
        f.write(f"  2. {model_output_file} - Data with MLR model descriptors only\n")
        f.write(f"  3. {info_file} - Descriptor information\n")
        f.write(f"  4. {summary_file} - This summary file\n\n")

        if invalid_compounds:
            f.write("Invalid Compounds:\n")
            f.write("-" * 40 + "\n")
            for name, smiles in invalid_compounds:
                f.write(f"  {name}: {smiles}\n")

        if missing:
            f.write("\nMissing MLR Model Descriptors:\n")
            f.write("-" * 40 + "\n")
            for desc in missing:
                f.write(f"  {desc}\n")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total compounds processed: {len(df)}")
    print(f"Invalid SMILES: {len(invalid_compounds)}")
    print(f"Total Mordred descriptors generated: {descriptor_df.shape[1] - 1}")
    print(f"MLR model descriptors required: {len(selected_bits)}")
    print(f"MLR model descriptors available: {len(available)}")
    print(f"MLR model descriptors missing: {len(missing)}")
    print(f"\nOutput files saved to '{output_dir}' directory:")
    print(f"  1. {os.path.basename(output_file)} - Full data with all descriptors")
    print(f"  2. {os.path.basename(model_output_file)} - Data with MLR model descriptors")
    print(f"  3. {os.path.basename(info_file)} - Descriptor information")
    print(f"  4. {os.path.basename(summary_file)} - Processing summary")

    if invalid_compounds:
        print(f"\nInvalid compounds (first 5):")
        for name, smiles in invalid_compounds[:5]:
            print(f"  {name}: {smiles}")
        if len(invalid_compounds) > 5:
            print(f"  ... and {len(invalid_compounds) - 5} more")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Use 'my_test_dyes_model_descriptors.xlsx' for MLR prediction")
    print("2. Run the MLR prediction script:")
    print("   python candidate_comparison_plot.py")
    print("3. Make sure to update the input file path in prediction script")
    print("=" * 70)

    return model_result_df


def show_preview(result_df):
    """显示生成数据的预览"""
    if result_df is not None:
        print("\nPreview of generated data (first 3 rows):")
        print("-" * 70)

        # 选择要显示的列
        display_cols = ['Name', 'SMILES']

        # 添加前3个描述符列
        if len(selected_bits) >= 3:
            display_cols.extend([f'Md_{selected_bits[0]}',
                                 f'Md_{selected_bits[1]}',
                                 f'Md_{selected_bits[2]}'])

        # 检查这些列是否存在于结果中
        available_cols = [col for col in display_cols if col in result_df.columns]

        if available_cols:
            print(result_df[available_cols].head(3).to_string(index=False))
        else:
            print("No preview columns available")

        # Check for experimental values
        exp_cols = [col for col in result_df.columns if 'experimental' in col.lower() or
                    'λmax' in col.lower() or 'absorption' in col.lower()]
        if exp_cols:
            print(f"\nFound experimental value columns: {exp_cols}")


if __name__ == "__main__":
    # Check if Mordred and RDKit are installed
    try:
        from rdkit import Chem
        from mordred import Calculator, descriptors

        print("✓ Mordred and RDKit are installed")
    except ImportError as e:
        print("❌ Error: Required packages not installed!")
        print("Please install Mordred and RDKit:")
        print("pip install mordred rdkit")
        sys.exit(1)

    # Check if input file exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'my_test_dyes.xlsx')
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please make sure 'my_test_dyes.xlsx' is in the same directory as this script.")
        print("\nDirectory contents:")
        for file in os.listdir(base_dir):
            if file.endswith('.xlsx'):
                print(f"  - {file}")
        sys.exit(1)

    # Run main function
    print("=" * 70)
    print("Starting Mordred Descriptor Generation")
    print("=" * 70)

    result_df = main()

    # Show preview
    show_preview(result_df)

    print("\n" + "=" * 70)
    print("Script completed successfully!")
    print("=" * 70)